"""GUI のサーバー（FastAPI）。画面は static/ の静的ファイルを配るだけ。

守り（ほかのサイトから解析を投げ込まれない・結果を読まれないため）:
- Host が 127.0.0.1:PORT か localhost:PORT でなければ 400（DNS リバインディング）
- /api/ の下は合言葉の Cookie（または X-Chordgen-Token）が合っていなければ 403
- 状態を変える要求（POST・PUT・PATCH・DELETE）は、Origin があればそれも確かめる
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import cookie_parser
from starlette.staticfiles import StaticFiles

from chordgen import paths
from chordgen.gui import (
    ENV_FAKE,
    MAX_LYRICS_CHARS,
    MAX_UPLOAD_BYTES,
    TOKEN_COOKIE,
    TOKEN_HEADER,
)
from chordgen.gui import info as info_mod
from chordgen.gui import outputs as outputs_mod
from chordgen.gui.jobs import Job, JobManager, JobOptions, NotCancellable, python_command
from chordgen.pipeline import (
    BASS_MODES,
    CHORD_INPUT_MODES,
    CHORD_MODELS,
    MAX_BARS_PER_LINE,
    PipelineOptions,
    safe_filename,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_FIELD_BYTES = 4 * 1024 * 1024  # ファイル以外の欄（歌詞など）の大きさの上限
MAX_TEXT_FIELD = 300  # 題名・アーティストの文字数の上限
MAX_URL_LENGTH = 4096
SSE_POLL_SEC = 0.2
_LANGUAGE_RE = re.compile(r"^[A-Za-z][A-Za-z_-]{0,31}$")
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,10}$")
_STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass
class ServerConfig:
    port: int
    host: str = "127.0.0.1"
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    fake: bool = field(default_factory=lambda: os.environ.get(ENV_FAKE, "") == "1")
    manager: JobManager | None = None  # テストで差し替える
    device_probe: info_mod.DeviceProbe | None = None
    request_shutdown: Callable[[], None] | None = None  # サーバーを終わらせる（__main__ が uvicorn に向けて設定する）
    open_path: Callable[[Path, bool], None] | None = None  # フォルダを開く（テストで差し替える）
    ping_interval: float = 15.0

    def allowed_hosts(self) -> set[str]:
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if self.host not in ("127.0.0.1", "localhost", "0.0.0.0", "::", ""):
            hosts.add(f"{self.host}:{self.port}")
        return hosts

    def allowed_origins(self) -> set[str]:
        return {f"http://{h}" for h in self.allowed_hosts()}


def api_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status,
                        headers={"Cache-Control": "no-store"})


def _token_ok(given: str | None, token: str) -> bool:
    return bool(given) and secrets.compare_digest(given.encode("utf-8"), token.encode("utf-8"))


class _Guard:
    """Host・合言葉・Origin を確かめ、すべての応答に守りのヘッダを足す ASGI のミドルウェア。"""

    def __init__(self, app, config: ServerConfig):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        path: str = scope.get("path", "")
        method: str = scope.get("method", "GET").upper()
        is_api = path == "/api" or path.startswith("/api/")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                hs = list(message.get("headers") or [])
                names = {k.lower() for k, _ in hs}
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"frame-ancestors 'none'"),
                    (b"cross-origin-resource-policy", b"same-origin"),
                ]
                if b"cache-control" not in names:
                    extra.append((b"cache-control", b"no-store" if is_api or path == "/" else b"no-cache"))
                hs += [(k, v) for k, v in extra if k not in names]
                message = {**message, "headers": hs}
            await send(message)

        if headers.get("host", "").lower() not in self.config.allowed_hosts():
            resp = Response("Bad Host header\n", status_code=400, media_type="text/plain")
            return await resp(scope, receive, send_with_headers)

        if method in _STATE_CHANGING:
            origin = headers.get("origin")
            if origin is not None and origin.lower() not in self.config.allowed_origins():
                resp = api_error(403, "BAD_ORIGIN", "ほかのサイトからの要求は受け付けません。")
                return await resp(scope, receive, send_with_headers)

        if is_api:
            cookie = cookie_parser(headers.get("cookie", "")).get(TOKEN_COOKIE)
            given = headers.get(TOKEN_HEADER.lower()) or cookie
            if not _token_ok(given, self.config.token):
                resp = api_error(403, "FORBIDDEN", "合言葉が合いません。コンソールに出ている URL から開き直してください。")
                return await resp(scope, receive, send_with_headers)

        return await self.app(scope, receive, send_with_headers)


_NEED_TOKEN_PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>chordgen</title>
<style>body{{font-family:system-ui,sans-serif;max-width:40em;margin:4em auto;padding:0 1em;line-height:1.7;color:#222;background:#fff}}
@media (prefers-color-scheme: dark){{body{{color:#ddd;background:#1b1b1b}}}}code{{word-break:break-all}}</style></head>
<body><h1>chordgen</h1><p>{message}</p>
<p>run.bat を実行したコンソールに <code>http://127.0.0.1:…/?t=…</code> の形の URL が出ています。その URL をブラウザで開いてください。</p>
</body></html>
"""


def _page(message: str, status: int) -> HTMLResponse:
    return HTMLResponse(_NEED_TOKEN_PAGE.format(message=message), status_code=status)


# ---- フォームの読み取り ----


class FormError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _text_field(form, name: str, *, limit: int = MAX_TEXT_FIELD) -> str | None:
    v = form.get(name)
    if v is None:
        return None
    if not isinstance(v, str):
        raise FormError("INVALID_FIELD", f"{name} は文字で送ってください。")
    v = unicodedata.normalize("NFC", v).strip()
    if len(v) > limit:
        raise FormError("INVALID_FIELD", f"{name} が長すぎます（{limit} 文字まで）。")
    if any(ord(c) < 0x20 for c in v):
        raise FormError("INVALID_FIELD", f"{name} に使えない文字が入っています。")
    return v or None


def _choice_field(form, name: str, choices: tuple[str, ...], default: str, label: str) -> str:
    v = _text_field(form, name)
    if v is None:
        return default
    if v not in choices:
        raise FormError("INVALID_OPTION", f"{label}の値が正しくありません: {v}（{' / '.join(choices)} のどれか）")
    return v


def _bool_field(form, name: str, default: bool, label: str) -> bool:
    v = _text_field(form, name)
    if v is None:
        return default
    s = v.lower()
    if s in ("true", "1", "on", "yes"):
        return True
    if s in ("false", "0", "off", "no"):
        return False
    raise FormError("INVALID_OPTION", f"{label}は true か false で指定してください: {v}")


def parse_options(form) -> JobOptions:
    from chordgen.gui.info import static_info

    d = static_info()["defaults"]
    o = JobOptions(
        chord_input=_choice_field(form, "chordInput", CHORD_INPUT_MODES, d["chordInput"], "コード認識に入れる音"),
        bass=_choice_field(form, "bass", BASS_MODES, d["bass"], "ベースの使い方"),
        snap=_bool_field(form, "snap", d["snap"], "拍にそろえるか"),
        model=_choice_field(form, "model", CHORD_MODELS, d["model"], "コード認識のモデル"),
        keep_work=_bool_field(form, "keepWork", False, "作業ファイルを残すか"),
    )
    lang = _text_field(form, "language", limit=32)
    if lang is not None:
        if not _LANGUAGE_RE.match(lang):
            raise FormError("INVALID_OPTION", f"歌詞の言語の書き方が正しくありません: {lang}（例 ja）")
        o.language = lang
    bpl = _text_field(form, "barsPerLine", limit=8)
    if bpl is not None:
        try:
            n = int(bpl)
        except ValueError:
            raise FormError("INVALID_OPTION", f"1 行の小節数は整数で指定してください: {bpl}") from None
        if not 1 <= n <= MAX_BARS_PER_LINE:
            raise FormError("INVALID_OPTION", f"1 行の小節数は 1〜{MAX_BARS_PER_LINE} にしてください: {n}")
        o.bars_per_line = n
    try:
        PipelineOptions(chord_input=o.chord_input, bass_mode=o.bass, snap=o.snap, chord_model=o.model,
                        lyrics_language=o.language, bars_per_line=o.bars_per_line).validate()
    except ValueError as e:
        raise FormError("INVALID_OPTION", str(e)) from None
    return o


def upload_name(filename: str) -> str:
    """アップロードされたファイルの名前から、保存に使える名前を作る（拡張子は残す）。"""
    base = PureWindowsPath(filename.replace("/", "\\")).name  # どちらの区切りでも最後の部分だけ
    base = unicodedata.normalize("NFC", base).lstrip(".")  # . で始まる名前（隠しファイル・..）にしない
    stem, dot, ext = base.rpartition(".")
    suffix = f".{ext}" if dot and stem and _SAFE_SUFFIX_RE.match(f".{ext}") else ""
    if not suffix:
        stem = base
    return safe_filename(stem, max_len=100, fallback="audio") + suffix.lower()


def _display_name(filename: str) -> str:
    base = PureWindowsPath(filename.replace("/", "\\")).name
    base = "".join(c for c in unicodedata.normalize("NFC", base) if ord(c) >= 0x20)
    return base[:255] or "audio"


def _copy_upload(src, dest: Path, limit: int) -> int:
    """アップロードの一時ファイルを dest に写す。limit を超えたら FormError。"""
    src.seek(0)
    n = 0
    with dest.open("wb") as f:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            n += len(chunk)
            if n > limit:
                raise FormError("FILE_TOO_LARGE", f"音源ファイルが大きすぎます（{limit // (1024 * 1024)} MB まで）。")
            f.write(chunk)
    return n


# ---- フォルダを開く ----


def open_in_file_manager(path: Path, select: bool) -> None:
    """エクスプローラーなどでフォルダを開く。select なら path のファイルを選んだ状態にする。"""
    if os.name == "nt":
        if select:
            # explorer は /select, の後ろを自分で解釈するので、一つの文字列で渡す（shell は使わない）
            subprocess.Popen(f'explorer /select,"{path}"', shell=False)
        else:
            subprocess.Popen(["explorer", str(path)], shell=False)
        return
    if sys.platform == "darwin":
        cmd = ["open", "-R", str(path)] if select else ["open", str(path)]
    else:
        opener = shutil.which("xdg-open")
        if opener is None:
            raise OSError("xdg-open が見つかりません")
        cmd = [opener, str(path.parent if select else path)]
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     shell=False, start_new_session=True)


# ---- アプリ ----


def _sse(name: str, data: Any, seq: int | None = None) -> str:
    head = f"id: {seq}\n" if seq is not None else ""
    return f"{head}event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(config: ServerConfig) -> FastAPI:
    gui_dir = paths.work_dir() / "gui"
    if config.manager is None:
        command = python_command("chordgen.gui.fake_cli") if config.fake else None
        config.manager = JobManager(gui_dir=gui_dir, command=command, cwd=paths.app_home())
    manager = config.manager
    if config.device_probe is None:
        config.device_probe = (
            info_mod.DeviceProbe(fixed={"kind": "cpu", "name": "偽のジョブ（CHORDGEN_GUI_FAKE）", "torchAvailable": False},
                                 delay=1.0)
            if config.fake else info_mod.DeviceProbe()
        )
    probe = config.device_probe
    opener = config.open_path or open_in_file_manager
    shutdown_started = threading.Event()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        probe.start()
        try:
            yield
        finally:
            await run_in_threadpool(manager.close)

    app = FastAPI(title="chordgen", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.config = config

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path == "/api" or request.url.path.startswith("/api/"):
            code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
            msg = {404: "見つかりません。", 405: "この操作はできません。"}.get(exc.status_code, str(exc.detail))
            return api_error(exc.status_code, code, msg)
        return Response(str(exc.detail), status_code=exc.status_code, media_type="text/plain")

    # ---- 画面 ----
    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        t = request.query_params.get("t")
        if t is not None:
            if not _token_ok(t, config.token):
                return _page("合言葉が違います（サーバーを起動し直すと変わります）。", 403)
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(TOKEN_COOKIE, config.token, httponly=True, samesite="strict", path="/")
            return resp
        if not _token_ok(request.cookies.get(TOKEN_COOKIE), config.token):
            return _page("コンソールに出ている URL から開いてください。", 403)
        index_html = STATIC_DIR / "index.html"
        if not index_html.is_file():
            return _page("画面のファイル（chordgen/gui/static/index.html）がありません。", 500)
        return FileResponse(index_html, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})

    @app.get("/index.html", include_in_schema=False)
    async def index_html_redirect():
        return RedirectResponse("/", status_code=303)

    # ---- 情報 ----
    @app.get("/api/info")
    async def api_info():
        d = info_mod.static_info()
        d["device"] = probe.get()
        d["mediaDlAvailable"] = True if config.fake else await run_in_threadpool(info_mod.media_dl_available)
        d["limits"] = info_mod.limits()
        d["fake"] = config.fake
        return d

    # ---- ジョブ ----
    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request):
        ctype = request.headers.get("content-type", "")
        if not ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            return api_error(400, "BAD_FORM", "multipart/form-data で送ってください。")
        try:
            if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES + MAX_FIELD_BYTES:
                return api_error(400, "FILE_TOO_LARGE", f"音源ファイルが大きすぎます（{MAX_UPLOAD_BYTES // (1024 * 1024)} MB まで）。")
        except ValueError:
            return api_error(400, "BAD_FORM", "Content-Length が正しくありません。")
        try:
            form = await request.form(max_files=2, max_fields=40, max_part_size=MAX_FIELD_BYTES)
        except Exception as e:  # multipart の形が壊れている・大きすぎる欄
            return api_error(400, "BAD_FORM", f"フォームを読めませんでした: {e}")
        upload_dir: Path | None = None
        try:
            file = form.get("file")
            has_file = isinstance(file, UploadFile) and bool(file.filename)
            if file is not None and not isinstance(file, UploadFile) and str(file).strip():
                raise FormError("BAD_FORM", "file はファイルとして送ってください。")
            url = _text_field(form, "url", limit=MAX_URL_LENGTH)
            if has_file and url:
                raise FormError("BOTH_SOURCES", "音源ファイルと URL は、どちらか一方だけを指定してください。")
            if not has_file and not url:
                raise FormError("NO_SOURCE", "音源ファイルか URL を指定してください。")
            if url:
                from chordgen.fetch import is_url

                if not is_url(url):
                    raise FormError("INVALID_URL", "URL は http:// か https:// で始めてください。")
            options = parse_options(form)
            title = _text_field(form, "title")
            artist = _text_field(form, "artist")
            lyrics = form.get("lyrics")
            if lyrics is not None and not isinstance(lyrics, str):
                raise FormError("INVALID_FIELD", "lyrics は文字で送ってください。")
            if lyrics is not None and len(lyrics) > MAX_LYRICS_CHARS:
                raise FormError("LYRICS_TOO_LONG", f"歌詞が長すぎます（{MAX_LYRICS_CHARS} 文字まで）。")
            lyrics = lyrics if lyrics and lyrics.strip() else None

            job_id = manager.new_id()
            upload_dir = manager.upload_dir(job_id)
            upload_dir.mkdir(parents=True, exist_ok=True)
            lyrics_path = None
            if lyrics is not None:
                lyrics_path = upload_dir / "lyrics.txt"
                lyrics_path.write_text(lyrics.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")
            if has_file:
                assert isinstance(file, UploadFile)
                name = upload_name(file.filename or "audio")
                if lyrics_path is not None and name.lower() == lyrics_path.name:
                    name = f"audio_{name}"
                dest = upload_dir / name
                size = await run_in_threadpool(_copy_upload, file.file, dest, MAX_UPLOAD_BYTES)
                if size == 0:
                    raise FormError("EMPTY_FILE", "音源ファイルが空です。")
                source, kind, display = str(dest.resolve()), "file", _display_name(file.filename or name)
            else:
                assert url is not None
                source, kind, display = url, "url", url
            options.has_lyrics = lyrics_path is not None
            job = Job(id=job_id, source_kind=kind, source=source, source_name=display, options=options,
                      title=title, artist=artist, lyrics_path=lyrics_path, upload_dir=upload_dir,
                      work_dir=manager.job_work_dir(job_id))
            try:
                manager.submit(job)
            except RuntimeError as e:
                raise FormError("SHUTTING_DOWN", str(e)) from None
        except FormError as e:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return api_error(400, e.code, e.message)
        except BaseException:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        finally:
            await form.close()
        return JSONResponse({"id": job.id, "job": manager.job_api(job.id)}, status_code=201)

    @app.get("/api/jobs")
    async def list_jobs():
        return manager.list_api()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        d = manager.job_api(job_id, with_events=True)
        if d is None:
            return api_error(404, "JOB_NOT_FOUND", "そのジョブはありません。")
        return d

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        try:
            await run_in_threadpool(manager.cancel, job_id)
        except KeyError:
            return api_error(404, "JOB_NOT_FOUND", "そのジョブはありません。")
        except NotCancellable:
            return api_error(409, "NOT_CANCELLABLE", "このジョブはもう終わっています。")
        return manager.job_api(job_id)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        snap = manager.snapshot(job_id)
        if snap is None:
            return api_error(404, "JOB_NOT_FOUND", "そのジョブはありません。")
        last_id = request.headers.get("last-event-id", "")
        resume = int(last_id) if last_id.isdigit() else None

        async def stream():
            entries, status, terminal, last = snap
            if resume is None:
                # 繋いだ時点までの記録を流し直す（古い status は送らず、最後に今の status を 1 回）
                for seq, name, data in entries:
                    if name in ("progress", "warning", "log"):
                        yield _sse(name, data, seq)
                yield _sse("status", status, last)
                cursor = last
                if terminal:
                    return
            else:
                cursor = resume
            last_ping = time.monotonic()
            while True:
                new, status, terminal = manager.entries_after(job_id, cursor)
                for seq, name, data in new:
                    if name != "error":
                        yield _sse(name, data, seq)
                    cursor = seq
                if terminal:
                    if not new:
                        yield _sse("status", status, cursor)
                    return
                if manager.closing:
                    yield _sse("status", status, cursor)
                    return
                if time.monotonic() - last_ping >= config.ping_interval:
                    last_ping = time.monotonic()
                    yield ": ping\n\n"
                await asyncio.sleep(SSE_POLL_SEC)

        return StreamingResponse(stream(), media_type="text/event-stream; charset=utf-8",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    # ---- 出力 ----
    @app.get("/api/outputs")
    async def list_outputs():
        return await run_in_threadpool(outputs_mod.list_outputs, paths.output_dir())

    def _resolve(name: str) -> Path:
        return outputs_mod.resolve_output(paths.output_dir(), name)

    @app.get("/api/outputs/{name}")
    async def get_output(name: str, download: str | None = None):
        try:
            p = _resolve(name)
        except outputs_mod.BadName:
            return api_error(400, "BAD_NAME", "ファイル名が正しくありません。")
        if not p.is_file():
            return api_error(404, "OUTPUT_NOT_FOUND", "そのファイルはありません。")
        headers = {"Cache-Control": "no-store"}
        if download not in (None, "", "0", "false"):
            headers["Content-Disposition"] = outputs_mod.content_disposition(p.name)
        return FileResponse(p, media_type="application/json; charset=utf-8", headers=headers)

    @app.delete("/api/outputs/{name}")
    async def delete_output(name: str):
        try:
            p = _resolve(name)
        except outputs_mod.BadName:
            return api_error(400, "BAD_NAME", "ファイル名が正しくありません。")
        if not p.is_file():
            return api_error(404, "OUTPUT_NOT_FOUND", "そのファイルはありません。")
        try:
            p.unlink()
        except OSError as e:
            return api_error(500, "DELETE_FAILED", f"消せませんでした: {e}")
        return {"deleted": name}

    @app.post("/api/open-folder")
    async def open_folder(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return api_error(400, "BAD_BODY", "JSON のオブジェクトで送ってください。")
        target, name = body.get("target"), body.get("name")
        if target == "output":
            base = paths.output_dir()
        elif target == "work":
            base = paths.work_dir()
        else:
            return api_error(400, "BAD_TARGET", "target は output か work にしてください。")
        path, select = base, False
        if name is not None:
            if target != "output" or not isinstance(name, str):
                return api_error(400, "BAD_NAME", "name は target が output のときだけ指定できます。")
            try:
                path = _resolve(name)
            except outputs_mod.BadName:
                return api_error(400, "BAD_NAME", "ファイル名が正しくありません。")
            if not path.is_file():
                return api_error(404, "OUTPUT_NOT_FOUND", "そのファイルはありません。")
            select = True
        try:
            await run_in_threadpool(opener, path, select)
        except OSError as e:
            return api_error(500, "OPEN_FAILED", f"フォルダを開けませんでした: {e}（場所: {path}）")
        return {"ok": True, "path": str(path)}

    @app.post("/api/shutdown")
    async def shutdown():
        if not shutdown_started.is_set():
            shutdown_started.set()

            def stop() -> None:
                time.sleep(0.3)  # 先に応答を返す
                manager.cancel_all()
                if config.request_shutdown is not None:
                    config.request_shutdown()

            threading.Thread(target=stop, name="chordgen-shutdown", daemon=True).start()
        return {"ok": True}

    # 画面の静的ファイル（app.js など）。合言葉は要らない（秘密は入っていない。API は守ってある）
    app.mount("/", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
    app.add_middleware(_Guard, config=config)
    return app
