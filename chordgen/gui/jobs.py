"""GUI の解析ジョブ。キューで 1 件ずつ、子プロセスの CLI（--progress-json）で走らせる。

子プロセスにするのは、中止したときにプロセスツリーごと止めて GPU のメモリを確実に空けるためと、
パイプラインに手を入れずに済ませるため。fastapi には依存しない（テストしやすいように）。
"""

from __future__ import annotations

import codecs
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from chordgen.pipeline import (
    DEFAULT_BARS_PER_LINE,
    DEFAULT_BASS_MODE,
    DEFAULT_CHORD_INPUT,
    DEFAULT_CHORD_MODEL,
    DEFAULT_SNAP,
    STAGE_LABELS,
)
from chordgen.proc import kill_tree, new_group_kwargs

log = logging.getLogger(__name__)

STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
MAX_KEPT_JOBS = 100  # jobs.json に残す、終わったジョブの数
LOG_KEEP = 1000  # 1 ジョブで覚えておく log の行数（SSE の流し直しに使う）
LOG_LINES_PER_SEC = 20  # これより多い行は間引く
TRANSIENT_LOG_INTERVAL = 1.0  # \r で上書きされる行（tqdm など）は、この秒数に 1 行だけ
STDERR_TAIL = 200
EVENT_KINDS = ("progress", "warning", "error")  # GET /api/jobs/{id} の events に入れるもの


def now_iso(ts: float | None = None) -> str:
    """ISO 8601（UTC、ミリ秒、末尾 Z）。"""
    dt = datetime.fromtimestamp(ts, timezone.utc) if ts is not None else datetime.now(timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def python_command(module: str) -> list[str]:
    """今の Python で `-m module` を起動するコマンド。-P・-s は親に合わせる（run.bat は -P -s で起動する）。"""
    cmd = [sys.executable]
    if getattr(sys.flags, "safe_path", False):
        cmd.append("-P")
    if sys.flags.no_user_site:
        cmd.append("-s")
    return [*cmd, "-m", module]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # chordgen をインストールせずに PYTHONPATH で読む形（ポータブル版）でも子が import できるように
    pkg_parent = str(Path(__file__).resolve().parent.parent.parent)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if pkg_parent not in parts:
        env["PYTHONPATH"] = os.pathsep.join([pkg_parent, *parts])
    return env


def remove_tree(path: Path | None, base: Path) -> None:
    """base の下にある path を消す。Windows で子を止めた直後はファイルが掴まれていることがあるので数回試す。"""
    if path is None:
        return
    try:
        p = path.resolve()
        if not p.is_relative_to(base.resolve()) or p == base.resolve():
            return
    except OSError:
        return
    for i in range(5):
        if not p.exists():
            return
        shutil.rmtree(p, ignore_errors=True)
        if not p.exists():
            return
        time.sleep(0.3 * (i + 1))
    log.warning("作業ファイルを消せませんでした: %s", p)


@dataclass
class JobOptions:
    chord_input: str = DEFAULT_CHORD_INPUT
    bass: str = DEFAULT_BASS_MODE
    snap: bool = DEFAULT_SNAP
    model: str = DEFAULT_CHORD_MODEL
    language: str | None = None
    bars_per_line: int = DEFAULT_BARS_PER_LINE
    keep_work: bool = False
    has_lyrics: bool = False

    def to_api(self) -> dict[str, Any]:
        return {
            "chordInput": self.chord_input,
            "bass": self.bass,
            "snap": self.snap,
            "model": self.model,
            "language": self.language,
            "barsPerLine": self.bars_per_line,
            "keepWork": self.keep_work,
            "hasLyrics": self.has_lyrics,
        }

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "JobOptions":
        o = cls()
        o.chord_input = str(d.get("chordInput", o.chord_input))
        o.bass = str(d.get("bass", o.bass))
        o.snap = bool(d.get("snap", o.snap))
        o.model = str(d.get("model", o.model))
        o.language = d.get("language") if isinstance(d.get("language"), str) else None
        try:
            o.bars_per_line = int(d.get("barsPerLine", o.bars_per_line))
        except (TypeError, ValueError):
            pass
        o.keep_work = bool(d.get("keepWork", False))
        o.has_lyrics = bool(d.get("hasLyrics", False))
        return o


@dataclass
class Job:
    id: str
    source_kind: str  # "file" | "url"
    source: str  # 音源ファイルの絶対パスか URL
    source_name: str  # 表示用（アップロードしたときのファイル名か URL）
    options: JobOptions = field(default_factory=JobOptions)
    title: str | None = None
    artist: str | None = None
    status: str = "queued"
    created_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    current_stage: dict[str, Any] | None = None
    output_name: str | None = None
    output_path: str | None = None
    error: dict[str, Any] | None = None
    lyrics_path: Path | None = None
    upload_dir: Path | None = None
    work_dir: Path | None = None
    order: int = 0  # 作った順（一覧の並びに使う）
    # 進捗などの記録。(seq, 名前, データ)。名前は progress / warning / error / log / status
    entries: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)
    log_count: int = 0
    proc: subprocess.Popen | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def to_api(self, queue_position: int | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "sourceName": self.source_name,
            "sourceKind": self.source_kind,
            "title": self.title,
            "artist": self.artist,
            "options": self.options.to_api(),
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "queuePosition": queue_position,
            "currentStage": self.current_stage,
            "outputName": self.output_name,
            "error": self.error,
        }

    def events(self) -> list[dict[str, Any]]:
        return [data for _, name, data in self.entries if name in EVENT_KINDS]

    def to_record(self) -> dict[str, Any]:
        rec = self.to_api(None)
        rec["source"] = self.source if self.source_kind == "url" else None
        rec["events"] = self.events()
        return rec


class NotCancellable(Exception):
    pass


class _LogThrottle:
    """子の標準エラーの行を間引く。"""

    def __init__(self, per_sec: int = LOG_LINES_PER_SEC, transient_interval: float = TRANSIENT_LOG_INTERVAL):
        self.per_sec = per_sec
        self.transient_interval = transient_interval
        self._window = 0.0
        self._count = 0
        self._dropped = 0
        self._last_transient = 0.0

    def feed(self, line: str, transient: bool) -> list[str]:
        """出す行の一覧を返す（間引いた行の数の知らせを含む）。"""
        now = time.monotonic()
        if transient:
            if now - self._last_transient < self.transient_interval:
                return []
            self._last_transient = now
            return [line]
        if now - self._window >= 1.0:
            self._window = now
            self._count = 0
        if self._count >= self.per_sec:
            self._dropped += 1
            return []
        self._count += 1
        out = []
        if self._dropped:
            out.append(f"（ログが多いので {self._dropped} 行を省きました）")
            self._dropped = 0
        out.append(line)
        return out

    def flush(self) -> list[str]:
        """最後に、まだ知らせていない省いた行の数を返す。"""
        if not self._dropped:
            return []
        n, self._dropped = self._dropped, 0
        return [f"（ログが多いので {n} 行を省きました）"]


class JobManager:
    """ジョブの記録・キュー・実行。メソッドはどのスレッドから呼んでもよい。"""

    def __init__(
        self,
        *,
        gui_dir: Path,
        command: list[str] | None = None,
        state_path: Path | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        on_change: Callable[[], None] | None = None,
    ):
        self.gui_dir = Path(gui_dir)
        self.command = list(command) if command else python_command("chordgen.cli")
        self.state_path = state_path if state_path is not None else self.gui_dir / "jobs.json"
        self.env = env
        self.cwd = cwd
        self.on_change = on_change
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._seq = itertools.count(1)
        self._order = itertools.count(1)
        self._closing = False
        self._thread: threading.Thread | None = None

    # ---- 置き場所 ----
    @property
    def uploads_root(self) -> Path:
        return self.gui_dir / "uploads"

    @property
    def work_root(self) -> Path:
        return self.gui_dir / "jobs"

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def upload_dir(self, job_id: str) -> Path:
        return self.uploads_root / job_id

    def job_work_dir(self, job_id: str) -> Path:
        return self.work_root / job_id

    # ---- 起動と終了 ----
    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._load()
            self._sweep()
            self._thread = threading.Thread(target=self._worker, name="chordgen-jobs", daemon=True)
            self._thread.start()
        self._save()

    @property
    def closing(self) -> bool:
        return self._closing

    def close(self, timeout: float = 15.0) -> None:
        """キューの残りを中止し、実行中のジョブを止めて、記録を書く。何度呼んでもよい。"""
        with self._cond:
            if self._closing and self._thread is None:
                return
            self._closing = True
            queued = list(self._queue)
            running = [j for j in self._jobs.values() if j.status == "running"]
        for job_id in queued:
            try:
                self.cancel(job_id)
            except (KeyError, NotCancellable):
                pass
        for job in running:
            try:
                self.cancel(job.id)
            except (KeyError, NotCancellable):
                pass
        with self._cond:
            self._cond.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        with self._lock:
            self._thread = None
        self._save()

    # ---- 参照 ----
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _queue_position(self, job: Job) -> int | None:
        if job.status != "queued":
            return None
        try:
            return self._queue.index(job.id) + 1
        except ValueError:
            return None

    def job_api(self, job_id: str, *, with_events: bool = False) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            d = job.to_api(self._queue_position(job))
            if with_events:
                d["events"] = job.events()
            return d

    def list_api(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: (j.created_at, j.order), reverse=True)
            return [j.to_api(self._queue_position(j)) for j in jobs]

    def snapshot(self, job_id: str) -> tuple[list[tuple[int, str, dict]], dict[str, Any], bool, int] | None:
        """（記録の全部, 今の Job, 終わったか, 最後の seq）。SSE の流し直しに使う。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            last = job.entries[-1][0] if job.entries else 0
            return list(job.entries), job.to_api(self._queue_position(job)), job.terminal, last

    def entries_after(self, job_id: str, seq: int) -> tuple[list[tuple[int, str, dict]], dict[str, Any], bool]:
        """seq より後の記録と、今の Job と、終わったか。"""
        with self._lock:
            job = self._jobs[job_id]
            # entries は seq の昇順。後ろから探す（新しいものだけ欲しいことが多い）
            i = len(job.entries)
            while i > 0 and job.entries[i - 1][0] > seq:
                i -= 1
            return job.entries[i:], job.to_api(self._queue_position(job)), job.terminal

    # ---- 投入と中止 ----
    def submit(self, job: Job) -> Job:
        with self._cond:
            if self._closing:
                raise RuntimeError("サーバーを終了しているところです。")
            job.order = next(self._order)
            job.status = "queued"
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._push_status(job)
            self._prune()
            self._cond.notify_all()
        self._save_and_notify()
        return job

    def cancel(self, job_id: str) -> Job:
        """queued ならキューから外し、running なら子を止める。終わったジョブは NotCancellable。"""
        proc = None
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "queued":
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.status = "cancelled"
                job.finished_at = now_iso()
                self._push_status(job)
                self._push_queue_positions()
                self._cleanup(job)
            elif job.status == "running":
                job.cancel_requested.set()
                proc = job.proc
            else:
                raise NotCancellable(job.status)
        if proc is not None:
            kill_tree(proc)
        self._save_and_notify()
        return job

    def cancel_all(self) -> None:
        with self._lock:
            ids = [j.id for j in self._jobs.values() if not j.terminal]
        for job_id in ids:
            try:
                self.cancel(job_id)
            except (KeyError, NotCancellable):
                pass

    # ---- 記録 ----
    def _append(self, job: Job, name: str, data: dict[str, Any]) -> None:
        """記録を足す（ロックを持って呼ぶ）。"""
        if name == "progress" and data.get("status") == "progress" and job.entries:
            # 取得中の progress は最後の 1 件だけ残す
            _, last_name, last = job.entries[-1]
            if last_name == "progress" and last.get("status") == "progress" and last.get("stage") == data.get("stage"):
                job.entries.pop()
        job.entries.append((next(self._seq), name, data))
        if name == "log":
            job.log_count += 1
            if job.log_count > LOG_KEEP * 3 // 2:
                drop = job.log_count - LOG_KEEP
                kept = []
                for e in job.entries:
                    if e[1] == "log" and drop > 0:
                        drop -= 1
                        continue
                    kept.append(e)
                job.entries = kept
                job.log_count = LOG_KEEP

    def _push_status(self, job: Job) -> None:
        self._append(job, "status", job.to_api(self._queue_position(job)))

    def _push_queue_positions(self) -> None:
        for job_id in self._queue:
            self._push_status(self._jobs[job_id])

    def _prune(self) -> None:
        """終わったジョブは新しい MAX_KEPT_JOBS 件だけ残す。"""
        done = sorted((j for j in self._jobs.values() if j.terminal), key=lambda j: (j.created_at, j.order))
        for j in done[: max(0, len(done) - MAX_KEPT_JOBS)]:
            del self._jobs[j.id]

    def _save(self) -> None:
        with self._save_lock:  # 書く順を守り、一時ファイルを取り合わない
            with self._lock:
                jobs = sorted(self._jobs.values(), key=lambda j: (j.created_at, j.order), reverse=True)
                records = [j.to_record() for j in jobs]
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.state_path.with_name(self.state_path.name + ".tmp")
                text = json.dumps({"version": 1, "jobs": records}, ensure_ascii=False, indent=1)
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.state_path)
            except OSError:
                log.warning("ジョブの記録を書けませんでした: %s", self.state_path, exc_info=True)

    def _save_and_notify(self) -> None:
        self._save()
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                log.debug("on_change が失敗しました", exc_info=True)

    def _load(self) -> None:
        """前の起動の記録を読む。実行中・待ちのまま止まったジョブは failed にする。"""
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            log.warning("ジョブの記録を読めませんでした（読み飛ばします）: %s", self.state_path)
            return
        records = data.get("jobs") if isinstance(data, dict) else None
        if not isinstance(records, list):
            return
        for rec in reversed(records):  # 古い順に入れる（order を作った順にそろえる）
            job = self._from_record(rec)
            if job is None or job.id in self._jobs:
                continue
            if not job.terminal:
                stage = (job.current_stage or {}).get("stage")
                was_running = job.status == "running"
                job.status = "failed"
                job.finished_at = job.finished_at or now_iso()
                job.error = {
                    "code": "SERVER_STOPPED",
                    "message": "解析の途中でサーバーが止まったので、中断しました。" if was_running
                    else "解析を始める前にサーバーが止まったので、中断しました。",
                    "stage": stage if was_running else None,
                }
                self._append(job, "error", {"type": "error", **job.error, "exitCode": None, "at": job.finished_at})
                self._cleanup(job)
            self._jobs[job.id] = job
        self._prune()

    def _from_record(self, rec: Any) -> Job | None:
        if not isinstance(rec, dict) or not isinstance(rec.get("id"), str) or not re.fullmatch(r"[0-9a-f]{6,32}", rec["id"]):
            return None
        status = rec.get("status")
        if status not in STATUSES:
            return None
        kind = rec.get("sourceKind") if rec.get("sourceKind") in ("file", "url") else "file"
        job = Job(
            id=rec["id"],
            source_kind=kind,
            source=str(rec.get("source") or ""),
            source_name=str(rec.get("sourceName") or ""),
            options=JobOptions.from_api(rec.get("options") or {}),
            title=rec.get("title"),
            artist=rec.get("artist"),
            status=status,
            created_at=str(rec.get("createdAt") or now_iso()),
            started_at=rec.get("startedAt"),
            finished_at=rec.get("finishedAt"),
            current_stage=rec.get("currentStage") if isinstance(rec.get("currentStage"), dict) else None,
            output_name=rec.get("outputName"),
            error=rec.get("error") if isinstance(rec.get("error"), dict) else None,
            upload_dir=self.upload_dir(rec["id"]),
            work_dir=self.job_work_dir(rec["id"]),
            order=next(self._order),
        )
        for ev in rec.get("events") or []:
            if isinstance(ev, dict) and ev.get("type") in EVENT_KINDS:
                job.entries.append((next(self._seq), ev["type"], ev))
        return job

    def _sweep(self) -> None:
        """前の起動の残り（keep_work のジョブ以外のアップロード・作業フォルダ）を消す（起動のときだけ）。"""
        keep = {j.id for j in self._jobs.values() if j.options.keep_work}
        for root in (self.uploads_root, self.work_root):
            try:
                children = list(root.iterdir())
            except OSError:
                continue
            for d in children:
                if d.is_dir() and d.name not in keep:
                    remove_tree(d, root)

    def _cleanup(self, job: Job) -> None:
        """アップロードした音源と作業ファイルを消す（keep_work のときは残す）。"""
        if job.options.keep_work:
            return
        remove_tree(job.upload_dir, self.uploads_root)
        remove_tree(job.work_dir, self.work_root)

    # ---- 実行 ----
    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closing:
                    self._cond.wait()
                if self._closing:
                    return
                job = self._jobs[self._queue.popleft()]
                job.status = "running"
                job.started_at = now_iso()
                self._push_status(job)
                self._push_queue_positions()
            self._save_and_notify()
            try:
                self._run(job)
            except Exception as e:  # 想定外の失敗でもキューは止めない
                log.exception("ジョブの実行に失敗しました: %s", job.id)
                self._cleanup(job)
                with self._lock:
                    if not job.terminal:
                        self._finish(job, "failed", {"code": "UNEXPECTED", "message": f"{type(e).__name__}: {e}",
                                                     "stage": (job.current_stage or {}).get("stage")})
                self._save_and_notify()

    def cli_args(self, job: Job) -> list[str]:
        o = job.options
        args = [
            "--progress-json",
            "--chord-input", o.chord_input,
            "--bass", o.bass,
            "--model", o.model,
            "--bars-per-line", str(o.bars_per_line),
        ]
        if job.work_dir is not None:
            args.append(f"--work-dir={job.work_dir}")
        if not o.snap:
            args.append("--no-snap")
        if o.language:
            args.append(f"--language={o.language}")
        # 「-」で始まる題名もオプションと取り違えないよう、= でつなぐ
        if job.title:
            args.append(f"--title={job.title}")
        if job.artist:
            args.append(f"--artist={job.artist}")
        if job.lyrics_path is not None:
            args.append(f"--lyrics={job.lyrics_path}")
        if o.keep_work:
            args.append("--keep-work")
        return [*args, "--", job.source]

    def _run(self, job: Job) -> None:
        if job.work_dir is not None:
            job.work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [*self.command, *self.cli_args(job)]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                env=self.env if self.env is not None else _child_env(),
                cwd=str(self.cwd) if self.cwd else None,
                **new_group_kwargs(),
            )
        except OSError as e:
            self._cleanup(job)
            with self._lock:
                self._finish(job, "failed", {"code": "START_FAILED", "message": f"解析の処理を起動できませんでした: {e}",
                                             "stage": None})
            self._save_and_notify()
            return

        with self._lock:
            job.proc = proc
            cancel_now = job.cancel_requested.is_set()
        if cancel_now:
            kill_tree(proc)

        state: dict[str, Any] = {"done": None, "error": None}
        tail: deque[str] = deque(maxlen=STDERR_TAIL)
        t_out = threading.Thread(target=self._read_stdout, args=(job, proc, state), daemon=True)
        t_err = threading.Thread(target=self._read_stderr, args=(job, proc, tail), daemon=True)
        t_out.start()
        t_err.start()
        rc = proc.wait()
        t_out.join(10)
        t_err.join(10)
        for s in (proc.stdout, proc.stderr):
            try:
                if s:
                    s.close()
            except OSError:
                pass

        # 片付けてから終わりの状態にする（status を受けた画面が、消えたはずのファイルを見ないように）
        self._cleanup(job)
        with self._lock:
            job.proc = None
            if job.cancel_requested.is_set():
                self._finish(job, "cancelled", None)
            elif rc == 0 and state["done"] is not None:
                out = state["done"].get("output")
                job.output_path = out if isinstance(out, str) else None
                job.output_name = Path(out).name if isinstance(out, str) and out else None
                t = state["done"].get("title")
                if isinstance(t, str) and t:
                    job.title = t
                self._finish(job, "done", None)
            else:
                err = state["error"]
                stage = (job.current_stage or {}).get("stage")
                if err is not None:
                    error = {"code": str(err.get("code") or "FAILED"), "message": str(err.get("message") or ""),
                             "stage": err.get("stage") if err.get("stage") is not None else None}
                else:
                    last = next((s for s in reversed(tail) if s.strip()), "")
                    msg = f"解析の処理が終了コード {rc} で終わりました。" + (f"\n{last}" if last else "")
                    error = {"code": "PROCESS_FAILED", "message": msg, "stage": stage}
                    self._append(job, "error", {"type": "error", **error, "exitCode": rc, "at": now_iso()})
                self._finish(job, "failed", error)
        self._save_and_notify()

    def _finish(self, job: Job, status: str, error: dict[str, Any] | None) -> None:
        """ジョブを終わりの状態にする（ロックを持って呼ぶ。作業ファイルの片付けはロックの外で _cleanup）。"""
        job.status = status
        job.error = error
        job.finished_at = now_iso()
        self._push_status(job)
        self._prune()

    def _on_event(self, job: Job, obj: dict[str, Any], state: dict[str, Any]) -> None:
        kind = obj.get("type")
        data = {**obj, "at": now_iso()}
        with self._lock:
            if kind == "progress":
                stage = obj.get("stage")
                if isinstance(stage, str):
                    data.setdefault("label", STAGE_LABELS.get(stage, stage))
                    if obj.get("status") in ("start", "done", "skipped", "failed") or job.current_stage is None:
                        changed = (job.current_stage or {}).get("stage") != stage
                        job.current_stage = {"stage": stage, "label": data.get("label"),
                                             "index": obj.get("index"), "total": obj.get("total")}
                        self._append(job, "progress", data)
                        if changed:
                            self._push_status(job)
                        return
                self._append(job, "progress", data)
            elif kind == "warning":
                self._append(job, "warning", {"type": "warning", "message": str(obj.get("message") or ""), "at": data["at"]})
            elif kind == "error":
                state["error"] = obj
                self._append(job, "error", data)
            elif kind == "done":
                state["done"] = obj

    def _read_stdout(self, job: Job, proc: subprocess.Popen, state: dict[str, Any]) -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("{"):
                    if line:
                        self._log_lines(job, [line])
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    self._log_lines(job, [line])
                    continue
                if isinstance(obj, dict):
                    self._on_event(job, obj, state)
        except (OSError, ValueError):
            pass

    def _read_stderr(self, job: Job, proc: subprocess.Popen, tail: deque[str]) -> None:
        """標準エラーを \\n と \\r で区切って log にする。\\r で上書きされる行は間引く。"""
        assert proc.stderr is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        throttle = _LogThrottle()
        buf = ""
        read = getattr(proc.stderr, "read1", proc.stderr.read)
        try:
            while True:
                chunk = read(8192)
                if not chunk:
                    break
                buf += decoder.decode(chunk)
                buf = self._split_log(job, buf, throttle, tail, final=False)
            buf += decoder.decode(b"", final=True)
            self._split_log(job, buf, throttle, tail, final=True)
        except (OSError, ValueError):
            pass
        rest = throttle.flush()
        if rest:
            self._log_lines(job, rest)

    def _split_log(self, job: Job, buf: str, throttle: _LogThrottle, tail: deque[str], *, final: bool) -> str:
        out: list[str] = []
        while True:
            m = re.search(r"\r\n|\n|\r", buf)
            if m is None:
                break
            if m.group(0) == "\r" and m.end() == len(buf) and not final:
                break  # 次の塊が \n で始まるかもしれない
            line, buf = buf[: m.start()], buf[m.end():]
            if not line.strip():
                continue
            tail.append(line)
            out.extend(throttle.feed(line, transient=m.group(0) == "\r"))
        if final and buf.strip():
            tail.append(buf)
            out.extend(throttle.feed(buf, transient=False))
            buf = ""
        if out:
            self._log_lines(job, out)
        return buf

    def _log_lines(self, job: Job, lines: list[str]) -> None:
        at = now_iso()
        with self._lock:
            for line in lines:
                self._append(job, "log", {"line": line, "at": at})
