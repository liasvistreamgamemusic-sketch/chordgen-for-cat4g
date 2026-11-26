"""python -m chordgen.gui [--port N] [--no-browser] [--host 127.0.0.1]

ローカルのサーバーを立ち上げ、ブラウザで画面を開く。終わるときは Ctrl+C か画面の「終了」。
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chordgen.gui import DEFAULT_PORT, ENV_FAKE, TOKEN_HEADER

PORT_SEARCH = 50  # 既定のポートが使用中なら、この数だけ先まで探す（それでもだめなら OS に選ばせる）


def build_parser() -> argparse.ArgumentParser:
    from chordgen.cli import _JapaneseHelpFormatter

    p = argparse.ArgumentParser(
        prog="python -m chordgen.gui",
        description="chordgen の画面（ローカルのサーバー）を起動し、ブラウザで開きます。",
        formatter_class=_JapaneseHelpFormatter,
        add_help=False,
    )
    p._optionals.title = "オプション"
    p.add_argument("-h", "--help", action="help", help="このヘルプを表示して終わります")
    p.add_argument("--port", type=int, metavar="N",
                   help=f"待ち受けるポート（既定 {DEFAULT_PORT}。使用中なら空いているポートを探します。"
                        "指定したポートが使用中ならエラーにします）")
    p.add_argument("--host", default="127.0.0.1", metavar="アドレス",
                   help="待ち受けるアドレス（既定 127.0.0.1。ほかにすると同じネットワークの人からも繋がるので勧めません）")
    p.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    return p


def _bind(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            # Windows はほかのプロセスが SO_REUSEADDR で使っているポートにも bind できてしまうので、排他にする
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    sock.set_inheritable(False)
    return sock


def open_socket(host: str, port: int | None) -> socket.socket:
    """port が None なら既定のポートから空いているものを探す。"""
    if port is not None:
        return _bind(host, port)
    for p in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SEARCH):
        try:
            return _bind(host, p)
        except OSError:  # 使用中・予約済み（Windows の Hyper-V など）
            continue
    return _bind(host, 0)


def _missing_deps_message(e: ImportError) -> str:
    return (
        f"エラー: 画面に要るライブラリが入っていません（{e.name or e}）。\n"
        "  setup.bat を実行して入れ直してください（済んだ手順は飛ばすので、すぐ終わります）。\n"
        "  開発の環境なら: uv pip install -e \".[dev]\""
    )


def pid_alive(pid: int) -> bool:
    """そのプロセスが生きているか。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class InstanceLock:
    """同じアプリのフォルダで画面を二重に起動しないための鍵（work/gui/server.lock）。

    起動中のサーバーの URL（合言葉つき）は server.json に書いておき、二つ目の起動はそれをブラウザで開いて終わる。
    server.json は本人だけが読める権限（POSIX で 0o600）で書き、終わるとき（atexit でも）消す。
    """

    def __init__(self, gui_dir: Path):
        self.gui_dir = gui_dir
        self.lock_path = gui_dir / "server.lock"
        self.info_path = gui_dir / "server.json"
        self._f = None

    def acquire(self) -> bool:
        self.gui_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.lock_path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self._f = f
        self._remove_info()  # 鍵が取れたなら、残っている server.json は前の起動の古いもの
        return True

    def publish(self, url: str, port: int) -> None:
        """server.json を書く。一時ファイルを 0o600 で作ってから置き換える。

        Windows では mode は効かないが、置き場所はアプリのフォルダ（work/gui）の中で、権限はそのフォルダに従う。
        """
        data = json.dumps({"url": url, "port": port, "pid": os.getpid()}).encode("utf-8")
        tmp = self.info_path.with_name(self.info_path.name + ".tmp")
        try:
            tmp.unlink(missing_ok=True)  # 既にあるファイルには O_CREAT の mode が効かないので消しておく
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            os.replace(tmp, self.info_path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def running_url(self) -> str | None:
        """起動中のサーバーの URL。書いた pid が生きていて、その URL の /api/info が合言葉つきで応えたときだけ返す。

        pid が止まっている・読めない server.json は消す。
        """
        try:
            data = json.loads(self.info_path.read_text(encoding="utf-8"))
            url, port, pid = str(data["url"]), int(data["port"]), int(data["pid"])
            token = parse_qs(urlparse(url).query)["t"][0]
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            self._remove_info()
            return None
        if not pid_alive(pid):
            self._remove_info()
            return None
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/info", headers={TOKEN_HEADER: token})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # プロキシの設定を通さない
            with opener.open(req, timeout=5) as r:
                if r.status == 200:
                    return url
        except (OSError, ValueError):
            pass
        return None

    def _remove_info(self) -> None:
        for p in (self.info_path, self.info_path.with_name(self.info_path.name + ".tmp")):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def release(self) -> None:
        """server.json を消して鍵を外す。何度呼んでもよい（finally と atexit の両方から呼ぶ）。"""
        if self._f is None:
            return
        self._remove_info()
        try:
            self._f.close()  # 閉じれば鍵も外れる
        except OSError:
            pass
        self._f = None


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception as e:  # ブラウザが開けなくても URL は出してある
        print(f"ブラウザを開けませんでした（{e}）。上の URL を手で開いてください。", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    from chordgen.cli import _setup_stdio

    _setup_stdio()
    args = build_parser().parse_args(argv)
    try:
        import uvicorn

        from chordgen.gui.server import ServerConfig, create_app
    except ImportError as e:
        print(_missing_deps_message(e), file=sys.stderr, flush=True)
        return 1

    from chordgen.paths import work_dir

    lock = InstanceLock(work_dir() / "gui")
    if not lock.acquire():
        url = lock.running_url()
        if url is None:
            print("エラー: ほかの chordgen の画面がこのフォルダで動いているようですが、応えません。\n"
                  "  そのウィンドウを閉じてから、もう一度起動してください。", file=sys.stderr, flush=True)
            return 1
        print("chordgen の画面はもう起動しています。ブラウザで開きます:", flush=True)
        print(f"  {url}", flush=True)
        if not args.no_browser:
            _open_browser(url)
        return 0
    atexit.register(lock.release)  # 異常終了でも合言葉つきの URL を残さない

    try:
        if args.host != "127.0.0.1":
            print(f"警告: 127.0.0.1 以外（{args.host}）で待ち受けます。同じネットワークのほかの機器から繋がる"
                  "おそれがあります。ふだんは --host を指定しないでください。", file=sys.stderr, flush=True)
        try:
            sock = open_socket(args.host, args.port)
        except OSError as e:
            print(f"エラー: {args.host}:{args.port or DEFAULT_PORT} で待ち受けられませんでした: {e}\n"
                  "  --port で別のポートを指定してください。", file=sys.stderr, flush=True)
            return 1
        port = sock.getsockname()[1]
        token = secrets.token_urlsafe(32)
        fake = os.environ.get(ENV_FAKE, "") == "1"
        config = ServerConfig(port=port, host=args.host, token=token, fake=fake)
        app = create_app(config)
        server = uvicorn.Server(uvicorn.Config(
            app, log_level="warning", access_log=False, timeout_graceful_shutdown=3, lifespan="on",
        ))
        config.request_shutdown = lambda: setattr(server, "should_exit", True)
        url_host = args.host if args.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
        url = f"http://{url_host}:{port}/?t={token}"

        def announce() -> None:
            for _ in range(600):  # 起動を待ってから知らせる（最大 60 秒）
                if server.started or server.should_exit:
                    break
                time.sleep(0.1)
            if not server.started:
                return
            lock.publish(url, port)
            print("", flush=True)
            print("chordgen の画面を起動しました。ブラウザで次の URL を開いてください（自動で開きます）:", flush=True)
            print(f"  {url}", flush=True)
            if fake:
                print(f"（{ENV_FAKE}=1: 本物の解析の代わりに偽のジョブを走らせます）", flush=True)
            print("終わるときは、画面の「終了」を押すか、このウィンドウで Ctrl+C を押してください。", flush=True)
            print("", flush=True)
            if not args.no_browser:
                _open_browser(url)

        threading.Thread(target=announce, name="chordgen-announce", daemon=True).start()
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            pass
        finally:
            sock.close()
        print("chordgen の画面を終了しました。", flush=True)
        return 0
    finally:
        lock.release()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
