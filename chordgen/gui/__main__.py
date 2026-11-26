"""python -m chordgen.gui [--port N] [--no-browser] [--host 127.0.0.1]

ローカルのサーバーを立ち上げ、ブラウザで画面を開く。終わるときは Ctrl+C か画面の「終了」。
"""

from __future__ import annotations

import argparse
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser

from chordgen.gui import DEFAULT_PORT, ENV_FAKE

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
    print("", flush=True)
    print("chordgen の画面を起動しました。ブラウザで次の URL を開いてください（自動で開きます）:", flush=True)
    print(f"  {url}", flush=True)
    if fake:
        print(f"（{ENV_FAKE}=1: 本物の解析の代わりに偽のジョブを走らせます）", flush=True)
    print("終わるときは、画面の「終了」を押すか、このウィンドウで Ctrl+C を押してください。", flush=True)
    print("", flush=True)

    if not args.no_browser:
        def open_browser() -> None:
            for _ in range(300):  # 起動を待ってから開く（最大 30 秒）
                if server.started or server.should_exit:
                    break
                time.sleep(0.1)
            if server.started:
                try:
                    webbrowser.open(url)
                except Exception as e:  # ブラウザが開けなくても URL は出してある
                    print(f"ブラウザを開けませんでした（{e}）。上の URL を手で開いてください。", file=sys.stderr)

        threading.Thread(target=open_browser, name="chordgen-browser", daemon=True).start()

    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    print("chordgen の画面を終了しました。", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
