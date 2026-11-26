"""子プロセスの扱いの共通部分（fetch と GUI で使う）。"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def new_group_kwargs() -> dict[str, Any]:
    """子を別のプロセスグループで起動する Popen の引数。

    端末の Ctrl+C が子に直接届かないようにし、kill_tree でグループごと止められるようにする。
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _is_group_leader(pid: int) -> bool:
    try:
        return os.getpgid(pid) == pid
    except (OSError, AttributeError):
        return False


def kill_tree(proc: subprocess.Popen) -> None:
    """子とその子（yt-dlp・ffmpeg など）を止める。

    Windows は taskkill /T /F でツリーごと。POSIX は子がグループの長（new_group_kwargs で起動）なら
    グループごと SIGKILL、そうでなければ子だけを kill する。
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif _is_group_leader(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
