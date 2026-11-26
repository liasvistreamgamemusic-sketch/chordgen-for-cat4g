"""GET /api/info の中身（版・デバイス・選択肢・既定値）。"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from typing import Any

from chordgen.pipeline import (
    BASS_MODES,
    CHORD_INPUT_MODES,
    CHORD_MODELS,
    DEFAULT_BARS_PER_LINE,
    DEFAULT_BASS_MODE,
    DEFAULT_CHORD_INPUT,
    DEFAULT_CHORD_MODEL,
    DEFAULT_SNAP,
    MAX_BARS_PER_LINE,
    STAGE_LABELS,
)

log = logging.getLogger(__name__)

DEVICE_PROBE_TIMEOUT = 300.0  # 初回の torch の import は Windows の ROCm でとても遅いことがある

CHORD_INPUT_CHOICES = {
    "original": ("原音のまま", "音源をそのままコード認識に入れます。いちばん速く、既定です。"),
    "no_drums": ("ドラムを除く", "音源を分離してドラムを除いた音で認識します。分離の分だけ時間がかかります。"),
    "no_drums_vocals": ("ドラムとボーカルを除く", "ドラムと歌を除いた伴奏だけで認識します。分離の分だけ時間がかかります。"),
    "boost_other": ("ギター・ピアノを強める", "分離した other（ギター・ピアノなど）を大きくして組み直した音で認識します。"),
}
BASS_CHOICES = {
    "off": ("使わない", "ベースの認識結果を使いません。既定です。"),
    "root_fix": ("ルートを補正する", "分離したベースの音で、コードのルートを直します。音源の分離が要ります。"),
    "slash": ("分数コードにする", "ベースの音がルートと違えば、C/E のような分数コードにします。音源の分離が要ります。"),
}
MODEL_CHOICES = {
    "chordmini": ("ChordMini（BTC・CL full）", "ChordMini で学習し直した BTC。既定です。"),
    "btc": ("元の BTC（比較用）", "BTC-ISMIR19 の元の重み。比べるときに使います。"),
}

# torch を GUI のサーバーには読み込まない（重いうえ、GPU のメモリを掴むことがある）。子プロセスで調べる
_PROBE_SCRIPT = """
import json
try:
    import torch
except Exception as e:
    print(json.dumps({"kind": "cpu", "name": None, "torchAvailable": False, "detail": f"{type(e).__name__}: {e}"}))
    raise SystemExit(0)
try:
    ok = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if ok else None
except Exception as e:
    ok, name = False, None
print(json.dumps({"kind": "cuda" if ok else "cpu", "name": name, "torchAvailable": True}))
"""


def _choices(table: dict[str, tuple[str, str]], order: tuple[str, ...]) -> list[dict[str, str]]:
    return [{"value": k, "label": table[k][0], "description": table[k][1]} for k in order if k in table]


def static_info() -> dict[str, Any]:
    """デバイスと media-dl 以外（起動中に変わらないもの）。"""
    from chordgen import __version__

    return {
        "version": __version__,
        "defaults": {
            "chordInput": DEFAULT_CHORD_INPUT,
            "bass": DEFAULT_BASS_MODE,
            "snap": DEFAULT_SNAP,
            "model": DEFAULT_CHORD_MODEL,
            "barsPerLine": DEFAULT_BARS_PER_LINE,
            "language": None,
            "keepWork": False,
        },
        "choices": {
            "chordInput": _choices(CHORD_INPUT_CHOICES, CHORD_INPUT_MODES),
            "bass": _choices(BASS_CHOICES, BASS_MODES),
            "model": _choices(MODEL_CHOICES, CHORD_MODELS),
        },
        "stages": [{"stage": k, "label": v} for k, v in STAGE_LABELS.items()],
    }


UNKNOWN_DEVICE = {"kind": "unknown", "name": None, "torchAvailable": False}


class DeviceProbe:
    """別スレッドで（子プロセスで torch を読み込んで）デバイスを調べる。分かるまでは kind: unknown。"""

    def __init__(self, *, fixed: dict[str, Any] | None = None, delay: float = 0.0, command: list[str] | None = None):
        self._fixed = fixed
        self._delay = delay
        self._command = command
        self._result: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="chordgen-device-probe", daemon=True)
                self._thread.start()

    def get(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._result) if self._result is not None else dict(UNKNOWN_DEVICE)

    def _run(self) -> None:
        if self._delay:
            time.sleep(self._delay)
        result = dict(self._fixed) if self._fixed is not None else self._probe()
        with self._lock:
            self._result = result

    def _probe(self) -> dict[str, Any]:
        from chordgen.gui.jobs import _child_env

        flags = ["-s"] if sys.flags.no_user_site else []
        cmd = self._command or [sys.executable, *flags, "-c", _PROBE_SCRIPT]
        try:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=DEVICE_PROBE_TIMEOUT,
                               env=_child_env(), shell=False)
            for line in reversed(p.stdout.decode("utf-8", errors="replace").splitlines()):
                if line.startswith("{"):
                    d = json.loads(line)
                    return {"kind": d.get("kind", "cpu"), "name": d.get("name"),
                            "torchAvailable": bool(d.get("torchAvailable"))}
        except (OSError, subprocess.TimeoutExpired, ValueError):
            log.debug("デバイスを調べられませんでした", exc_info=True)
        return {"kind": "cpu", "name": None, "torchAvailable": False}


def media_dl_available() -> bool:
    try:
        from chordgen.fetch import FetchError, find_media_dl
    except ImportError:
        return False
    try:
        find_media_dl()
    except FetchError:
        return False
    except Exception:
        return False
    return True


def limits() -> dict[str, int]:
    from chordgen.gui import MAX_LYRICS_CHARS, MAX_UPLOAD_BYTES

    return {"maxUploadBytes": MAX_UPLOAD_BYTES, "maxLyricsChars": MAX_LYRICS_CHARS, "maxBarsPerLine": MAX_BARS_PER_LINE}
