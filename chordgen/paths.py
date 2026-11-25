"""アプリのフォルダの中の置き場所。フォルダの外には書き出さない。

CHORDGEN_HOME（run.bat が設定する）があればそれを、なければリポジトリの直下を使う。
"""

from __future__ import annotations

import os
from pathlib import Path


def app_home() -> Path:
    env = os.environ.get("CHORDGEN_HOME")
    return Path(env) if env else Path(__file__).resolve().parent.parent


def _sub(name: str) -> Path:
    p = app_home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:  # モデルの重み
    return _sub("models")


def work_dir() -> Path:  # 取得した音源・分離した音などの作業ファイル
    return _sub("work")


def output_dir() -> Path:  # .cat4g.json
    return _sub("output")


def third_party_dir() -> Path:  # ChordMini のソースなど、初回に取得するコード
    return _sub("third_party")


def schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "specs" / "format" / "cat4g-song.v1.schema.json"
