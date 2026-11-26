"""ブラウザで使う画面（ローカルのサーバー）。起動は `python -m chordgen.gui`。

fastapi・uvicorn・python-multipart は、このパッケージの下のモジュール（server など）の中でだけ
import する。入っていなくても `import chordgen`・CLI・ほかのテストは動く。
"""

ENV_FAKE = "CHORDGEN_GUI_FAKE"  # "1" なら本物の CLI の代わりに偽のジョブを走らせる（画面の開発・テスト用）
ENV_FAKE_DELAY = "CHORDGEN_GUI_FAKE_DELAY"  # 偽のジョブの 1 段あたりの秒数
DEFAULT_PORT = 8765
TOKEN_COOKIE = "chordgen_token"
TOKEN_HEADER = "X-Chordgen-Token"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # アップロードする音源の大きさの上限（1 GiB）
MAX_LYRICS_CHARS = 200_000  # 歌詞の文字数の上限
