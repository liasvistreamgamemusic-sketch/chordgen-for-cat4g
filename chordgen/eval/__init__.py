"""コード認識の精度を条件ごとに比べる（python -m chordgen.eval）。

データセット
    リポジトリの外のフォルダ（例 D:\\eval）に、曲ごとに音源と正解を同じ名前で置く。

        D:\\eval\\曲名A.wav     音源（.wav / .flac / .mp3 / .m4a / .ogg / .aac / .opus）
        D:\\eval\\曲名A.lab     正解のコード

    .lab は UTF-8 のテキストで、1 行 1 区間の「開始秒 終了秒 ラベル」（空白かタブで区切る）。
    空行と # で始まる行は読み飛ばす。すき間は N（コードなし）として扱う。

        0.000   2.512   N
        2.512   4.980   A:min7
        4.980   7.433   C:maj/3

    ラベルは Harte 記法（A:min7、C:maj/3、N、X）のほか、cat4g の表記（Am7、C/E、F#m7-5、
    G7(b9)、N.C.）でもよい。X（不明）の区間は採点から除く。

使い方
    python -m chordgen.eval --dataset D:\\eval
    python -m chordgen.eval --dataset D:\\eval --conditions input=original,no_drums model=chordmini
    python -m chordgen.eval --list-conditions

    結果は標準出力の表と、作業フォルダの eval/results/<日時>/ の results.csv・results.json。
    重い処理の結果は eval/cache/ に残り、条件を変えて走らせ直すときに使い回す。

指標は mir_eval.chord の root、majmin、thirds、triads、sevenths、tetrads、mirex と、分数ベースまで見る
majmin_inv、sevenths_inv（どれも長さで重みづけ）。
"""

from chordgen.eval.conditions import AXES, Condition, apply_condition, enumerate_conditions, parse_filters
from chordgen.eval.labfile import cat4g_to_harte, format_lab, parse_lab, read_lab, to_harte, write_lab
from chordgen.eval.metrics import METRICS, SongScore, aggregate, score_song

__all__ = [
    "AXES",
    "Condition",
    "METRICS",
    "SongScore",
    "aggregate",
    "apply_condition",
    "cat4g_to_harte",
    "enumerate_conditions",
    "format_lab",
    "parse_filters",
    "parse_lab",
    "read_lab",
    "score_song",
    "to_harte",
    "write_lab",
]
