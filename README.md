# chordgen（cat4g 用のコード譜の下書きを作るアプリ）

曲の音源（ファイルか URL）を解析し、拍・キー・コードを推定して、cat4g に読み込める
CaT4G Song Format v1 のファイル（`.cat4g.json`）を書き出します。歌詞のテキストを渡すと、
Whisper で歌詞の各文字の時刻を求め、コードを歌詞の位置に置きます。

出力は人が直す前提の下書きです（コード認識の正解率は三和音で 8 割弱）。cat4g の編集画面で、
歌詞・コード・奏法を直して完成させてください。

Windows のポータブル版です。Python・Node.js・PyTorch・モデルの重み・出力はすべてこのフォルダの中に置き、
Windows に入れるのは GPU のドライバだけです。WSL は使いません。

## 必要なもの

| 項目 | 内容 |
|---|---|
| OS | Windows 11（AMD の ROCm 10.2.0 は Windows 11 25H2 が対象） |
| ディスク | 約 10 GB（PyTorch と ROCm / CUDA の部品で数 GB、モデルの重みで数 GB） |
| ネット | 初回のセットアップと、モデルの重みの初回ダウンロードに要ります |
| AMD の GPU | **AMD Software: Adrenalin Edition 26.11.2**（ROCm 10.2.0 の互換表で対応とされている版）。ROCm 7.2.1 に戻すときは 26.2.2 |
| NVIDIA の GPU | ドライバ R595 以上を推奨（CUDA 13.2。CUDA 13.x の互換で 580 以上でも動きます）。Turing（RTX 20 / GTX 16）以降。ドライバを上げられないときは `setup.bat --cuda 12.6`（525 以上） |
| GPU なし | CPU でも動きます（とても時間がかかります） |

- AMD の **HIP SDK for Windows** が入っていると（`%HIP_PATH%` や `C:\Program Files\AMD\ROCm` がある）、
  pip で入れる ROCm と食い違うことがあります。セットアップは見つけると警告します。GPU が使えないときは外してください。
- フォルダは `C:\chordgen` のような**短いパス**に置いてください。PyTorch のファイルのパスが長く、
  Windows の 260 文字の上限を超えると入りません（セットアップが長すぎるときに警告します）。
- パスに空白や日本語が入っていても動きます。

### 対応する AMD の GPU（ROCm 10.2.0）

セットアップは GPU の名前から gfx ターゲットを決めて、合う PyTorch（`torch[device-gfxXXXX]`）を入れます。
名前から決められないときは `setup.bat --gpu amd --amd-device gfx1201` のように指定します。

| gfx | GPU（AMD の PyTorch のインストール手順の表から） |
|---|---|
| gfx1201 | Radeon RX 9070 XT / 9070 GRE / 9070、Radeon AI PRO R9700 / R9600 |
| gfx1200 | Radeon RX 9060 XT / 9060 / 9050 |
| gfx1100 | Radeon RX 7900 XTX / XT / GRE、PRO W7900 / W7800 |
| gfx1101 | Radeon RX 7800 XT / 7700 XT / 7700、PRO W7700 |
| gfx1102 | Radeon RX 7600 |
| gfx1030 | Radeon RX 6800 / 6900 / 6950（ROCm 10.2.0 の互換表に Windows の対象としてある） |
| gfx1151 / gfx1150 / gfx1152 / gfx1153 / gfx1103 | Ryzen AI Max（8060S など）/ Ryzen AI 9（890M・880M）/ Ryzen AI 7・5（860M・840M）/ Ryzen 7040・8040（780M など）の内蔵 GPU。840M は gfx1152 と gfx1153 の両方があるので `--amd-device` で指定 |

開発に使っているのは RX 9070 XT（gfx1201）です。Demucs・ChordMini の BTC・Whisper が Windows の ROCm で
動くかは、まだすべては確かめていません。

## 使い方

1. このフォルダを `C:\chordgen` などに置きます（git clone でも zip の展開でも）。
2. `run.bat` をダブルクリックします。初回は自動でセットアップが走ります（`setup.bat` を先に実行しても同じです）。
3. ブラウザで画面が開きます（下の「画面（GUI）」）。音源のファイルを選ぶか URL を入れて、解析を始めます。

- 出力は `output\` に、取得した音源などの作業ファイルは `work\` に置かれます。

### 画面（GUI）

`run.bat` を引数なしで実行すると、ローカルのサーバー（`python -m chordgen.gui`）が立ち上がり、ブラウザで画面が開きます。
ComfyUI などと同じ形で、画面はこのパソコンの中だけで動きます（外部のサイトや CDN には繋ぎません）。

- コンソールに `http://127.0.0.1:8765/?t=…` の形の URL が出ます。ブラウザが開かないときや、閉じてしまったときは、
  この URL を開いてください。`t=` の後ろは起動のたびに変わる合言葉で、ほかのサイトから解析を投げ込まれたり、
  結果を読まれたりしないためのものです（合言葉なしで開くと「コンソールに出ている URL から開いてください」と出ます）。
- 終わるときは、画面の「終了」を押すか、コンソールで Ctrl+C を押します（実行中の解析は止めます）。
- 解析は 1 件ずつ順に走ります（GPU を取り合わないため）。続けて投入すると待ちの列に並びます。中止もできます。
- 8765 番のポートが使われているときは、空いているポートを探して使います。もう起動しているときに `run.bat` を
  もう一度実行すると、起動中の画面をブラウザで開いて終わります。
- 画面から投入した音源は `work\gui\uploads\` に置き、解析が終わったら消します（「作業ファイルを残す」のときは残します）。
  ジョブの一覧は `work\gui\jobs.json` に書いておき、起動し直しても新しい 100 件まで残ります。

```bat
run.bat --gui --port 8800          :: ポートを指定する（使用中ならエラー）
run.bat --gui --no-browser         :: ブラウザを自動で開かない
```

セットアップを済ませた後に画面の部品（fastapi・uvicorn・python-multipart）が足されたので、前から使っている
フォルダで「画面に要るライブラリが入っていません」と出たら、`setup.bat` をもう一度実行してください。

### コマンドライン

引数をつけて `run.bat` を実行すると、画面を使わずに解析します。

```bat
run.bat 自作の曲.wav
run.bat 自作の曲.wav --lyrics 歌詞.txt --title "曲名" --artist "名前"
run.bat "https://example.com/watch?v=xxxx"
```

- URL は `&` を含むことが多いので、必ず `"` で囲んでください。
- 引数の一覧は `run.bat --help`（`python -m chordgen.cli --help` と同じ）で見てください。

### セットアップ（`setup.bat`）

`setup.bat` は何度実行してもかまいません。済んだ手順は飛ばし、途中で失敗したら続きから進みます。

| 手順 | 内容 |
|---|---|
| 0 | Python 3.12.16（[python-build-standalone](https://github.com/astral-sh/python-build-standalone) の install_only 版）と Node.js 24.22.3（LTS）を `runtime\` に取得し、sha256 を確かめて展開 |
| 1 | GPU を見分けて PyTorch を入れる（下の表） |
| 2 | 解析に使うライブラリ（`pyproject.toml` と `portable\requirements-*.txt`）を入れる |
| 3 | media-dl を `npm install` し、`media-dl setup` で yt-dlp・ffmpeg・ffprobe・deno を `runtime\media-dl\bin` に入れる |
| 4 | `torch.cuda.is_available()` を確かめて表示する |

| 種類 | 既定 | 切り替え |
|---|---|---|
| AMD | ROCm 10.2.0、torch 2.15.0、torchaudio 2.11.1（`https://stable.repo.amd.com/rocm/whl-next/`） | `--rocm 7.2.1`: ROCm 7.2.1、torch 2.9.1、torchaudio 2.9.1（`https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`） |
| NVIDIA | CUDA 13.2、torch 2.15.0、torchaudio 2.11.1（`https://download.pytorch.org/whl/cu132`） | `--cuda 12.6`: torch 2.11.0、torchaudio 2.11.0 |
| CPU | torch 2.15.0、torchaudio 2.11.1（`https://download.pytorch.org/whl/cpu`） | |

```bat
setup.bat --gpu nvidia            :: 自動判定を上書き（auto / amd / nvidia / cpu）
setup.bat --gpu amd --amd-device gfx1201
setup.bat --rocm 7.2.1            :: ROCm 10.2 で問題が出たときに戻す
setup.bat --force                 :: すべてやり直す
setup.bat --skip-media-dl         :: URL の入力を使わない
```

引数の代わりに環境変数 `CHORDGEN_GPU`・`CHORDGEN_AMD_DEVICE`・`CHORDGEN_ROCM`・`CHORDGEN_CUDA` でも指定できます。
NVIDIA と AMD の単体 GPU が両方あるときは、自動では決めずに `--gpu` の指定を求めます。
torchvision は解析で使わないので入れません。版と取得元は `portable\sources.json` にまとめてあります。

## フォルダの外に書き出さない仕組み

`run.bat` は、次の環境変数をこのフォルダの中に向けてから起動します。`setlocal` の中で変えるので、
Windows の設定は変わりません。

| 書き出すもの | 環境変数 | 置き場所 |
|---|---|---|
| media-dl の yt-dlp・ffmpeg・ffprobe・deno | `MEDIA_DL_BIN_DIR` | `runtime\media-dl\bin` |
| yt-dlp のキャッシュ | `XDG_CACHE_HOME`、`HOME` | `cache\xdg`、`cache\home` |
| 設定ファイル | `XDG_CONFIG_HOME` | `cache\config` |
| MIOpen | `MIOPEN_USER_DB_PATH`、`MIOPEN_CUSTOM_CACHE_DIR` | `cache\miopen\` |
| モデルの重み | `HF_HOME`、`TORCH_HOME` | `models\huggingface`、`models\torch` |
| Triton・pip・deno のキャッシュ | `TRITON_CACHE_DIR`、`PIP_CACHE_DIR`、`DENO_DIR` | `cache\` |
| 一時ファイル | `TEMP`、`TMP` | `work\tmp` |
| ユーザー側の Python パッケージを読まない | `PYTHONNOUSERSITE=1` | |

ほかに npm（`npm_config_cache` など）・CUDA の ComputeCache・matplotlib・numba・torch の拡張と inductor の
キャッシュも `cache\` に向けています。`PATH` の先頭には同梱の Python・Node.js・`MEDIA_DL_BIN_DIR` を置くので、
ffmpeg は media-dl と解析の側で同じものを使います。chordgen 自体はインストールせず `PYTHONPATH` で読むので、
フォルダを移しても動きます。

環境変数を見ずに書き出すライブラリが混ざっていないかは、次の道具で確かめられます
（`%USERPROFILE%\.cache`・`%LOCALAPPDATA%`・`%APPDATA%`・`%USERPROFILE%` の直下を実行の前後で比べます）。

```bat
runtime\python\python.exe portable\check_outside_writes.py run -- run.bat 自作の曲.wav
```

## モデルの重み

重みはリポジトリにも配布物にも入れず、初回の解析のときに自動でダウンロードします。

| モデル | 置き場所 | 取得元 | ライセンス |
|---|---|---|---|
| コード認識: ChordMini の BTC（CL full、`btc_model_best.pth`） | `models\chords\` | [ptnghia-j/ChordMini](https://github.com/ptnghia-j/ChordMini) | 重みのライセンスは明記されていない。元の BTC から作られているので **CC-BY-NC-SA 4.0（非商用）** として扱う |
| コード認識（比較用）: 元の BTC（`btc_model_large_voca.pt`） | `models\chords\` | [jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) | **CC-BY-NC-SA 4.0（非商用）** |
| 拍・ダウンビート: Beat This! | `models\beat_this\` | [CPJKU/beat_this](https://github.com/CPJKU/beat_this) | 配布元で確認 |
| 音源分離: Demucs（HTDemucs） | `models\huggingface\` | [adefossez/demucs](https://github.com/adefossez/demucs) | 配布元で確認 |
| 歌詞の位置合わせ: Whisper | `models\whisper\` | [openai/whisper](https://github.com/openai/whisper) | MIT |

BTC と ChordMini の重みは非商用として扱い、商用には使わないでください。

## cat4g への読み込み

`output\` に書き出された `.cat4g.json` を、cat4g の読み込み（インポート）から開きます。
ファイルは書き出す前に毎回 JSON Schema（`specs\format\cat4g-song.v1.schema.json`）で検証しています。

## 精度の評価

自分で耳コピしたコード譜（正解）を用意し、`mir_eval` で条件ごとの正解率を比べます。

```bat
run.bat --python -m chordgen.eval --help
```

`run.bat --python <引数>` は、run.bat の環境変数を設定したうえで同梱の Python を呼びます
（同じく `run.bat --media-dl update` で同梱の media-dl を呼べます。yt-dlp が古いと言われたときに使います）。
著作物の曲を使うときは、音源と正解のコード譜をこのフォルダの外の手元に置いてください。

## ライセンスと注意

- 著作物の音源や歌詞をリポジトリに入れないでください。テストには自作の音源と歌詞を使います。
- URL から音源を取得してよいかどうかは、各サイトの利用規約と著作権法に従って判断してください。
- media-dl の `setup` が取得する ffmpeg は [ffmpeg-static](https://github.com/eugeneware/ffmpeg-static) の配布物（GPL）です。
  このアプリを人に配るときは、ライセンスの条件を配布元で確かめてください。
- ROCm の Windows 版はプレリリースの部分があり、安定性に影響することがあります。

## 開発

WSL / Linux で開発し、テストは pytest で動かします（GPU は要りません）。

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/python -m pytest
```

| ファイル | 役目 |
|---|---|
| `run.bat` | 環境変数をこのフォルダに向けて、引数なしなら `python -m chordgen.gui`、引数があれば `python -m chordgen.cli` を起動。未セットアップならセットアップを走らせる。cmd.exe はコンソールのコードページで .bat を読むので ASCII だけで書く |
| `setup.bat` | `run.bat --setup` を呼ぶだけ |
| `portable\bootstrap.ps1` | Python と Node.js の取得・展開（Windows PowerShell 5.1。UTF-8 BOM 付き） |
| `portable\install.py` | GPU の判定・PyTorch・依存・media-dl・確認。`--dry-run` で手順だけ表示できる（Linux でも動く） |
| `portable\sources.json` | Python・Node.js・PyTorch の版・URL・sha256 |
| `portable\requirements-*.txt` | 解析のライブラリ。torch は書かない。install.py が `runtime\torch-constraints.txt` を `-c` で渡して torch を固定する。依存を解決させたくないものは `requirements-nodeps.txt`（`--no-deps`） |
| `portable\check_outside_writes.py` | フォルダの外への書き出しを調べる |
| `package.json` | media-dl（GitHub Release の tarball）。版を替えるときはこの URL だけを直す |
| `chordgen\gui\` | 画面のサーバー（FastAPI）。API は `chordgen\gui\API.md`。画面は `static\` の静的ファイルで、ビルドの手順はない |

画面を作るときは、本物の解析の代わりに偽のジョブを走らせられます（数秒で全段の進捗を出し、サンプルを元にした文書を書く。
URL や題名に `fail` を入れると失敗、`slow` で遅く、`warn` で警告）。

```bash
CHORDGEN_GUI_FAKE=1 .venv/bin/python -m chordgen.gui
```

### まだ確かめていないこと

- Windows の実機での起動（WSL の中から cmd.exe で run.bat の環境変数の設定までは確かめた）。
  画面のサーバーの起動・ブラウザの自動起動・中止のときの `taskkill /T /F` も、Windows の実機ではまだ確かめていません。
- media-dl の tarball は、まだ GitHub Release に添付されていません。`package.json` の URL
  （タグ `v1.5.0` は仮）は、添付されたら実際のタグに直してください。取得できない間は、セットアップは
  media-dl を飛ばして続け、ファイルの入力だけが使えます。
- ROCm 10.2.0・torch 2.15.0 の上で Demucs・Beat This!・stable-ts（torchaudio 2.11）が動くか。
