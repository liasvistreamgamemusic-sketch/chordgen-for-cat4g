# CaT4G - 曲の形式(CaT4G Song Format)

**状態: 版 1**

CaT4G の曲 1 つを、全情報入りの JSON 文書として表す形式。アプリの保存形式であり、ファイルの書き出し・読み込みにも使う。
この文書は、**音源を解析してこの形式を出力する仕組みを作る人**向けに、各項目の意味と決まりを説明する。
アプリのソースコードを読まなくても、正しいファイルを作れることを目標にする。

| 公開物 | 場所 |
|------|------|
| JSON Schema(1 曲) | `specs/format/cat4g-song.v1.schema.json` |
| JSON Schema(ライブラリ) | `specs/format/cat4g-library.v1.schema.json` |
| サンプル(U-Fret 相当: 位置だけ) | `specs/format/samples/ufret-like.cat4g.json` |
| サンプル(解析結果相当: 小節・拍・時刻・歌詞タイミングつき) | `specs/format/samples/analyzed.cat4g.json` |

アプリの読み込み時の検査と JSON Schema は同じ内容にしてあり、テスト(`src/lib/songFormat/schemaParity.test.ts`)で一致を確かめている。
JSON Schema だけでは書けない決まりは「[アプリだけが検査すること](#アプリだけが検査すること)」にまとめた。

---

## 1. ファイルの種類

| ファイル | 拡張子 | `format` | 中身 |
|---------|--------|----------|------|
| 1 曲 | `.cat4g.json` | `"cat4g-song"` | 曲 1 つ |
| ライブラリ全体 | `.cat4g-library.json` | `"cat4g-library"` | 全曲・プレイリスト・押さえ方の全体設定 |

- 文字コードは UTF-8 の JSON。解析の仕組みが作るのは、通常は 1 曲ファイル。
- アプリの読み込みは**常に追加**。既存の曲を上書きも統合もしない。同じ題名の曲があっても別の曲として増える。

## 2. 封筒(ファイルの最上位)

```json
{
  "format": "cat4g-song",
  "version": 1,
  "exportedAt": "2026-11-11T09:00:00.000Z",
  "appVersion": "my-analyzer 0.3.0",
  "song": { "...": "曲の文書(4 章)" }
}
```

| 項目 | 型 | 内容 |
|------|----|------|
| `format` | 文字列 | `"cat4g-song"` 固定 |
| `version` | 整数 | 形式の版。今は `1` |
| `exportedAt` | 文字列 | 作った日時。ISO 8601 の `YYYY-MM-DDTHH:MM:SS`(小数秒は任意)+ `Z` か `±HH:MM`(`2026-11-11T09:00:00.000Z`、`2026-11-11T18:00:00+09:00` など) |
| `appVersion` | 文字列 | 作ったプログラムの名前・版。自由な文字列(アプリが書き出すときはアプリの版) |
| `song` | オブジェクト | 曲の文書 |

**どの階層でも、仕様にない項目はエラーになる**(`$schema` なども不可)。独自の値は 12 章の `extensions` に入れる。

## 3. 全体の構造

```text
song                     曲
├─ メタ情報              title, artist, originalKey, bpm, timeSignature, capo, transpose,
│                        playbackSpeed, tuning, difficulty, sourceUrl, notes, tags
├─ voicings              押さえる形の名前 → 押さえ方(11 章)
├─ audio (任意)          time の基準になる音源の情報(10 章)
├─ extensions (任意)
└─ sections[]            区間(配列の順 = 曲の順)
   ├─ 区間設定           name, repeatCount, capo, transpose, bpm, playbackSpeed,
   │                     timeSignature, tempoChange, repeatMark
   ├─ id (任意)          アプリ内部の識別子(書き出すときは取り除く)
   ├─ extensions (任意)
   └─ lines[]            行
      ├─ lyrics, measures, memo
      ├─ lyricTimings (任意)   歌詞の各位置の小節・拍・時刻
      ├─ id (任意)       区間の id と同じ扱い
      ├─ extensions (任意)
      └─ chords[]        コード
         ├─ 位置         position(必須) / bar・beat(任意) / time(任意)
         ├─ 奏法         method, strokePattern, arpeggioOrder, subdivision
         ├─ 詳細         duration, techniques, dynamics, tieToNext, annotation
         ├─ voicing (任意)   このコードだけの押さえ方
         └─ extensions (任意)
```

「必須」は、項目そのものを書かなければならないもの。値に `null` を書けるものは、空のときも `null` を書く(項目を省略しない)。
「任意」の項目は、省略できる(`null` は書けない)。

## 4. 曲(song)

| 項目 | 必須 | 型 | 内容 |
|------|------|----|------|
| `title` | ○ | 文字列(1〜1000 文字) | 題名 |
| `artist` | ○ | 文字列 / null | アーティスト名 |
| `originalKey` | ○ | 文字列 / null | 原曲のキー(例 `"Am"`) |
| `bpm` | ○ | 数(1〜999)/ null | テンポ |
| `timeSignature` | ○ | `{ beats, unit }` | 拍子。4/4 → `{ "beats": 4, "unit": 4 }`。`beats` は 1〜32 の整数、`unit` は 1 / 2 / 4 / 8 / 16 |
| `capo` | ○ | 整数(-12〜24) | 曲のカポ。負の値はチューニングを下げる意味(-2 = 全音下げ) |
| `transpose` | ○ | 整数(-24〜24) | 表示用の移調(半音)。コード名そのものは書き換えない |
| `playbackSpeed` | ○ | 数(0 より大きく 10 以下) | スクロール速度の倍率。普通は `1` |
| `tuning` | ○ | 列挙 | `standard` / `half-step-down` / `whole-step-down` / `drop-d` / `dadgad` / `open-g` / `open-d` / `custom` |
| `difficulty` | ○ | 列挙 / null | `beginner` / `intermediate` / `advanced` |
| `sourceUrl` | ○ | 文字列 / null | 取り込み元の URL など |
| `notes` | ○ | 文字列 / null | 曲のメモ(長さの上限なし) |
| `tags` | ○ | 文字列の配列 | タグ名。読み込み時、同じ名前のタグがなければ作られる |
| `voicings` | ○ | オブジェクト | 押さえる形の名前 → 押さえ方(11 章)。無ければ `{}` |
| `audio` | | オブジェクト | 10 章 |
| `sections` | ○ | 配列(500 個まで) | 区間 |
| `extensions` | | オブジェクト | 12 章 |

数の範囲は、アプリの画面で選べる範囲(例: カポ -2〜12、移調 -12〜12)より広い。既存のデータを弾かないための広さで、画面の範囲外の値も読み込める。

## 5. 区間(section)

| 項目 | 必須 | 型 | 内容 |
|------|------|----|------|
| `name` | ○ | 文字列 | 区間名(例 `"Aメロ"`、`"サビ"`) |
| `repeatCount` | ○ | 整数(1〜99) | 繰り返し回数 |
| `capo` | ○ | 整数(-12〜24)/ null | この区間のカポ(**絶対値**)。null は曲のカポを使う |
| `transpose` | ○ | 整数(-24〜24)/ null | この区間だけの表示用の移調(半音、曲の移調への**足し算**)。null はなし |
| `bpm` | ○ | 数(1〜999)/ null | この区間のテンポ。null は曲のテンポ |
| `playbackSpeed` | ○ | 数(0 より大きく 10 以下)/ null | この区間のスクロール速度。null は曲の値 |
| `timeSignature` | ○ | `{ beats, unit }` / null | この区間の拍子。null は曲の拍子 |
| `tempoChange` | ○ | 列挙 / null | `rit` / `accel` / `a tempo` |
| `repeatMark` | ○ | 列挙 / null | `D.C.` / `D.S.` / `D.C. al Fine` / `D.S. al Fine` / `D.C. al Coda` / `D.S. al Coda` / `Fine` / `Segno` / `Coda` |
| `lines` | ○ | 配列(2000 個まで) | 行 |
| `id` | | 文字列(1〜100 文字) | アプリ内部の識別子。書き出すときは取り除き、読み込むときは振り直す。作る側は省略してよい |
| `extensions` | | オブジェクト | 12 章 |

## 6. 行(line)

| 項目 | 必須 | 型 | 内容 |
|------|------|----|------|
| `lyrics` | ○ | 文字列(10000 文字まで) | 歌詞。コードだけの行(イントロなど)は空文字列 `""` |
| `measures` | ○ | 整数(1〜64) | この行の小節数 |
| `memo` | ○ | 文字列 / null | 行メモ(長さの上限なし) |
| `chords` | ○ | 配列(200 個まで) | コード(7 章)。並び順は位置の順を推奨 |
| `lyricTimings` | | 配列 | 歌詞の各位置が、どの小節・拍・時刻で歌われるか(10 章) |
| `id` | | 文字列(1〜100 文字) | 区間の `id` と同じ扱い |
| `extensions` | | オブジェクト | 12 章 |

## 7. コード(chord)

| 項目 | 必須 | 型 | 内容 |
|------|------|----|------|
| `chord` | ○ | 文字列(200 文字まで。空も可) | **実際に鳴る音(実音)のコード名**(8 章)。空はアプリの画面で空のまま保存できるため許す |
| `position` | ○ | 数(0 以上) | 歌詞上の位置。**コードポイント単位**、小数可(9 章) |
| `bar` | | 整数(1 以上) | 区間の先頭から数えた小節番号(10 章) |
| `beat` | | 数(1 以上) | 小節内の拍。1 始まり、小数可。拍の単位は有効な拍子の `unit`(10 章) |
| `duration` | | 数(0 より大きい) | 長さ。`beat` と同じ単位の拍数(10 章) |
| `time` | | 数(0 以上) | 時刻(秒)。音源の再生位置は `audio.offsetSec + time`(10 章) |
| `method` | | 列挙 | 奏法。`stroke` / `arpeggio` |
| `strokePattern` | | 列挙の配列(64 個まで) | ストロークの並び。各要素 `down` / `up` / `mute` / `rest` |
| `arpeggioOrder` | | 配列 | 弾く弦の順番。要素は弦番号(1〜6 の整数)か、同時に弾く弦番号の配列(例 `[[6, 4], 3, 2, 1]`) |
| `subdivision` | | 列挙 | パターン 1 要素の音価。`4` / `8` / `16` / `8t`(8 分 3 連)/ `16t`(16 分 3 連)。省略時は「並びだけ」 |
| `techniques` | | 列挙の配列 | `hammer-on` / `pull-off` / `slide-up` / `slide-down` / `bend` / `vibrato` / `palm-mute` / `harmonic` / `let-ring` / `accent` |
| `dynamics` | | 列挙 | `ppp` / `pp` / `p` / `mp` / `mf` / `f` / `ff` / `fff` |
| `tieToNext` | | 真偽値 | 次のコードとタイでつなぐ |
| `annotation` | | 文字列 | このコードへの注釈(長さの上限なし) |
| `voicing` | | `{ shape, fingering }` | このコードだけの押さえ方(11 章) |
| `extensions` | | オブジェクト | 12 章 |

位置の情報は 3 段階あり、互いに独立している。あるものだけを書けばよい。

| 段階 | 書く項目 | 例 |
|------|---------|----|
| 歌詞上の位置だけ | `position` | U-Fret などのコード譜サイトから取り込んだ曲 |
| 小節・拍 | `position` + `bar` / `beat`(+ `duration`) | 解析結果、楽譜からの入力 |
| 音源の時刻 | `position` + `time` | 解析結果 |

## 8. コード名は実音の名前

**形式上のコード名は、すべて実際に鳴る音(実音)の名前**とする。音源から検出したコードは、そのまま書ける。
画面に出す「押さえる形」は、アプリが表示するときに次の式で計算する。

```text
押さえる形 = コード名 − 有効カポ + song.transpose + (section.transpose ?? 0)
有効カポ   = section.capo ?? song.capo
```

(単位はすべて半音。`??` は「左が null なら右」)

| 実音 `chord` | `song.capo` | `section.capo` | `song.transpose` | `section.transpose` | 押さえる形 |
|-------------|-------------|----------------|------------------|---------------------|-----------|
| `A` | 0 | null | 0 | null | A |
| `A` | 2 | null | 0 | null | G |
| `A` | 2 | 4 | 0 | null | F |
| `A` | 2 | null | 0 | 2 | A |
| `F#m` | 2 | null | 1 | null | Fm |

- 解析の仕組みは、検出した実音をそのまま `chord` に書き、弾きやすくしたいときは `capo` を設定する。コード名を形に直して書かないこと。
- コード名の書き方: ルート音(`C` `C#` `Db` … `B`。`♯` `♭` も可)+ 種類(`m` `7` `maj7` `m7` `sus4` `add9` `dim` `aug` `m7b5` など)+ 任意の分数ベース(`/F#`)。
- アプリが解釈できない文字列(`N.C.` など)はそのまま表示され、移調も押さえ方の表示もされない。

## 9. 位置(position)はコードポイント単位

`position` は「歌詞の何文字目の上にコードを置くか」を、**Unicode のコードポイント単位、0 始まり**で表す。

| 歌詞 | コードを置く文字 | `position` |
|------|----------------|-----------|
| `そらにうかぶ` | `そ` | 0 |
| `そらにうかぶ` | `う` | 3 |
| `🌂かさをさそう` | `か` | 1(`🌂` は 1 コードポイント。UTF-16 では 2 単位だが、数えるのはコードポイント) |

- 小数を書ける(例 `3.5` は 4 文字目と 5 文字目の間、`3.45` も可)。
- 歌詞の長さを超えてよい(歌詞の後ろのコード、歌詞が空の行のコード)。
- 数えるのはコードポイントで、見た目の 1 文字(書記素)ではない。結合文字や `👨‍👩‍👧` のような絵文字の並びは、複数のコードポイントとして数える。
- 数え方の例: Python は `len(text)`、JavaScript は `Array.from(text).length` がコードポイント数。

## 10. 小節・拍・時刻

- **`bar`**: 区間の先頭から数えた小節番号。**1 始まり**で、区間ごとに 1 から数え直す。
- **`beat`**: その小節の中の拍。**1 始まり**、小数可(`2.5` は 2 拍目の裏)。拍の単位は、その区間で有効な拍子(`section.timeSignature ?? song.timeSignature`)の `unit`(4/4 なら 4 分音符、6/8 なら 8 分音符)。
- **`duration`**: コードの長さ。`beat` と同じ単位の拍数。
- 行の `measures` を区間の先頭から順に足すと、各行が何小節目から始まるかが分かる。`bar` はこれと食い違わないように書くことを勧める。
- **`time`**: 音源上の時刻(秒)。基準は `song.audio`:

```text
音源の再生位置(秒) = audio.offsetSec + time      (offsetSec が無ければ 0)
```

- `repeatCount` で繰り返す区間でも、各コードが持てる `time` は 1 つだけ。繰り返しごとに時刻が違うときは、区間をまとめずに並べて書くことを勧める。
- **`lyricTimings`**: 行の歌詞のうち、`at` 文字目(9 章と同じコードポイント単位、0 以上、小数可)がどの小節・拍・時刻で歌われるか。要素は `{ "at": 数, "bar"?: 整数, "beat"?: 数, "time"?: 数 }`。`bar` / `beat` / `time` の意味はコードと同じ。

音源の情報 `audio`(任意): `{ "title"?: 文字列, "url"?: 文字列, "durationSec"?: 0 以上の数, "offsetSec"?: 数 }`。
音源そのものはファイルに含めない。今のアプリは保持と書き出しだけを行う(再生には使わない)。

## 11. 押さえ方(voicings / voicing)

押さえ方は ID ではなく実データで持つ。

| 項目 | 型 | 内容 |
|------|----|------|
| `frets` | 6 要素の配列 | 各弦のフレット番号(0〜24 の整数、0 は開放)か null(弾かない弦)。**1 弦 → 6 弦の順** |
| `fingers` | 6 要素の配列 | 各弦の指番号(0 = 押さえない、1〜4 = 人差し指〜小指)か null。1 弦 → 6 弦の順 |
| `barreAt` | 整数(0〜24)/ null | バレーするフレット |
| `barreStrings` | `[始め, 終わり]` / null | バレーする弦の範囲。**0 始まりの位置**(0 = 1 弦、5 = 6 弦)。全弦なら `[0, 5]` |
| `baseFret` | 整数(1〜24) | ダイアグラムの表示を始めるフレット(1 = ナットから) |
| `muted` | 6 要素の真偽値の配列 | ミュートする弦。1 弦 → 6 弦の順 |
| `name` | 文字列(任意) | 押さえ方の名前 |

- `frets` のフレット番号は `baseFret` からの相対ではなく、ネック上の位置そのもの。
- **弾かない弦**は、`frets` を `null`、`muted` を `true`、`fingers` を `null` にして 3 つを揃える(アプリのデータはこの形)。
  ダイアグラムの × は `muted` だけで決まるので、`frets` が `null` でも `muted` が `false` だと × が出ない。
- **開放弦**は `frets` が `0`、`muted` が `false`、`fingers` が `null`。押さえる弦は `frets` にフレット番号、`fingers` に指番号。

例: 開放の C(6 弦から書くと x32010)。配列は 1 弦 → 6 弦の順なので、6 弦(弾かない)が最後に来る。

```json
{
  "frets": [0, 1, 0, 2, 3, null],
  "fingers": [null, 1, null, 2, 3, null],
  "barreAt": null,
  "barreStrings": null,
  "baseFret": 1,
  "muted": [false, false, false, false, false, true]
}
```

例: 2 フレットの全弦バレーの Bm(6 弦から書くと 224432)なら `"frets": [2, 3, 4, 4, 2, 2]`、`"fingers": [1, 2, 4, 3, 1, 1]`、`"barreAt": 2`、`"barreStrings": [0, 5]`、`"baseFret": 2`、`muted` はすべて `false`。

押さえ方は、実音のコード名ではなく**画面に出る「押さえる形」の名前**(8 章の式の結果)に結びつく。

- `song.voicings`: `{ "形の名前": 押さえ方 }`。カポ 2 で実音 A を G の形で押さえるなら、キーは `"G"`。その曲の中で、その形が表示されるすべての箇所に効く。キーは 1〜200 文字(空は不可)。
- `chord.voicing`: `{ "shape": "形の名前(1〜200 文字)", "fingering": 押さえ方 }`。そのコード 1 か所だけに効く。表示中の形の名前が `shape` と一致するときだけ使う(移調して形が変われば使わない)。
- 表示の優先順位: そのコードの `voicing` > 曲の `voicings` > ユーザーの全体設定 > アプリの既定。
- 解析の仕組みが押さえ方を決めないなら、`voicings` は `{}` にし、`voicing` は書かない(アプリの既定やユーザーの設定で表示される)。

## 12. 拡張欄(extensions)

曲・区間・行・コードのそれぞれに、任意の `extensions`(自由な JSON オブジェクト)を置ける。
仕様にない値(確信度、解析器の名前など)はここへ入れる。**アプリは中身を解釈せず、読み込み・保存・書き出しでそのまま保つ**(下の `cat4g.` で始まるキーを除く)。

```json
"extensions": { "analyzer.confidence": 0.91, "analyzer.model": "v3" }
```

- 配列や null ではなく、オブジェクトにする。中の値は任意の JSON。入れ子は、`extensions` 自体を 1 段目として、オブジェクト・配列を **64 段まで**(それより深いとエラー。アプリだけが検査する)。
- キーは、他の仕組みと重ならないよう `"解析器の名前.項目"` のように名前空間をつけることを勧める。
- **どの深さでも、`__proto__` / `constructor` / `prototype` はキーにできない**(読み込み時にエラー)。
- `cat4g.` で始まるキーはアプリが使う。作る側は書かないこと。今使っているのは、コードの `extensions["cat4g.legacyChordName"]`(旧データのコード名の綴りの記録。200 文字以内の文字列でなければエラー)と、曲の次の 3 つ。
  - `extensions["cat4g.playSettingsBaseline"]`: 閲覧画面のバーで初めて設定を変える直前の値。「元の設定に戻す」に使う。`{ "capo": 0, "transpose": 0, "bpm": null, "timeSignature": "4/4", "playbackSpeed": 1, "playbackMode": "scroll" }` の形で、拍子は `"12/8"` など文書の拍子と同じ書き方の文字。`playbackMode` は、そのとき曲が `cat4g.playbackMode` を持っていたときだけ入る(無い控えも正しい)。読めない値(項目が足りない・型が違う・曲の同じ項目の範囲の外・知らない進め方)はエラーにせず、無いものとして扱う。曲に読める値が無いときだけ書き、編集画面で保存すると消える。
  - `extensions["cat4g.playbackMode"]`: 閲覧画面のバーで選んだ進め方。`"scroll"`(一定速度)か `"measure"`(小節単位)。
  - `extensions["cat4g.viewMode"]`: 閲覧・編集画面のバーで選んだ表示モード。`"compact"` / `"standard"` / `"detailed"` / `"lyrics-only"`。
  - 進め方と表示モードは画面の進め方・見せ方で、コード譜の中身ではないので、曲の項目にせず拡張欄に置く。利用者が選んだときだけ書き(既定と同じ値でも書く)、無い曲・知らない値の曲はエラーにせず、アプリの既定で開く。進め方は、開いた端末に前の版のアプリが保存した進め方があればそれ、無ければ一定速度(`"scroll"`)。表示モードは標準(`"standard"`)。

## 13. 版の扱い

- `version` は整数。この文書が説明するのは版 `1`。
- 項目を足す・変えるときは `version` を上げる(アプリは仕様にない項目をエラーにするので、項目を足すだけでも古いアプリは読めない)。
- アプリは、自分より新しい版のファイルを読まずに「このファイルは新しい版のアプリで作られています。アプリを更新してください。」と示して拒否する。このとき中身は検査しない。
- `version` が整数でない・1 未満のファイルはエラー。

## 14. 検証で弾く値

読み込み時、どこが悪いか(例 `song.sections[1].lines[0].chords[2].position`)を示して、ファイル全体を拒否する(何も保存しない)。

| 対象 | 決まり |
|------|--------|
| 項目 | 仕様にない項目はエラー(`extensions` の中を除く)。必須の項目が無いとエラー |
| 数 | すべて有限(NaN・Infinity は不可。`1e999` のように Infinity になる書き方も不可) |
| 文字数 | コードポイントで数える。`title` 1〜1000、`lyrics` 10000 まで、`chord` と `extensions["cat4g.legacyChordName"]` 200 まで(空も可)、`voicing.shape`・`voicings` のキー・`chordPreferences[].chordName` 1〜200、区間・行の `id` 1〜100。`notes`・`memo`・`annotation` などのそれ以外の文字列には上限なし(ファイルの大きさの上限だけがかかる) |
| 個数 | 区間 500、区間あたりの行 2000、行あたりのコード 200、`strokePattern` 64 |
| 整数 | `capo`(-12〜24)、`transpose`(-24〜24)、`measures`(1〜64)、`repeatCount`(1〜99)、`bar`(1 以上)、`timeSignature.beats`(1〜32) |
| 小数可 | `bpm`(1〜999)、`playbackSpeed`(0 より大きく 10 以下)、`position`・`at`(0 以上)、`beat`(1 以上)、`duration`(0 より大きい)、`time`・`durationSec`(0 以上)、`offsetSec`(制限なし) |
| 拍子 | `unit` は 1 / 2 / 4 / 8 / 16 |
| 押さえ方 | `frets` / `fingers` / `muted` は 6 要素。`frets` と `barreAt` は 0〜24 の整数か null、`fingers` は 0〜4 の整数か null、`barreStrings` は 0〜5 の整数 2 つか null、`baseFret` は 1〜24 の整数。`arpeggioOrder` の弦番号は 1〜6 |
| 日時 | `exportedAt`、ライブラリの `createdAt` / `updatedAt` は ISO 8601 の形(2 章)で、日時として読めること |
| キー | `extensions` の中のどこにも `__proto__` / `constructor` / `prototype` を使えない。`voicings` のキーにも使えない |

### アプリだけが検査すること

次は JSON Schema では表せないため、アプリだけが検査する。

| 決まり | 内容 |
|--------|------|
| ファイルの大きさ | 1 曲ファイルは 5 MB、ライブラリファイルは 200 MB まで |
| 新しすぎる版 | 版が新しいときは、中身を検査せず更新を促す文言だけを返す(JSON Schema では単に `version` が `1` でないエラー) |
| ライブラリの不正な曲 | ライブラリの曲が 1 つ不正でも、その曲だけを飛ばして残りを読み込む(JSON Schema ではファイル全体が不合格) |
| `extensions` の入れ子の深さ | `extensions` 自体を 1 段目として、オブジェクト・配列の入れ子は 64 段まで。深すぎると保存の処理が失敗するため |
| 日時として読めること | `2026-13-45T25:61:61Z` のように、形は ISO 8601 でも日時として読めない値は拒否する(JavaScript の `Date.parse` が NaN になる値)。Supabase が返す `2026-02-03T04:05:06.123456+00:00`(マイクロ秒・`+00:00`)は受け付ける |

数が有限であることは、JSON Schema の検証器によって扱いが違う(ajv は既定で NaN・Infinity を数と認めない)。
そもそも `NaN`・`Infinity` は JSON の値ではない。Python で書き出すときは `json.dumps(data, ensure_ascii=False, allow_nan=False)` を使う
(`allow_nan=False` で NaN・Infinity を書こうとするとその場でエラーになる。`ensure_ascii=False` で日本語を `\uXXXX` にせずそのまま書く)。

## 15. ライブラリファイル

```json
{
  "format": "cat4g-library",
  "version": 1,
  "exportedAt": "2026-11-11T09:00:00.000Z",
  "appVersion": "0.1.0",
  "songs": [
    { "id": "song-1", "playCount": 3, "createdAt": "…", "updatedAt": "…", "song": { "...": "曲の文書" } }
  ],
  "playlists": [{ "name": "朝の練習", "description": null, "songIds": ["song-1"] }],
  "chordPreferences": [{ "chordName": "G", "fingering": { "...": "押さえ方(11 章)" } }]
}
```

- `songs[]`: `id`(このファイルの中だけで使う 1〜100 文字の文字列)、`playCount`(0 以上の整数)、`createdAt` / `updatedAt`(ISO 8601)、`song`(4 章の文書)。
- `playlists[]`: `name`(文字列)、`description`(文字列 / null)、`songIds`(収録順の `songs[].id`)。
- `chordPreferences[]`: ユーザーの全体設定「コード名ごとの既定の押さえ方」。`chordName`(1〜200 文字)と `fingering`。復元するとき、同じコードの設定がすでにあれば既存を残す(異名同音の `A#` と `Bb`、種類の表記ゆれの `Cmaj7` と `CM7` なども同じコードとみなす)。
- 封筒(曲以外)が不正ならファイル全体を拒否する。曲が不正なら、その曲だけを理由と題名つきで飛ばす。

## 16. 例(注釈つき)

カポ 2 の曲。実音 A のコードは、画面には G の形で出る。

```json
{
  "format": "cat4g-song",
  "version": 1,
  "exportedAt": "2026-11-11T09:00:00.000Z",
  "appVersion": "my-analyzer 0.3.0",
  "song": {
    "title": "しろいくも",
    "artist": null,
    "originalKey": "A",
    "bpm": 90,
    "timeSignature": { "beats": 4, "unit": 4 },
    "capo": 2,
    "transpose": 0,
    "playbackSpeed": 1,
    "tuning": "standard",
    "difficulty": null,
    "sourceUrl": null,
    "notes": null,
    "tags": [],
    "voicings": {},
    "audio": { "durationSec": 64, "offsetSec": 1.2 },
    "sections": [
      {
        "name": "Aメロ",
        "repeatCount": 1,
        "capo": null,
        "transpose": null,
        "bpm": null,
        "playbackSpeed": null,
        "timeSignature": null,
        "tempoChange": null,
        "repeatMark": null,
        "lines": [
          {
            "lyrics": "そらにうかぶ しろいくも",
            "measures": 2,
            "memo": null,
            "chords": [
              { "chord": "A", "position": 0, "bar": 1, "beat": 1, "duration": 4, "time": 0,
                "extensions": { "analyzer.confidence": 0.93 } },
              { "chord": "D", "position": 7, "bar": 2, "beat": 1, "duration": 2, "time": 2.667 }
            ],
            "lyricTimings": [{ "at": 7, "bar": 2, "beat": 1, "time": 2.667 }]
          }
        ]
      }
    ]
  }
}
```

| 箇所 | 意味 |
|------|------|
| `"capo": 2` と `"chord": "A"` | 実音 A。押さえる形は A − 2 = **G**。画面には G と出る |
| `"chord": "D"` | 実音 D。押さえる形は **C** |
| `"position": 7` | 歌詞の 8 文字目(0 始まりで 7)の `し` の上。空白も 1 文字 |
| `"bar": 2, "beat": 1` | この区間の 2 小節目の 1 拍目 |
| `"duration": 2` | 2 拍(4/4 なので 4 分音符 2 つ分) |
| `"time": 2.667` と `"offsetSec": 1.2` | 音源の 3.867 秒目(1.2 + 2.667)。BPM 90 の 1 小節は 2.667 秒 |
| `lyricTimings` の `"at": 7` | `し` が 2 小節目の 1 拍目に歌われる |
| `"analyzer.confidence"` | 解析器の確信度。アプリは解釈せず保つ |
| 区間の `null` | 曲の設定(カポ・テンポ・拍子など)をそのまま使う |

## 17. JSON Schema での検証

- Schema は JSON Schema draft-07。1 曲の Schema(`cat4g-song.v1.schema.json`)の `$id` はファイル名と同じ。
- ライブラリの Schema は、曲の部分を `cat4g-song.v1.schema.json#/definitions/...` で参照する。2 つのファイルを同じフォルダに置き、検証器に両方を登録する。

```ts
// ajv 8 の例
import Ajv from 'ajv';
import songSchema from './cat4g-song.v1.schema.json';
import librarySchema from './cat4g-library.v1.schema.json';

const ajv = new Ajv({ allowUnionTypes: true });
const validateSong = ajv.compile(songSchema);
const validateLibrary = ajv.compile(librarySchema); // songSchema を先に compile しておく
```

```python
# Python(jsonschema)の例
import json
from pathlib import Path
from jsonschema import Draft7Validator

schema = json.loads(Path("cat4g-song.v1.schema.json").read_text(encoding="utf-8"))
Draft7Validator(schema).validate(data)  # data は書き出す dict。不正なら jsonschema.ValidationError

# NaN・Infinity は JSON の値ではないので allow_nan=False で書き出す(含まれていればここでエラー)
text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)
Path("output.cat4g.json").write_text(text, encoding="utf-8")
```

JSON Schema だけでは、「アプリだけが検査すること」(14 章)は確かめられない。とくに `extensions` の入れ子の深さと、日時として読めることは作る側でも守る。
