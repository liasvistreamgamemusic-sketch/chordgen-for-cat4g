# chordgen GUI の API（最終版）

サーバーは `python -m chordgen.gui [--port N] [--no-browser] [--host 127.0.0.1]` で起動する
（`run.bat` を引数なしで実行しても同じ）。画面は `chordgen/gui/static/` の静的ファイルで、
`/` が `index.html`、ほかは `/app.js` のようにそのままのパスで配る。

- JSON のキーは camelCase。時刻は ISO 8601（UTC、ミリ秒つき、末尾 `Z`。例 `2026-11-26T08:38:12.745Z`）。
- 誤りは `{"error": {"code": "…", "message": "…（日本語）"}}`。
- 開発・テスト用に `CHORDGEN_GUI_FAKE=1` で起動すると、本物の解析の代わりに偽のジョブが走る（下の「偽のジョブ」）。

## 守り

| 確かめること | 合わないとき |
|---|---|
| `Host` が `127.0.0.1:PORT` か `localhost:PORT`（`--host` に IP を指定したときはその IP も） | 400（text/plain） |
| `/api/` の下は、Cookie `chordgen_token` か ヘッダ `X-Chordgen-Token` が合言葉と同じ | 403 `FORBIDDEN` |
| POST・PUT・PATCH・DELETE は、`Origin` があれば `http://127.0.0.1:PORT` か `http://localhost:PORT` | 403 `BAD_ORIGIN` |

- 合言葉は起動のたびに作り直す。コンソールに出る URL は `http://127.0.0.1:PORT/?t=<合言葉>`。
- `GET /?t=<合言葉>` は Cookie（HttpOnly、SameSite=Strict、Path=/）を付けて `/` へ 303 で移す。
  合言葉が違えば 403 の短いページ。Cookie がないまま `/` を開くと「コンソールに出ている URL から開いてください」の 403 のページ。
- `/index.html` は `/` へ移す。静的ファイル（`/app.js` など）は合言葉なしで配る（秘密は入っていない）。
- 画面は同じオリジンの `fetch` と `EventSource` を使えば Cookie が自動で付く。
- すべての応答に `X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer`、
  `Content-Security-Policy: frame-ancestors 'none'` を付ける。`/api/` と `/` は `Cache-Control: no-store`。

## GET /api/info

```json
{
  "version": "0.1.0",
  "device": {"kind": "cuda" | "cpu" | "unknown", "name": "AMD Radeon RX 9070 XT" | null, "torchAvailable": true},
  "mediaDlAvailable": true,
  "defaults": {"chordInput": "original", "bass": "off", "snap": true, "model": "chordmini",
               "barsPerLine": 4, "language": null, "keepWork": false},
  "choices": {
    "chordInput": [{"value": "original", "label": "原音のまま", "description": "…"}, …],
    "bass": [{"value": "off", "label": "使わない", "description": "…"}, …],
    "model": [{"value": "chordmini", "label": "ChordMini（BTC・CL full）", "description": "…"}, …]
  },
  "stages": [{"stage": "fetch", "label": "音源の取得"}, …],
  "limits": {"maxUploadBytes": 1073741824, "maxLyricsChars": 200000, "maxBarsPerLine": 64},
  "fake": false
}
```

- `device` は、起動のあと別スレッドで子プロセスに torch を読み込ませて調べる（サーバーには torch を読み込まない）。
  分かるまでは `{"kind": "unknown", "name": null, "torchAvailable": false}`。画面はしばらくして取り直す。
- `mediaDlAvailable` は `chordgen.fetch.find_media_dl` で判定する（偽のジョブのときは常に true）。
- `stages` は `plan_stages` が出しうる全段（fetch, probe, separate, beats, chords, key, snap, bass, lyrics, write）。
  実際に走る段はジョブごとに違う（progress の `index` / `total` を見る）。

## POST /api/jobs（multipart/form-data）

| 欄 | 内容 |
|---|---|
| `file` | 音源ファイル。`url` とどちらか一方だけ |
| `url` | `http://` か `https://` の URL |
| `lyrics` | 歌詞のテキスト（任意。空なら歌詞なし） |
| `title`, `artist` | 任意（300 文字まで） |
| `chordInput`, `bass`, `model` | `choices` の `value` のどれか |
| `snap`, `keepWork` | `"true"` / `"false"`（`1`/`0`、`on`/`off` も可） |
| `language` | 歌詞の言語（例 `ja`）。空なら自動判定 |
| `barsPerLine` | 1〜64 |

- 空の欄は省いたのと同じ（既定値）。
- 返り値は 201 で `{"id": "…", "job": Job}`。
- 誤りは 400。`code` は `BAD_FORM`（multipart でない・壊れている）、`NO_SOURCE`、`BOTH_SOURCES`、`INVALID_URL`、
  `INVALID_OPTION`、`INVALID_FIELD`、`LYRICS_TOO_LONG`、`FILE_TOO_LARGE`（1 GiB を超える）、`EMPTY_FILE`、`SHUTTING_DOWN`。
- 音源は `work/gui/uploads/<id>/<安全な名前>`、歌詞は同じフォルダの `lyrics.txt` に置き、
  ジョブが終わったら消す（`keepWork` のときは残す）。解析の作業フォルダは `work/gui/jobs/<id>/`（同じく消す）。

## Job

```json
{
  "id": "c1e255572695",
  "status": "queued" | "running" | "done" | "failed" | "cancelled",
  "sourceName": "私の曲.wav",
  "sourceKind": "file" | "url",
  "title": "曲名" | null,
  "artist": "名前" | null,
  "options": {"chordInput": "original", "bass": "off", "snap": true, "model": "chordmini",
              "language": null, "barsPerLine": 4, "keepWork": false, "hasLyrics": true},
  "createdAt": "…Z", "startedAt": "…Z" | null, "finishedAt": "…Z" | null,
  "queuePosition": 1 | null,
  "currentStage": {"stage": "chords", "label": "コード認識", "index": 3, "total": 6} | null,
  "outputName": "曲名.cat4g.json" | null,
  "error": {"code": "STAGE_FAILED", "message": "…", "stage": "chords" | null} | null
}
```

- `queuePosition` は queued のときだけ。1 が次に走るもの。
- `title` は指定したもの。指定がなければ null で、解析が終わると書き出した文書の題名になる。
- `currentStage` は最後に始まった段（終わった後も残る）。
- `error.code` は CLI の `--progress-json` の error の code（`STAGE_FAILED`、`VALIDATION_FAILED`、media-dl の code など）か、
  サーバーが付けるもの（`PROCESS_FAILED`: error を出さずに子が終わった、`START_FAILED`、`SERVER_STOPPED`: 前の起動で
  実行中・待ちのままサーバーが止まった、`UNEXPECTED`）。cancelled のときは null。

## GET /api/jobs, GET /api/jobs/{id}

- `GET /api/jobs` → `[Job]`（新しい順）。終わったジョブは `work/gui/jobs.json` に書いておき、再起動しても新しい 100 件まで残る。
- `GET /api/jobs/{id}` → Job に `events` を足したもの。ないときは 404 `JOB_NOT_FOUND`。
- `events` はこれまでの progress・warning・error の配列。どれも `at`（サーバーが受け取った時刻）を足してある。
  - `{"type": "progress", "stage", "label", "index", "total", "status": "start" | "done" | "skipped" | "failed" | "progress",
    "elapsed": 秒 | null, "message": str | null, "data": {…} | null, "at"}`
    （`status: "progress"` は URL の取得の途中。`data` は media-dl の progress（`percent`、`downloadedBytes`、`totalBytes`、
    `speedBps`、`etaSec` など）。取得中の progress は最後の 1 件だけ残す）
  - `{"type": "warning", "message", "at"}`
  - `{"type": "error", "code", "message", "stage", "exitCode", "at"}`

## GET /api/jobs/{id}/events（Server-Sent Events）

- 繋いだ時点で、これまでの progress・warning・log を全部流し直し、続けて今の `status` を 1 回送る。その後は起きた順に流す。
- イベント:
  - `progress`: data は上の progress と同じ
  - `warning`: data は上の warning と同じ
  - `status`: data は Job の全体。状態・キューの順番・段が変わるたびに送る
  - `log`: data は `{"line": "子プロセスの標準エラーの 1 行", "at"}`。1 秒に 20 行を超える分は間引き、
    `\r` で上書きされる行（tqdm など）は 1 秒に 1 行だけ。省いたときは「（ログが多いので N 行を省きました）」の行を送る。
    流し直すのは最後の 1000 行程度
- error はイベントにしない（`status` の `error` を見る）。
- 各イベントに `id:` を付ける。`Last-Event-ID` を付けて繋ぎ直すと、その続きから流す（流し直しはしない）。
- 15 秒ごとに `: ping` を送る。
- ジョブが終わったら最後の `status` を送って閉じる。**EventSource は閉じられると自動で繋ぎ直すので、
  画面は status が done / failed / cancelled になったら `close()` すること。**

## POST /api/jobs/{id}/cancel

queued ならキューから外し、running なら子プロセスをツリーごと止めて cancelled にする（running の止まり方は非同期。
返り値の Job はまだ running のことがあるので、status イベントか GET で確かめる）。
返り値は 200 で Job。終わったジョブは 409 `NOT_CANCELLABLE`、ないときは 404。

## GET /api/outputs

`output/` の `*.cat4g.json` を新しい順で返す。読めないファイルは飛ばす。

```json
[{"name": "曲名.cat4g.json", "title": "曲名", "artist": "名前" | null, "originalKey": "A" | null, "bpm": 90 | null,
  "durationSec": 64 | null, "chordCount": 8, "modifiedAt": "…Z", "size": 5739}]
```

## GET /api/outputs/{name}, DELETE /api/outputs/{name}

- `GET` はその JSON（`application/json; charset=utf-8`）。`?download=1` なら
  `Content-Disposition: attachment; filename="<ASCII にした名前>"; filename*=UTF-8''<名前を % で書いたもの>`。
- `DELETE` は消して `{"deleted": name}`。
- `name` はファイル名だけ（URL では encodeURIComponent する）。`/`・`\`・`:`・制御文字を含むもの、`.` で始まるもの
  （`..` を含む）、`.cat4g.json` で終わらないものは 400 `BAD_NAME`。`output/` の中に解決されることも確かめる。
  ないときは 404 `OUTPUT_NOT_FOUND`。

## POST /api/open-folder

body は JSON `{"target": "output" | "work"}`。`{"target": "output", "name": "曲名.cat4g.json"}` ならそのファイルを選んだ状態で開く。
Windows は `explorer`（ファイルは `explorer /select,"…"`）、macOS は `open`（`-R`）、ほかは `xdg-open`（ファイルのときはそのフォルダ）。
返り値は `{"ok": true, "path": "…"}`。開けなければ 500 `OPEN_FAILED`、誤りは 400（`BAD_BODY`、`BAD_TARGET`、`BAD_NAME`）。

## POST /api/shutdown

`{"ok": true}` を返してから、待ち・実行中のジョブを中止し（cancelled）、サーバーを終わる。

## 偽のジョブ（CHORDGEN_GUI_FAKE=1）

本物の CLI の代わりに `python -m chordgen.gui.fake_cli`（同じ引数・同じ `--progress-json` の形）を子プロセスで走らせる。
中止はプロセスを止める本物と同じ流れ。

- 走る段は本物と同じ `plan_stages`（URL なら fetch から、歌詞やベースがあれば separate など）。1 段に
  `CHORDGEN_GUI_FAKE_DELAY` 秒（既定 0.5）。fetch では progress（`percent` が 0〜100）を出す。
- `specs/format/samples/analyzed.cat4g.json` を元に、題名とアーティストを入れ替えた文書を `output/` に書く。
- 音源の名前・URL・題名に次の語が入っていると振る舞いが変わる:
  - `fail`: 失敗する（URL なら fetch で `UNAVAILABLE`、ファイルなら chords で `STAGE_FAILED`）
  - `warn`: key の段で警告を出す
  - `slow`: 1 段の時間を 4 倍にする（中止を試すとき）
- `/api/info` は `fake: true`、`mediaDlAvailable: true`、`device` は 1 秒後に `{"kind": "cpu", "name": "偽のジョブ（CHORDGEN_GUI_FAKE）", …}`。

## CLI の --progress-json（GUI が子プロセスから読むもの）

`python -m chordgen.cli <音源> --progress-json …` は標準出力に 1 行 1 JSON で次を出す（人向けの表示は標準エラー）。
ライブラリが標準出力に print したものも標準エラーに回す。終了コードは `--progress-json` なしと同じ。

- `{"type": "progress", "stage", "label", "index", "total", "status", "elapsed", "message", "data"}`（取得中の progress は 0.25 秒に 1 回まで）
- `{"type": "warning", "message"}`
- `{"type": "done", "output": "絶対パス", "title": "題名" | null}`
- `{"type": "error", "code", "message", "stage": "段" | null, "exitCode"}`
  - code: `USAGE`（引数の誤り）、`SOURCE_NOT_FOUND`、`LYRICS_NOT_FOUND`、`LYRICS_NOT_UTF8`、`LYRICS_UNREADABLE`、`INVALID_OPTIONS`、
    `STAGE_FAILED`、`VALIDATION_FAILED`、`UNEXPECTED`、`INTERRUPTED`、または media-dl の code（URL の取得の失敗。stage は fetch）

GUI は CLI に `--work-dir=<work/gui/jobs/<id>>` も渡す（作業フォルダを作る場所。CLI にも足した）。
