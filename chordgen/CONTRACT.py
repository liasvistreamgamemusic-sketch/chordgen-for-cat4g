"""モジュールごとの公開関数の一覧（実装は各モジュール）。import はしない、署名の控え。

chordgen.audio_io
    probe_audio(path: Path) -> AudioInfo                      # 長さ・タグ・ファイル名から題名
    load_mono(path: Path, sample_rate: int) -> np.ndarray     # float32 モノラル
chordgen.fetch
    fetch_info(url: str) -> TrackMeta
    fetch_audio(url: str, dest_dir: Path, on_progress: Callable[[dict], None] | None = None) -> FetchResult
    class FetchError(Exception): code: str; user_message: str; exit_code: int
chordgen.separation
    separate(audio_path: Path, out_dir: Path, device: str) -> Stems
    make_chord_input(audio_path: Path, stems: Stems, mode: ChordInputMode, out_dir: Path) -> Path
chordgen.bass
    detect_bass(bass_path: Path, beats: BeatGrid) -> list[BassNote]     # 拍ごと
chordgen.beats
    detect_beats(audio_path: Path, device: str) -> BeatGrid
chordgen.key
    estimate_key(audio_path: Path, chords: list[ChordSegment] | None = None) -> KeyEstimate
chordgen.chords
    recognize_chords(audio_path: Path, device: str, model: str = "chordmini") -> list[ChordSegment]
        # model: "chordmini"（CL full、既定）/ "btc"（元の BTC、比較用）
chordgen.postprocess
    snap_to_beats(chords: list[ChordSegment], beats: BeatGrid) -> list[ChordSegment]
    apply_bass(chords: list[ChordSegment], bass: list[BassNote], mode: BassMode) -> list[ChordSegment]
chordgen.lyrics
    align_lyrics(audio_path: Path, lyrics_text: str, device: str, language: str | None = None) -> list[AlignedLine]
chordgen.songformat
    harte_to_cat4g(label: str, key: KeyEstimate | None = None) -> str | None   # N / X は None
    build_document(result: AnalysisResult, *, bars_per_line: int = 4) -> dict
    validate_document(doc: dict) -> None                                        # 通らなければ例外
    write_document(doc: dict, path: Path) -> None                               # 検証してから書く
chordgen.pipeline
    analyze(source: str | Path, *, lyrics_text=None, title=None, artist=None, options: PipelineOptions) -> AnalysisResult
chordgen.cli
    main(argv: list[str] | None = None) -> int
chordgen.eval
    python -m chordgen.eval ...   # mir_eval で条件ごとの正解率を比べる
"""
