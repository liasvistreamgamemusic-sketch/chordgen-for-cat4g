"""portable/check_outside_writes.py の差分の出し方。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "portable" / "check_outside_writes.py"
    spec = importlib.util.spec_from_file_location("chordgen_check_outside_writes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cow = _load()


def test_watch_roots_uses_windows_folders():
    env = {"USERPROFILE": r"C:\Users\u", "LOCALAPPDATA": r"C:\Users\u\AppData\Local", "APPDATA": r"C:\Users\u\AppData\Roaming"}
    roots = dict(cow.watch_roots(env))
    assert roots[str(Path(r"C:\Users\u") / ".cache")] == 2
    assert roots[r"C:\Users\u\AppData\Local"] == 2
    assert roots[r"C:\Users\u\AppData\Roaming"] == 2
    assert roots[r"C:\Users\u"] == 1


def test_snapshot_and_diff(tmp_path):
    local = tmp_path / "Local"
    (local / "Existing" / "old").mkdir(parents=True)
    roots = [(str(local), 2)]
    before = cow.snapshot_tree(roots)

    (local / "huggingface" / "hub" / "deep").mkdir(parents=True)  # 新しいフォルダ（中身はまとめる）
    (local / "Existing" / "new.txt").write_text("x")
    (local / "Temp" / "tmpabc").mkdir(parents=True)
    after = cow.snapshot_tree(roots)

    new = cow.diff_snapshots(before, after)
    assert new == sorted([str(local / "Existing" / "new.txt"), str(local / "Temp"), str(local / "huggingface")])

    main, noisy = cow.classify(new, app_dir=tmp_path / "app")
    assert str(local / "huggingface") in main
    assert noisy == [str(local / "Temp")]
    text = cow.report(new, app_dir=tmp_path / "app")
    assert "huggingface" in text


def test_report_when_nothing_new(tmp_path):
    assert "できていません" in cow.report([], app_dir=tmp_path)


def test_inside_app_is_ignored(tmp_path):
    app = tmp_path / "app"
    assert cow.classify([str(app / "cache" / "x")], app_dir=app) == ([], [])


def test_snapshot_skips_missing_root(tmp_path):
    entries = cow.snapshot_tree([(str(tmp_path / "missing"), 2)])
    assert entries == {str(tmp_path / "missing"): {"dir": False, "root": True}}


def test_cli_snapshot_and_diff(tmp_path, monkeypatch, capsys):
    local = tmp_path / "Local"
    local.mkdir()
    monkeypatch.setattr(cow, "watch_roots", lambda environ=None: [(str(local), 2)])
    snap = tmp_path / "snap.json"
    assert cow.main(["snapshot", "--out", str(snap)]) == 0
    assert cow.main(["diff", "--snapshot", str(snap)]) == 0
    (local / "media-dl").mkdir()
    assert cow.main(["diff", "--snapshot", str(snap)]) == 1
    assert "media-dl" in capsys.readouterr().out
