"""GUI の画面（chordgen/gui/static/）の静的な検査。ブラウザは使わない。"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "chordgen" / "gui" / "static"
JS_FILES = sorted(STATIC.glob("*.js"))


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_files_exist():
    for name in ("index.html", "app.css", "app.js"):
        assert (STATIC / name).is_file(), name


def test_index_links_local_assets():
    html = _read("index.html")
    assert re.search(r'<link[^>]+href="app\.css"', html)
    assert re.search(r'<script[^>]+type="module"[^>]+src="app\.js"', html)
    assert '<html lang="ja">' in html


@pytest.mark.parametrize("path", [STATIC / "index.html", STATIC / "app.css", *JS_FILES], ids=lambda p: p.name)
def test_no_external_resources(path: Path):
    """オフラインで動くこと: CDN・Web フォントを読まない。"""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", text)
    assert not re.search(r"@import|url\(\s*['\"]?(https?:)?//", text)
    assert "fonts.googleapis" not in text


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_imports_resolve(path: Path):
    text = path.read_text(encoding="utf-8")
    for spec in re.findall(r"""^import .* from ["'](.+?)["'];""", text, flags=re.M):
        assert spec.startswith("./"), spec
        assert (STATIC / spec[2:]).is_file(), spec


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
def test_no_blocking_dialogs(path: Path):
    """削除などの確認は画面の中で行う（confirm / alert / prompt を使わない）。"""
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"(?<![\w.])(confirm|alert|prompt)\(", text)
    assert not re.search(r"window\.(confirm|alert|prompt)\(", text)


def test_ids_used_by_script_exist():
    html = _read("index.html")
    js = "\n".join(p.read_text(encoding="utf-8") for p in JS_FILES)
    html_ids = set(re.findall(r'\bid="([^"]+)"', html))
    created = set(re.findall(r"""\bid:\s*["']([^"']+)["']""", js))
    used = set(re.findall(r"""\$\(["']([^"']+)["']\)""", js))
    missing = used - html_ids - created
    assert not missing, f"index.html にない id: {sorted(missing)}"


def test_fetch_same_origin():
    """API は同じオリジンへ Cookie つきで送る。"""
    api = _read("api.js")
    assert 'credentials: "same-origin"' in api
    assert not re.search(r"""fetch\(\s*["'](https?:)?//""", api)


def test_color_schemes_and_focus():
    css = _read("app.css")
    assert "prefers-color-scheme: dark" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css


def test_position_counts_code_points():
    """position はコードポイントで数える（仕様 9 章）。UTF-16 の length で数えていないこと。"""
    js = _read("preview.js")
    assert "Array.from(" in js
    assert "Intl.Segmenter" in js  # 書記素で並べ、絵文字の並びを崩さない
