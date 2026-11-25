"""CaT4G Song Format v1（specs/11_song_format.md）の文書を組み立て、検証して書き出す。"""

from chordgen.songformat.build import build_document
from chordgen.songformat.chordname import harte_to_cat4g, key_to_cat4g
from chordgen.songformat.validate import SongFormatError, validate_document, write_document

__all__ = [
    "SongFormatError",
    "build_document",
    "harte_to_cat4g",
    "key_to_cat4g",
    "validate_document",
    "write_document",
]
