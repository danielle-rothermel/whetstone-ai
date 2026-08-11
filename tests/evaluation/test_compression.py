from __future__ import annotations

import zstandard

from whetstone.evaluation.compression import (
    ZSTD_LEVEL,
    zstd_compressed_utf8_byte_length,
)

ENCODER_TEXT = "def f(x):\n    return x + 1\n" * 8


def test_zstd_byte_length_matches_level19_utf8_compression() -> None:
    expected = len(
        zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            ENCODER_TEXT.encode("utf-8")
        )
    )
    assert zstd_compressed_utf8_byte_length(ENCODER_TEXT) == expected
    assert ZSTD_LEVEL == 19


def test_zstd_byte_length_is_nonnegative_integer() -> None:
    value = zstd_compressed_utf8_byte_length("")
    assert isinstance(value, int)
    assert value >= 0


def test_zstd_byte_length_uses_exact_utf8_bytes() -> None:
    text = "print('π —')"
    expected = len(
        zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(
            text.encode("utf-8")
        )
    )
    assert zstd_compressed_utf8_byte_length(text) == expected
