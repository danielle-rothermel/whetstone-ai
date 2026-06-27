from __future__ import annotations

import bz2
import gzip
import lzma
import zlib
from enum import StrEnum

import zstandard
from pydantic import BaseModel, ConfigDict


class CompressionMethod(StrEnum):
    RAW = "raw"
    ZLIB = "zlib"
    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"
    ZSTD = "zstd"


class CompressionMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: CompressionMethod
    ground_truth_bytes: int
    representation_bytes: int
    compressed_bytes: int
    ratio_to_ground_truth: float | None
    percent_reduction_vs_ground_truth: float | None


def compressed_bytes(value: bytes, method: CompressionMethod) -> bytes:
    if method is CompressionMethod.RAW:
        return value
    if method is CompressionMethod.ZLIB:
        return zlib.compress(value)
    if method is CompressionMethod.GZIP:
        return gzip.compress(value)
    if method is CompressionMethod.BZ2:
        return bz2.compress(value)
    if method is CompressionMethod.LZMA:
        return lzma.compress(value)
    if method is CompressionMethod.ZSTD:
        return zstandard.ZstdCompressor().compress(value)
    raise ValueError(f"unsupported compression method: {method}")


def compression_metrics(
    *,
    ground_truth_code: str,
    representation_text: str,
    methods: tuple[CompressionMethod, ...] = tuple(CompressionMethod),
) -> list[CompressionMetric]:
    ground_truth_bytes = len(ground_truth_code.encode("utf-8"))
    representation = representation_text.encode("utf-8")
    representation_bytes = len(representation)
    metrics: list[CompressionMetric] = []
    for method in methods:
        size = len(compressed_bytes(representation, method))
        ratio = size / ground_truth_bytes if ground_truth_bytes else None
        metrics.append(
            CompressionMetric(
                method=method,
                ground_truth_bytes=ground_truth_bytes,
                representation_bytes=representation_bytes,
                compressed_bytes=size,
                ratio_to_ground_truth=ratio,
                percent_reduction_vs_ground_truth=(
                    (1.0 - ratio) * 100.0 if ratio is not None else None
                ),
            )
        )
    return metrics
