from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

DEFAULT_COST_SIGNIFICANT_DIGITS = 6
PRICE_PER_THOUSAND_SAMPLE_MULTIPLIER = 1000.0


def variance_or_none(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.variance(values)


def average_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def format_cost(
    value: float | None,
    *,
    significant_digits: int = DEFAULT_COST_SIGNIFICANT_DIGITS,
) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    decimals = max(
        significant_digits - math.floor(math.log10(abs(value))) - 1,
        0,
    )
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def align_decimal_column(formatted_values: Sequence[str]) -> list[str]:
    split_values = [
        formatted_value.partition(".")
        for formatted_value in formatted_values
        if formatted_value
    ]
    integer_width = max(
        (len(integer) for integer, _, _ in split_values), default=0
    )
    fractional_width = max(
        (len(fractional) for _, _, fractional in split_values),
        default=0,
    )

    aligned_values = []
    for formatted_value in formatted_values:
        if not formatted_value:
            aligned_values.append("")
            continue
        integer, separator, fractional = formatted_value.partition(".")
        aligned_integer = integer.rjust(integer_width)
        if not separator:
            if fractional_width == 0:
                aligned_values.append(aligned_integer)
                continue
            if "e" in integer.lower():
                aligned_values.append(
                    aligned_integer + "".ljust(fractional_width + 1)
                )
                continue
            aligned_values.append(
                f"{aligned_integer}.{''.ljust(fractional_width, '0')}"
            )
            continue
        aligned_values.append(
            f"{aligned_integer}.{fractional.ljust(fractional_width, '0')}"
        )
    return aligned_values


def format_float_column(values: Sequence[float | None]) -> list[str]:
    return align_decimal_column([format_float(value) for value in values])


def format_cost_column(values: Sequence[float | None]) -> list[str]:
    return align_decimal_column([format_cost(value) for value in values])


def price_per_thousand_samples(value: float | None) -> float | None:
    if value is None:
        return None
    return value * PRICE_PER_THOUSAND_SAMPLE_MULTIPLIER


def sum_present_float(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)
