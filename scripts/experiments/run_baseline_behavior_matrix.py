#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from whetstone.envs.ed1_behavior_matrix import (
    DEFAULT_CONCURRENCY,
    run_ed1_baseline_behavior_matrix,
)


def _baseline_provider_routes():
    routes_path = (
        Path(__file__).resolve().with_name("code_comp_matrix_routes.py")
    )
    spec = importlib.util.spec_from_file_location(
        "code_comp_matrix_routes",
        routes_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load matrix routes from {routes_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.baseline_provider_routes()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed ED1 baseline behavior matrix."
    )
    parser.add_argument("--evaluation-python", required=True, type=Path)
    parser.add_argument("--snapshot-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run all four routes on one task, one repeat, unbudgeted",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_ed1_baseline_behavior_matrix(
        provider_routes=_baseline_provider_routes()(),
        evaluation_python=args.evaluation_python,
        snapshot_path=args.snapshot_path,
        output_dir=args.output_dir,
        resume=args.resume,
        concurrency=args.concurrency,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
