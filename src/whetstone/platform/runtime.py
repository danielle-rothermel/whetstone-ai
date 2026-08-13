from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dr_platform.runtime.database import upgrade_platform_schema as upgrade_platform_schema
    from dr_platform.runtime.dbos import (
        PlatformDbosConfig,
        initialize_dbos_runtime,
    )


def _require_platform_extra() -> None:
    try:
        import dr_platform  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Platform runtime requires the optional platform extra: "
            "pip install 'whetstone-ai[platform]'"
        ) from exc


def __getattr__(name: str):
    _require_platform_extra()
    if name == "PlatformDbosConfig":
        from dr_platform.runtime.dbos import PlatformDbosConfig

        return PlatformDbosConfig
    if name == "initialize_dbos_runtime":
        from dr_platform.runtime.dbos import initialize_dbos_runtime

        return initialize_dbos_runtime
    if name == "upgrade_platform_schema":
        from dr_platform.runtime.database import upgrade_platform_schema

        return upgrade_platform_schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PlatformDbosConfig",
    "initialize_dbos_runtime",
    "upgrade_platform_schema",
]
