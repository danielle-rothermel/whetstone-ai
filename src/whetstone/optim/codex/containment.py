"""D13 containment posture for the Codex-direct optimizer.

One owner for the containment facts the control pins, the runner enforces,
and the output artifact reports: which Codex CLI features are denied, and
what the process, network, and filesystem posture is.
"""

from __future__ import annotations

from enum import UNIQUE, StrEnum, verify

#: D13, stated explicitly. Network is allowed because the Codex CLI must
#: reach its own model. The only writable roots are the per-run scratch
#: directory and the run's own state paths.
CODEX_NETWORK_POLICY = "allowed"
CODEX_FILESYSTEM_POLICY = "scratch_only"
CODEX_CONTAINMENT_PROFILE = "process_boundary_only"

#: The auth material the Codex CLI accepts, in the order it looks for it.
#: One owner, because two consumers must not drift: the preflight checks
#: that a usable auth source exists, and the runner stages exactly these
#: files into each run's scratch CODEX_HOME. A file the preflight accepts
#: and the runner never copies would pass the check and then fail the run.
CODEX_AUTH_FILENAMES = ("auth.json", ".credentials.json")
#: The environment variable that is an auth source on its own.
CODEX_AUTH_ENV_KEY = "OPENAI_API_KEY"

#: Default cap on the retained Codex stdout+stderr payload. Codex emits one
#: JSONL event per turn, so a run that overruns this is producing output the
#: adapter would not read anyway.
CODEX_DEFAULT_MAX_OUTPUT_BYTES = 16 * 1024 * 1024


@verify(UNIQUE)
class CodexDeniedFeature(StrEnum):
    APPS = "apps"
    BROWSER_USE = "browser_use"
    BROWSER_USE_EXTERNAL = "browser_use_external"
    CODE_MODE = "code_mode"
    CODE_MODE_HOST = "code_mode_host"
    CODE_MODE_ONLY = "code_mode_only"
    COLLABORATION_MODES = "collaboration_modes"
    COMPUTER_USE = "computer_use"
    DEFAULT_MODE_REQUEST_USER_INPUT = "default_mode_request_user_input"
    GOALS = "goals"
    HOOKS = "hooks"
    IMAGE_GENERATION = "image_generation"
    IN_APP_BROWSER = "in_app_browser"
    MEMORIES = "memories"
    MULTI_AGENT = "multi_agent"
    MULTI_AGENT_V2 = "multi_agent_v2"
    PLUGINS = "plugins"
    REMOTE_PLUGIN = "remote_plugin"
    REQUEST_PERMISSIONS_TOOL = "request_permissions_tool"
    SHELL_SNAPSHOT = "shell_snapshot"
    SHELL_TOOL = "shell_tool"
    SKILL_MCP_DEPENDENCY_INSTALL = "skill_mcp_dependency_install"
    SKILL_SEARCH = "skill_search"
    STANDALONE_WEB_SEARCH = "standalone_web_search"
    TOOL_CALL_MCP_ELICITATION = "tool_call_mcp_elicitation"
    TOOL_SUGGEST = "tool_suggest"
    UNIFIED_EXEC = "unified_exec"
    WEB_SEARCH_CACHED = "web_search_cached"
    WEB_SEARCH_REQUEST = "web_search_request"
    WORKSPACE_DEPENDENCIES = "workspace_dependencies"


#: The frozen ordered deny list passed to the CLI as ``--disable`` flags and
#: recorded verbatim in the control and the artifact isolation block.
CODEX_DENIED_FEATURES: tuple[str, ...] = tuple(
    feature.value for feature in CodexDeniedFeature
)


__all__ = [
    "CODEX_CONTAINMENT_PROFILE",
    "CODEX_DEFAULT_MAX_OUTPUT_BYTES",
    "CODEX_DENIED_FEATURES",
    "CODEX_FILESYSTEM_POLICY",
    "CODEX_NETWORK_POLICY",
    "CodexDeniedFeature",
]
