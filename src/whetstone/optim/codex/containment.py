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

#: ``code_mode_host`` is deliberately NOT denied. It reads like an agent
#: capability, but from Codex 0.148 it is the host that *routes MCP tool
#: calls*: with it disabled the agent is told "Code Mode is unavailable"
#: and sees no server-specific MCP tools at all, so ``evaluate_candidate``
#: is invisible and the agent can only return an empty artifact. Denying it
#: did not contain the agent; it silently removed the one tool the run
#: exists to drive. Code Mode itself stays denied through ``code_mode`` and
#: ``code_mode_only``.

#: Web search is enabled by default from Codex 0.148, and the
#: ``web_search_cached`` / ``web_search_request`` feature flags that used to
#: gate it are deprecated: passing them disables nothing and makes the real
#: CLI emit a deprecation ``error`` item into the JSONL transcript. The
#: top-level config key is the only thing that still turns it off, so the
#: containment posture names it here and the runner writes it on every run.
CODEX_WEB_SEARCH_CONFIG_KEY = "web_search"
CODEX_WEB_SEARCH_DISABLED = "disabled"

#: The Codex CLI resolves agent-extension roots -- the skills loader's
#: ``~/.agents/skills`` among them -- from ``HOME``, and it scans them at
#: startup before it reads any config. Nothing in the CLI's configuration
#: turns that scan off: ``skills.config=[]``,
#: ``skills.bundled.enabled=false``, ``skills.include_instructions=false``,
#: and the ``skill_search`` / ``skill_mcp_dependency_install`` feature
#: denials were each measured against the real 0.148 binary and left the
#: scan running.
#:
#: So the run moves the root instead of trying to silence the scan. Each
#: run points ``HOME`` at its own scratch directory, which makes the
#: agent's home an empty directory it already owns. This is containment,
#: not cosmetics: the real home holds the user's dotfiles, credentials,
#: and the skill trees that ``~/.agents/skills`` symlinks into, and the
#: agent's single MCP tool is meant to be its only capability.
#:
#: The alternative -- granting the profile a read rule over the scanned
#: paths -- was measured and rejected: the entries are symlinks into the
#: dotfiles repository, so satisfying the scan would have handed the
#: untrusted agent read access to that tree.
#:
#: Under ``(deny default)`` the un-redirected scan also failed with EPERM
#: and logged a ``failed to scan skill path`` ERROR line. The run still
#: succeeded, but the line landed in the stderr tail quoted by unrelated
#: failures, where it read like their cause.
CODEX_AGENT_HOME_ENV_KEY = "HOME"
#: The scratch subdirectory ``HOME`` points at, relative to a run's root.
CODEX_AGENT_HOME_DIR_NAME = "agent-home"

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
    WORKSPACE_DEPENDENCIES = "workspace_dependencies"


#: The frozen ordered deny list passed to the CLI as ``--disable`` flags and
#: recorded verbatim in the control and the artifact isolation block.
CODEX_DENIED_FEATURES: tuple[str, ...] = tuple(
    feature.value for feature in CodexDeniedFeature
)


__all__ = [
    "CODEX_AGENT_HOME_DIR_NAME",
    "CODEX_AGENT_HOME_ENV_KEY",
    "CODEX_CONTAINMENT_PROFILE",
    "CODEX_DEFAULT_MAX_OUTPUT_BYTES",
    "CODEX_DENIED_FEATURES",
    "CODEX_FILESYSTEM_POLICY",
    "CODEX_NETWORK_POLICY",
    "CODEX_WEB_SEARCH_CONFIG_KEY",
    "CODEX_WEB_SEARCH_DISABLED",
    "CodexDeniedFeature",
]
