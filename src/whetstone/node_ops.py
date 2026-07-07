"""App-side node op vocabulary for graph specs.

dr-graph treats ``op`` as an open string; whetstone's executor
dispatches on this value and its builders stamp it into specs. The
string is persisted spec content covered by graph digests — frozen.
"""

from __future__ import annotations

from typing import Final

LLM_CALL_OP: Final = "llm_call"
