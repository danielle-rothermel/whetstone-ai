"""Consumer sketch 1: nl_latents' seed -> run -> read loop.

Design-validation artifact for the dr-platform facade (platform.md).
Not executable: dr_platform does not exist yet; nl_latents' domain
models are stand-ins. The test this sketch applies to the facade
(per overall.md house rule 5): it must stay small, and it must import
no library internals — only the public facade.

Lineage being fixed: the real nl_latents imported ~20 low-level
dr_llm.pool symbols and hand-rolled an axis-metadata catalog, a
filtered claim backend, a config-drift repair pass, and a bespoke
seeder. Everything it re-implemented appears here as one facade call.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from pydantic import BaseModel
from sqlalchemy import create_engine

# The entire import surface a consumer needs. No internals beyond the
# public facade. (6d validation: signatures below match the shipped
# dr-platform API.)
from dr_platform import (
    LocalDirArtifactStore,
    PlatformSchema,
    ProjectionSpec,
    await_operation,
    load_operation_snapshot,
    load_projection_frame,
    rebuild_projection,
    stable_item_id,
    submit_batch,
)

# One schema handle per app: physical naming (fresh adopters keep the
# neutral defaults).
PLATFORM = PlatformSchema()

# --- app-side domain (stays in nl_latents) --------------------------------

MODELS = ("gemma-2-2b", "llama-3.2-3b")
LAYERS = (4, 12, 20)
PROMPT_SEED = 7
RUN_NAME = "nl-latents-2026-07"
ID_NAMESPACE = "nl-latents-v1"  # app owns axis names + their stability


class LatentProbeItem(BaseModel):
    """Typed work item — satisfies dr_platform.SubmittableItem
    structurally (item_id / order_key / group_key properties)."""

    model: str
    layer: int
    prompt_id: str
    prompt_text: str

    @property
    def item_id(self) -> str:
        # Library provides the hashing *mechanism*; the app owns the
        # axis names and their stability (platform.md ownership rule).
        return stable_item_id(
            ID_NAMESPACE,
            axes={
                "model": self.model,
                "layer": self.layer,
                "prompt_id": self.prompt_id,
            },
        )

    @property
    def order_key(self) -> str:
        # Interleave models so no provider/model starves the sweep.
        return stable_item_id(
            ID_NAMESPACE,
            axes={"order": [self.prompt_id, self.layer, self.model]},
        )

    @property
    def group_key(self) -> str:
        return RUN_NAME


def sample_prompts(seed: int) -> list[tuple[str, str]]:
    """App-side sampling; deterministic under seed."""
    raise NotImplementedError("domain code, not part of the sketch")


def start_probe_workflow(item_id: str) -> "EnqueueOutcome":
    """App-side enqueue target: starts the durable DBOS workflow for
    one item (dr_platform.dedup_enqueue makes this a one-liner) and
    reports the outcome. The app owns workflows and steps; the library
    never sees a step definition."""
    raise NotImplementedError("domain code, not part of the sketch")


# --- 1) seed: declare axes, build typed items ------------------------------


def build_items() -> list[LatentProbeItem]:
    prompts = sample_prompts(PROMPT_SEED)
    return [
        LatentProbeItem(
            model=model,
            layer=layer,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
        )
        for model, layer, (prompt_id, prompt_text) in product(
            MODELS, LAYERS, prompts
        )
    ]


# --- 2) run: durable, idempotent, resumable submission ---------------------


def run(database_url: str) -> None:
    engine = create_engine(database_url)
    items = build_items()

    # Re-runnable: operation key + stable item ids make re-submission
    # reconcile instead of duplicate (declarative seeding — the fix for
    # dr-llm gen-2's ON CONFLICT DO NOTHING under-seeding).
    result = submit_batch(
        engine,
        operation_key=f"{RUN_NAME}-seed",
        group_key=RUN_NAME,
        items=items,
        enqueue=start_probe_workflow,
        schema=PLATFORM,
    )
    print(result.enqueued_count, result.already_scheduled_count)

    # Await work completion (workflow ids are deterministic + recorded,
    # so the library can watch them without domain knowledge).
    await_operation(
        engine,
        operation_key=f"{RUN_NAME}-seed",
        schema=PLATFORM,
        poll_interval_seconds=30.0,
        timeout_seconds=6 * 3600.0,
    )
    with engine.connect() as connection:
        snapshot = load_operation_snapshot(
            connection,
            operation_key=f"{RUN_NAME}-seed",
            schema=PLATFORM,
        )
    print(snapshot.model_dump() if snapshot else None)


# --- 3) read: rebuildable typed projection ---------------------------------


class ProbeRow(BaseModel):
    item_id: str
    model: str
    layer: int
    prompt_id: str
    activation_sparsity: float
    artifact_sha256: str  # offloaded tensor blob (ArtifactRef.sha256)


def probe_rows(connection: Any) -> list[ProbeRow]:
    """App-side query over the app's append-only outcome rows."""
    raise NotImplementedError("domain code, not part of the sketch")


PROBE_PROJECTION = ProjectionSpec(
    name="nl_latents_probe",
    version="v1",  # bump + rebuild instead of migrating
    row_model=ProbeRow,
    build=probe_rows,
)


def read(database_url: str) -> None:
    engine = create_engine(database_url)
    rebuild_projection(engine, PROBE_PROJECTION, schema=PLATFORM)
    frame = load_projection_frame(engine, PROBE_PROJECTION, schema=PLATFORM)

    store = LocalDirArtifactStore(root="~/.nl-latents/artifacts")
    heavy = frame.loc[frame["activation_sparsity"] < 0.01]
    for sha in heavy["artifact_sha256"]:
        tensor_bytes = store.get_bytes(sha)  # verify-on-read
        _ = tensor_bytes
