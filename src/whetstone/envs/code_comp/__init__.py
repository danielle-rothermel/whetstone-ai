"""HumanEval code-compression environment family.

The ``code_comp`` package groups direct-generation (d1), encoder-decoder
(ed1), and behavioral-mutant (ed1m) modes that share HumanEval scoring,
dataset loading, and enc-dec rollout infrastructure. Legacy import paths
under ``whetstone.envs.d1``, ``ed1``, and ``ed1m`` remain as shims during
the migration toward a single ``code_comp`` env identity.
"""
