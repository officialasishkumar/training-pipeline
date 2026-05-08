"""Synthetic data generation: turn logs into seeds, then drive a model + mock
tools to produce diverse trajectories.

The mentor's framing on the OpenAgriNet RFP is that data *creation* is the
critical work, not log ETL. This package adds a generation layer on top of
the existing ingest/redact/tag/validate plumbing:

1. ``seeds.SeedExtractor`` clusters user questions across canonical logs and
   keeps one representative per cluster — a diverse seed pool that drops
   the heavy near-duplication common in production traces.
2. ``mock_tools.MockToolRegistry`` returns deterministic (or deliberately
   broken) tool observations so trajectories can include recovery,
   timeouts, and partial-data branches we rarely see organically.
3. ``generator.TrajectoryGenerator`` drives an LLM through propose →
   validate → observe → continue and emits canonical ``Trajectory`` rows
   that the rest of the pipeline already knows how to redact, tag, and
   export.
4. ``difficulty`` assigns difficulty tiers and edge-case categories so a
   downstream stratifier can cap easy/dominant categories and keep the
   rare-but-valuable ones.

Heavy ML dependencies (``sentence-transformers``, ``scikit-learn``,
``transformers``, ``vllm``) are imported lazily and live behind the
``generate`` extra. Lightweight fallbacks (hash-based embeddings, a stub
LLM) keep the CLI and tests working without them.
"""

from training_pipeline.generate.difficulty import (
    DifficultyTier,
    EdgeCase,
    assign_difficulty,
    flag_edge_cases,
)
from training_pipeline.generate.generator import (
    GenerationStep,
    LLMBackend,
    StubLLMBackend,
    TrajectoryGenerator,
)
from training_pipeline.generate.mock_tools import (
    FailureMode,
    MockToolRegistry,
    ToolResult,
)
from training_pipeline.generate.seeds import Seed, SeedExtractor

__all__ = [
    "DifficultyTier",
    "EdgeCase",
    "FailureMode",
    "GenerationStep",
    "LLMBackend",
    "MockToolRegistry",
    "Seed",
    "SeedExtractor",
    "StubLLMBackend",
    "ToolResult",
    "TrajectoryGenerator",
    "assign_difficulty",
    "flag_edge_cases",
]
