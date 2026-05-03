"""Export trajectories to SFT and DPO datasets."""

from training_pipeline.export.dpo import (
    DPOPairStrategy,
    build_dpo_record,
    export_dpo_jsonl,
)
from training_pipeline.export.sft import (
    build_sft_record,
    export_sft_jsonl,
    trajectory_to_messages,
)
from training_pipeline.export.shards import (
    ShardWriter,
    write_dataset_card,
)
from training_pipeline.export.templates import (
    KNOWN_TEMPLATES,
    apply_template,
    template_for,
)

__all__ = [
    "DPOPairStrategy",
    "KNOWN_TEMPLATES",
    "ShardWriter",
    "apply_template",
    "build_dpo_record",
    "build_sft_record",
    "export_dpo_jsonl",
    "export_sft_jsonl",
    "template_for",
    "trajectory_to_messages",
    "write_dataset_card",
]
