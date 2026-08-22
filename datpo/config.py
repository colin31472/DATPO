"""Configuration objects for DATPO training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    policy_model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # Qwen/Qwen3-4B-Thinking-2507, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    value_model: Optional[str] = None
    tokenizer: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    eos_token: Optional[str] = None
    pad_token: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    load_in_8bit: bool = False
    use_reward_model: bool = False


@dataclass
class TreeSearchConfig:
    """Hyper-parameters controlling the adaptive tree search."""

    root_rollouts: int = 4
    adaptive: bool = True
    er_max: int = 1
    bp_max: int = 4
    bn_max: int = 4
    divergence_alpha: float = 0
    divergence_alpha_min: float = 0
    fork_selection: str = "sent-entropy"
    entropy_smoothing: float = 1e-4
    one_step_off_policy: bool = False
    reward_on_correct: float = 1.0
    reward_on_incorrect: float = 0.0
    early_stop_min_value: float = 1e-3


__all__ = [
    "ModelConfig",
    "TreeSearchConfig",
]
