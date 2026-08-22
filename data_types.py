from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]
    advantage: Optional[float] = None
    old_logprobs: Optional[torch.Tensor] = None


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[str]
    target: List[str]
    levels: List[Optional[int]]
