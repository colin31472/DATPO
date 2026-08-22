# datpo/training/tree_search.py
"""Tree search utilities implementing DATPO rollouts."""
from __future__ import annotations

import math
import random
import re
import signal
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# sympy parsing utils (same as math.py AnswerChecker)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from ..config import ModelConfig, TreeSearchConfig
from ..utils.token_stats import (
    rolling_delta,
    sequence_entropy,
    split_into_sentences,
    topk_indices,
)

LOGGER = logging.getLogger(__name__)


# -------------------------
# Answer Checking (SAME as math.py)
# -------------------------
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("SymPy parsing reached timeout")


@dataclass
class AnswerChecker:
    def _is_answer_correct(self, generated: str, answer: str) -> bool:
        gen_clean = self._prepare_answer(generated)
        ans_clean = self._prepare_answer(answer)

        if self._normalize_for_match(gen_clean) == self._normalize_for_match(ans_clean):
            return True

        return self._check_numerical_equivalence(gen_clean, ans_clean)

    def _prepare_answer(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        boxed = self._extract_last_boxed(text)
        return boxed if boxed is not None else text

    def _normalize_for_match(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r"\\(left|right|big|Big|bigg|Bigg)", "", text)
        text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
        text = re.sub(r"\\(text|mbox|mathrm|textbf|textit)\{([^}]+)\}", r"\2", text)
        text = re.sub(r"\\?\$|\\?%|\^\\circ|degrees?", "", text)
        text = text.replace(",", "").replace("!", "").replace("\\ ", "")
        text = text.replace(r"\in", "")
        text = re.sub(r"^[a-z]\s*=\s*", "", text)
        text = re.sub(r"_\d+|_{\d+}", "", text)
        return text.replace(" ", "").replace("\\", "")

    def _check_numerical_equivalence(self, gen: str, ans: str) -> bool:
        def extract_numbers(text: str) -> List[float]:
            # 1) Limit full input length.
            if len(text) > 300:
                return []

            # frac -> division
            text = re.sub(r"\\f?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)

            parts = re.split(r"[,;]", text)
            nums: List[float] = []

            transformations = standard_transformations + (implicit_multiplication_application,)

            for p in parts:
                # 2) Limit each part length.
                clean_p = re.sub(r"[^0-9\.\-\+\*\/\(\)]", "", p)
                if not clean_p or len(clean_p) > 50:
                    continue

                # 3) Signal-based timeout (Linux/Unix).
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=SyntaxWarning)
                        expr = parse_expr(clean_p, transformations=transformations)
                        val = float(expr.evalf())
                        nums.append(val)
                except (TimeoutException, Exception):
                    continue
                finally:
                    signal.alarm(0)

            return nums

        gen_nums = extract_numbers(gen)
        ans_nums = extract_numbers(ans)

        if not gen_nums or not ans_nums or len(gen_nums) != len(ans_nums):
            return False

        try:
            return all(
                abs(g - a) < 1e-6 for g, a in zip(sorted(gen_nums), sorted(ans_nums))
            )
        except Exception:
            return False

    @staticmethod
    def _extract_last_boxed(text: str) -> Optional[str]:
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return None
        i = idx + len("\\boxed{")
        depth, j, n = 1, i, len(text)
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j]
            j += 1
        return None


# Make a singleton checker (avoid constructing per call)
_ANSWER_CHECKER = AnswerChecker()


# -------------------------
# Core data structures
# -------------------------
@dataclass
class RolloutResult:
    """Stores rollout information for PPO updates."""
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    token_scores: Optional[torch.Tensor]  # Optional to avoid storing heavy logits.
    rewards: float
    value: float
    answer: str
    text: str
    advantage: Optional[float] = None
    attention_scores: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None  # Precomputed entropy.


@dataclass
class BlockSample:
    """Block-level PPO sample extracted from a rollout segment."""
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantage: float
    reward: float
    entropy: Optional[torch.Tensor] = None


@dataclass
class TreeStatistics:
    """Aggregated statistics computed for a single search tree."""
    average_response_length: float
    average_entropy: float
    leaf_count: int
    generated_token_count: int
    tree_pass_rate: float = 0.0


@dataclass
class ForkNode:
    """A node in the adaptive tree search."""
    prefix_ids: torch.Tensor
    prefix_token_count: int
    value: float
    rollouts: List[RolloutResult] = field(default_factory=list)
    children: List["ForkNode"] = field(default_factory=list)
    parent: Optional["ForkNode"] = None
    parent_rollout: Optional[RolloutResult] = None
    fork_token_index: Optional[int] = None
    block_from_parent: Optional["BlockInfo"] = None
    blocks: List["BlockInfo"] = field(default_factory=list)
    expanded: bool = False
    training_blocked: bool = False  # kept for compatibility (now disabled)

    def is_terminal(self) -> bool:
        return math.isclose(self.value, 1.0, rel_tol=1e-6)


@dataclass
class BlockInfo:
    """Metadata describing a contiguous block within a rollout.

    IMPORTANT (Plan B fix):
    - start_offset/end_offset are interpreted as offsets in *block.parent_rollout.actions* (absolute in that rollout),
      NOT "relative to node prefix".
    """
    parent: ForkNode
    parent_rollout: RolloutResult
    start_offset: int
    end_offset: int
    block_type: str
    child: Optional[ForkNode] = None
    target_rollouts: List[RolloutResult] = field(default_factory=list)
    text: str = ""
    divergence: float = 0.0


@dataclass(eq=False)
class ExpansionPlan:
    """Plan describing a single fork expansion to be executed in batch."""
    parent_node: ForkNode
    parent_rollout: RolloutResult
    start_offset: int
    end_offset: int
    fork_pos: int
    prefix_ids: torch.Tensor
    answer: str
    bn: int
    block_info: BlockInfo
    child_node: ForkNode


class PolicyBackend(Protocol):
    """Protocol describing the policy operations required by the search."""
    @property
    def device(self) -> torch.device: ...

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]: ...

    def compute_logprobs(
        self, prompt_ids: torch.Tensor, response_ids: torch.Tensor
    ) -> torch.Tensor: ...

    def snapshot(self) -> "PolicyBackend": ...


class PolicySnapshot:
    """Frozen copy of the policy used for on-policy rollouts."""
    def __init__(self, backend: PolicyBackend, tokenizer: AutoTokenizer, config: ModelConfig) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        **generate_kwargs,
    ) -> Dict[str, torch.Tensor]:
        return self.backend.generate(
            input_ids,
            attention_mask,
            max_new_tokens,
            **generate_kwargs,
        )

    def compute_logprobs(
        self, prompt_ids: torch.Tensor, response_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.backend.compute_logprobs(prompt_ids, response_ids)


class TreeSearchEngine:
    """Implements adaptive tree search following the DATPO algorithm."""

    def __init__(
        self,
        backend: PolicyBackend,
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        tree_config: TreeSearchConfig,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.model_config = config
        self.config = tree_config
        self.config.adaptive = self._coerce_bool(self.config.adaptive)
        self.device = backend.device
        self.snapshot = PolicySnapshot(backend.snapshot(), tokenizer, config)
        self.divergence_enabled = not (
            math.isclose(tree_config.divergence_alpha, 0.0)
            and math.isclose(tree_config.divergence_alpha_min, 0.0)
        )
        self.embedder = None
        if self.divergence_enabled:
            self.embedder = SentenceTransformer(
                config.embedding_model,
                trust_remote_code=True,
                device=str(self.device),
            )

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"false", "0", "no", "off", "n"}:
                return False
            if normalized in {"true", "1", "yes", "on", "y"}:
                return True
        return bool(value)

    def _adaptive_enabled(self) -> bool:
        return self._coerce_bool(getattr(self.config, "adaptive", True))

    def refresh_snapshot(self) -> None:
        self.snapshot = PolicySnapshot(self.backend.snapshot(), self.tokenizer, self.model_config)

    def _scheduled_alpha(self, progress: Optional[float]) -> float:
        alpha_max = self.config.divergence_alpha
        alpha_min = self.config.divergence_alpha_min
        if progress is None:
            return alpha_max
        progress_clamped = max(0.0, min(1.0, progress))
        return alpha_max + (alpha_min - alpha_max) * progress_clamped

    def _encode_problem(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        return input_ids, attention_mask

    # ---------------------------------------------------------------------
    # Helpers: "effective" generated region (exclude padding tail)
    # ---------------------------------------------------------------------
    def _get_pad_token_id(self) -> Optional[int]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        return int(pad_id) if pad_id is not None else None

    def _effective_generated_length(self, rollout: RolloutResult) -> int:
        actions = rollout.actions
        if actions is None:
            return 0
        tokens = actions
        while tokens.dim() > 1:
            tokens = tokens[0]
        if tokens.numel() == 0:
            return 0

        pad_id = self._get_pad_token_id()
        if pad_id is None:
            return int(tokens.numel())

        pad_pos = (tokens == pad_id).nonzero(as_tuple=False)
        if pad_pos.numel() == 0:
            return int(tokens.numel())
        return int(pad_pos[0].item())

    def _get_effective_actions_1d(self, rollout: RolloutResult) -> torch.Tensor:
        actions = rollout.actions
        if actions is None:
            return torch.empty((0,), dtype=torch.long)
        tokens = actions
        while tokens.dim() > 1:
            tokens = tokens[0]
        L = self._effective_generated_length(rollout)
        return tokens[:L]

    def _get_effective_entropy_1d(self, rollout: RolloutResult) -> Optional[torch.Tensor]:
        if rollout.entropy is None:
            return None
        e = rollout.entropy
        while e.dim() > 1:
            e = e[0]
        L = self._effective_generated_length(rollout)
        return e[:L]

    def _get_effective_attention_1d(self, rollout: RolloutResult) -> Optional[torch.Tensor]:
        if rollout.attention_scores is None:
            return None
        scores = rollout.attention_scores
        while scores.dim() > 1:
            scores = scores[0]
        L = self._effective_generated_length(rollout)
        return scores[:L].to(torch.float32)

    def _uses_attention_fci(self) -> bool:
        return str(getattr(self.config, "fork_selection", "")).lower() in {
            "attn-fci",
            "attention-fci",
            "fci",
        }

    def _tokenized_text_len(self, text: str) -> int:
        if not text:
            return 0
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if ids and isinstance(ids[0], list):
                return len(ids[0])
            return len(ids)
        except Exception:
            return 0

    def _step_token_ranges(self, rollout: RolloutResult) -> List[Tuple[int, int]]:
        eff_len = self._effective_generated_length(rollout)
        if eff_len <= 0:
            return []

        text = rollout.text or ""
        ranges: List[Tuple[int, int]] = []
        if text.strip():
            spans: List[Tuple[int, int]] = []
            last = 0
            for match in re.finditer(r"(?:\r?\n\s*){2,}", text):
                if match.start() > last:
                    spans.append((last, match.start()))
                last = match.end()
            if last < len(text):
                spans.append((last, len(text)))

            for start_char, end_char in spans:
                start_tok = self._tokenized_text_len(text[:start_char])
                end_tok = self._tokenized_text_len(text[:end_char])
                start_tok = max(0, min(start_tok, eff_len))
                end_tok = max(start_tok, min(end_tok, eff_len))
                if end_tok > start_tok:
                    ranges.append((start_tok, end_tok))

        if ranges:
            return ranges

        eff_actions = self._get_effective_actions_1d(rollout)
        token_ids = eff_actions.detach().cpu().tolist()
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids) if token_ids else []
        sentences = split_into_sentences(
            token_strs,
            tokenizer=self.tokenizer,
            token_ids=token_ids,
            text=rollout.text,
        )
        for idxs in sentences:
            if not idxs:
                continue
            start = max(0, min(int(idxs[0]), eff_len))
            end = max(start, min(int(idxs[-1]) + 1, eff_len))
            if end > start:
                ranges.append((start, end))
        return ranges if ranges else [(0, eff_len)]

    # ---------------------------------------------------------------------

    def _generate_rollout_batch(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer: str,
        count: int,
    ) -> List[RolloutResult]:
        if count <= 0:
            return []

        LOGGER.info(
            "[TreeSearch._generate_rollout_batch] Generating %d rollouts (n=%d, K=20)",
            count, count
        )

        outputs = self.snapshot.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.model_config.max_new_tokens,
            n=count,
            logprobs=20,
            return_attention=self._uses_attention_fci(),
            attention_token_gap=4,
        )

        responses = outputs["responses"]
        generated_entropy = outputs.get("generated_entropy")
        token_logprobs_flat = outputs.get("token_logprobs")
        attention_scores = outputs.get("attention_scores")
        del outputs

        if responses.dim() == 1:
            responses = responses.unsqueeze(0)

        batch_size = responses.size(0)
        prompt_batch = prompt_ids.repeat(batch_size, 1)

        texts = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        rollouts: List[RolloutResult] = []
        for idx, text in enumerate(texts):
            prompt_tensor = prompt_batch[idx: idx + 1].detach().cpu()
            response_tensor = responses[idx: idx + 1].detach().cpu()

            if token_logprobs_flat is not None:
                logprob_tensor = token_logprobs_flat[idx: idx + 1].detach().cpu()
            else:
                logprob_tensor = torch.zeros(
                    (1, response_tensor.shape[-1]),
                    dtype=torch.float32,
                    device="cpu",
                )

            if generated_entropy is not None:
                entropy_tensor = generated_entropy[idx: idx + 1].detach().cpu()
            else:
                entropy_tensor = torch.zeros(
                    (1, response_tensor.shape[-1]),
                    dtype=torch.float32,
                    device="cpu",
                )

            is_correct = self._is_answer_correct(text, answer)
            reward = self.config.reward_on_correct if is_correct else self.config.reward_on_incorrect

            rollout = RolloutResult(
                observations=prompt_tensor,
                actions=response_tensor,
                logprobs=logprob_tensor,
                token_scores=None,
                entropy=entropy_tensor,
                attention_scores=(
                    attention_scores[idx: idx + 1].detach().cpu()
                    if attention_scores is not None
                    else None
                ),
                rewards=reward,
                value=float(is_correct),
                answer=answer,
                text=text,
                advantage=None,
            )
            rollouts.append(rollout)

        del prompt_batch, prompt_ids, attention_mask
        return rollouts

    def _sample_root_rollouts(self, prompt: str, correct_answer: str) -> Tuple[float, List[RolloutResult]]:
        input_ids, attention_mask = self._encode_problem(prompt)
        rollouts = self._generate_rollout_batch(
            input_ids,
            attention_mask,
            correct_answer,
            self.config.root_rollouts,
        )
        successes = [rollout.value for rollout in rollouts]
        value = float(np.mean(successes)) if successes else 0.0
        return value, rollouts

    def _is_answer_correct(self, generated: str, answer: str) -> bool:
        return _ANSWER_CHECKER._is_answer_correct(generated, answer)

    def _select_fork_positions(
        self, rollout: RolloutResult, fork_method: str, k: int
    ) -> List[int]:
        """
        When selecting fork positions:
        - Exclude all positions after PAD.
        - Use only generated-token spans where entropy/logprob values are meaningful.
        """
        if k <= 0:
            return []

        eff_len = self._effective_generated_length(rollout)
        if eff_len <= 0:
            return []

        if fork_method in {"attn-fci", "attention-fci", "fci"}:
            return self._select_attention_fci_positions(rollout, min(k, eff_len))

        if rollout.entropy is not None:
            entropies = self._get_effective_entropy_1d(rollout)
            if entropies is None:
                return []
        elif rollout.token_scores is not None:
            logits = rollout.token_scores
            if logits.dim() == 3:
                logits = logits.squeeze(0)
            full_ent = sequence_entropy(logits, smoothing=self.config.entropy_smoothing)
            entropies = full_ent[:eff_len]
        else:
            return []

        if entropies.numel() == 0:
            return []

        max_entropy = entropies.max().clamp(min=1.0)
        entropies = entropies / max_entropy

        num_tokens = int(entropies.shape[0])
        max_k = min(k, num_tokens)

        eff_actions = self._get_effective_actions_1d(rollout)
        token_ids = eff_actions.detach().cpu().tolist()
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids) if token_ids else []

        if fork_method == "random":
            positions = list(range(1, num_tokens + 1))
            random.shuffle(positions)
            return sorted(positions[:max_k])

        if fork_method == "tok-entropy":
            indices = topk_indices(entropies, max_k).tolist()
            return sorted({idx + 1 for idx in indices})

        if fork_method == "tok-delta":
            delta = rolling_delta(entropies)
            indices = topk_indices(delta.abs(), max_k).tolist()
            return sorted({idx + 1 for idx in indices})

        if fork_method in {"sent-entropy", "sent-delta"}:
            sentences = split_into_sentences(
                token_strs,
                tokenizer=self.tokenizer,
                token_ids=token_ids,
                text=rollout.text,
            )
            if not sentences:
                if fork_method == "sent-entropy":
                    indices = topk_indices(entropies, max_k).tolist()
                else:
                    delta = rolling_delta(entropies)
                    indices = topk_indices(delta.abs(), max_k).tolist()
                return sorted({idx + 1 for idx in indices})

            sent_entropies = torch.stack([entropies[idxs].mean() for idxs in sentences])

            if fork_method == "sent-entropy":
                chosen = topk_indices(sent_entropies, min(max_k, len(sentences))).tolist()
                positions = {max(1, min(num_tokens, sentences[i][0] + 1)) for i in chosen}
                return sorted(positions)

            sent_deltas = torch.zeros_like(sent_entropies)
            if sent_deltas.numel() > 1:
                sent_deltas[1:] = (sent_entropies[1:] - sent_entropies[:-1]).abs()
            chosen = topk_indices(sent_deltas, min(max_k, len(sentences))).tolist()
            positions = {max(1, min(num_tokens, sentences[i][0] + 1)) for i in chosen}
            return sorted(positions)

        if fork_method == "fixed-segment":
            if num_tokens == 1:
                return [1]
            segment_positions: List[int] = []
            for i in range(1, max_k + 1):
                raw_pos = int(round(num_tokens * i / (max_k + 1)))
                pos = max(1, min(num_tokens, raw_pos))
                if pos not in segment_positions:
                    segment_positions.append(pos)
            if len(segment_positions) < max_k:
                for pos in range(1, num_tokens + 1):
                    if pos not in segment_positions:
                        segment_positions.append(pos)
                    if len(segment_positions) >= max_k:
                        break
            return sorted(segment_positions[:max_k])

        raise ValueError(f"Unknown fork selection method: {fork_method}")

    def _normalize_fork_positions(
        self, positions: Sequence[int], num_tokens: int, k: int
    ) -> List[int]:
        max_k = min(max(int(k), 0), max(int(num_tokens), 0))
        if max_k <= 0:
            return []

        normalized: List[int] = []
        seen: set[int] = set()

        def add_position(pos: int) -> None:
            if len(normalized) >= max_k:
                return
            pos = max(1, min(num_tokens, int(pos)))
            if pos in seen:
                return
            seen.add(pos)
            normalized.append(pos)

        for pos in positions:
            add_position(int(pos))

        if not self._adaptive_enabled() and len(normalized) < max_k:
            for i in range(1, max_k + 1):
                add_position(int(round(num_tokens * i / (max_k + 1))))
            for pos in range(1, num_tokens + 1):
                add_position(pos)

        return sorted(normalized[:max_k])

    def _select_attention_fci_positions(self, rollout: RolloutResult, k: int) -> List[int]:
        scores = self._get_effective_attention_1d(rollout)
        if scores is None or scores.numel() == 0 or k <= 0:
            return []

        ranges = self._step_token_ranges(rollout)
        if len(ranges) <= 1:
            return []

        step_scores: List[float] = []
        for start, end in ranges:
            if end <= start:
                step_scores.append(0.0)
            else:
                step_scores.append(float(scores[start:end].mean().item()))

        score_tensor = torch.tensor(step_scores, dtype=torch.float32)
        if score_tensor.numel() == 0 or float(score_tensor.max().item()) <= 0.0:
            return []

        threshold = float(torch.quantile(score_tensor, 0.8).item())
        candidate_indices = [
            idx
            for idx, score in enumerate(step_scores)
            if score >= threshold and ranges[idx][0] > 0
        ]
        if not candidate_indices:
            candidate_indices = [
                idx
                for idx in torch.argsort(score_tensor, descending=True).tolist()
                if ranges[idx][0] > 0
            ]

        limit = min(k, 2)
        candidate_indices.sort(key=lambda idx: ranges[idx][0])
        positions: List[int] = []
        seen: set[int] = set()
        num_tokens = int(scores.shape[0])
        for idx in candidate_indices:
            pos = max(1, min(num_tokens, int(ranges[idx][0])))
            if pos in seen:
                continue
            seen.add(pos)
            positions.append(pos)
            if len(positions) >= limit:
                break
        return sorted(positions)

    def _adaptive_counts(self, value: float) -> Tuple[int, int, int]:
        if not self._adaptive_enabled():
            return (
                max(int(self.config.er_max), 0),
                max(int(self.config.bp_max), 0),
                max(int(self.config.bn_max), 0),
            )
        er = int(math.ceil(-self.config.er_max * value + self.config.er_max))
        bp = int(math.ceil(-self.config.bp_max * value + self.config.bp_max))
        bn = int(math.ceil(-self.config.bn_max * value + self.config.bn_max))
        er = max(er, 0)
        bp = max(bp, 0)
        bn = max(bn, 0)
        return er, bp, bn

    def _build_padded_prefix_batch(
        self, prefixes: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        max_len = max(t.shape[1] for t in prefixes)
        batch = torch.full(
            (len(prefixes), max_len),
            pad_id,
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(batch)
        for idx, prefix in enumerate(prefixes):
            prefix = prefix.to(self.device)
            length = prefix.shape[1]
            batch[idx, :length] = prefix[0]
            attention_mask[idx, :length] = 1
        return batch, attention_mask

    def _decode_block_text(self, rollout: RolloutResult, start: int, end: int) -> str:
        eff_tokens = self._get_effective_actions_1d(rollout)
        if eff_tokens.numel() == 0:
            return ""
        start_clamped = max(0, min(start, int(eff_tokens.shape[-1])))
        end_clamped = max(start_clamped, min(end, int(eff_tokens.shape[-1])))
        if end_clamped <= start_clamped:
            return ""
        block_tokens = eff_tokens[start_clamped:end_clamped].detach().cpu().tolist()
        if not block_tokens:
            return ""
        return self.tokenizer.decode(block_tokens, skip_special_tokens=True)

    # ---------------------------------------------------------------------
    # [MOD] Plan B (Chain Structure Correction):
    # - child.parent == real_parent (chain)
    # - block stored in real_parent.blocks, and block.parent == real_parent
    # - block offsets are ABSOLUTE offsets in parent_rollout.actions (the rollout we are slicing)
    # ---------------------------------------------------------------------
    def _collect_expansion_plans(
        self,
        node: ForkNode,
        bn: int,
        bp: int,
        answer: str,
    ) -> Tuple[List[ExpansionPlan], List[ForkNode]]:
        should_expand = (
            bn > 0
            and bp > 0
            and (not self._adaptive_enabled() or not node.is_terminal())
        )
        child_nodes: List[ForkNode] = []
        plans: List[ExpansionPlan] = []

        for rollout in node.rollouts:
            actions = rollout.actions
            if actions is None:
                continue

            tokens = actions
            while tokens.dim() > 1:
                tokens = tokens[0]

            num_tokens = self._effective_generated_length(rollout)

            if num_tokens == 0:
                block_type = "root-leaf" if node.parent is None else "fork-leaf"
                block_info = BlockInfo(
                    parent=node,
                    parent_rollout=rollout,
                    start_offset=0,
                    end_offset=0,
                    block_type=block_type,
                    child=None,
                    target_rollouts=[rollout],
                    text="",
                )
                node.blocks.append(block_info)
                continue

            if should_expand:
                positions = self._select_fork_positions(
                    rollout, self.config.fork_selection, bp
                )
                positions = self._normalize_fork_positions(positions, num_tokens, bp)
            else:
                positions = []

            filtered_positions: List[int] = []
            current = 0
            for pos in positions:
                if pos <= current:
                    continue
                filtered_positions.append(pos)
                current = pos

            current_start = 0
            previous_fork: Optional[ForkNode] = None

            for pos in filtered_positions:
                if pos <= current_start:
                    continue

                # [MOD] real parent in chain
                real_parent = previous_fork if previous_fork is not None else node

                # prefix_ids for the child is still constructed from *this rollout's* prefix extension
                prefix_extension = tokens[:pos].unsqueeze(0).to(node.prefix_ids.device)
                prefix_ids = torch.cat([node.prefix_ids, prefix_extension], dim=-1)

                block_type = "root-fork" if real_parent.parent is None else "fork-fork"
                block_text = self._decode_block_text(rollout, current_start, pos)

                child_node = ForkNode(
                    prefix_ids=prefix_ids,
                    prefix_token_count=node.prefix_token_count + pos,
                    value=0.0,
                    rollouts=[],
                    parent=real_parent,
                    parent_rollout=rollout,
                    fork_token_index=pos,
                )

                # [MOD] store the edge (block) on the *real_parent*
                block_info = BlockInfo(
                    parent=real_parent,
                    parent_rollout=rollout,
                    start_offset=current_start,  # ABS offset in rollout.actions
                    end_offset=pos,              # ABS offset in rollout.actions
                    block_type=block_type,
                    child=child_node,
                    target_rollouts=[rollout],
                    text=block_text,
                )
                child_node.block_from_parent = block_info

                real_parent.blocks.append(block_info)
                real_parent.children.append(child_node)

                plan = ExpansionPlan(
                    parent_node=real_parent,
                    parent_rollout=rollout,
                    start_offset=current_start,
                    end_offset=pos,
                    fork_pos=pos,
                    prefix_ids=prefix_ids,
                    answer=answer,
                    bn=bn,
                    block_info=block_info,
                    child_node=child_node,
                )
                plans.append(plan)
                child_nodes.append(child_node)

                previous_fork = child_node
                current_start = pos

            # tail leaf segment attaches to last real parent in chain
            if current_start < num_tokens or not filtered_positions:
                real_parent = previous_fork if previous_fork is not None else node
                block_type = "root-leaf" if real_parent.parent is None else "fork-leaf"
                block_text = self._decode_block_text(rollout, current_start, num_tokens)
                block_info = BlockInfo(
                    parent=real_parent,
                    parent_rollout=rollout,
                    start_offset=current_start,   # ABS offset
                    end_offset=num_tokens,        # ABS offset
                    block_type=block_type,
                    child=None,
                    target_rollouts=[rollout],
                    text=block_text,
                )
                real_parent.blocks.append(block_info)

        node.expanded = True
        return plans, child_nodes

    def _execute_expansion_plans(
        self, plans: Sequence[ExpansionPlan]
    ) -> Dict[ExpansionPlan, List[RolloutResult]]:
        """
        OOM guard:
        - Flatten prefixes by bn so the generation count is sum(plan.bn), with n=1.
        - Chunk sum(plan.bn) by max_total_sequences.
        """
        if not plans:
            return {}

        max_total_sequences = int(getattr(self.config, "expansion_max_total_sequences", 384))
        if max_total_sequences <= 0:
            max_total_sequences = 384

        active_plans: List[ExpansionPlan] = [p for p in plans if p.bn and p.bn > 0]
        if not active_plans:
            return {p: [] for p in plans}

        plan_to_rollouts: Dict[ExpansionPlan, List[RolloutResult]] = {p: [] for p in plans}

        flat_plan_refs: List[ExpansionPlan] = []
        flat_prefixes: List[torch.Tensor] = []
        for p in active_plans:
            for _ in range(int(p.bn)):
                flat_plan_refs.append(p)
                flat_prefixes.append(p.prefix_ids)

        total = len(flat_prefixes)

        start = 0
        while start < total:
            end = min(total, start + max_total_sequences)
            chunk_prefixes = flat_prefixes[start:end]
            chunk_plans = flat_plan_refs[start:end]

            input_batch, attn_batch = self._build_padded_prefix_batch(chunk_prefixes)

            outputs = self.snapshot.generate(
                input_ids=input_batch,
                attention_mask=attn_batch,
                max_new_tokens=self.model_config.max_new_tokens,
                n=1,
                logprobs=20,
                return_attention=self._uses_attention_fci(),
                attention_token_gap=4,
            )

            responses = outputs["responses"]  # CPU
            generated_entropy = outputs.get("generated_entropy")
            token_logprobs_flat = outputs.get("token_logprobs")
            attention_scores = outputs.get("attention_scores")
            del outputs

            if responses.dim() == 1:
                responses = responses.unsqueeze(0)

            decoded_texts = self.tokenizer.batch_decode(
                responses,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, (plan, text) in enumerate(zip(chunk_plans, decoded_texts)):
                response_tensor = responses[i: i + 1].detach().cpu()

                if token_logprobs_flat is not None:
                    logprob_tensor = token_logprobs_flat[i: i + 1].detach().cpu()
                else:
                    logprob_tensor = torch.zeros(
                        (1, response_tensor.shape[-1]),
                        dtype=torch.float32,
                        device="cpu",
                    )

                if generated_entropy is not None:
                    entropy_tensor = generated_entropy[i: i + 1].detach().cpu()
                else:
                    entropy_tensor = torch.zeros(
                        (1, response_tensor.shape[-1]),
                        dtype=torch.float32,
                        device="cpu",
                    )

                prompt_tensor = plan.prefix_ids.detach().cpu()
                is_correct = self._is_answer_correct(text, plan.answer)
                reward = self.config.reward_on_correct if is_correct else self.config.reward_on_incorrect

                rollout = RolloutResult(
                    observations=prompt_tensor,
                    actions=response_tensor,
                    logprobs=logprob_tensor,
                    token_scores=None,
                    entropy=entropy_tensor,
                    attention_scores=(
                        attention_scores[i: i + 1].detach().cpu()
                        if attention_scores is not None
                        else None
                    ),
                    rewards=reward,
                    value=float(is_correct),
                    answer=plan.answer,
                    text=text,
                    advantage=None,
                )
                plan_to_rollouts[plan].append(rollout)

            del input_batch, attn_batch
            start = end

        return plan_to_rollouts

    def _ensure_leaf_blocks(self, node: ForkNode) -> None:
        for rollout in node.rollouts:
            length = self._effective_generated_length(rollout)
            uncovered = self._rollout_uncovered_intervals_in_subtree(node, rollout, length)
            if not uncovered:
                continue

            block_type = "root-leaf" if node.parent is None else "fork-leaf"
            for start, end in uncovered:
                block_info = BlockInfo(
                    parent=node,
                    parent_rollout=rollout,
                    start_offset=start,
                    end_offset=end,
                    block_type=block_type,
                    child=None,
                    target_rollouts=[rollout],
                    text=self._decode_block_text(rollout, start, end),
                )
                node.blocks.append(block_info)

    @staticmethod
    def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []

        merged: List[Tuple[int, int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _collect_rollout_intervals_in_subtree(
        self,
        node: ForkNode,
        rollout: RolloutResult,
    ) -> List[Tuple[int, int]]:
        intervals: List[Tuple[int, int]] = []
        stack: List[ForkNode] = [node]
        seen_nodes: set[int] = set()

        while stack:
            current = stack.pop()
            current_id = id(current)
            if current_id in seen_nodes:
                continue
            seen_nodes.add(current_id)

            for block in current.blocks:
                if block.parent_rollout is not rollout:
                    continue
                start = max(0, int(block.start_offset))
                end = max(start, int(block.end_offset))
                intervals.append((start, end))

            stack.extend(current.children)

        return intervals

    def _rollout_uncovered_intervals_in_subtree(
        self,
        node: ForkNode,
        rollout: RolloutResult,
        length: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        target_length = self._effective_generated_length(rollout) if length is None else int(length)
        if target_length <= 0:
            return []

        intervals = self._collect_rollout_intervals_in_subtree(node, rollout)
        if not intervals:
            return [(0, target_length)]

        merged = self._merge_intervals(intervals)
        uncovered: List[Tuple[int, int]] = []
        cursor = 0

        for start, end in merged:
            start = max(0, min(start, target_length))
            end = max(start, min(end, target_length))
            if cursor < start:
                uncovered.append((cursor, start))
            cursor = max(cursor, end)

        if cursor < target_length:
            uncovered.append((cursor, target_length))

        return uncovered

    def _compute_block_divergence(self, blocks: Sequence[BlockInfo]) -> None:
        for block in blocks:
            block.divergence = 0.0
        if not self.divergence_enabled:
            return
        valid_blocks = [block for block in blocks if block.text.strip()]
        if len(valid_blocks) <= 1:
            return
        assert self.embedder is not None
        embeddings = self.embedder.encode(
            [block.text for block in valid_blocks],
            convert_to_tensor=True,
            device=self.device,
        )
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        cosine = embeddings @ embeddings.T
        for idx, block in enumerate(valid_blocks):
            mask = torch.ones(len(valid_blocks), dtype=torch.bool, device=cosine.device)
            mask[idx] = False
            if mask.sum() == 0:
                block.divergence = 0.0
            else:
                mean_cos = cosine[idx][mask].mean()
                block.divergence = float(1.0 - mean_cos.item())

    def _compute_rollout_text_divergence(self, rollouts: Sequence[RolloutResult]) -> Dict[int, float]:
        out: Dict[int, float] = {id(r): 0.0 for r in rollouts}
        if not self.divergence_enabled:
            return out
        valid: List[RolloutResult] = [r for r in rollouts if isinstance(r.text, str) and r.text.strip()]
        if len(valid) <= 1:
            return out

        texts = [r.text for r in valid]
        assert self.embedder is not None
        embeddings = self.embedder.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
        )
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        cosine = embeddings @ embeddings.T

        m = len(valid)
        for i in range(m):
            mask = torch.ones(m, dtype=torch.bool, device=cosine.device)
            mask[i] = False
            if mask.sum() == 0:
                div = 0.0
            else:
                mean_cos = cosine[i][mask].mean()
                div = float(1.0 - mean_cos.item())
            out[id(valid[i])] = div

        return out

    def _compute_sibling_divergence(self, node: ForkNode) -> None:
        # 1) child blocks (immediate children of this node)
        child_blocks: List[BlockInfo] = []
        for child in node.children:
            block = child.block_from_parent
            if block is not None and block.text.strip():
                child_blocks.append(block)
        if child_blocks:
            self._compute_block_divergence(child_blocks)

        # 2) leaf blocks: divergence among rollouts at this node
        rollout_div = self._compute_rollout_text_divergence(node.rollouts)
        for b in node.blocks:
            if b.child is not None:
                continue
            rid = id(b.parent_rollout)
            b.divergence = float(rollout_div.get(rid, 0.0))

    def _estimate_node_value(self, node: ForkNode) -> float:
        values = [float(rollout.value) for rollout in node.rollouts]
        if node.parent_rollout is not None:
            values.append(float(node.parent_rollout.value))
        return float(np.mean(values)) if values else 0.0


    # ---------------------------------------------------------------------
    # [MOD] Early termination disabled:
    # - keep method/field for compatibility, but never sets training_blocked
    # ---------------------------------------------------------------------
    def _mark_training_blocked(self, root: ForkNode) -> None:
        # Early termination disabled: do nothing.
        # Fields are kept to avoid breaking other code that may access training_blocked.
        return

    # ---------------------------------------------------------------------
    # [MOD] Build samples correctly for Plan B:
    # - Use rollout.observations (the prefix used to generate rollout.actions) as the base prompt.
    # - Slice rollout.actions/logprobs/entropy using ABS offsets in that rollout.
    # - DO NOT re-concatenate node.prefix_ids + rollout.actions[:start] (that causes duplication/misalignment in chain mode).
    # ---------------------------------------------------------------------
    def _build_block_sample(
        self, node: ForkNode, block: BlockInfo, advantage: float
    ) -> Optional[BlockSample]:
        rollout = block.parent_rollout
        actions = rollout.actions
        logprobs = rollout.logprobs
        entropy = rollout.entropy
        if actions is None or logprobs is None:
            return None

        actions_1d = actions.squeeze(0) if actions.dim() > 1 else actions
        logprobs_1d = logprobs.squeeze(0) if logprobs.dim() > 1 else logprobs
        entropy_1d = entropy.squeeze(0) if entropy is not None and entropy.dim() > 1 else entropy

        eff_len = self._effective_generated_length(rollout)

        start = max(0, int(block.start_offset))
        end = max(start, int(block.end_offset))
        if start >= eff_len:
            return None
        end = min(end, eff_len)
        if start >= end:
            return None

        # Base prompt that actually generated this rollout
        prompt_tensor = rollout.observations.detach().cpu().view(-1)

        # Observations for this block: prompt + actions[:start]
        prefix_tokens = actions_1d[:start].detach().cpu()
        observations = torch.cat([prompt_tensor, prefix_tokens], dim=-1)

        action_slice = actions_1d[start:end].detach().cpu()
        logprob_slice = logprobs_1d[start:end].detach().cpu()
        entropy_slice = entropy_1d[start:end].detach().cpu() if entropy_1d is not None else None

        reward = rollout.rewards if block.child is None else float(block.child.value)

        return BlockSample(
            observations=observations,
            actions=action_slice,
            logprobs=logprob_slice,
            advantage=float(advantage),
            reward=float(reward),
            entropy=entropy_slice,
        )

    # ---------------------------------------------------------------------
    # [MOD] Early termination disabled:
    # - remove filtering by training_blocked
    # ---------------------------------------------------------------------
    def _assign_advantages(self, root: ForkNode, alpha: float) -> List[BlockSample]:
        block_samples: List[BlockSample] = []
        stack: List[ForkNode] = [root]
        while stack:
            node = stack.pop()

            # Early termination disabled -> do not skip nodes
            if self.divergence_enabled:
                self._compute_sibling_divergence(node)

            for block in node.blocks:
                d = getattr(block, "divergence", 0.0)
                if block.child is None:
                    base_advantage = block.parent_rollout.rewards - node.value
                else:
                    base_advantage = block.child.value - node.value

                advantage = base_advantage
                if base_advantage > 0.0:
                    advantage += alpha * d

                for _ in block.target_rollouts:
                    sample = self._build_block_sample(node, block, advantage)
                    if sample is not None:
                        block_samples.append(sample)

            stack.extend(node.children)

        return block_samples

    def search(
        self,
        prompt: str,
        answer: str,
        progress: Optional[float] = None,
        precomputed_root: Optional[Tuple[float, List[RolloutResult]]] = None,
    ) -> Tuple[
        List[RolloutResult],
        List[RolloutResult],
        TreeStatistics,
        List[RolloutResult],
        List[BlockSample],
    ]:
        if precomputed_root is None:
            root_value, root_rollouts = self._sample_root_rollouts(prompt, answer)
        else:
            root_value, root_rollouts = precomputed_root

        prefix_ids, _ = self._encode_problem(prompt)
        root_node = ForkNode(
            prefix_ids=prefix_ids,
            prefix_token_count=0,
            value=root_value,
            rollouts=root_rollouts,
        )

        all_rollouts: List[RolloutResult] = root_rollouts.copy()
        fork_rollouts: List[RolloutResult] = []
        alpha = self._scheduled_alpha(progress)

        er_global, bp_initial, bn_initial = self._adaptive_counts(root_value)
        nodes_to_expand: List[Tuple[ForkNode, int, int]] = [(root_node, bn_initial, bp_initial)]
        all_nodes: List[ForkNode] = [root_node]

        for round_idx in range(er_global):
            expansion_plans: List[ExpansionPlan] = []
            child_nodes_round: List[ForkNode] = []
            next_round: List[Tuple[ForkNode, int, int]] = []

            for node, bn_local, bp_local in nodes_to_expand:
                plans, child_nodes = self._collect_expansion_plans(
                    node=node,
                    bn=bn_local,
                    bp=bp_local,
                    answer=answer,
                )
                expansion_plans.extend(plans)
                child_nodes_round.extend(child_nodes)
                all_nodes.extend(child_nodes)

            plan_results = self._execute_expansion_plans(expansion_plans)
            for plan, rollouts in plan_results.items():
                child = plan.child_node
                child.rollouts = rollouts
                for rollout in rollouts:
                    fork_rollouts.append(rollout)
                    all_rollouts.append(rollout)
                child.value = self._estimate_node_value(child)

            if round_idx + 1 >= er_global:
                break

            for child in child_nodes_round:
                if self._adaptive_enabled() and child.is_terminal():
                    continue
                _, bp_child, bn_child = self._adaptive_counts(child.value)
                if bn_child <= 0 or bp_child <= 0:
                    continue
                next_round.append((child, bn_child, bp_child))

            if not next_round:
                break
            nodes_to_expand = next_round

        for node in all_nodes:
            self._ensure_leaf_blocks(node)

        # [MOD] Early termination disabled: do not mark training_blocked
        # self._mark_training_blocked(root_node)

        block_samples = self._assign_advantages(root_node, alpha)
        stats = self._compute_tree_statistics(all_rollouts, root_node)

        return all_rollouts, fork_rollouts, stats, root_rollouts, block_samples

    def _compute_tree_statistics(
        self, rollouts: Sequence[RolloutResult], root: Optional[ForkNode] = None
    ) -> TreeStatistics:
        if not rollouts:
            token_count = self._count_generated_tokens(root)
            return TreeStatistics(0.0, 0.0, 0, token_count)

        lengths = [float(r.actions.shape[-1]) for r in rollouts]

        entropies: List[float] = []
        for rollout in rollouts:
            e = self._get_effective_entropy_1d(rollout)
            if e is None or e.numel() == 0:
                continue
            entropies.append(float(e.mean().item()))
        avg_entropy = float(np.mean(entropies)) if entropies else 0.0

        leaf_count = len(rollouts)
        token_count = self._count_generated_tokens(root)

        avg_length = float(np.mean(lengths)) if lengths else 0.0
        tree_pass_rate = float(any(float(getattr(r, "value", 0.0)) > 0.0 for r in rollouts))
        return TreeStatistics(
            average_response_length=avg_length,
            average_entropy=avg_entropy,
            leaf_count=leaf_count,
            generated_token_count=token_count,
            tree_pass_rate=tree_pass_rate,
        )

    def _count_generated_tokens(self, root: Optional[ForkNode]) -> int:
        if root is None:
            return 0

        seen: set[int] = set()
        total = 0
        stack: List[ForkNode] = [root]
        while stack:
            current = stack.pop()
            for rollout in current.rollouts:
                rid = id(rollout)
                if rid in seen:
                    continue
                seen.add(rid)
                total += int(self._effective_generated_length(rollout))
            stack.extend(current.children)

        return total


__all__ = ["TreeSearchEngine", "RolloutResult", "TreeStatistics", "BlockSample"]
