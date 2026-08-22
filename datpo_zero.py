# datpo_zero.py
"""DATPO rollout utilities built on top of the GRPO-Zero stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from transformers import AutoTokenizer

from math_dataset import reward_function
from data_types import Episode
from datpo.config import ModelConfig, TreeSearchConfig
from datpo.training.tree_search import (
    BlockSample,
    ExpansionPlan,
    ForkNode,
    PolicyBackend,
    RolloutResult,
    TreeSearchEngine,
    TreeStatistics,
)


@dataclass
class DatpoConfig:
    """Configuration block for enabling DATPO training."""

    enabled: bool = True
    adv_normalization: bool = False
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    root_rollouts: int = 4
    adaptive: bool = True
    er_max: int = 1
    bp_max: int = 4
    bn_max: int = 4
    divergence_alpha: float = 0.0
    divergence_alpha_min: float = 0.0
    fork_selection: str = "sent-entropy"
    entropy_smoothing: float = 1e-4
    one_step_off_policy: bool = False
    verbose_generation: bool = True


class GRPOPolicyBackend(PolicyBackend):
    """Policy backend that adapts the GRPO-Zero Transformer to TreeSearchEngine."""

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: torch.device,
        dtype: torch.dtype,
        verbose_generation: bool = False,
    ) -> None:
        self.model = model
        self._tokenizer = tokenizer
        self._device = device
        self._dtype = dtype
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self._verbose_generation = verbose_generation

        # --- stop token ids (multiple) ---
        self.stop_token_ids: Set[int] = set()

        # eos
        if self.eos_token_id is not None:
            self.stop_token_ids.add(int(self.eos_token_id))

        # Pad can appear as an end token, so include it for generated tokens only.
        if self.pad_token_id is not None:
            self.stop_token_ids.add(int(self.pad_token_id))

        # Common end tokens, including Qwen variants.
        candidate_stop_tokens = [
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
            "<|eot_id|>",
            "<|eot|>",
        ]
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_str in candidate_stop_tokens:
            try:
                tid = tokenizer.convert_tokens_to_ids(token_str)
            except Exception:
                tid = None
            if tid is None:
                continue
            if unk_id is not None and tid == unk_id:
                continue
            self.stop_token_ids.add(int(tid))

        # Also include eos_token if it is exposed only as a string.
        eos_tok = getattr(tokenizer, "eos_token", None)
        if eos_tok is not None:
            try:
                tid = tokenizer.convert_tokens_to_ids(eos_tok)
            except Exception:
                tid = None
            if tid is not None and not (unk_id is not None and tid == unk_id):
                self.stop_token_ids.add(int(tid))

        self._stop_ids_tensor: Optional[torch.Tensor] = None

        # Fallback when pad_token_id is None.
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id if self.eos_token_id is not None else 0

    @property
    def device(self) -> torch.device:
        return self._device

    def snapshot(self) -> "GRPOPolicyBackend":
        return self

    def _get_stop_ids_tensor(self) -> Optional[torch.Tensor]:
        if not self.stop_token_ids:
            return None
        if self._stop_ids_tensor is None or self._stop_ids_tensor.device != self.device:
            self._stop_ids_tensor = torch.tensor(
                sorted(self.stop_token_ids),
                device=self.device,
                dtype=torch.long,
            )
        return self._stop_ids_tensor

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        IMPORTANT:
        - Match grpo.py rollout behavior:
          * teacher forcing by attention_mask (prompt tokens only)
          * start generation at min_prompt_len across batch
          * stop early when stop_ids is generated (not in prompt region)
          * responses/logprobs/entropy are returned for tokens AFTER EACH SAMPLE'S TRUE PROMPT LENGTH
        """
        count = int(kwargs.get("n", 1))

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        batch_size = input_ids.size(0)
        max_prompt_len = input_ids.size(1)

        total_batch = batch_size * count
        total_len = max_prompt_len + max_new_tokens

        pad_id = self.pad_token_id
        if pad_id is None:
            pad_id = self.eos_token_id if self.eos_token_id is not None else 0

        # [total_batch, total_len]
        tokens = torch.full(
            (total_batch, total_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )

        # prompt_mask=True where real prompt tokens exist (teacher-forcing region)
        prompt_mask = torch.zeros((total_batch, total_len), dtype=torch.bool, device=self.device)

        # expand prompts by n=count
        for idx in range(batch_size):
            start = idx * count
            end = start + count
            tokens[start:end, :max_prompt_len] = input_ids[idx].expand(count, -1)
            prompt_mask[start:end, :max_prompt_len] = attention_mask[idx].bool().expand(count, -1)

        # per-sample prompt lengths
        prompt_lengths = prompt_mask[:, :max_prompt_len].sum(dim=1).to(torch.long)  # [total_batch]
        if prompt_lengths.numel() == 0:
            responses = tokens[:, max_prompt_len : max_prompt_len + max_new_tokens]
            return {
                "responses": responses.cpu(),
                "token_logprobs": torch.zeros((total_batch, max_new_tokens), dtype=torch.float32).cpu(),
                "generated_entropy": torch.zeros((total_batch, max_new_tokens), dtype=torch.float32).cpu(),
            }

        min_prompt_len = int(prompt_lengths.min().item())
        if min_prompt_len <= 0:
            min_prompt_len = 1 if max_prompt_len > 0 else 0

        # storage for generated token stats (aligned to "after true prompt length")
        logprobs = torch.zeros((total_batch, max_new_tokens), device=self.device, dtype=torch.float32)
        entropies = torch.zeros_like(logprobs)
        collect_attention = bool(kwargs.get("return_attention", False) or kwargs.get("output_attentions", False))
        attention_scores = torch.zeros_like(logprobs) if collect_attention else None
        attention_token_gap = max(0, int(kwargs.get("attention_token_gap", 4)))

        is_finished = torch.zeros((total_batch,), dtype=torch.bool, device=self.device)
        stop_ids = self._get_stop_ids_tensor()

        # avoid creating a new tensor every step
        pad_fill = torch.full((total_batch,), int(pad_id), dtype=torch.long, device=self.device)

        # init kv cache
        self.model.init_kv_cache(
            max_batch_size=total_batch,
            max_seq_len=total_len,
            device=self.device,
            dtype=self._dtype,
        )

        if self._verbose_generation:
            print(
                f"* DATPO Generating batch={total_batch} n={count} "
                f"max_new_tokens={max_new_tokens} max_prompt_len={max_prompt_len} "
                f"min_prompt_len={min_prompt_len}",
                flush=True,
            )

        prev_pos = 0
        steps = 0

        # grpo-style loop: generate from min_prompt_len -> total_len-1 with teacher forcing
        for cur_pos in range(min_prompt_len, total_len):
            with torch.autocast(device_type=self.device.type, dtype=self._dtype):
                if collect_attention:
                    logits, attn_to_keys = self.model.inference(
                        tokens[:, prev_pos:cur_pos],
                        prev_pos,
                        return_attention_scores=True,
                    )
                else:
                    logits = self.model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
                    attn_to_keys = None

            if collect_attention and attention_scores is not None and attn_to_keys is not None:
                query_abs_pos = cur_pos - 1
                query_gen_index = query_abs_pos - prompt_lengths
                valid_query_rows = (
                    (query_gen_index >= 0)
                    & (query_gen_index < max_new_tokens)
                    & (~is_finished)
                ).nonzero(as_tuple=False).flatten()
                if valid_query_rows.numel() > 0:
                    key_abs = torch.arange(attn_to_keys.shape[-1], device=self.device, dtype=torch.long)
                    for row in valid_query_rows.tolist():
                        key_gen_index = key_abs - prompt_lengths[row]
                        valid_keys = (
                            (key_gen_index >= 0)
                            & (key_gen_index < max_new_tokens)
                            & ((query_gen_index[row] - key_gen_index) >= attention_token_gap)
                        )
                        if valid_keys.any():
                            cols = key_gen_index[valid_keys].to(torch.long)
                            vals = attn_to_keys[row, valid_keys].to(torch.float32)
                            attention_scores[row].index_add_(0, cols, vals)

            # ------------------------------------------------------------------
            # FIX: compute probs/logp/entropy in float32 so assignment into
            # float32 logprobs/entropies never dtype-mismatches (index_put)
            # ------------------------------------------------------------------
            step_logits = logits[:, -1, :].float()  # [B, V] float32
            probs = torch.softmax(step_logits, dim=-1)  # float32

            # sample
            next_token = torch.multinomial(probs, num_samples=1).reshape(-1)  # [B], long

            # teacher forcing for prompt region
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)

            # if finished, fill pad
            next_token = torch.where(is_finished, pad_fill, next_token)

            tokens[:, cur_pos] = next_token

            # compute logprob/entropy (float32)
            gathered = torch.gather(probs, 1, next_token[:, None]).squeeze(1)  # float32
            token_logp = torch.log(gathered + 1e-12)  # [B] float32
            token_ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [B] float32

            # store only for generated tokens (not prompt-forced), and only if within max_new_tokens window
            active_mask = ~is_finished
            gen_mask = active_mask & (~prompt_mask[:, cur_pos])

            if gen_mask.any():
                rows = gen_mask.nonzero(as_tuple=False).squeeze(1)
                gen_index = (cur_pos - prompt_lengths[rows]).to(torch.long)  # per-row index
                valid = (gen_index >= 0) & (gen_index < max_new_tokens)
                if valid.any():
                    rows = rows[valid]
                    cols = gen_index[valid]
                    # token_logp/token_ent are float32 already -> safe for index_put
                    logprobs[rows, cols] = token_logp[rows]
                    entropies[rows, cols] = token_ent[rows]

            # update finished flags if generated stop token
            if stop_ids is not None:
                is_end = torch.isin(next_token, stop_ids)
                is_gen = ~prompt_mask[:, cur_pos]
                is_finished = is_finished | (is_end & is_gen)

            prev_pos = cur_pos
            steps += 1

            if is_finished.all():
                break

        if self._verbose_generation:
            unfinished = int((~is_finished).sum().item())
            print(
                f"* DATPO Generation done: steps={steps}, unfinished={unfinished}/{total_batch}",
                flush=True,
            )

        self.model.del_kv_cache()

        # build responses AFTER each sample's TRUE prompt length
        arange = torch.arange(max_new_tokens, device=self.device, dtype=torch.long)[None, :]  # [1, T]
        pos = prompt_lengths[:, None] + arange  # [B, T], always < total_len
        responses = tokens.gather(1, pos)  # [B, T]

        out = {
            "responses": responses.cpu(),
            "token_logprobs": logprobs.cpu(),
            "generated_entropy": entropies.cpu(),
        }
        if attention_scores is not None:
            out["attention_scores"] = attention_scores.cpu()

        del tokens, prompt_mask, logprobs, entropies, logits, step_logits, probs, next_token, pad_fill
        if attention_scores is not None:
            del attention_scores
        return out

    @torch.no_grad()
    def compute_logprobs(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        combined = torch.cat([prompt_ids, response_ids], dim=1).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self._dtype):
            logits = self.model.forward(combined[:, :-1]).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        target = combined[:, 1:]
        gathered = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        prompt_length = prompt_ids.size(1)
        return gathered[:, prompt_length:]


class MathDatasetTreeSearchEngine(TreeSearchEngine):
    """Adapts the TreeSearchEngine to the Math dataset and GRPO stack."""

    def __init__(
        self,
        backend: GRPOPolicyBackend,
        tokenizer: AutoTokenizer,
        model_config: ModelConfig,
        tree_config: TreeSearchConfig,
    ) -> None:
        super().__init__(backend, tokenizer, model_config, tree_config)
        self._reward_kwargs: Dict[str, object] = {}

    def _collect_attention_for_fci(self) -> bool:
        return str(getattr(self.config, "fork_selection", "")).lower() in {
            "attn-fci",
            "attention-fci",
            "fci",
        }

    def _encode_problem(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        return input_ids, attention_mask

    def _generate_rollout_batch(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        answer: str,
        count: int,
    ) -> List[RolloutResult]:
        reward_kwargs = getattr(self, "_reward_kwargs", {})

        outputs = self.snapshot.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.model_config.max_new_tokens,
            n=count,
            logprobs=20,
            return_attention=self._collect_attention_for_fci(),
            attention_token_gap=4,
        )

        responses = outputs["responses"]  # CPU
        generated_entropy = outputs.get("generated_entropy")  # CPU
        token_logprobs_flat = outputs.get("token_logprobs")  # CPU
        attention_scores = outputs.get("attention_scores")  # CPU
        del outputs

        if responses.dim() == 1:
            responses = responses.unsqueeze(0)

        batch_size = responses.size(0)
        prompt_batch = prompt_ids.repeat(batch_size, 1)
        texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        rollouts: List[RolloutResult] = []
        for idx, text in enumerate(texts):
            prompt_tensor = prompt_batch[idx : idx + 1].detach().cpu()
            response_tensor = responses[idx : idx + 1].detach().cpu()

            logprob_tensor = (
                token_logprobs_flat[idx : idx + 1].detach().cpu()
                if token_logprobs_flat is not None
                else torch.zeros((1, response_tensor.shape[-1]), dtype=torch.float)
            )
            entropy_tensor = (
                generated_entropy[idx : idx + 1].detach().cpu()
                if generated_entropy is not None
                else torch.zeros((1, response_tensor.shape[-1]), dtype=torch.float)
            )

            rewards = reward_function(response=text, **reward_kwargs)
            rollout = RolloutResult(
                observations=prompt_tensor,
                actions=response_tensor,
                logprobs=logprob_tensor,
                token_scores=None,
                entropy=entropy_tensor,
                attention_scores=(
                    attention_scores[idx : idx + 1].detach().cpu()
                    if attention_scores is not None
                    else None
                ),
                rewards=rewards["reward"],
                value=rewards["reward"],
                answer=str(reward_kwargs.get("target", "")),
                text=text,
                advantage=None,
            )
            rollout.reward_info = rewards["reward_info"]  # type: ignore[attr-defined]
            rollouts.append(rollout)

        del prompt_batch, prompt_ids, attention_mask
        return rollouts

    def _sample_root_rollouts_batch(
        self,
        prompts: List[str],
        reward_kwargs_list: List[Dict[str, object]],
    ) -> List[Tuple[float, List[RolloutResult]]]:
        if not prompts:
            return []

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        reward_kwargs = getattr(self, "_reward_kwargs", {})

        outputs = self.snapshot.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.model_config.max_new_tokens,
            n=self.config.root_rollouts,
            logprobs=20,
            return_attention=self._collect_attention_for_fci(),
            attention_token_gap=4,
        )

        responses = outputs["responses"]
        generated_entropy = outputs.get("generated_entropy")
        token_logprobs_flat = outputs.get("token_logprobs")
        attention_scores = outputs.get("attention_scores")
        del outputs

        if responses.dim() == 1:
            responses = responses.unsqueeze(0)

        count = self.config.root_rollouts
        prompt_lengths = attention_mask.sum(dim=1)
        prompt_lengths_cpu = prompt_lengths.detach().cpu().tolist()

        rollouts_per_prompt: List[List[RolloutResult]] = [list() for _ in prompts]

        for idx in range(responses.size(0)):
            prompt_idx = idx // count
            text = self.tokenizer.decode(
                responses[idx],
                skip_special_tokens=True,
            )

            prompt_len = int(prompt_lengths_cpu[prompt_idx])
            prompt_tensor = input_ids[prompt_idx : prompt_idx + 1, :prompt_len].detach().cpu()
            response_tensor = responses[idx : idx + 1].detach().cpu()

            logprob_tensor = (
                token_logprobs_flat[idx : idx + 1].detach().cpu()
                if token_logprobs_flat is not None
                else torch.zeros((1, response_tensor.shape[-1]), dtype=torch.float, device="cpu")
            )
            entropy_tensor = (
                generated_entropy[idx : idx + 1].detach().cpu()
                if generated_entropy is not None
                else torch.zeros((1, response_tensor.shape[-1]), dtype=torch.float, device="cpu")
            )
            attention_tensor = (
                attention_scores[idx : idx + 1].detach().cpu()
                if attention_scores is not None
                else None
            )

            rollout_rewards = reward_function(
                response=text, **reward_kwargs_list[prompt_idx], **reward_kwargs
            )
            rollout = RolloutResult(
                observations=prompt_tensor,
                actions=response_tensor,
                logprobs=logprob_tensor,
                token_scores=None,
                entropy=entropy_tensor,
                attention_scores=attention_tensor,
                rewards=rollout_rewards["reward"],
                value=rollout_rewards["reward"],
                answer=str(reward_kwargs_list[prompt_idx].get("target", "")),
                text=text,
                advantage=None,
            )
            rollout.reward_info = rollout_rewards["reward_info"]  # type: ignore[attr-defined]
            rollouts_per_prompt[prompt_idx].append(rollout)

        del encoded, input_ids, attention_mask, prompt_lengths

        results: List[Tuple[float, List[RolloutResult]]] = []
        for rollouts in rollouts_per_prompt:
            successes = [rollout.value for rollout in rollouts]
            value = float(np.mean(successes)) if successes else 0.0
            results.append((value, rollouts))
        return results

    def search_episode(
        self,
        prompt: str,
        reward_kwargs: Dict[str, object],
        progress: Optional[float] = None,
        precomputed_root: Optional[Tuple[float, List[RolloutResult]]] = None,
    ) -> Tuple[List[RolloutResult], List[RolloutResult], TreeStatistics, List[RolloutResult], List[BlockSample]]:
        """
        Fix: pass the real target answer into TreeSearchEngine.search(...).
        """
        self._reward_kwargs = reward_kwargs
        correct_answer = str(reward_kwargs.get("target", ""))
        try:
            return super().search(
                prompt, answer=correct_answer, progress=progress, precomputed_root=precomputed_root
            )
        finally:
            self._reward_kwargs = {}

    def search_batch(
        self,
        prompts: List[str],
        reward_kwargs_list: List[Dict[str, object]],
        progress: Optional[float] = None,
        precomputed_roots: Optional[List[Tuple[float, List[RolloutResult]]]] = None,
    ) -> Tuple[List[List[BlockSample]], List[TreeStatistics]]:
        """
        Fix: pass each tree's target into _collect_expansion_plans(...).
        """
        if not prompts:
            return [], []

        alpha = self._scheduled_alpha(progress)

        if precomputed_roots is None:
            self._reward_kwargs = {}
            try:
                precomputed_roots = self._sample_root_rollouts_batch(
                    prompts=prompts, reward_kwargs_list=reward_kwargs_list
                )
            finally:
                self._reward_kwargs = {}

        if len(precomputed_roots) != len(prompts):
            raise ValueError(
                f"precomputed_roots length ({len(precomputed_roots)}) does not match prompts ({len(prompts)})"
            )

        tree_states: List[Dict[str, object]] = []
        nodes_to_expand_by_tree: Dict[int, List[Tuple[ForkNode, int, int]]] = {}
        tree_rounds_left: Dict[int, int] = {}

        for idx, (prompt, root_data) in enumerate(zip(prompts, precomputed_roots)):
            root_value, root_rollouts = root_data
            prefix_ids, _ = self._encode_problem(prompt)
            root_node = ForkNode(
                prefix_ids=prefix_ids,
                prefix_token_count=0,
                value=root_value,
                rollouts=root_rollouts,
            )
            tree_states.append(
                {
                    "root": root_node,
                    "all_nodes": [root_node],
                    "all_rollouts": list(root_rollouts),
                }
            )
            er_global, bp_initial, bn_initial = self._adaptive_counts(root_value)
            tree_rounds_left[idx] = er_global
            nodes_to_expand_by_tree[idx] = [(root_node, bn_initial, bp_initial)]

        while True:
            active_trees = {
                idx: nodes
                for idx, nodes in nodes_to_expand_by_tree.items()
                if nodes and tree_rounds_left.get(idx, 0) > 0
            }
            if not active_trees:
                break

            expansion_plans: List[ExpansionPlan] = []
            plan_tree_index: Dict[ExpansionPlan, int] = {}
            child_nodes_by_tree: Dict[int, List[ForkNode]] = {}

            for tree_idx, node_entries in active_trees.items():
                tree_answer = str(reward_kwargs_list[tree_idx].get("target", ""))

                for node, bn_local, bp_local in node_entries:
                    plans, child_nodes = self._collect_expansion_plans(
                        node=node,
                        bn=bn_local,
                        bp=bp_local,
                        answer=tree_answer,
                    )
                    for plan in plans:
                        plan_tree_index[plan] = tree_idx
                    expansion_plans.extend(plans)
                    child_nodes_by_tree.setdefault(tree_idx, []).extend(child_nodes)
                    tree_states[tree_idx]["all_nodes"].extend(child_nodes)

            plan_results = self._execute_expansion_plans(expansion_plans)
            for plan, rollouts in plan_results.items():
                tree_idx = plan_tree_index.get(plan)
                if tree_idx is None:
                    continue
                child = plan.child_node
                child.rollouts = rollouts
                tree_states[tree_idx]["all_rollouts"].extend(rollouts)
                child.value = float(np.mean([r.value for r in rollouts])) if rollouts else 0.0

            next_round: Dict[int, List[Tuple[ForkNode, int, int]]] = {}
            for tree_idx, children in child_nodes_by_tree.items():
                for child in children:
                    _, bp_child, bn_child = self._adaptive_counts(child.value)
                    if bn_child <= 0 or bp_child <= 0:
                        continue
                    next_round.setdefault(tree_idx, []).append((child, bn_child, bp_child))

            for tree_idx in active_trees:
                tree_rounds_left[tree_idx] = tree_rounds_left.get(tree_idx, 0) - 1

            nodes_to_expand_by_tree = {}
            for tree_idx, nodes in next_round.items():
                if tree_rounds_left.get(tree_idx, 0) > 0 and nodes:
                    nodes_to_expand_by_tree[tree_idx] = nodes

        block_samples_batch: List[List[BlockSample]] = []
        stats: List[TreeStatistics] = []
        for state in tree_states:
            root_node: ForkNode = state["root"]  # type: ignore[assignment]
            all_nodes: List[ForkNode] = state["all_nodes"]  # type: ignore[assignment]
            for node in all_nodes:
                self._ensure_leaf_blocks(node)
            self._mark_training_blocked(root_node)
            block_samples = self._assign_advantages(root_node, alpha)
            block_samples_batch.append(block_samples)
            stats.append(
                self._compute_tree_statistics(state["all_rollouts"], root_node)  # type: ignore[arg-type]
            )

        return block_samples_batch, stats

    def _execute_expansion_plans_and_sample_roots(
        self,
        plans: List[ExpansionPlan],
        root_prompts: List[str],
        root_reward_kwargs_list: List[Dict[str, object]],
    ) -> Tuple[Dict[ExpansionPlan, List[RolloutResult]], List[Tuple[float, List[RolloutResult]]]]:
        plan_to_rollouts: Dict[ExpansionPlan, List[RolloutResult]] = {p: [] for p in plans}
        root_rollouts_per_prompt: List[List[RolloutResult]] = [list() for _ in root_prompts]

        max_total_sequences = int(getattr(self.config, "expansion_max_total_sequences", 384))
        if max_total_sequences <= 0:
            max_total_sequences = 384

        flat_prefixes: List[torch.Tensor] = []
        flat_items: List[Tuple[str, object]] = []
        root_prefixes: List[torch.Tensor] = []
        for prompt in root_prompts:
            prefix_ids, _ = self._encode_problem(prompt)
            root_prefixes.append(prefix_ids)
        for prompt_idx, prefix_ids in enumerate(root_prefixes):
            for _ in range(int(self.config.root_rollouts)):
                flat_prefixes.append(prefix_ids)
                flat_items.append(("root", prompt_idx))
        for plan in plans:
            if plan.bn <= 0:
                continue
            for _ in range(int(plan.bn)):
                flat_prefixes.append(plan.prefix_ids)
                flat_items.append(("plan", plan))

        if not flat_prefixes:
            return plan_to_rollouts, [(0.0, rollouts) for rollouts in root_rollouts_per_prompt]

        reward_kwargs = getattr(self, "_reward_kwargs", {})
        start = 0
        while start < len(flat_prefixes):
            end = min(len(flat_prefixes), start + max_total_sequences)
            chunk_prefixes = flat_prefixes[start:end]
            chunk_items = flat_items[start:end]

            input_batch, attn_batch = self._build_padded_prefix_batch(chunk_prefixes)
            outputs = self.snapshot.generate(
                input_ids=input_batch,
                attention_mask=attn_batch,
                max_new_tokens=self.model_config.max_new_tokens,
                n=1,
                logprobs=20,
                return_attention=self._collect_attention_for_fci(),
                attention_token_gap=4,
            )

            responses = outputs["responses"]
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

            for local_idx, (item, text) in enumerate(zip(chunk_items, decoded_texts)):
                kind, payload = item
                response_tensor = responses[local_idx : local_idx + 1].detach().cpu()
                logprob_tensor = (
                    token_logprobs_flat[local_idx : local_idx + 1].detach().cpu()
                    if token_logprobs_flat is not None
                    else torch.zeros((1, response_tensor.shape[-1]), dtype=torch.float32, device="cpu")
                )
                entropy_tensor = (
                    generated_entropy[local_idx : local_idx + 1].detach().cpu()
                    if generated_entropy is not None
                    else torch.zeros((1, response_tensor.shape[-1]), dtype=torch.float32, device="cpu")
                )
                attention_tensor = (
                    attention_scores[local_idx : local_idx + 1].detach().cpu()
                    if attention_scores is not None
                    else None
                )

                if kind == "plan":
                    plan = payload
                    assert isinstance(plan, ExpansionPlan)
                    prompt_tensor = plan.prefix_ids.detach().cpu()
                    is_correct = self._is_answer_correct(text, plan.answer)
                    reward = self.config.reward_on_correct if is_correct else self.config.reward_on_incorrect
                    rollout = RolloutResult(
                        observations=prompt_tensor,
                        actions=response_tensor,
                        logprobs=logprob_tensor,
                        token_scores=None,
                        entropy=entropy_tensor,
                        attention_scores=attention_tensor,
                        rewards=reward,
                        value=float(is_correct),
                        answer=plan.answer,
                        text=text,
                        advantage=None,
                    )
                    rollout.reward_info = {"correct": bool(is_correct)}  # type: ignore[attr-defined]
                    plan_to_rollouts[plan].append(rollout)
                else:
                    prompt_idx = int(payload)
                    kwargs_for_prompt = root_reward_kwargs_list[prompt_idx]
                    rollout_rewards = reward_function(
                        response=text,
                        **kwargs_for_prompt,
                        **reward_kwargs,
                    )
                    rollout = RolloutResult(
                        observations=root_prefixes[prompt_idx].detach().cpu(),
                        actions=response_tensor,
                        logprobs=logprob_tensor,
                        token_scores=None,
                        entropy=entropy_tensor,
                        attention_scores=attention_tensor,
                        rewards=rollout_rewards["reward"],
                        value=rollout_rewards["reward"],
                        answer=str(kwargs_for_prompt.get("target", "")),
                        text=text,
                        advantage=None,
                    )
                    rollout.reward_info = rollout_rewards["reward_info"]  # type: ignore[attr-defined]
                    root_rollouts_per_prompt[prompt_idx].append(rollout)

            del input_batch, attn_batch
            start = end

        root_results: List[Tuple[float, List[RolloutResult]]] = []
        for rollouts in root_rollouts_per_prompt:
            successes = [rollout.value for rollout in rollouts]
            value = float(np.mean(successes)) if successes else 0.0
            root_results.append((value, rollouts))
        return plan_to_rollouts, root_results

    def search_batch_with_next_roots(
        self,
        prompts: List[str],
        reward_kwargs_list: List[Dict[str, object]],
        progress: Optional[float] = None,
        precomputed_roots: Optional[List[Tuple[float, List[RolloutResult]]]] = None,
        next_prompts: Optional[List[str]] = None,
        next_reward_kwargs_list: Optional[List[Dict[str, object]]] = None,
    ) -> Tuple[List[List[BlockSample]], List[TreeStatistics], Optional[List[Tuple[float, List[RolloutResult]]]]]:
        if not prompts:
            next_roots = None
            if next_prompts is not None and next_reward_kwargs_list is not None:
                _, next_roots = self._execute_expansion_plans_and_sample_roots(
                    [], next_prompts, next_reward_kwargs_list
                )
            return [], [], next_roots

        alpha = self._scheduled_alpha(progress)

        if precomputed_roots is None:
            self._reward_kwargs = {}
            try:
                precomputed_roots = self._sample_root_rollouts_batch(
                    prompts=prompts, reward_kwargs_list=reward_kwargs_list
                )
            finally:
                self._reward_kwargs = {}

        if len(precomputed_roots) != len(prompts):
            raise ValueError(
                f"precomputed_roots length ({len(precomputed_roots)}) does not match prompts ({len(prompts)})"
            )

        tree_states: List[Dict[str, object]] = []
        nodes_to_expand_by_tree: Dict[int, List[Tuple[ForkNode, int, int]]] = {}
        tree_rounds_left: Dict[int, int] = {}

        for idx, (prompt, root_data) in enumerate(zip(prompts, precomputed_roots)):
            root_value, root_rollouts = root_data
            prefix_ids, _ = self._encode_problem(prompt)
            root_node = ForkNode(
                prefix_ids=prefix_ids,
                prefix_token_count=0,
                value=root_value,
                rollouts=root_rollouts,
            )
            tree_states.append(
                {
                    "root": root_node,
                    "all_nodes": [root_node],
                    "all_rollouts": list(root_rollouts),
                }
            )
            er_global, bp_initial, bn_initial = self._adaptive_counts(root_value)
            tree_rounds_left[idx] = er_global
            nodes_to_expand_by_tree[idx] = [(root_node, bn_initial, bp_initial)]

        next_roots: Optional[List[Tuple[float, List[RolloutResult]]]] = None
        next_roots_done = next_prompts is None or next_reward_kwargs_list is None

        while True:
            active_trees = {
                idx: nodes
                for idx, nodes in nodes_to_expand_by_tree.items()
                if nodes and tree_rounds_left.get(idx, 0) > 0
            }
            if not active_trees:
                break

            expansion_plans: List[ExpansionPlan] = []
            plan_tree_index: Dict[ExpansionPlan, int] = {}
            child_nodes_by_tree: Dict[int, List[ForkNode]] = {}

            for tree_idx, node_entries in active_trees.items():
                tree_answer = str(reward_kwargs_list[tree_idx].get("target", ""))

                for node, bn_local, bp_local in node_entries:
                    plans, child_nodes = self._collect_expansion_plans(
                        node=node,
                        bn=bn_local,
                        bp=bp_local,
                        answer=tree_answer,
                    )
                    for plan in plans:
                        plan_tree_index[plan] = tree_idx
                    expansion_plans.extend(plans)
                    child_nodes_by_tree.setdefault(tree_idx, []).extend(child_nodes)
                    tree_states[tree_idx]["all_nodes"].extend(child_nodes)

            if not next_roots_done:
                plan_results, next_roots = self._execute_expansion_plans_and_sample_roots(
                    expansion_plans,
                    next_prompts or [],
                    next_reward_kwargs_list or [],
                )
                next_roots_done = True
            else:
                plan_results = self._execute_expansion_plans(expansion_plans)

            for plan, rollouts in plan_results.items():
                tree_idx = plan_tree_index.get(plan)
                if tree_idx is None:
                    continue
                child = plan.child_node
                child.rollouts = rollouts
                tree_states[tree_idx]["all_rollouts"].extend(rollouts)
                child.value = float(np.mean([r.value for r in rollouts])) if rollouts else 0.0

            next_round: Dict[int, List[Tuple[ForkNode, int, int]]] = {}
            for tree_idx, children in child_nodes_by_tree.items():
                for child in children:
                    _, bp_child, bn_child = self._adaptive_counts(child.value)
                    if bn_child <= 0 or bp_child <= 0:
                        continue
                    next_round.setdefault(tree_idx, []).append((child, bn_child, bp_child))

            for tree_idx in active_trees:
                tree_rounds_left[tree_idx] = tree_rounds_left.get(tree_idx, 0) - 1

            nodes_to_expand_by_tree = {}
            for tree_idx, nodes in next_round.items():
                if tree_rounds_left.get(tree_idx, 0) > 0 and nodes:
                    nodes_to_expand_by_tree[tree_idx] = nodes

        if not next_roots_done:
            _, next_roots = self._execute_expansion_plans_and_sample_roots(
                [],
                next_prompts or [],
                next_reward_kwargs_list or [],
            )

        block_samples_batch: List[List[BlockSample]] = []
        stats: List[TreeStatistics] = []
        for state in tree_states:
            root_node: ForkNode = state["root"]  # type: ignore[assignment]
            all_nodes: List[ForkNode] = state["all_nodes"]  # type: ignore[assignment]
            for node in all_nodes:
                self._ensure_leaf_blocks(node)
            self._mark_training_blocked(root_node)
            block_samples = self._assign_advantages(root_node, alpha)
            block_samples_batch.append(block_samples)
            stats.append(
                self._compute_tree_statistics(state["all_rollouts"], root_node)  # type: ignore[arg-type]
            )

        return block_samples_batch, stats, next_roots


def _make_reward_kwargs_list(engine: MathDatasetTreeSearchEngine, batch) -> List[Dict[str, object]]:
    return [
        {
            "numbers": batch.numbers[idx],
            "target": batch.target[idx],
            "end_token": engine.tokenizer.eos_token,
        }
        for idx in range(len(batch.prefix))
    ]


def datpo_rollout(
    engine: MathDatasetTreeSearchEngine,
    batch,
    progress: float,
) -> Tuple[List[Episode], List[TreeStatistics]]:
    """
    Fix: set Episode(prefix_token_ids=...) from BlockSample.observations (Q + history).
    """
    episodes: List[Episode] = []
    stats: List[TreeStatistics] = []

    reward_kwargs_list = _make_reward_kwargs_list(engine, batch)

    precomputed_roots = engine._sample_root_rollouts_batch(
        prompts=list(batch.prefix), reward_kwargs_list=reward_kwargs_list
    )

    block_samples_batch, stats = engine.search_batch(
        prompts=list(batch.prefix),
        reward_kwargs_list=reward_kwargs_list,
        progress=progress,
        precomputed_roots=precomputed_roots,
    )

    pad_id = engine.tokenizer.pad_token_id

    for q_idx, (prompt, block_samples) in enumerate(zip(batch.prefix, block_samples_batch)):
        for block in block_samples:
            # -------- actions --------
            block_actions = block.actions.detach().cpu().view(-1).tolist()
            if pad_id is not None and pad_id in block_actions:
                block_actions = block_actions[: block_actions.index(pad_id)]

            old_logprobs = block.logprobs.detach().cpu().view(-1)
            old_logprobs = old_logprobs[: len(block_actions)]

            # -------- observations (Q + history) --------
            obs_ids = block.observations.detach().cpu().view(-1).tolist()
            if pad_id is not None and pad_id in obs_ids:
                obs_ids = obs_ids[: obs_ids.index(pad_id)]

            decoded_full_text = engine.tokenizer.decode(
                obs_ids + block_actions, skip_special_tokens=True
            )

            episodes.append(
                Episode(
                    prefix=prompt,
                    text=decoded_full_text,
                    prefix_token_ids=obs_ids,
                    prefix_tokens=batch.prefix_tokens[q_idx] if hasattr(batch, "prefix_tokens") else [],
                    generated_token_ids=block_actions,
                    is_finished=True,
                    reward=float(block.reward),
                    reward_info={
                        "answer_reward": float(block.reward),
                        "format_reward": 0.0,
                    },
                    advantage=float(block.advantage),
                    old_logprobs=old_logprobs,
                )
            )

    return episodes, stats


def attnrl_sample_initial_roots(
    engine: MathDatasetTreeSearchEngine,
    batch,
) -> List[Tuple[float, List[RolloutResult]]]:
    return engine._sample_root_rollouts_batch(
        prompts=list(batch.prefix),
        reward_kwargs_list=_make_reward_kwargs_list(engine, batch),
    )


def attnrl_rollout_from_roots(
    engine: MathDatasetTreeSearchEngine,
    batch,
    precomputed_roots: List[Tuple[float, List[RolloutResult]]],
    progress: float,
    next_batch=None,
) -> Tuple[List[Episode], List[TreeStatistics], Optional[List[Tuple[float, List[RolloutResult]]]]]:
    reward_kwargs_list = _make_reward_kwargs_list(engine, batch)
    next_reward_kwargs_list = (
        _make_reward_kwargs_list(engine, next_batch) if next_batch is not None else None
    )
    block_samples_batch, stats, next_roots = engine.search_batch_with_next_roots(
        prompts=list(batch.prefix),
        reward_kwargs_list=reward_kwargs_list,
        progress=progress,
        precomputed_roots=precomputed_roots,
        next_prompts=list(next_batch.prefix) if next_batch is not None else None,
        next_reward_kwargs_list=next_reward_kwargs_list,
    )

    pad_id = engine.tokenizer.pad_token_id
    episodes: List[Episode] = []
    for q_idx, (prompt, block_samples) in enumerate(zip(batch.prefix, block_samples_batch)):
        for block in block_samples:
            block_actions = block.actions.detach().cpu().view(-1).tolist()
            if pad_id is not None and pad_id in block_actions:
                block_actions = block_actions[: block_actions.index(pad_id)]

            old_logprobs = block.logprobs.detach().cpu().view(-1)
            old_logprobs = old_logprobs[: len(block_actions)]

            obs_ids = block.observations.detach().cpu().view(-1).tolist()
            if pad_id is not None and pad_id in obs_ids:
                obs_ids = obs_ids[: obs_ids.index(pad_id)]

            decoded_full_text = engine.tokenizer.decode(
                obs_ids + block_actions, skip_special_tokens=True
            )

            episodes.append(
                Episode(
                    prefix=prompt,
                    text=decoded_full_text,
                    prefix_token_ids=obs_ids,
                    prefix_tokens=batch.prefix_tokens[q_idx] if hasattr(batch, "prefix_tokens") else [],
                    generated_token_ids=block_actions,
                    is_finished=True,
                    reward=float(block.reward),
                    reward_info={
                        "answer_reward": float(block.reward),
                        "format_reward": 0.0,
                    },
                    advantage=float(block.advantage),
                    old_logprobs=old_logprobs,
                )
            )

    return episodes, stats, next_roots
