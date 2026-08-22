import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    stop_ids = torch.tensor(list(tokenizer.stop_token_ids), device=device)
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        is_end_token = torch.isin(next_token, stop_ids) 
        is_generated_token = ~input_text_mask[:, cur_pos]
        is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[str(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def groupwise_normalize(values: List[float], keys: List[List[int]]) -> List[float]:
    """Normalize arbitrary per-episode values grouped by their prefix tokens."""

    groups = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[tuple(key)].append(idx)

    normalized = [0.0 for _ in values]
    for indices in groups.values():
        group_vals = [values[i] for i in indices]
        mean_val = float(np.mean(group_vals))
        std_val = float(np.std(group_vals))
        for i, episode_idx in enumerate(indices):
            normalized[episode_idx] = (group_vals[i] - mean_val) / (std_val + 1e-4)

    return normalized


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    mini_batch: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    clip_eps: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update the policy using mini-batch PPO/GRPO objective."""

    episodes = list(mini_batch)
    if len(episodes) == 0:
        return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0, "clip_fraction": 0.0}

    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    optimizer.zero_grad(set_to_none=True)

    token_counts = [len(ep.generated_token_ids) for ep in episodes]
    total_target_tokens = sum(token_counts)
    if total_target_tokens == 0:
        return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0, "clip_fraction": 0.0}

    total_loss_value = 0.0
    total_entropy_value = 0.0
    total_clip_count = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        batch_episodes = episodes[i : i + micro_batch_size]
        batch_lengths = [len(ep.prefix_token_ids) + len(ep.generated_token_ids) for ep in batch_episodes]
        batch_max_length = max(batch_lengths)

        batch_token_ids = []
        batch_prefix_lens = []
        batch_advantages = []
        batch_old_logprobs = []
        for ep in batch_episodes:
            full_ids = ep.prefix_token_ids + ep.generated_token_ids
            batch_token_ids.append(full_ids + [pad_token_id] * (batch_max_length - len(full_ids)))
            batch_prefix_lens.append(len(ep.prefix_token_ids))
            batch_advantages.append(float(ep.advantage) if ep.advantage is not None else float(ep.reward))
            if ep.old_logprobs is not None:
                batch_old_logprobs.append(ep.old_logprobs.to(torch.float32))
            else:
                batch_old_logprobs.append(None)

        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        input_ids = batch_token_ids[:, :-1]

        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.forward(input_ids)

        log_probs = torch.log_softmax(logits.float(), dim=-1)

        micro_loss_sum = torch.tensor(0.0, device=device)

        for row, ep in enumerate(batch_episodes):
            gen_len = len(ep.generated_token_ids)
            if gen_len == 0:
                continue

            prefix_len = batch_prefix_lens[row]
            start = max(prefix_len - 1, 0)
            end = start + gen_len

            target_ids = torch.tensor(ep.generated_token_ids, device=device, dtype=torch.long)
            row_log_probs = log_probs[row, start:end]
            new_logprobs = torch.gather(row_log_probs, 1, target_ids.unsqueeze(-1)).squeeze(-1)

            if batch_old_logprobs[row] is not None:
                old_logprobs = batch_old_logprobs[row].to(device)
                assert new_logprobs.shape == old_logprobs.shape, (
                    f"new_logprobs shape {new_logprobs.shape} != old_logprobs shape {old_logprobs.shape}"
                )
                ratio = torch.exp(new_logprobs - old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                clip_mask = (ratio - clipped_ratio).abs() > 1e-8
            else:
                ratio = torch.ones_like(new_logprobs)
                clipped_ratio = ratio
                clip_mask = torch.zeros_like(new_logprobs, dtype=torch.bool)

            adv = torch.tensor(batch_advantages[row], device=device, dtype=torch.float32)
            surr1 = ratio * adv
            surr2 = clipped_ratio * adv
            token_losses = -torch.minimum(surr1, surr2)

            micro_loss_sum = micro_loss_sum + token_losses.sum()
            total_loss_value += float(token_losses.sum().item())
            total_entropy_value += float(compute_entropy(logits[row, start:end]).sum().item())
            total_clip_count += float(clip_mask.to(torch.float32).sum().item())

        micro_loss = micro_loss_sum / total_target_tokens
        micro_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss_value / total_target_tokens,
        "grad_norm": grad_norm.item(),
        "entropy": total_entropy_value / total_target_tokens,
        "clip_fraction": total_clip_count / total_target_tokens,
    }
