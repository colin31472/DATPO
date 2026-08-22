import html
import random
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from math_dataset import MathDataset, reward_function
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer
from datpo_zero import (
    attnrl_rollout_from_roots,
    attnrl_sample_initial_roots,
    MathDatasetTreeSearchEngine,
    GRPOPolicyBackend,
    DatpoConfig,
    datpo_rollout,
)
from datpo.config import ModelConfig, TreeSearchConfig
from transformers import AutoTokenizer


def get_split_path(config, split: str) -> str:
    root = Path(config["data"]["root_path"])
    filename = (
        config["data"]["train_file"] if split == "train" else config["data"]["test_file"]
    )
    return str(root / filename)


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = MathDataset(
        data_path=get_split_path(config, "test"),
        tokenizer=tokenizer,
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device="cpu").manual_seed(config["training"]["random_seed"])
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=MathDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"],
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([float(ep.reward_info["correct"]) for ep in episodes])
    return float(np.mean(success)) if len(success) > 0 else 0.0

def _accumulate_tokens_per_question(
    *,
    episodes,
    num_questions_in_batch: int,
    k: int,
    total_generated_tokens_over_questions: int,
    search_stats=None,
):
    """
    Accumulate total generated tokens per question.

    For DATPO runs, we use per-tree token counts from search statistics, which now
    represent the total tokens generated across all rollouts in the search tree.
    For GRPO runs, we fall back to summing the generated tokens across K samples per
    question.

    Assumption for GRPO fallback:
      - episodes are appended in order: question i -> answers j=0..k-1
    """
    if search_stats is not None:
        for qi in range(min(num_questions_in_batch, len(search_stats))):
            total_generated_tokens_over_questions += search_stats[qi].generated_token_count
        return total_generated_tokens_over_questions

    for qi in range(num_questions_in_batch):
        s = qi * k
        e = s + k
        if s >= len(episodes):
            break
        q_eps = episodes[s: min(e, len(episodes))]
        q_token_sum = sum(len(ep.generated_token_ids) for ep in q_eps)
        total_generated_tokens_over_questions += q_token_sum
    return total_generated_tokens_over_questions


def _accumulate_training_tokens_per_question(
    *,
    episodes_used_for_update,
    num_questions_in_batch: int,
    total_training_tokens_over_questions: int,
):
    """
    Accumulate tokens actually USED for training (i.e., tokens in episodes passed to update_policy),
    per question.

    - DATPO: episodes are block-level samples created ONLY for non-blocked nodes (early termination excluded).
      So summing episode token lengths here excludes early-terminated parts automatically.
    - GRPO: this becomes "tokens from finished episodes actually used for update" (if skip_unfinished enabled).
    """
    if not episodes_used_for_update:
        return total_training_tokens_over_questions

    # Group by question/prefix text. (Batch prefixes are typically unique.)
    per_prefix_sum = defaultdict(int)
    for ep in episodes_used_for_update:
        per_prefix_sum[str(ep.prefix)] += len(ep.generated_token_ids)

    # Add per-question totals (cap by num_questions_in_batch defensively)
    added = 0
    for _, v in per_prefix_sum.items():
        total_training_tokens_over_questions += int(v)
        added += 1
        if added >= num_questions_in_batch:
            break

    return total_training_tokens_over_questions


def _normalize_episode_advantages_global(episodes, eps: float = 1e-8):
    if not episodes:
        return None

    values = [
        float(episode.advantage) if episode.advantage is not None else float(episode.reward)
        for episode in episodes
    ]
    advantage_tensor = torch.tensor(values, dtype=torch.float32)
    mean = advantage_tensor.mean()
    std = advantage_tensor.std(unbiased=False)

    if not torch.isfinite(std) or std.item() == 0.0:
        normalized = advantage_tensor - mean
    else:
        normalized = (advantage_tensor - mean) / (std + eps)

    for idx, episode in enumerate(episodes):
        episode.advantage = float(normalized[idx].item())

    return float(mean.item()), float(std.item())


def _iter_with_lookahead(iterable):
    iterator = iter(iterable)
    try:
        current = next(iterator)
    except StopIteration:
        return
    for next_item in iterator:
        yield current, next_item
        current = next_item
    yield current, None


def _iter_one_step_delayed(iterable):
    iterator = iter(iterable)
    try:
        previous = next(iterator)
    except StopIteration:
        return

    yield None, previous
    for current in iterator:
        yield previous, current
        previous = current
    yield previous, None


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    datpo_cfg = DatpoConfig(**config.get("datpo", {}))

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    torch.random.manual_seed(config["training"]["random_seed"])

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["training"]["random_seed"])

    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH
    MINI_BATCH_SIZE = int(config["training"]["mini_batch_size"])
    MICRO_BATCH_SIZE = int(config["training"]["micro_batch_size"])
    PPO_EPOCHS = int(config["training"]["ppo_epochs"])
    CLIP_EPS = float(config["training"]["clip_eps"])
    assert MINI_BATCH_SIZE % MICRO_BATCH_SIZE == 0, "mini_batch_size must be divisible by micro_batch_size"

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    train_dataset = MathDataset(
        data_path=get_split_path(config, "train"),
        tokenizer=tokenizer,
        test_size=config["data"]["test_size"],
        train_samples=config["data"].get("train_samples"),
    )
    generator = torch.Generator(device="cpu").manual_seed(config["training"]["random_seed"])
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=MathDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    EPOCHS = int(config["training"].get("epochs", 1))
    global_step = 0

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    # --- DATPO engine init (kept as-is) ---
    tree_engine = None
    if datpo_cfg.enabled:
        tree_model_config = ModelConfig(
            max_new_tokens=config["training"]["max_gen_len"],
            embedding_model=datpo_cfg.embedding_model,
            pad_token=hf_tokenizer.pad_token,
            eos_token=hf_tokenizer.eos_token,
        )
        tree_search_config = TreeSearchConfig(
            root_rollouts=datpo_cfg.root_rollouts,
            adaptive=datpo_cfg.adaptive,
            er_max=datpo_cfg.er_max,
            bp_max=datpo_cfg.bp_max,
            bn_max=datpo_cfg.bn_max,
            divergence_alpha=datpo_cfg.divergence_alpha,
            divergence_alpha_min=datpo_cfg.divergence_alpha_min,
            fork_selection=datpo_cfg.fork_selection,
            entropy_smoothing=datpo_cfg.entropy_smoothing,
            one_step_off_policy=datpo_cfg.one_step_off_policy,
        )
        backend = GRPOPolicyBackend(
            model=model,
            tokenizer=hf_tokenizer,
            device=device,
            dtype=dtype,
            verbose_generation=datpo_cfg.verbose_generation,
        )
        tree_engine = MathDatasetTreeSearchEngine(
            backend=backend,
            tokenizer=hf_tokenizer,
            model_config=tree_model_config,
            tree_config=tree_search_config,
        )

    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # [NEW] Best-eval checkpoint tracking
    # ----------------------------
    best_eval_score = -float("inf")
    best_eval_step = 0

    # --- training time + token tracking ---
    train_start_time = time.time()
    step_start_time = train_start_time
    total_questions_seen = 0
    # (A) metric: all generated tokens per question (DATPO: all rollouts in search tree; GRPO: K samples)
    total_generated_tokens_over_questions = 0
    # (B) metric: tokens actually used for training per question (DATPO: excludes early-terminated/blocked; GRPO: excludes skipped unfinished)
    total_training_tokens_over_questions = 0
    level_leaf_totals = defaultdict(int)
    level_pass_totals = defaultdict(float)
    level_question_counts = defaultdict(int)

    # --- optional initial eval ---
    eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
    print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
    tb_writer.add_scalar("success_rate/eval", eval_success_rate, 0)

    # ----------------------------
    # [NEW] Initialize best from initial eval, save best checkpoint at step 0
    # ----------------------------
    best_eval_score = eval_success_rate
    best_eval_step = 0
    best_path = ckpt_dir / "best.pt"
    torch.save(model.state_dict(), best_path)
    print(f"Saved BEST checkpoint to {best_path} (score={best_eval_score:.4f}, step={best_eval_step})")
    tb_writer.add_scalar("success_rate/best_eval", best_eval_score, 0)

    use_one_step_off_policy = tree_engine is not None and datpo_cfg.one_step_off_policy

    for epoch in range(1, EPOCHS + 1):
        pending_one_step_roots = None
        step_iterator = (
            _iter_one_step_delayed(train_dataloader)
            if use_one_step_off_policy
            else train_dataloader
        )
        for step_in_epoch, step_item in enumerate(step_iterator, start=1):
            if use_one_step_off_policy:
                batch, next_batch = step_item
                if batch is None:
                    pending_one_step_roots = attnrl_sample_initial_roots(
                        tree_engine, next_batch
                    )
                    continue
            else:
                batch = step_item
                next_batch = None
            global_step += 1

            # ----------------------------
            # Track progress + actual DATPO alpha used by engine
            # ----------------------------
            progress = None
            current_alpha = None

            if tree_engine is not None:
                # progress in [0,1] across the whole run (epoch-aware)
                total_steps = len(train_dataloader) * max(EPOCHS, 1)
                progress = min(1.0, global_step / max(total_steps, 1))

                # Log the engine's scheduled alpha directly.
                current_alpha = float(tree_engine._scheduled_alpha(progress))

                if use_one_step_off_policy:
                    if pending_one_step_roots is None:
                        pending_one_step_roots = attnrl_sample_initial_roots(tree_engine, batch)
                    episodes, search_stats, pending_one_step_roots = attnrl_rollout_from_roots(
                        engine=tree_engine,
                        batch=batch,
                        precomputed_roots=pending_one_step_roots,
                        progress=progress,
                        next_batch=next_batch,
                    )
                else:
                    episodes, search_stats = datpo_rollout(
                        engine=tree_engine,
                        batch=batch,
                        progress=progress,
                    )
            else:
                episodes = rollout(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    max_gen_len=config["training"]["max_gen_len"],
                    num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                    reward_function=reward_function,
                    device=device,
                    dtype=dtype,
                )
                search_stats = None

            # --- accumulate leaf counts by problem level (DATPO only) ---
            if tree_engine is not None and search_stats:
                for idx, level in enumerate(batch.levels):
                    if level is None:
                        continue
                    if idx >= len(search_stats):
                        break
                    level_leaf_totals[level] += search_stats[idx].leaf_count
                    level_pass_totals[level] += float(
                        getattr(search_stats[idx], "tree_pass_rate", 0.0)
                    )
                    level_question_counts[level] += 1

            # --- metric (A): avg generated tokens per question ---
            k = NUM_ANSWERS_PER_QUESTION
            num_questions_in_batch = len(batch.prefix)  # == NUM_QUESTIONS_PER_BATCH
            total_generated_tokens_over_questions = _accumulate_tokens_per_question(
                episodes=episodes,
                num_questions_in_batch=num_questions_in_batch,
                k=k,
                total_generated_tokens_over_questions=total_generated_tokens_over_questions,
                search_stats=search_stats,
            )
            total_questions_seen += num_questions_in_batch
            avg_tokens_per_question_so_far = (
                total_generated_tokens_over_questions / total_questions_seen
                if total_questions_seen > 0
                else 0.0
            )

            # --- select episodes actually used for update (affects training-token metric) ---
            episodes_for_update = episodes
            if config["training"]["skip_unfinished_episodes"]:
                episodes_for_update = [ep for ep in episodes_for_update if ep.is_finished]

            # --- metric (B): avg training-used tokens per question ---
            total_training_tokens_over_questions = _accumulate_training_tokens_per_question(
                episodes_used_for_update=episodes_for_update,
                num_questions_in_batch=num_questions_in_batch,
                total_training_tokens_over_questions=total_training_tokens_over_questions,
            )
            avg_training_tokens_per_question_so_far = (
                total_training_tokens_over_questions / total_questions_seen
                if total_questions_seen > 0
                else 0.0
            )

            if len(episodes_for_update) == 0:
                print(
                    f"\rEpoch {epoch}/{EPOCHS} | Step {global_step} "
                    f"(epoch_step={step_in_epoch}) skipped (no finished episodes)"
                    + " " * 20
                )
                continue

            if tree_engine is not None and datpo_cfg.adv_normalization:
                norm_stats = _normalize_episode_advantages_global(episodes_for_update)
                if norm_stats is not None:
                    mean, std = norm_stats
                    print(
                        f"\rDATPO advantage normalized (mean={mean:.6f}, std={std:.6f})"
                        + " " * 10
                    )

            ppo_results = []
            for _ in range(PPO_EPOCHS):
                shuffled_episodes = list(episodes_for_update)
                random.shuffle(shuffled_episodes)
                for mb_start in range(0, len(shuffled_episodes), MINI_BATCH_SIZE):
                    mini_batch = shuffled_episodes[mb_start : mb_start + MINI_BATCH_SIZE]
                    if not mini_batch:
                        continue
                    ppo_results.append(
                        update_policy(
                            model=model,
                            optimizer=optimizer,
                            mini_batch=mini_batch,
                            micro_batch_size=MICRO_BATCH_SIZE,
                            pad_token_id=tokenizer.pad_token_id,
                            max_grad_norm=config["training"]["max_grad_norm"],
                            clip_eps=CLIP_EPS,
                            device=device,
                            dtype=dtype,
                        )
                    )

            results = {
                "loss": float(np.mean([r["loss"] for r in ppo_results])) if ppo_results else 0.0,
                "grad_norm": float(np.mean([r["grad_norm"] for r in ppo_results])) if ppo_results else 0.0,
                "entropy": float(np.mean([r["entropy"] for r in ppo_results])) if ppo_results else 0.0,
                "clip_fraction": float(np.mean([r["clip_fraction"] for r in ppo_results])) if ppo_results else 0.0,
            }

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            duration = end_time - step_start_time
            step_start_time = end_time

            reward = [episode.reward for episode in episodes_for_update]
            answer_reward = [episode.reward_info["answer_reward"] for episode in episodes_for_update]
            num_finished_episodes = sum(episode.is_finished for episode in episodes_for_update)

            std_reward = float(np.std(reward)) if len(reward) > 0 else 0.0
            success_rate = float(np.mean(answer_reward)) if len(answer_reward) > 0 else 0.0

            grad_norm = results["grad_norm"]
            entropy = results["entropy"]
            lr = optimizer.param_groups[0]["lr"]
            loss = results["loss"]
            clip_fraction = results["clip_fraction"]

            print(
                f"\rEpoch {epoch}/{EPOCHS} | Step {global_step} "
                f"train success_rate: {success_rate:.2f}, "
                f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
                f"num_finished_episodes: {num_finished_episodes}, "
                f"avg_tokens_per_question: {avg_tokens_per_question_so_far:.2f}, "
                f"avg_training_tokens_per_question: {avg_training_tokens_per_question_so_far:.2f}, "
                f"entropy: {entropy:.2f}, "
                f"clip_fraction: {clip_fraction:.3f}",
            )

            # ----------------------------
            # Eval + (NEW) Best checkpoint saving
            # ----------------------------
            if global_step % config["training"]["eval_interval"] == 0:
                eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
                print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
                tb_writer.add_scalar("success_rate/eval", eval_success_rate, global_step)

                # Save checkpoint when eval achieves a new best.
                if eval_success_rate > best_eval_score:
                    best_eval_score = float(eval_success_rate)
                    best_eval_step = int(global_step)

                    # always overwrite "best.pt" with latest best
                    best_path = ckpt_dir / "best.pt"
                    torch.save(model.state_dict(), best_path)

                    print(
                        f"New BEST eval={best_eval_score:.4f} at step={best_eval_step} "
                        f"-> saved {best_path}"
                    )

            # --- TensorBoard scalars ---
            tb_writer.add_scalar("loss", loss, global_step)
            tb_writer.add_scalar("ppo/clip_fraction", clip_fraction, global_step)
            tb_writer.add_scalar("std_reward", std_reward, global_step)
            tb_writer.add_scalar("success_rate/train", success_rate, global_step)
            tb_writer.add_scalar("grad_norm", grad_norm, global_step)
            tb_writer.add_scalar("duration", duration, global_step)
            tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, global_step)
            tb_writer.add_scalar("learning_rate", lr, global_step)
            tb_writer.add_scalar("entropy", entropy, global_step)
            tb_writer.add_scalar("gen/avg_tokens_per_question", avg_tokens_per_question_so_far, global_step)
            tb_writer.add_scalar(
                "gen/avg_training_tokens_per_question",
                avg_training_tokens_per_question_so_far,
                global_step,
            )

            # --- Log actual alpha/progress used by DATPO engine ---
            if tree_engine is not None and progress is not None and current_alpha is not None:
                tb_writer.add_scalar("datpo/divergence_alpha", current_alpha, global_step)
                tb_writer.add_scalar("datpo/progress", float(progress), global_step)

            for i, episode in enumerate(episodes_for_update):
                text = html.escape(episode.text)
                tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", global_step)

            # --- save checkpoint (interval) ---
            if global_step % config["training"]["ckpt_save_interval"] == 0:
                output_file = ckpt_dir / f"ckpt_{global_step:06d}.pt"
                torch.save(model.state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")

    # --- save checkpoint (final) ---
    output_file = ckpt_dir / f"ckpt_{global_step:06d}.pt"
    torch.save(model.state_dict(), output_file)
    print(f"Saved checkpoint to {output_file}")

    # --- final summary ---
    total_train_time = time.time() - train_start_time
    avg_tokens_per_question = (
        total_generated_tokens_over_questions / total_questions_seen
        if total_questions_seen > 0
        else 0.0
    )
    avg_training_tokens_per_question = (
        total_training_tokens_over_questions / total_questions_seen
        if total_questions_seen > 0
        else 0.0
    )

    level_leaf_summary = []
    for level in sorted(level_leaf_totals.keys()):
        total_leaves = level_leaf_totals[level]
        total_passes = level_pass_totals[level]
        question_count = level_question_counts.get(level, 0)
        avg_leaves = (total_leaves / question_count) if question_count > 0 else 0.0
        pass_rate = (total_passes / question_count) if question_count > 0 else 0.0
        level_leaf_summary.append((level, total_leaves, avg_leaves, pass_rate, question_count))

    print("\n" + "=" * 60)
    print(
        f"Total training time: {total_train_time:.2f} sec "
        f"({total_train_time/60:.2f} min, {total_train_time/3600:.2f} hr)"
    )
    avg_tokens_desc = (
        "total tokens generated across all rollouts in the search tree"
        if tree_engine is not None
        else f"sum over {NUM_ANSWERS_PER_QUESTION} samples"
    )
    print(
        f"Avg generated tokens per question ({avg_tokens_desc}): {avg_tokens_per_question:.2f}"
    )
    print(
        f"Avg TRAINING-used tokens per question (episodes passed to update, DATPO excludes early-terminated): "
        f"{avg_training_tokens_per_question:.2f}"
    )
    print(
        f"   (questions_seen={total_questions_seen}, "
        f"total_generated_tokens={total_generated_tokens_over_questions}, "
        f"total_training_tokens={total_training_tokens_over_questions})"
    )
    if level_leaf_summary:
        print("Level-wise leaf statistics (DATPO search):")
        for level, total_leaves, avg_leaves, pass_rate, question_count in level_leaf_summary:
            print(
                f"   Level {level}: total_leaf_count={total_leaves}, "
                f"avg_leaf_count_per_question={avg_leaves:.2f}, "
                f"pass_rate={pass_rate:.4f} "
                f"(questions={question_count})"
            )
    elif tree_engine is not None:
        print("No level-wise leaf statistics collected (no level data found).")
    print("=" * 60 + "\n")

    tb_writer.add_scalar("time/total_hrs", total_train_time / 3600, global_step)
    tb_writer.add_scalar("gen/avg_tokens_per_question", avg_tokens_per_question, global_step)
    tb_writer.add_scalar("gen/avg_training_tokens_per_question", avg_training_tokens_per_question, global_step)
    for level, _, avg_leaves, pass_rate, _ in level_leaf_summary:
        tb_writer.add_scalar(
            f"datpo/avg_leaf_count_level_{level}", avg_leaves, global_step
        )
        tb_writer.add_scalar(
            f"datpo/pass_rate_level_{level}", pass_rate, global_step
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
