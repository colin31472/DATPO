# math.py
import re
import signal
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

# sympy parsing utils (used in AnswerChecker)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from data_types import MiniBatch
from tokenizer import Tokenizer


# -------------------------
# Prompting (Math)
# -------------------------
_SYSTEM_PROMPT = (
    "You are an expert competition mathematician who explains solutions clearly and rigorously. "
    "Your response must follow a strict structure: first provide the step-by-step reasoning, "
    "and then provide the final answer in a LaTeX \\boxed{} block. "
    "Once you have provided the \\boxed{} answer, stop immediately. "
    "Do NOT add any summaries, repetitions, or further explanations after the boxed answer."
)

_USER_INSTRUCTIONS = (
    "Solve the problem step by step, show your reasoning, and conclude with the final answer "
    "enclosed in LaTeX \\boxed{} notation. After the answer is provided, do not repeat your reasoning."
)


def format_math_prompt(raw_problem: str, tokenizer=None) -> str:
    problem = (raw_problem or "").strip()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"{problem}\n\n{_USER_INSTRUCTIONS}"},
    ]

    # If HF-style tokenizer exists
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback (string prompt)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Problem:\n{problem}\n\n"
        f"{_USER_INSTRUCTIONS}\n\n"
        f"Reasoning:\n"
    )


# -------------------------
# Answer Checking (Accuracy Reward)
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
# Dataset
# -------------------------
class MathDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
        train_samples: Optional[int] = None,
    ):
        path = Path(data_path)

        # 1) If data_path is a parquet file, use it directly.
        if path.is_file() and path.suffix == ".parquet":
            data = pd.read_parquet(path)

            # Already split into train/test, so do not slice by test_size.
            self.data = data

        # 2) If data_path is a directory, select train/test parquet by split.
        elif path.is_dir():
            split_file = path / f"{split}.parquet"
            legacy_file = path / "data"  # Backward compatibility with the old layout.

            if split_file.exists():
                data = pd.read_parquet(split_file)
                self.data = data
            else:
                # 3) Legacy fallback: split a single dataset by test_size.
                data = pd.read_parquet(legacy_file)
                self.data = (
                    data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
                )

        else:
            raise FileNotFoundError(f"Invalid data_path: {data_path}")

        # train subsample
        if split == "train" and train_samples is not None:
            self.data = self.data.iloc[:train_samples]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        # Expecting columns: "problem", "answer", optional "level"
        problem = item["problem"]
        answer = item["answer"]
        level = item.get("level")

        item["nums"] = problem
        item["target"] = answer

        item["level"] = level
        item.update(self.encode_prefix(problem))
        return item

    def encode_prefix(self, problem: str):
        """
        Prefix is the *actual* input to the model.
        """
        # Build a chat-formatted prompt. Use HF tokenizer if available,
        # otherwise fall back to plain string prompt.
        prompt = format_math_prompt(problem, getattr(self.tokenizer, "hf_tokenizer", None))

        prefix = prompt
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        numbers = [item["nums"] for item in batch]   # problems
        target = [item["target"] for item in batch]  # answers
        levels = [item.get("level") for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]

        return MiniBatch(
            numbers=numbers,
            target=target,
            levels=levels,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


# -------------------------
# Reward Function (Accuracy only, format reward removed)
# -------------------------
def reward_function(
    response: str,
    numbers: List[Any] = None,  # kept for interface; contains "problem" strings
    target: Any = None,         # kept for interface; contains "answer" strings
    end_token: str = None,
) -> Dict[str, Any]:
    """
    Accuracy-only reward for Math.
    - numbers: unused (kept for compatibility)
    - target: ground-truth answer string
    """
    if end_token and isinstance(response, str) and response.endswith(end_token):
        response = response[: -len(end_token)]

    gt_answer = target if isinstance(target, str) else (str(target) if target is not None else "")

    correct = _ANSWER_CHECKER._is_answer_correct(response, gt_answer)
    reward = 1.0 if correct else 0.0

    return {
        "reward": reward,
        "reward_info": {
            "correct": correct,
        },
    }
