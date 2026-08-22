"""Utilities for entropy-based fork selection."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import pysbd


def sequence_entropy(logits: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """Compute token-wise entropies from logits.

    Args:
        logits: Tensor of shape (seq_len, vocab_size) with logit values.
        smoothing: Optional smoothing factor mixed with a uniform
            distribution to avoid zero-probability bins when computing the
            entropy. When set to zero, the standard entropy is returned.

    Returns:
        Tensor of shape (seq_len,) containing entropies.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    if smoothing > 0:
        smoothing = float(min(max(smoothing, 0.0), 1.0))
        vocab = probs.shape[-1]
        uniform_prob = 1.0 / float(vocab)
        probs = (1 - smoothing) * probs + smoothing * uniform_prob
        log_probs = torch.log(probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def rolling_delta(values: torch.Tensor) -> torch.Tensor:
    """Finite differences for sequences."""
    if values.numel() < 2:
        return torch.zeros_like(values)
    diff = values[1:] - values[:-1]
    return torch.cat([diff.new_zeros(1), diff])


def topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the top-k values."""
    k = min(k, values.shape[0])
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    topk = torch.topk(values, k=k)
    return topk.indices.sort().values


def _pysbd_sentence_spans(
    text: str,
    *,
    language: str = "en",
    clean: bool = False,
) -> List[Tuple[int, int]]:
    """
    Compute sentence character spans using PySBD.

    PySBD returns sentence strings (not spans), so we recover spans by scanning the
    original text left-to-right. This is stable even when the same sentence text
    repeats, because we always search starting from the last end position.
    """
    seg = pysbd.Segmenter(language=language, clean=clean)
    sents = seg.segment(text)

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for sent in sents:
        if not sent:
            continue

        start = text.find(sent, cursor)
        if start < 0:
            # If mismatch happens (rare; e.g., due to normalization), skip this chunk
            # rather than returning wrong spans.
            continue

        end = start + len(sent)
        spans.append((start, end))
        cursor = end

    if not spans and text:
        spans = [(0, len(text))]

    return spans


def _build_token_spans_by_single_decode(
    tokenizer,
    token_ids: Sequence[int],
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Fallback path: build full_text and per-token char spans by decoding each token id
    individually and concatenating. This keeps a 1:1 mapping with the original token_ids
    even when re-tokenization of the decoded text would mismatch.
    """
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for tid in token_ids:
        piece = tokenizer.decode(
            [int(tid)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        start = cursor
        cursor += len(piece)
        end = cursor
        parts.append(piece)
        spans.append((start, end))
    return "".join(parts), spans


def _align_sentences_with_offsets(
    sent_char_spans: Iterable[Tuple[int, int]],
    offsets: List[Tuple[int, int]],
) -> List[List[int]]:
    """
    Given sentence char spans and token offsets (start,end) in the same text,
    return token-index groups per sentence by overlap.
    """
    boundaries: List[List[int]] = []
    n = len(offsets)

    # Two-pointer sweep because both are in increasing order.
    tok_i = 0
    for sent_start, sent_end in sent_char_spans:
        current: List[int] = []

        # Advance tokens that end before sentence starts
        while tok_i < n and offsets[tok_i][1] <= sent_start:
            tok_i += 1

        j = tok_i
        while j < n:
            t_start, t_end = offsets[j]

            # If token starts after sentence ends, stop
            if t_start >= sent_end:
                break

            # Overlap check (ignore empty offsets like (0,0) sometimes used for special tokens)
            if t_end > sent_start and t_start < sent_end and not (t_start == 0 and t_end == 0):
                current.append(j)

            j += 1

        if current:
            boundaries.append(current)

        # Keep tok_i where it is (not necessarily j) to be safe;
        # but moving tok_i forward helps performance without harming correctness
        tok_i = max(tok_i, j)

    # If anything remains, append as trailing group (robustness)
    if boundaries:
        last_end = boundaries[-1][-1] + 1
    else:
        last_end = 0

    if last_end < n:
        remaining = list(range(last_end, n))
        if remaining:
            boundaries.append(remaining)

    return boundaries


def split_into_sentences(
    tokens: List[str],
    *,
    tokenizer=None,
    token_ids: Optional[Sequence[int]] = None,
    text: Optional[str] = None,
    language: str = "en",
    clean: bool = False,
) -> List[List[int]]:
    """
    Return sentence boundaries as token-index lists.

    Preferred (Option B):
      - Provide tokenizer + token_ids (or tokenizer + text)
      - Use PySBD on text to get sentence char-spans
      - Use HF fast tokenizer offsets to map sentence spans -> token indices

    Backward compatible:
      - If tokenizer/token_ids/text not provided, fall back to a naive concat-based approach.

    Args:
        tokens: list of token strings (used for backward compatibility and sanity checks).
        tokenizer: HuggingFace tokenizer (preferably fast).
        token_ids: original token ids aligned with `tokens`.
        text: decoded string (optional; if not provided, derived from token_ids).
        language: PySBD language code (default "en").
        clean: PySBD clean option (default False to preserve offsets).

    Returns:
        List[List[int]]: each inner list contains token indices belonging to one sentence.
    """
    # ---- Option B path (offset mapping) ----
    if tokenizer is not None and (token_ids is not None or text is not None):
        if text is None:
            # IMPORTANT: disable cleanup to keep offsets stable
            text = tokenizer.decode(
                list(map(int, token_ids)),  # type: ignore[arg-type]
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        # Sentence char spans from PySBD
        sent_char_spans = _pysbd_sentence_spans(text, language=language, clean=clean)

        # If we have a fast tokenizer, we can get offsets
        is_fast = bool(getattr(tokenizer, "is_fast", False))
        if is_fast:
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = [(int(a), int(b)) for (a, b) in enc["offset_mapping"]]

            # If token_ids were provided, verify that re-tokenization matches 1:1.
            # If mismatch, offsets won't align with original entropies -> fallback.
            if token_ids is not None:
                enc_ids = enc.get("input_ids", [])
                if len(enc_ids) != len(token_ids) or any(
                    int(a) != int(b) for a, b in zip(enc_ids, token_ids)
                ):
                    # Fallback: build offsets that are guaranteed aligned with original token_ids
                    text2, spans = _build_token_spans_by_single_decode(tokenizer, token_ids)
                    sent_char_spans2 = _pysbd_sentence_spans(text2, language=language, clean=clean)
                    return _align_sentences_with_offsets(sent_char_spans2, spans)

            return _align_sentences_with_offsets(sent_char_spans, offsets)

        # No fast tokenizer -> fallback span building
        if token_ids is not None:
            text2, spans = _build_token_spans_by_single_decode(tokenizer, token_ids)
            sent_char_spans2 = _pysbd_sentence_spans(text2, language=language, clean=clean)
            return _align_sentences_with_offsets(sent_char_spans2, spans)

    # ---- Legacy fallback (kept for compatibility) ----
    # NOTE: This is the old-style concat without real spacing rules; used only when we
    # cannot do offset mapping.
    full_text = ""
    token_spans: List[Tuple[int, int]] = []
    current_pos = 0

    for tok in tokens:
        if tok is None:
            token_spans.append((current_pos, current_pos))
            continue
        start = current_pos
        full_text += tok
        current_pos += len(tok)
        end = current_pos
        token_spans.append((start, end))

    sent_char_spans = _pysbd_sentence_spans(full_text, language=language, clean=clean)
    return _align_sentences_with_offsets(sent_char_spans, token_spans)


__all__ = ["sequence_entropy", "rolling_delta", "topk_indices", "split_into_sentences"]