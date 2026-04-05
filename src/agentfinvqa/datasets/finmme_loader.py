"""FinMME dataset loader → PerceivedSample list.

Dataset: luojunyu/FinMME
Key columns:
  id                int        – unique identifier
  image             Image      – financial chart image
  question_text     str        – natural language query
  question_type     str        – dataset-provided category label
  options           str/list   – MCQ options (optional)
  answer            str        – reference answer
  unit              str        – unit metadata (optional)
  tolerance         float      – numeric tolerance metadata (optional)
  verified_caption  str        – model-verified caption text
  related_sentences list[str]  – supporting context sentences
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, List, Optional

from .image_utils import save_image_data
from .perceived_sample import (
    UNANSWERABLE_ANSWERS,
    UNANSWERABLE_TOKEN,
    PerceivedSample,
    QuestionType,
)


def _map_question_type(raw: Optional[str]) -> QuestionType:
    """Translate dataset labels into internal QuestionType."""
    if not raw:
        return QuestionType.STANDARD
    label = raw.strip().lower()
    if any(term in label for term in ("choice", "mcq", "multiple")):
        return QuestionType.MCQ
    if "conversation" in label or "conversational" in label:
        return QuestionType.CONVERSATIONAL
    if "hypothetical" in label or "what if" in label:
        return QuestionType.HYPOTHETICAL
    if label in {"unanswerable", "cannot answer", "not answerable"}:
        return QuestionType.UNANSWERABLE
    return QuestionType.STANDARD


def _normalize_answer(answer: Any, qtype: QuestionType) -> str:
    """Convert reference answers into canonical strings."""
    text = "" if answer is None else str(answer)
    if qtype == QuestionType.UNANSWERABLE:
        return UNANSWERABLE_TOKEN
    if text.strip().lower() in UNANSWERABLE_ANSWERS:
        return UNANSWERABLE_TOKEN
    return text.strip()


def _clean_options(values: List[str]) -> Optional[List[str]]:
    cleaned = [opt.strip() for opt in values if isinstance(opt, str) and opt.strip()]
    return cleaned or None


def _parse_options(raw: Any) -> Optional[List[str]]:
    """Best-effort parser for options stored as string/list/JSON."""
    if raw is None:
        return None

    cleaned: Optional[List[str]] = None
    if isinstance(raw, list):
        cleaned = _clean_options([str(opt) for opt in raw])
    elif isinstance(raw, dict):
        cleaned = _clean_options([str(v) for v in raw.values()])
    else:
        text = str(raw).strip()
        if text:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    cleaned = _clean_options([str(opt) for opt in parsed])
                    break
            if cleaned is None:
                labeled = re.findall(r"(?:^|\n)\s*[A-H][\).:\-]\s*([^\n]+)", text)
                if labeled:
                    cleaned = _clean_options(labeled)
                else:
                    parts = re.split(r"(?:\|\||\||;|\n)+", text)
                    cleaned = _clean_options(parts)

    return cleaned


def _build_sample(row_idx: int, row: dict, img_dir: Path) -> PerceivedSample:
    """Normalize a FinMME dataset row into a PerceivedSample."""
    question = (row.get("question_text") or "").strip()
    if not question:
        raise ValueError(f"Row {row_idx} missing question_text")

    qtype_raw = row.get("question_type") or ""
    qtype = _map_question_type(qtype_raw)
    answer = _normalize_answer(row.get("answer"), qtype)
    options = _parse_options(row.get("options"))

    label_map: dict[str, str] | None = None
    if options:
        labels = [chr(ord("A") + idx) for idx in range(len(options))]
        label_map = dict(zip(labels, options))

    row_id = row.get("id")
    row_id_int = row_idx
    if row_id is not None:
        try:
            row_id_int = int(row_id)
        except (TypeError, ValueError):
            row_id_int = row_idx

    image_path = ""
    image = row.get("image")
    if image is not None:
        image_path = save_image_data(image, img_dir / f"finmme_{row_id_int:06d}.png")

    answer_raw = row.get("answer")
    answer_label = str(answer_raw).strip() if answer_raw is not None else ""
    answer = _normalize_answer(answer_raw, qtype)
    if label_map and answer_label:
        normalized_label = answer_label.replace(" ", "").upper()
        if normalized_label in label_map:
            answer = label_map[normalized_label]
        elif all((ch in label_map) for ch in normalized_label):
            answer = " + ".join(label_map[ch] for ch in normalized_label)

    metadata = {
        "dataset": "FinMME",
        "question_type_raw": qtype_raw,
        "unit": row.get("unit"),
        "tolerance": row.get("tolerance"),
        "verified_caption": row.get("verified_caption"),
        "related_sentences": row.get("related_sentences"),
        "row_idx": row_idx,
        "sample_id": row_id,
    }
    if label_map:
        metadata["choice_map"] = label_map
        metadata["answer_label"] = answer_label
        metadata["choices_labeled"] = [{"label": label, "text": text} for label, text in label_map.items()]
        metadata["expected_answer_text"] = answer

    return PerceivedSample(
        sample_id=f"finmme_{row_id_int:06d}",
        image_path=image_path,
        question=question,
        expected_output=answer,
        question_type=qtype,
        choices=options,
        metadata=metadata,
    )


def load_finmme(
    split: str = "train",
    n: Optional[int] = None,
    image_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> List[PerceivedSample]:
    """
    Load FinMME samples as PerceivedSample objects.

    Parameters
    ----------
    split : str, default 'train'
        Dataset split to load.
    n : int, optional
        Limit of samples to materialize.
    image_dir : str, optional
        Directory for cached chart images.
    cache_dir : str, optional
        HuggingFace datasets cache override.
    hf_token : str, optional
        Token for gated/private datasets.

    Returns
    -------
    list[PerceivedSample]
        Normalized FinMME records.
    """
    if split.lower().startswith("test"):
        suffix = split[4:]
        mapped_split = "train" + suffix
        if suffix and not suffix.startswith(":"):
            mapped_split = "train" + suffix
        print(f"FinMME dataset only provides a 'train' split. Using '{mapped_split}' instead of '{split}'.")
        split = mapped_split

    if image_dir is None:
        image_dir = "data/finmme_images"
    img_dir = Path(image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if hf_token:
        kwargs["token"] = hf_token

    from datasets import load_dataset  # noqa: PLC0415

    print(f"Loading luojunyu/FinMME split={split} …")
    ds = load_dataset("luojunyu/FinMME", split=split, **kwargs)
    total_rows = len(ds)

    samples: List[PerceivedSample] = []
    for row_idx, row in enumerate(ds):
        try:
            sample = _build_sample(row_idx, row, img_dir)
        except Exception as exc:
            print(f"  Warning: skipping row {row_idx}: {exc}")
            continue
        samples.append(sample)
        if n is not None and len(samples) >= n:
            break

    print(f"  → {len(samples)} samples loaded (from {total_rows} rows)")
    return samples
