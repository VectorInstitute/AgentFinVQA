"""Register supported dataset samples as a Langfuse Dataset.

Usage:
    python -m agentfinvqa.langfuse_integration.dataset \
        --split test --n 25
"""

import argparse
from collections.abc import Callable
from typing import List, Optional, Sequence, TypedDict

from ..datasets.chartqapro_loader import load_chartqapro
from ..datasets.finmme_loader import load_finmme
from ..datasets.perceived_sample import PerceivedSample
from .client import get_client


DatasetLoader = Callable[..., List[PerceivedSample]]


class DatasetLoaderConfig(TypedDict):
    """Configuration describing how to load and label a dataset for Langfuse."""

    loader: DatasetLoader
    display_name: str
    default_image_dir: str


DATASET_LOADERS: dict[str, DatasetLoaderConfig] = {
    "chartqapro": {
        "loader": load_chartqapro,
        "display_name": "ChartQAPro",
        "default_image_dir": "data/chartqapro_images",
    },
    "finmme": {
        "loader": load_finmme,
        "display_name": "FinMME",
        "default_image_dir": "data/finmme_images",
    },
}


def register_dataset(
    samples: Sequence[PerceivedSample],
    dataset_name: str = "ChartQAPro",
    split: str = "test",
) -> Optional[str]:
    """
    Upload a collection of samples as a Langfuse Dataset.

    Allows for versioned dataset management and evaluation in the Langfuse UI.

    Parameters
    ----------
    samples : list of PerceivedSample
        The data samples to register.
    dataset_name : str, default 'ChartQAPro'
        The base name for the dataset.
    split : str, default 'test'
        The split identifier (e.g., 'train', 'val').

    Returns
    -------
    str or None
        The name of the created dataset if successful, else None.
    """
    client = get_client()
    if client is None:
        return None

    name = f"{dataset_name}_{split}"
    try:
        client.create_dataset(name=name)
        for s in samples:
            client.create_dataset_item(
                dataset_name=name,
                input={
                    "source_id": s.sample_id,  # stored as data field; Langfuse auto-generates UUID v7 id
                    "question": s.question,
                    "question_type": s.question_type.value,
                    "image_path": s.image_path or "",
                    "choices": s.choices or [],
                },
                expected_output=s.expected_output,
            )
        print(f"[langfuse] Registered {len(samples)} samples → dataset '{name}'")
        return name
    except Exception as exc:
        print(f"[langfuse] Dataset registration failed: {exc}")
        return None


def main() -> None:
    """
    Command-line interface for registering supported datasets.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Register dataset samples as Langfuse dataset")
    parser.add_argument("--dataset", default="chartqapro", choices=sorted(DATASET_LOADERS.keys()))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    dataset_key = args.dataset.lower()
    cfg = DATASET_LOADERS[dataset_key]
    loader = cfg["loader"]
    display_name = cfg["display_name"]
    image_dir = args.image_dir or cfg["default_image_dir"]

    samples = loader(split=args.split, n=args.n, image_dir=image_dir, cache_dir=args.cache_dir)
    register_dataset(samples, dataset_name=display_name, split=args.split)


if __name__ == "__main__":
    main()
