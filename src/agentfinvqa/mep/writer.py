"""MEP file I/O utilities."""

import json
from pathlib import Path
from typing import Iterator

from .schema import MEP


def write_mep(mep: MEP, out_dir: str) -> str:
    """Serialize MEP to JSON and write to <out_dir>/<sample_id>.json. Returns path."""
    assert mep.sample is not None, "MEP must have a sample before writing"
    path = Path(out_dir) / f"{mep.sample.sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(mep.to_dict(), f, indent=2, default=str)
    return str(path)


def read_mep(path: str) -> dict:
    """Read a MEP JSON file from disk and return its content as a dict."""
    with open(path) as f:
        return json.load(f)


def iter_meps(mep_dir: str) -> Iterator[dict]:
    """Yield all MEP dicts from a directory, sorted by filename."""
    for p in sorted(Path(mep_dir).glob("*.json")):
        try:
            yield read_mep(str(p))
        except Exception as e:
            print(f"Warning: could not read MEP {p}: {e}")
