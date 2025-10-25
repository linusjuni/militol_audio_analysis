"""Create train/val CSV stubs for the MAD dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


CLASS_NAMES = ("Gunshot", "Footsteps", "Shelling", "Vehicle", "Helicopter", "Fighter")


def _load_or_stub(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        missing = {"path", "start_sec", "stop_sec", "class"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        return df
    rows = [
        {"path": "audio/example_001.wav", "start_sec": 0.0, "stop_sec": 3.0, "class": CLASS_NAMES[0]},
        {"path": "audio/example_002.wav", "start_sec": 1.0, "stop_sec": 4.0, "class": CLASS_NAMES[1]},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strong-label CSVs for the AST SED trainer.")
    parser.add_argument("--mad_root", type=Path, required=True, help="Path to MAD_dataset root.")
    parser.add_argument("--strong_annotations", type=Path, default=None, help="Existing strong CSV (defaults to mad_root/annotations/strong.csv).")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    strong_path = args.strong_annotations or (args.mad_root / "annotations" / "strong.csv")
    df = _load_or_stub(strong_path)
    df["path"] = df["path"].apply(lambda p: str((args.mad_root / p).resolve()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "train.csv", index=False)

    if 0.0 < args.val_fraction < 1.0 and len(df) > 1:
        val_df = df.sample(frac=args.val_fraction, random_state=42)
    else:
        val_df = df
    val_df.to_csv(args.out_dir / "val.csv", index=False)

    print(f"Wrote train/val CSVs to {args.out_dir}. Edit them to match your data paths.")


if __name__ == "__main__":
    main()
