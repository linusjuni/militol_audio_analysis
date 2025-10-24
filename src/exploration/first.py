"""Quick exploratory analysis of the MAD dataset.

This module loads the training and test metadata shipped in ``data/MAD_dataset``
and prints a handful of useful diagnostics: record counts, label distribution,
YouTube source variety, missing audio files, and a few duration estimates if
``librosa`` is available. Run it with:

    uv run python -m src.exploration.first
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - provide a human-friendly error
    raise RuntimeError(
        "pandas is required for this exploration script.\n"
        "Install it with `uv pip install pandas` or add it to your dependencies."
    ) from exc

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]

# Resolve repository root from this file's location: src/exploration/first.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "MAD_dataset"


def _read_split(split: str) -> pd.DataFrame:
    """Load a dataset split from CSV and enrich it with absolute audio paths."""
    csv_path = DATASET_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find metadata file at {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df = df.rename(columns={"youtube title": "youtube_title", "youtube url": "youtube_url"})
    df["audio_path"] = df["path"].apply(lambda rel: (DATASET_DIR / rel).resolve())
    return df


def _summarise_labels(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().sort_index()
    parts = [f"  Label {label}: {count} clips" for label, count in counts.items()]
    return "\n".join(parts)


def _find_missing_audio(df: pd.DataFrame) -> list[Path]:
    return [path for path in _iter_audio_paths(df) if not path.exists()]


def _iter_audio_paths(df: pd.DataFrame) -> Iterable[Path]:
    return (Path(path) for path in df["audio_path"].tolist())


def _sample_durations(df: pd.DataFrame, sample_size: int = 5) -> list[str]:
    """Measure durations for a handful of clips if librosa is available."""
    if librosa is None or df.empty:
        return []

    sample = df.sample(n=min(sample_size, len(df)), random_state=0)
    durations: list[str] = []
    for _, row in sample.iterrows():
        path: Path = row["audio_path"]
        try:
            audio, sr = librosa.load(path, sr=None, mono=True)
        except Exception as exc:  # pragma: no cover - diagnostics only
            durations.append(f"  {path.name}: failed to load ({exc})")
            continue

        seconds = len(audio) / sr if sr else 0.0
        durations.append(f"  {path.name}: {seconds:0.2f}s at {sr}Hz")
    return durations


def explore_split(split: str, df: pd.DataFrame) -> str:
    """Generate a human-readable summary for a dataset split."""
    missing_audio = _find_missing_audio(df)
    youtube_titles = df["youtube_title"].nunique(dropna=True)
    youtube_urls = df["youtube_url"].nunique(dropna=True)

    lines = [
        f"{split.title()} split",
        f"- Total clips: {len(df)}",
        f"- Unique labels: {df['label'].nunique()}",
        f"- Unique YouTube titles: {youtube_titles}",
        f"- Unique YouTube URLs: {youtube_urls}",
        f"- Label distribution:\n{_summarise_labels(df)}",
        f"- Missing audio files: {len(missing_audio)}",
    ]
    if missing_audio:
        preview = random.sample(missing_audio, k=min(len(missing_audio), 3))
        lines.append("  Examples: " + ", ".join(path.name for path in preview))

    duration_snippets = _sample_durations(df)
    if duration_snippets:
        lines.append("- Sample durations:")
        lines.extend(duration_snippets)
    elif librosa is None:
        lines.append("- Sample durations: skipped (librosa not installed)")

    return "\n".join(lines)


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Expected dataset directory at {DATASET_DIR}")

    train_df = _read_split("training")
    test_df = _read_split("test")

    print(f"Dataset directory: {DATASET_DIR}")
    print()
    print(explore_split("training", train_df))
    print()
    print(explore_split("test", test_df))


if __name__ == "__main__":
    main()
