"""Plot sample waveforms from the MAD dataset.

Run with (adjust arguments as desired):

    uv run python -m src.exploration.plot_waveforms --split training --count 6 \
        --output outputs/exploration/waveforms.png

If ``--output`` is omitted the script attempts to display the figure window.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - provide a helpful error
    raise RuntimeError(
        "numpy is required for plotting waveforms.\n"
        "Install it with `uv pip install numpy` or add it to your dependencies."
    ) from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "pandas is required for plotting waveforms.\n"
        "Install it with `uv pip install pandas` or add it to your dependencies."
    ) from exc

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "librosa is required for plotting waveforms.\n"
        "Install it with `uv pip install librosa` or add it to your dependencies."
    ) from exc

import matplotlib

# Resolve repository root from this file's location: src/exploration/plot_waveforms.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "MAD_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise audio waveforms from the MAD dataset.")
    parser.add_argument(
        "--split",
        default="training",
        choices=("training", "test"),
        help="Dataset split to sample from (default: training).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="Number of clips to plot (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling clips.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Resample audio to this rate before plotting (default: original rate).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Limit loaded audio to N seconds from the start (default: full clip).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="File path to save the figure. If omitted, a window is shown instead.",
    )
    return parser.parse_args()


def _read_split(split: str) -> pd.DataFrame:
    """Load metadata for a single dataset split."""
    csv_path = DATASET_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find metadata file at {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df = df.rename(columns={"youtube title": "youtube_title", "youtube url": "youtube_url"})
    df["audio_path"] = df["path"].apply(lambda rel: (DATASET_DIR / rel).resolve())
    return df


def _iter_existing_audio(df: pd.DataFrame) -> Iterable[Path]:
    return (path for path in df["audio_path"].tolist() if Path(path).exists())


def _filter_existing(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["audio_path"].apply(lambda p: Path(p).exists())
    missing = df.loc[~mask, "audio_path"]
    if not missing.empty:
        print(f"Warning: {len(missing)} audio files referenced in metadata are missing on disk.")
    return df.loc[mask].copy()


def _setup_matplotlib(output: Path | None):
    """Select an appropriate matplotlib backend and return pyplot."""
    if output:
        matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting waveforms.\n"
            "Install it with `uv pip install matplotlib` or add it to your dependencies."
        ) from exc
    return plt


def _load_audio(path: Path, sample_rate: int | None, duration: float | None) -> tuple[np.ndarray, int]:
    """Load a mono audio clip with librosa, optionally trimming to duration seconds."""
    audio, sr = librosa.load(path, sr=sample_rate, mono=True, duration=duration)
    return audio, sr


def _choose_samples(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    count = min(count, len(df))
    if count == len(df):
        return df
    return df.sample(n=count, random_state=seed)


def _plot_waveforms(
    df: pd.DataFrame,
    plt,
    *,
    sample_rate: int | None,
    duration: float | None,
) -> "matplotlib.figure.Figure":
    """Create a grid of waveform plots and return the matplotlib figure."""
    clip_count = len(df)
    cols = min(3, clip_count)
    rows = math.ceil(clip_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (_, row) in zip(axes_flat, df.iterrows(), strict=False):
        audio_path = Path(row["audio_path"])
        try:
            audio, sr = _load_audio(audio_path, sample_rate, duration)
        except Exception as exc:  # pragma: no cover - this is exploratory tooling
            ax.text(0.5, 0.5, f"Failed to load\n{audio_path.name}\n{exc}", ha="center", va="center")
            ax.set_axis_off()
            continue

        if audio.size == 0:
            ax.text(0.5, 0.5, f"No samples in {audio_path.name}", ha="center", va="center")
            ax.set_axis_off()
            continue

        timeline = np.linspace(0, audio.size / sr, num=audio.size, endpoint=False)
        ax.plot(timeline, audio, linewidth=0.8)
        ax.set_title(f"{audio_path.name} | label {row['label']}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)

    # Hide any surplus axes when the grid is larger than the number of clips.
    for ax in axes_flat[len(df) :]:
        ax.set_visible(False)

    fig.suptitle("MAD dataset waveform samples", fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


def main() -> None:
    args = parse_args()
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Expected dataset directory at {DATASET_DIR}")

    plt = _setup_matplotlib(args.output)
    df = _read_split(args.split)
    df = _filter_existing(df)

    if df.empty:
        raise RuntimeError("No audio files found to plot. Check that the dataset is downloaded.")

    sample_df = _choose_samples(df, args.count, args.seed)
    figure = _plot_waveforms(sample_df, plt, sample_rate=args.sample_rate, duration=args.duration)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        print(f"Saved waveform figure to {args.output}")
    else:  # pragma: no cover - depends on user environment
        plt.show()

    plt.close(figure)


if __name__ == "__main__":
    main()
