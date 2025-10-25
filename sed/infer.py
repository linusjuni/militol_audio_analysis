"""Run the fine-tuned AST model on long audio and emit JSON events."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

from src.audio_utils import ensure_sample_rate
from sed.utils import load_checkpoint, save_json


def load_model(ckpt_path: Path, device: torch.device) -> Dict[str, object]:
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    extra = checkpoint.extra
    pretrained = extra["pretrained"]
    class_names = extra["class_names"]
    config_dict = extra["config"]
    threshold = extra.get("threshold", 0.1)
    sample_rate = extra.get("sample_rate", 16000)

    config = AutoConfig.from_pretrained(pretrained)
    for key, value in config_dict.items():
        setattr(config, key, value)

    model = AutoModelForAudioClassification.from_pretrained(
        pretrained,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(checkpoint.model)
    model.to(device)
    model.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained)

    return {
        "model": model,
        "feature_extractor": feature_extractor,
        "class_names": class_names,
        "threshold": threshold,
        "sample_rate": sample_rate,
    }


def sliding_windows(num_samples: int, sample_rate: int, window_sec: float, hop_sec: float) -> List[tuple[int, int]]:
    window = int(window_sec * sample_rate)
    hop = int(hop_sec * sample_rate)
    indices: List[tuple[int, int]] = []
    start = 0
    while start < num_samples:
        end = min(start + window, num_samples)
        indices.append((start, end))
        if end == num_samples:
            break
        start += hop
    return indices


def decode_events(
    probs: List[np.ndarray],
    window_times: List[tuple[float, float]],
    class_names: List[str],
    threshold: float,
) -> List[Dict[str, float | str]]:
    events: List[Dict[str, float | str]] = []
    num_windows = len(probs)
    for class_idx, class_name in enumerate(class_names):
        current = None
        for idx in range(num_windows):
            start_time, end_time = window_times[idx]
            confidence = float(probs[idx][class_idx])
            if confidence >= threshold:
                if current is None:
                    current = {"class": class_name, "onset": start_time, "offset": end_time, "confidence": confidence}
                else:
                    current["offset"] = end_time
                    current["confidence"] = max(current["confidence"], confidence)
            else:
                if current is not None:
                    current["onset"] = round(current["onset"], 3)
                    current["offset"] = round(current["offset"], 3)
                    current["confidence"] = float(round(current["confidence"], 3))
                    events.append(current)
                    current = None
        if current is not None:
            current["onset"] = round(current["onset"], 3)
            current["offset"] = round(current["offset"], 3)
            current["confidence"] = float(round(current["confidence"], 3))
            events.append(current)
    events.sort(key=lambda item: item["onset"])
    return events


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bundle = load_model(args.ckpt, device)
    model: AutoModelForAudioClassification = bundle["model"]  # type: ignore
    feature_extractor: AutoFeatureExtractor = bundle["feature_extractor"]  # type: ignore
    class_names: List[str] = bundle["class_names"]  # type: ignore
    threshold: float = args.threshold if args.threshold is not None else bundle["threshold"]  # type: ignore
    sample_rate: int = bundle["sample_rate"]  # type: ignore

    waveform_np, sr = sf.read(args.wav)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=1)

    waveform_tensor = torch.from_numpy(np.asarray(waveform_np, dtype=np.float32)).unsqueeze(0)
    waveform_tensor = ensure_sample_rate(waveform_tensor, sr, sample_rate)
    waveform = waveform_tensor.squeeze(0).cpu().numpy()

    samples = waveform.shape[0]
    hop_sec = args.window_sec * (1.0 - args.overlap)
    hop_sec = args.window_sec if hop_sec <= 0 else hop_sec

    windows = sliding_windows(samples, sample_rate, args.window_sec, hop_sec)
    probabilities: List[np.ndarray] = []
    window_times: List[tuple[float, float]] = []

    with torch.no_grad():
        for start, end in windows:
            segment = waveform[start:end]
            start_time = start / sample_rate
            end_time = end / sample_rate
            inputs = feature_extractor(
                segment,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            probabilities.append(probs)
            window_times.append((start_time, end_time))

    events = decode_events(probabilities, window_times, class_names, threshold)

    

    mapped = [
        {
            "label":      str(ev["class"]).lower(),
            "start_s":    float(ev["onset"]),
            "end_s":      float(ev["offset"]),
            "p":          float(ev["confidence"]),
        }
        for ev in events
    ]

    payload = {
        "events": mapped,
        "meta": {
            "ckpt": str(args.ckpt),
            "window_sec": args.window_sec,
            "overlap": args.overlap,
            "threshold": threshold,
            "sample_rate": sample_rate,
            "device": args.device,
        }
    }

    save_json(args.out_json, payload)

    if args.print_summary:
        if events:
            summary = ", ".join(f"{ev['class']}@{ev['onset']:.2f}-{ev['offset']:.2f}" for ev in events)
            print(summary)
        else:
            print("No events detected.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AST-based SED inference and emit JSON events.")
    parser.add_argument("--wav", type=Path, required=True, help="Path to WAV file.")
    parser.add_argument("--ckpt", type=Path, default='runs/ast_mad/best.pt', help="Checkpoint produced by sed.train.")
    parser.add_argument("--out_json", type=Path, required=True, help="Destination for event JSON list.")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Sliding window size in seconds.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap ratio (0–1).")
    parser.add_argument("--threshold", type=float, default=0.1, help="Optional probability threshold override.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print_summary", action="store_true")
    return parser

def infer_ast(
    wav: Path,
    ckpt: Path,
    out_json: Path,
    window_sec: float = 5.0,
    overlap: float = 0.5,
    threshold: float | None = None,
    device: str | None = None,
    print_summary: bool = False,
) -> None:
    """Importable entrypoint so the pipeline can call AST directly (no subprocess)."""
    args = argparse.Namespace(
        wav=wav,
        ckpt=ckpt,
        out_json=out_json,
        window_sec=window_sec,
        overlap=overlap,
        threshold=threshold,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        print_summary=print_summary,
    )
    run(args)

def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
