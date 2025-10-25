"""Evaluate a fine-tuned AST checkpoint on the MAD test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sed.infer import load_model
from sed.train import build_dataloader
from sed.utils import get_logger, save_json


def tensor_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a batch of tensors to the requested device."""
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    logger = get_logger()
    device = torch.device(args.device)

    bundle = load_model(args.ckpt, device)
    model: torch.nn.Module = bundle["model"]  # type: ignore[assignment]
    feature_extractor = bundle["feature_extractor"]  # type: ignore[assignment]
    class_names: List[str] = bundle["class_names"]  # type: ignore[assignment]
    ckpt_sample_rate: int = bundle["sample_rate"]  # type: ignore[assignment]

    sample_rate = args.sample_rate if args.sample_rate is not None else ckpt_sample_rate
    if sample_rate != ckpt_sample_rate:
        logger.warning("Overriding checkpoint sample rate %d with %d", ckpt_sample_rate, sample_rate)

    dataloader = build_dataloader(
        "test",
        feature_extractor=feature_extractor,
        batch_size=args.batch_size,
        sample_rate=sample_rate,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    logger.info("Evaluating on %d test clips (batch size %d)", len(dataloader.dataset), args.batch_size)

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for features, labels in dataloader:
            inputs = tensor_to_device(features, device)
            labels = labels.to(device)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = _compute_metrics(all_labels, all_preds, class_names)
    logger.info("Overall accuracy: %.4f", metrics["accuracy"])
    for entry in metrics["per_class"]:  # type: ignore[index]
        logger.info(
            "%s | precision %.3f | recall %.3f | f1 %.3f | support %d",
            entry["class"],
            entry["precision"],
            entry["recall"],
            entry["f1"],
            entry["support"],
        )

    if args.out_json is not None:
        save_path = Path(args.out_json)
        save_json(save_path, metrics)
        logger.info("Saved evaluation metrics to %s", save_path)

    return metrics


def _compute_metrics(labels: List[int], preds: List[int], class_names: List[str]) -> Dict[str, object]:
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    per_class = []
    for name in class_names:
        stats = report[name]
        per_class.append(
            {
                "class": name,
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1": float(stats.get("f1-score", 0.0)),
                "support": int(stats.get("support", 0)),
            }
        )

    macro = report["macro avg"]
    weighted = report["weighted avg"]

    macro_avg = {
        "precision": float(macro.get("precision", 0.0)),
        "recall": float(macro.get("recall", 0.0)),
        "f1": float(macro.get("f1-score", 0.0)),
        "support": int(macro.get("support", 0)),
    }
    weighted_avg = {
        "precision": float(weighted.get("precision", 0.0)),
        "recall": float(weighted.get("recall", 0.0)),
        "f1": float(weighted.get("f1-score", 0.0)),
        "support": int(weighted.get("support", 0)),
    }

    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "confusion_matrix": np.asarray(conf_mat, dtype=int).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an AST checkpoint on the MAD test split.")
    parser.add_argument("--ckpt", type=Path, default='runs/ast_mad/best.pt', help="Checkpoint produced by sed.train.")
    parser.add_argument("--data_root", type=str, default=None, help="Optional path to MAD_dataset directory.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=None, help="Override sample rate used for resampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", type=Path, default=None, help="If provided, saves metrics to this JSON path.")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
