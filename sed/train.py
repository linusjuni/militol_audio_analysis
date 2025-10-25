"""Fine-tune AST on MAD clip labels and save a checkpoint for inference."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    get_cosine_schedule_with_warmup,
)

from sed.utils import get_logger, save_checkpoint, seed_everything

# Reuse existing dataset logic
from src.datasets import MADDataset


CLASS_NAMES = ["Gunshot", "Footsteps", "Shelling", "Vehicle", "Helicopter", "Fighter"]
LABEL_MAP = {idx + 1: idx for idx in range(len(CLASS_NAMES))}


def collate_batch(
    batch: List[Tuple[torch.Tensor, int]],
    feature_extractor: AutoFeatureExtractor,
    sample_rate: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Convert a batch of raw waveforms into AST inputs + mapped labels."""
    waveforms = []
    labels = []
    for waveform, label in batch:
        waveforms.append(waveform.squeeze(0).numpy())
        labels.append(LABEL_MAP[int(label)])
    features = feature_extractor(
        waveforms,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return features, label_tensor


def build_dataloader(
    split: str,
    feature_extractor: AutoFeatureExtractor,
    batch_size: int,
    sample_rate: int,
    num_workers: int,
    data_root: str | None,
) -> DataLoader:
    dataset = MADDataset(split=split, data_root=data_root, sample_rate=sample_rate)
    collate_fn = partial(collate_batch, feature_extractor=feature_extractor, sample_rate=sample_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    logger = get_logger()
    device = torch.device(args.device)

    logger.info("Loading backbone %s", args.backbone)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.backbone)
    config = AutoConfig.from_pretrained(
        args.backbone,
        num_labels=len(CLASS_NAMES),
        label2id={name: idx for idx, name in enumerate(CLASS_NAMES)},
        id2label={idx: name for idx, name in enumerate(CLASS_NAMES)},
        finetuning_task="audio-classification",
    )
    model = AutoModelForAudioClassification.from_pretrained(
        args.backbone,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_loader = build_dataloader(
        "train",
        feature_extractor,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )
    val_loader = build_dataloader(
        "val",
        feature_extractor,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        data_root=args.data_root,
    )

    if args.train_head_only:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    logger.info(
        "Data sizes | train %d clips | val %d clips | batch size %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        args.batch_size,
    )
    logger.info(
        "Starting training for %d epochs on %s (%d train batches, %d val batches)",
        args.epochs,
        device,
        len(train_loader),
        len(val_loader),
    )
    logger.info("Total steps %d (warmup %d)", total_steps, warmup_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (features, labels) in enumerate(train_loader, start=1):
            inputs = {key: value.to(device) for key, value in features.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()
            if step % args.log_every == 0 or step == len(train_loader):
                avg_loss = running / step
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Epoch %d | step %d/%d | loss %.4f | lr %.2e",
                    epoch,
                    step,
                    len(train_loader),
                    avg_loss,
                    current_lr,
                )

        train_loss = running / max(1, len(train_loader))
        logger.info("Epoch %d | train loss %.4f", epoch, train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                inputs = {key: value.to(device) for key, value in features.items()}
                labels = labels.to(device)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
        val_loss /= max(1, len(val_loader))
        logger.info("Epoch %d | val loss %.4f", epoch, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = save_dir / "best.pt"
            save_checkpoint(
                ckpt_path,
                epoch,
                model,
                optimizer,
                {
                    "pretrained": args.backbone,
                    "config": config.to_dict(),
                    "class_names": CLASS_NAMES,
                    "threshold": args.threshold,
                    "sample_rate": args.sample_rate,
                },
            )
            logger.info("Saved checkpoint to %s", ckpt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune AST on the MAD dataset.")
    parser.add_argument("--backbone", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--data_root", type=str, default=None, help="Path to MAD_dataset (optional if already downloaded).")
    parser.add_argument("--save_dir", type=Path, default=Path("runs/ast_mad"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--threshold", type=float, default=0.6, help="Default probability threshold for event decoding.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25, help="Interval (in steps) for intermediate training logs.")
    parser.add_argument("--train_head_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
