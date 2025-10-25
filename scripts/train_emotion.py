"""
Training script for emotion classifier on RAVDESS dataset.

Train Wav2Vec2-based emotion classifier with 8 emotion classes.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import RAVDESSDataset
from models.emotion_classifier import EmotionClassifier
from training.utils import (
    train_epoch,
    compute_accuracy,
    save_checkpoint,
    collate_fn_pad,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4

    # Paths
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading RAVDESS dataset...")

    train_dataset = RAVDESSDataset(split="train")
    val_dataset = RAVDESSDataset(split="val")
    test_dataset = RAVDESSDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_pad,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_pad,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_pad,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Initialize model
    logger.info("Initializing Wav2Vec2 emotion classifier...")
    model = EmotionClassifier(
        num_emotions=8,
        freeze_feature_extractor=False,
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        logger.info("-" * 40)

        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # Evaluate
        train_acc = compute_accuracy(model, train_loader, device)
        val_acc = compute_accuracy(model, val_loader, device)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Acc: {train_acc:.4f}")
        logger.info(f"Val Acc: {val_acc:.4f}")

        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = MODEL_DIR / "emotion_classifier_best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                accuracy=val_acc,
                save_path=str(checkpoint_path),
            )
            logger.success(f"Saved best model (val_acc: {val_acc:.4f})")

        # Save latest checkpoint
        checkpoint_path = MODEL_DIR / "emotion_classifier_latest.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=train_loss,
            accuracy=val_acc,
            save_path=str(checkpoint_path),
        )

    # Final evaluation on test set
    logger.info("\n" + "=" * 40)
    logger.info("Final evaluation on test set...")

    # Load best model
    best_checkpoint_path = MODEL_DIR / "emotion_classifier_best.pt"
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc = compute_accuracy(model, test_loader, device)
    logger.success(f"Test Accuracy: {test_acc:.4f}")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
