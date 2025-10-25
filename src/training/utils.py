"""
Training utilities for emotion classification.

Purely functional utilities for training, evaluation, and checkpointing.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple


def collate_fn_pad(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that pads audio waveforms to the same length.

    Handles variable-length audio by padding to the longest sequence in the batch.

    Args:
        batch: List of (waveform, label) tuples
               waveform shape: (1, num_samples) or (num_samples,)

    Returns:
        Tuple of (padded_waveforms, labels)
        padded_waveforms: (batch_size, max_length)
        labels: (batch_size,)
    """
    waveforms, labels = zip(*batch)

    # Get max length in batch
    max_length = max(w.shape[-1] for w in waveforms)

    # Pad all waveforms to max_length
    padded_waveforms = []
    for w in waveforms:
        # Remove channel dimension if present
        if w.dim() == 2:  # (1, samples)
            w = w.squeeze(0)  # -> (samples,)

        # Pad to max_length
        if w.shape[0] < max_length:
            padding = max_length - w.shape[0]
            w = F.pad(w, (0, padding), mode="constant", value=0.0)

        padded_waveforms.append(w)

    # Stack into batch
    waveforms_batch = torch.stack(padded_waveforms, dim=0)  # (batch, max_length)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return waveforms_batch, labels_batch


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def compute_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """
    Compute accuracy on a dataset.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to run on

    Returns:
        Accuracy as a float (0-1)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            logits = model(waveforms)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    save_path: str,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        save_path: Path to save checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }

    torch.save(checkpoint, save_path)
