import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import kagglehub
import soundfile as sf
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MADDataset(Dataset):
    """
    Military Audio Dataset (MAD) for background sound classification.

    Loads all data, performs stratified split, returns raw audio waveforms.

    Classes: 6 total (labels 1-6). Label 0 (conversation) is excluded.
    """

    def __init__(
        self,
        split: str = "train",  # "train", "val", or "test"
        data_root: str = None,  # If None, downloads via kagglehub
        sample_rate: int = 16000,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ):
        """
        Args:
            split: "train", "val", or "test"
            data_root: Path to MAD_dataset folder. If None, downloads via kagglehub
            sample_rate: Target sample rate for audio
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed for reproducibility
        """
        self.split = split
        self.sample_rate = sample_rate

        # Setup data path
        if data_root is None:
            logger.info("Downloading MAD dataset via kagglehub...")
            cache_path = kagglehub.dataset_download(
                "junewookim/mad-dataset-military-audio-dataset"
            )
            self.data_root = Path(cache_path) / "MAD_dataset"
        else:
            self.data_root = Path(data_root)

        # Load both CSVs and combine
        train_csv = pd.read_csv(self.data_root / "training.csv")
        test_csv = pd.read_csv(self.data_root / "test.csv")

        # Combine all data
        all_data = pd.concat([train_csv, test_csv], ignore_index=True)
        logger.info(f"Loaded {len(all_data)} total samples")

        # Remove label 0 samples
        all_data = all_data[all_data["label"] != 0].reset_index(drop=True)
        logger.info(f"After removing label 0: {len(all_data)} samples")

        # Stratified split: train/temp, then temp -> val/test
        train_df, temp_df = train_test_split(
            all_data,
            test_size=(val_size + test_size),
            stratify=all_data["label"],
            random_state=random_state,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size / (val_size + test_size),
            stratify=temp_df["label"],
            random_state=random_state,
        )

        # Select the appropriate split
        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

        self.df = self.df.reset_index(drop=True)
        logger.info(f"{split} split: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            waveform: torch.Tensor of shape (1, num_samples) - raw audio
            label: int
        """
        row = self.df.iloc[idx]

        # Load audio using soundfile
        audio_path = self.data_root / row["path"]
        data, sr = sf.read(str(audio_path))

        # Convert to torch tensor and ensure correct shape
        if data.ndim == 1:  # Mono
            waveform = torch.FloatTensor(data).unsqueeze(
                0
            )  # (samples,) -> (1, samples)
        else:  # Stereo - convert to mono
            waveform = (
                torch.FloatTensor(data).mean(dim=1, keepdim=True).t()
            )  # Average channels

        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Get label
        label = int(row["label"])

        return waveform, label


def get_dataloaders(batch_size=32, num_workers=4, **dataset_kwargs):
    """
    Convenience function to get train/val/test dataloaders.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        **dataset_kwargs: Arguments passed to MADDataset

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    train_dataset = MADDataset(split="train", **dataset_kwargs)
    val_dataset = MADDataset(split="val", **dataset_kwargs)
    test_dataset = MADDataset(split="test", **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    logger.info("Testing MAD Dataset...")

    # Load datasets
    train_dataset = MADDataset(split="train")
    val_dataset = MADDataset(split="val")
    test_dataset = MADDataset(split="test")

    # Get a sample
    waveform, label = train_dataset[0]
    logger.info(f"Sample waveform shape: {waveform.shape}")
    logger.info(f"Label: {label}")

    # Show split sizes
    logger.info("Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Val: {len(val_dataset)}")
    logger.info(f"  Test: {len(test_dataset)}")
