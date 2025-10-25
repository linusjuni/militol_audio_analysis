import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class EmotionClassifier(nn.Module):
    """
    Emotion classifier using facebook/wav2vec2-base-960h.

    8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
    """

    def __init__(
        self,
        num_emotions: int = 8,
        model_name: str = "facebook/wav2vec2-base-960h",
        freeze_feature_extractor: bool = False,
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            model_name: Pretrained Wav2Vec2 model name
            freeze_feature_extractor: If True, freeze Wav2Vec2 feature extractor
        """
        super().__init__()

        self.num_emotions = num_emotions

        # Load pretrained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # Optionally freeze feature extractor
        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # Classification head
        hidden_size = self.wav2vec2.config.hidden_size  # 768 for base model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions),
        )

    def forward(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with optional attention masking.

        Args:
            waveforms: Audio waveforms (batch, samples)
            attention_mask: Mask for padding (batch, samples), 1=real, 0=padding

        Returns:
            logits: (batch, num_emotions)
        """
        # Handle different input shapes
        if waveforms.dim() == 3:  # (batch, 1, samples)
            waveforms = waveforms.squeeze(1)  # -> (batch, samples)

        # Get Wav2Vec2 features
        outputs = self.wav2vec2(waveforms, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Masked mean pooling over time dimension
        if attention_mask is not None:
            # Wav2Vec2 downsamples audio, so we need to downsample the attention mask too
            # Wav2Vec2 has a stride of 320 (for 16kHz audio, each frame = 20ms)
            # Calculate expected sequence length
            seq_len = hidden_states.shape[1]

            # Downsample attention mask to match hidden_states sequence length
            # Use adaptive pooling to match dimensions
            attention_mask_downsampled = torch.nn.functional.adaptive_avg_pool1d(
                attention_mask.unsqueeze(1),  # (batch, 1, samples)
                seq_len,  # Target length
            ).squeeze(
                1
            )  # (batch, seq_len)

            # Binarize (values > 0.5 are considered valid)
            attention_mask_downsampled = (attention_mask_downsampled > 0.5).float()

            # Expand mask to hidden dimension
            mask_expanded = attention_mask_downsampled.unsqueeze(
                -1
            )  # (batch, seq_len, 1)

            # Apply mask and compute mean
            masked_hidden = hidden_states * mask_expanded
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_size)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            pooled = sum_hidden / count  # (batch, hidden_size)
        else:
            # Fall back to simple mean pooling if no mask provided
            pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_emotions)

        return logits