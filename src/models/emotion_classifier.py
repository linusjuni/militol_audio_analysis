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

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        # Handle different input shapes
        if waveforms.dim() == 3:  # (batch, 1, samples)
            waveforms = waveforms.squeeze(1)  # -> (batch, samples)

        # Get Wav2Vec2 features
        outputs = self.wav2vec2(waveforms)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Average pool over time dimension
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_size)

        # Classification
        logits = self.classifier(pooled)  # (batch, num_emotions)

        return logits
