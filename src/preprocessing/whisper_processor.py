import whisper
import torch
from pyannote.audio import Pipeline
from dataclasses import dataclass
from pathlib import Path
from settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float


@dataclass
class WhisperOutput:
    transcript: str
    speaker_segments: list[SpeakerSegment]
    word_timestamps: list[dict[str, any]]


class WhisperProcessor:
    """
    Speech-to-text with speaker diarization.
    
    - Whisper: transcription + word-level timestamps
    - Pyannote: speaker diarization
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = None,
    ):
        """
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on. If None, auto-detects
        """
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing Whisper + Diarization on {self.device}")

        logger.info(f"Loading Whisper model '{model_name}'...")
        self.whisper_model = whisper.load_model(model_name, device=self.device)

        logger.info("Loading speaker diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=settings.HF_TOKEN
        )
        self.diarization_pipeline.to(torch.device(self.device))

        logger.success("Models loaded successfully")

    def process(self, audio_path: str) -> WhisperOutput:
        """
        Process audio file through Whisper + Diarization pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            WhisperOutput with transcript, speaker segments, and word timestamps
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Processing audio: {audio_path.name}")

        # Step 1: Transcribe with Whisper
        logger.info("Transcribing audio with Whisper...")
        result = self.whisper_model.transcribe(
            str(audio_path), word_timestamps=True, verbose=False
        )

        transcript = result["text"]

        word_timestamps = []
        for segment in result["segments"]:
            if "words" in segment:
                for word_info in segment["words"]:
                    word_timestamps.append(
                        {
                            "word": word_info["word"],
                            "start": word_info["start"],
                            "end": word_info["end"],
                            "speaker": "UNKNOWN",
                        }
                    )

        logger.info(f"Transcription complete: {len(word_timestamps)} words")

        # Step 2: Load audio for diarization
        import soundfile as sf

        waveform_np, sample_rate = sf.read(str(audio_path))

        if waveform_np.ndim == 1:
            waveform = torch.FloatTensor(waveform_np).unsqueeze(0)
        else:
            waveform = torch.FloatTensor(waveform_np.T)

        audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

        # Step 3: Speaker Diarization
        logger.info("Performing speaker diarization...")
        diarization = self.diarization_pipeline(audio_dict)

        speaker_segments = []
        annotation = diarization.speaker_diarization

        for segment, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append(
                SpeakerSegment(speaker=speaker, start=segment.start, end=segment.end)
            )

        logger.info(f"Diarization complete: {len(speaker_segments)} speaker segments")

        # Step 4: Assign speakers to words
        logger.info("Assigning speakers to words...")
        for word_info in word_timestamps:
            word_start = word_info["start"]
            word_end = word_info["end"]
            word_mid = (word_start + word_end) / 2

            for seg in speaker_segments:
                if seg.start <= word_mid <= seg.end:
                    word_info["speaker"] = seg.speaker
                    break

        logger.success(
            f"Processing complete: {len(speaker_segments)} speakers, "
            f"{len(word_timestamps)} words"
        )

        return WhisperOutput(
            transcript=transcript,
            speaker_segments=speaker_segments,
            word_timestamps=word_timestamps,
        )
