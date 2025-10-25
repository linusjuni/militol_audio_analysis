import os
import json
from pathlib import Path

import torch
import librosa
import soundfile as sf

AUDIO_PATH = "data/raw/test_speech.mp3"
SAMPLE_RATE = 16000
THRESHOLD = 0.5
MIN_SPEECH_MS = 250
MIN_SILENCE_MS = 200
PAD_MS = 100

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    return Path.cwd().resolve()


def pick_latest_audio(raw_dir: Path) -> Path | None:
    """Picks latest audio. Only used if we dont specify an audio."""
    exts = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")
    files = [f for ext in exts for f in raw_dir.glob(ext)]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def run_vad_on_file(audio_path: Path, out_dir: Path):
    """Load audio, run VAD, save segments and speech-only wav."""
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    (get_speech_timestamps, _save_audio, _read_audio, VADIterator, collect_chunks) = utils

    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    wav = torch.from_numpy(y)

    speech_ts = get_speech_timestamps(
        wav, model, sampling_rate=SAMPLE_RATE,
        threshold=THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=PAD_MS
    )

    segments_sec = [
        {"start_s": round(s["start"]/SAMPLE_RATE, 3),
         "end_s":   round(s["end"]/SAMPLE_RATE, 3)}
        for s in speech_ts
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vad_segments.json").write_text(
        json.dumps({
            "audio": str(audio_path.resolve()),
            "sample_rate": SAMPLE_RATE,
            "segments": segments_sec
        }, indent=2)
    )

    if speech_ts:
        speech_wav = collect_chunks(speech_ts, wav, sampling_rate=SAMPLE_RATE)  # torch tensor
        sf.write(str(out_dir / "speech_only.wav"), speech_wav.numpy(), SAMPLE_RATE)

    print(f"\nInput: {audio_path}")
    print("Speech segments (s):")
    for s in segments_sec:
        print(f"{s['start_s']:.2f} → {s['end_s']:.2f}")
    print(f"\nSaved:\n- {out_dir/'vad_segments.json'}")
    if speech_ts:
        print(f"- {out_dir/'speech_only.wav'}")


def main():
    project_root = find_project_root(Path(__file__).parent)
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    out_root = data_dir / "processed" / "vad"

    audio = Path(AUDIO_PATH) if AUDIO_PATH else pick_latest_audio(raw_dir)
    if not audio or not audio.exists():
        print(f"No audio found. Put a file in {raw_dir} or set AUDIO_PATH.")
        return

    clip_stem = audio.stem
    out_dir = out_root / clip_stem

    run_vad_on_file(audio, out_dir)


if __name__ == "__main__":
    main()
