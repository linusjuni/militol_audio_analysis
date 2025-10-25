import re
from pathlib import Path
import sys
import json

SYS_SRC = Path(__file__).resolve().parents[1]   # -> <repo>/src
if str(SYS_SRC) not in sys.path:
    sys.path.insert(0, str(SYS_SRC))

# Project root for data paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # -> <repo>

from preprocessing.whisper_processor import WhisperProcessor  # your class
from utils.logger import get_logger

logger = get_logger(__name__)

# Set the file you want to use
AUDIO_PATH = Path("data/raw/test_speech.mp3")
MODEL_NAME = "base"

# Might wnat to experiment with this. The gap before a new segment is created
MAX_GAP_S = 0.75

def words_to_segments(words: list[dict], max_gap_s: float = MAX_GAP_S) -> list[dict]:
    """Collapse Whisper word timestamps into utterances per speaker."""
    ws = []
    for w in words:
        if w.get("start") is None or w.get("end") is None:
            continue
        ws.append({
            "word": str(w.get("word", "")),
            "start": float(w["start"]),
            "end": float(w["end"]),
            "speaker": str(w.get("speaker", "UNKNOWN") or "UNKNOWN"),
        })
    if not ws:
        return []

    segs: list[dict] = []
    cur = {
        "speaker": ws[0]["speaker"],
        "start_s": ws[0]["start"],
        "end_s":   ws[0]["end"],
        "text":    ws[0]["word"],
    }

    for w in ws[1:]:
        same_spk = (w["speaker"] == cur["speaker"])
        gap = w["start"] - cur["end_s"]
        if same_spk and gap <= max_gap_s:
            cur["end_s"] = max(cur["end_s"], w["end"])
            cur["text"] += w["word"]
        else:
            cur["text"] = re.sub(r"\s+", " ", cur["text"]).strip()
            segs.append(cur)
            cur = {
                "speaker": w["speaker"],
                "start_s": w["start"],
                "end_s":   w["end"],
                "text":    w["word"],
            }

    cur["text"] = re.sub(r"\s+", " ", cur["text"]).strip()
    segs.append(cur)
    return segs

def main():
    print("Test")
    audio = AUDIO_PATH if AUDIO_PATH.is_absolute() else Path.cwd() / AUDIO_PATH
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    clip = audio.stem
    out_dir = Path("data/processed/asr") / clip
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing + diarizing: {audio}")
    proc = WhisperProcessor(model_name=MODEL_NAME)
    out = proc.process(str(audio))

    segments = words_to_segments(out.word_timestamps, max_gap_s=MAX_GAP_S)

    asr_json = {
        "transcript": (out.transcript or "").strip(),
        "segments": segments,
    }

    json_path = out_dir / "asr_segments.json"
    json_path.write_text(json.dumps(asr_json, ensure_ascii=False, indent=2))
    logger.success(f"Wrote ASR JSON → {json_path}")

if __name__ == "__main__":
    main()
