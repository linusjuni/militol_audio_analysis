import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add <repo> and <repo>/src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]  # -> <repo>
SRC_DIR = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm.report import generate_report
from preprocessing.whisper_to_json import export_asr_json
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ProcessResult:
    intercept_id: str
    asr_path: Path
    background_events_path: Path
    report_path: Path
    report_markdown: str
    duration_s: Optional[float]


def ensure_bg_events(audio_path: Path, clip: str, ckpt_path: Path, force: bool = False) -> Path:
    """
    Ensure background events exist by running AST inference if needed.
    - If audio_path is not WAV, convert elsewhere or switch to a .wav path here.
    - Returns the path to events.json in MVP shape.
    """
    out_dir = ROOT_DIR / "data" / "processed" / "bg" / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "events.json"
    if out_json.exists() and not force:
        return out_json

    wav_in = audio_path
    if wav_in.suffix.lower() != ".wav":
        raise RuntimeError(
            f"AST expects WAV. Got {audio_path}. Convert upstream or enable the conversion."
        )

    from sed.infer import infer_ast  # your script with the importable function

    log.info("Running AST on %s → %s", wav_in, out_json)
    infer_ast(
        wav=wav_in,
        ckpt=ckpt_path,
        out_json=out_json,
        window_sec=5.0,
        overlap=0.5,
        threshold=0.1,
        device=None,
        print_summary=True,
    )
    return out_json


def detect_duration_seconds(audio_path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate and info.frames:
            return info.frames / float(info.samplerate)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Failed to compute duration for %s: %s", audio_path, exc)
    return None


def process_clip(
    audio_path: Path,
    clip: str,
    *,
    model_name: str = "base",
    max_gap_s: float = 0.75,
    ckpt_path: Optional[Path] = None,
    force: bool = False,
) -> ProcessResult:
    """
    Full pipeline for a single intercept.
    1. Run ASR/Diarization via Whisper (export JSON)
    2. Run AST background event detection
    3. Generate the Markdown intelligence report
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    clip_slug = clip.strip().replace(" ", "_")
    ckpt = ckpt_path or (ROOT_DIR / "runs" / "ast_mad" / "best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"AST checkpoint missing: {ckpt}")

    log.info("Processing intercept %s", clip_slug)
    asr_path = export_asr_json(
        audio_path,
        model_name=model_name,
        max_gap_s=max_gap_s,
        out_root=ROOT_DIR / "data" / "processed" / "asr",
    )
    bg_path = ensure_bg_events(audio_path, clip_slug, ckpt, force=force)
    report_path, report_markdown = generate_report(
        asr_path, bg_path, ROOT_DIR / "data" / "processed" / "reports"
    )
    duration_s = detect_duration_seconds(audio_path)

    log.success("Completed intercept %s", clip_slug)
    return ProcessResult(
        intercept_id=clip_slug,
        asr_path=asr_path,
        background_events_path=bg_path,
        report_path=report_path,
        report_markdown=report_markdown,
        duration_s=duration_s,
    )


def make_report():
    clip = "military_test"
    ckpt = ROOT_DIR / "runs" / "ast_mad" / "best.pt"
    asr_path = export_asr_json(
        ROOT_DIR / f"data/raw/{clip}.wav", model_name="base", max_gap_s=0.75
    )
    bg_path = ensure_bg_events(ROOT_DIR / f"data/raw/{clip}.wav", clip, ckpt)
    path, report = generate_report(
        asr_path, bg_path, ROOT_DIR / "data/processed/reports"
    )
    print(report)
    print("Report ->", path)
    return report


if __name__ == "__main__":
    make_report()
