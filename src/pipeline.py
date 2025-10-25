from pathlib import Path
import sys

# Add <repo> and <repo>/src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]   # -> <repo>
SRC_DIR  = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm.report import generate_report
from preprocessing.whisper_to_json import export_asr_json
from utils.logger import get_logger

log = get_logger(__name__)

def ensure_bg_events(audio_path: Path, clip: str, ckpt_path: Path) -> Path:
    """
    Ensure background events exist by running AST inference if needed.
    - If audio_path is not WAV, convert elsewhere or switch to a .wav path here.
    - Returns the path to events.json in MVP shape.
    """
    out_dir  = ROOT_DIR / "data" / "processed" / "bg" / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "events.json"
    if out_json.exists():
        return out_json

    # AST CLI expects a WAV; use your existing WAV (no need to force 16k here).
    wav_in = audio_path
    if wav_in.suffix.lower() != ".wav":
        # If you already convert earlier, point to that WAV here.
        # Otherwise un-comment the 2 lines below to convert in place:
        # from utils.audio import save_as_16k
        # wav_in = out_dir / f"{clip}_tmp.wav"; save_as_16k(audio_path, wav_in)
        raise RuntimeError(f"AST expects WAV. Got {audio_path}. Convert upstream or enable the conversion here.")

    from sed.infer import infer_ast  # your script with the importable function
    log.info(f"🎯 Running AST on {wav_in} → {out_json}")
    infer_ast(
        wav=wav_in,
        ckpt=ckpt_path,
        out_json=out_json,
        window_sec=5.0,
        overlap=0.5,
        threshold=0.1,          # use checkpoint's default unless you want to override
        device=None,             # auto-pick cuda/cpu
        print_summary=True,
    )
    return out_json

def main():
    clip = "military_test"
    ckpt = ROOT_DIR / "runs" / "ast_mad" / "best.pt"
    asr_path = export_asr_json(ROOT_DIR / f"data/raw/{clip}.wav",
                               model_name="base", max_gap_s=0.75)
    bg_path = ensure_bg_events(ROOT_DIR / f"data/raw/{clip}.wav", clip, ckpt)
    path, report   = generate_report(asr_path, bg_path, ROOT_DIR / "data/processed/reports")
    print(report)
    print("Report ->", path)

if __name__ == "__main__":
    main()