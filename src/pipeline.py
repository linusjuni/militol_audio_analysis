from pathlib import Path
import sys

# Add <repo> and <repo>/src to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]   # -> <repo>
SRC_DIR  = ROOT_DIR / "src"
for p in (str(ROOT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm.report import generate_report                  # now works: <repo>/llm/report.py
from preprocessing.whisper_to_json import export_asr_json
from utils.logger import get_logger

log = get_logger(__name__)

def main():
    clip = "test_speech"
    asr_path = export_asr_json(ROOT_DIR / f"data/raw/{clip}.mp3",
                               model_name="base", max_gap_s=0.75)
    bg_path  = ROOT_DIR / f"data/processed/bg/{clip}/events.json"
    path, report   = generate_report(asr_path, bg_path, ROOT_DIR / "data/processed/reports")
    print(report)
    print("Report ->", path)

if __name__ == "__main__":
    main()