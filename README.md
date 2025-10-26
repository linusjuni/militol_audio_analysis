# Militol Audio Analysis

Multi-modal deep learning pipeline for automated intelligence extraction from intercepted audio communications. Combines speech transcription, emotion detection, background sound classification, and speaker hierarchy analysis to generate actionable intelligence summaries.

Developed for European Defense Tech Hackathon - extracting comprehensive insights from audio beyond just words.

## Ops Console Interface

The repository now ships with a FastAPI-powered service and a React operations console that present intercepts in a military-grade UI. The API exposes endpoints for listing intercepts, fetching full analysis packages, downloading audio, and queuing new files for processing. The frontend visualizes the intercept log, synchronized transcript/background-event timelines, and the LLM-generated report.

### Running the Stack Locally

1. **Backend**
   ```bash
   uv sync
   uv run uvicorn app.main:app --reload --port 8000
   ```
   Ensure the AST checkpoint exists at `runs/ast_mad/best.pt` (or update the path in `src/pipeline.py`). The service stores intercept metadata under `data/processed/intercepts/index.json` and will auto-bootstrap from existing processed clips.

2. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   The Vite dev server proxies `/api/*` requests to the backend at `http://localhost:8000`.

3. **Usage**
   - Use the left-rail action **Queue New Intercept** to upload a `.wav`. The API writes to `data/raw/` and kicks off the full pipeline (Whisper ASR + AST background events + LLM report).
   - The intercept log polls every 10 seconds; once processing finishes the detail view refreshes automatically with the new report and synchronized timeline.


## Transformer Sound Event Detection MVP

The repository now includes a minimal sound-event detector (`sed/`) that
fine-tunes the pretrained AST backbone
`MIT/ast-finetuned-audioset-10-10-0.4593` on the six MAD classes
(*Gunshot, Footsteps, Shelling, Vehicle, Helicopter, Fighter*).  
Inference produces a JSON list of `{class, onset, offset, confidence}` events.

### Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

If you already have a `.python-version` pinned, `uv sync` will create the
virtualenv automatically; otherwise `uv venv` ensures a local `.venv` exists.
`uv run <command>` is equivalent to activating the environment manually.

### Training

```bash
python -m sed.train \
  --data_root /path/to/MAD_dataset \
  --save_dir runs/ast_mad \
  --epochs 5 \
  --batch_size 8
```

`data_root` should point to the folder containing `training.csv`, `test.csv`, and
the `audio/` directory. If omitted, the script falls back to downloading the
dataset via `kagglehub` just like `src/datasets.py`.

### Inference

```bash
python -m sed.infer \
  --wav /path/to/long_recording.wav \
  --ckpt runs/ast_mad/best.pt \
  --window_sec 5 \
  --overlap 0.5 \
  --out_json outputs/events.json \
  --print_summary
```

`outputs/events.json` will contain entries similar to:

```json
[
  {"class": "Gunshot", "onset": 12.5, "offset": 15.0, "confidence": 0.91},
  {"class": "Vehicle", "onset": 46.2, "offset": 55.0, "confidence": 0.87}
]
```

You can override the probability threshold per run with `--threshold`.
