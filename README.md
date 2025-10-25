# Militol Audio Analysis

Multi-modal deep learning pipeline for automated intelligence extraction from intercepted audio communications. Combines speech transcription, emotion detection, background sound classification, and speaker hierarchy analysis to generate actionable intelligence summaries.

Developed for European Defense Tech Hackathon - extracting comprehensive insights from audio beyond just words.

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
