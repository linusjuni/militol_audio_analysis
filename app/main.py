from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.registry import (
    InterceptMeta,
    derive_tags,
    delete_meta,
    get_meta,
    list_summaries,
    upsert_meta,
)
from app.schemas import InterceptDetail, InterceptSummary

app = FastAPI(title="Militol Audio Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_base_audio_url() -> Optional[str]:
    return None  # TODO: inject base URL via settings once hosting is defined


@app.get("/healthz")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/intercepts", response_model=list[InterceptSummary])
async def list_intercepts() -> list[InterceptSummary]:
    return list_summaries(base_url=get_base_audio_url())


def _build_audio_response(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path)


@app.get("/intercepts/{intercept_id}/audio")
async def download_audio(intercept_id: str):
    meta = get_meta(intercept_id)
    if not meta or not meta.audio_rel_path:
        raise HTTPException(status_code=404, detail="Audio not available for intercept")
    audio_path = ROOT_DIR / meta.audio_rel_path
    return _build_audio_response(audio_path)


def _load_intercept_detail(intercept_id: str) -> InterceptDetail:
    meta = get_meta(intercept_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Intercept not found")

    if meta.status != "ready":
        note = meta.executive_summary or (
            "Analysis in progress..." if meta.status == "processing" else "Analysis unavailable."
        )
        return InterceptDetail(
            meta=meta.to_summary(base_url=get_base_audio_url()),
            transcript=[],
            background_events=[],
            report_markdown=note,
        )

    asr_path = PROCESSED_DIR / "asr" / intercept_id / "asr_segments.json"
    bg_path = PROCESSED_DIR / "bg" / intercept_id / "events.json"
    report_path = PROCESSED_DIR / "reports" / f"{intercept_id}.md"

    if not asr_path.exists() or not report_path.exists():
        raise HTTPException(status_code=404, detail="Intercept artifacts incomplete")

    import json

    asr = json.loads(asr_path.read_text())
    transcript = [
        segment
        for segment in asr.get("segments", [])
        if segment.get("text")
    ]

    bg_events = []
    if bg_path.exists():
        bg_raw = json.loads(bg_path.read_text())
        for event in bg_raw.get("events", []):
            bg_events.append(
                {
                    "label": event.get("label"),
                    "start_s": event.get("start_s", event.get("t", 0.0)),
                    "end_s": event.get("end_s", event.get("start_s", event.get("t", 0.0))),
                    "probability": event.get("p", event.get("prob")),
                }
            )

    markdown = report_path.read_text()

    return InterceptDetail(
        meta=meta.to_summary(base_url=get_base_audio_url()),
        transcript=transcript,
        background_events=bg_events,
        report_markdown=markdown,
    )


@app.get("/intercepts/{intercept_id}", response_model=InterceptDetail)
async def get_intercept(intercept_id: str) -> InterceptDetail:
    return _load_intercept_detail(intercept_id)


def _process_pipeline(intercept_id: str, audio_path: Path, *, force: bool = False) -> None:
    """Run the pipeline and update registry once finished."""
    from src.pipeline import process_clip

    try:
        result = process_clip(audio_path, intercept_id, force=force)
        summary_line: Optional[str] = None
        for line in result.report_markdown.splitlines():
            if line.startswith("- "):
                summary_line = line.removeprefix("- ").strip()
                break
        meta = get_meta(intercept_id)
        if meta:
            meta.status = "ready"
            meta.updated_at = _now()
            meta.executive_summary = summary_line
            meta.duration_s = result.duration_s
            meta.tags = derive_tags(intercept_id, report_text=result.report_markdown)
            if meta.tags and "high-threat" in meta.tags:
                meta.priority = "high"
            else:
                meta.priority = None
            upsert_meta(meta)
    except Exception as exc:  # pylint: disable=broad-except
        meta = get_meta(intercept_id)
        if meta:
            meta.status = "failed"
            meta.updated_at = _now()
            meta.executive_summary = f"Processing failed: {exc}"
            upsert_meta(meta)
        raise


async def _process_upload(intercept_id: str, file_path: Path) -> None:
    """Background task entrypoint for newly uploaded files."""
    _process_pipeline(intercept_id, file_path, force=False)


@app.post("/intercepts", response_model=InterceptSummary, status_code=201)
async def upload_intercept(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> InterceptSummary:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file requires a filename")

    original_name = file.filename

    suffix = Path(original_name).suffix.lower()
    if suffix not in {".wav", ".m4a"}:
        raise HTTPException(
            status_code=400, detail="Only .wav or .m4a files are supported at this time"
        )

    slug = Path(original_name).stem.lower().replace(" ", "_")
    intercept_id = f"{slug}-{int(_now().timestamp())}"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DIR / f"{intercept_id}.wav"

    contents = await file.read()
    if suffix == ".wav":
        dest_path.write_bytes(contents)
    else:
        temp_input = RAW_DIR / f"{intercept_id}_upload{suffix}"
        temp_input.write_bytes(contents)

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            temp_input.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Audio conversion unavailable: ffmpeg not installed on the server.",
            )
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(temp_input),
                    str(dest_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            temp_input.unlink(missing_ok=True)
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500, detail="Failed to convert uploaded audio to WAV."
            ) from exc
        finally:
            temp_input.unlink(missing_ok=True)
    await file.close()

    display_title = Path(original_name).stem.strip() or intercept_id

    meta = InterceptMeta(
        intercept_id=intercept_id,
        title=display_title,
        status="processing",
        created_at=_now(),
        updated_at=_now(),
        tags=[],
        executive_summary="Processing...",
        audio_rel_path=str(dest_path.relative_to(ROOT_DIR)),
    )
    upsert_meta(meta)

    background_tasks.add_task(_process_upload, intercept_id, dest_path)

    return meta.to_summary(base_url=get_base_audio_url())


def _cleanup_intercept_files(intercept_id: str, meta: InterceptMeta) -> None:
    targets = [
        PROCESSED_DIR / "asr" / intercept_id,
        PROCESSED_DIR / "bg" / intercept_id,
        PROCESSED_DIR / "reports" / f"{intercept_id}.md",
    ]
    for target in targets:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)

    if meta.audio_rel_path:
        audio_path = ROOT_DIR / meta.audio_rel_path
        if audio_path.exists():
            audio_path.unlink(missing_ok=True)


@app.delete("/intercepts/{intercept_id}", status_code=204)
async def delete_intercept(intercept_id: str) -> None:
    meta = delete_meta(intercept_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Intercept not found")
    _cleanup_intercept_files(intercept_id, meta)


@app.post("/intercepts/{intercept_id}/rerun", response_model=InterceptSummary)
async def rerun_intercept(background_tasks: BackgroundTasks, intercept_id: str) -> InterceptSummary:
    meta = get_meta(intercept_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Intercept not found")
    if not meta.audio_rel_path:
        raise HTTPException(status_code=400, detail="Audio path not recorded for intercept")

    audio_path = ROOT_DIR / meta.audio_rel_path
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Source audio not found")

    meta.status = "processing"
    meta.updated_at = _now()
    meta.executive_summary = "Re-processing pipeline..."
    upsert_meta(meta)

    background_tasks.add_task(_process_pipeline, intercept_id, audio_path, force=True)

    return meta.to_summary(base_url=get_base_audio_url())
