"""Utilities to persist and retrieve intercept metadata for the API."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.schemas import InterceptSummary

ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "data" / "processed" / "intercepts" / "index.json"


@dataclass
class InterceptMeta:
    intercept_id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    duration_s: Optional[float] = None
    priority: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = None  # type: ignore[assignment]
    executive_summary: Optional[str] = None
    audio_rel_path: Optional[str] = None

    def to_summary(self, base_url: Optional[str] = None) -> InterceptSummary:
        audio_url = None
        if base_url and self.audio_rel_path:
            audio_url = f"{base_url.rstrip('/')}/{self.audio_rel_path}"
        return InterceptSummary(
            intercept_id=self.intercept_id,
            title=self.title,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            duration_s=self.duration_s,
            priority=self.priority,
            source=self.source,
            tags=self.tags or [],
            executive_summary=self.executive_summary,
            audio_url=audio_url,
        )


def ensure_registry_dir() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def _datetime_from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _datetime_to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _compute_duration(audio_path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate and info.frames:
            return info.frames / float(info.samplerate)
    except Exception:
        return None
    return None


def derive_tags(
    intercept_id: str,
    *,
    report_text: Optional[str] = None,
    events: Optional[List[dict]] = None,
) -> List[str]:
    tags: set[str] = set()

    if events is None:
        events_path = ROOT_DIR / "data" / "processed" / "bg" / intercept_id / "events.json"
        if events_path.exists():
            try:
                payload = json.loads(events_path.read_text(encoding="utf-8"))
                events = payload.get("events", [])
            except Exception:
                events = []

    if events:
        tags.add("background-alert")
        high_prob = False
        for event in events:
            if not isinstance(event, dict):
                continue
            label = str(event.get("label", "")).lower().strip()
            if label:
                tags.add(label)
            prob = event.get("p", event.get("prob"))
            try:
                prob_val = float(prob) if prob is not None else None
            except (TypeError, ValueError):
                prob_val = None
            if prob_val is not None and prob_val >= 0.4:
                high_prob = True
        if high_prob:
            tags.add("high-threat")

    if report_text is None:
        report_path = ROOT_DIR / "data" / "processed" / "reports" / f"{intercept_id}.md"
        if report_path.exists():
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except Exception:
                report_text = None

    if report_text:
        lower = report_text.lower()
        if "civilian" in lower:
            tags.add("civilian-watch")
        if "urgent" in lower or "evacuate" in lower:
            tags.add("priority")

    return sorted(tags)


def _load_from_disk() -> Dict[str, InterceptMeta]:
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open("r", encoding="utf-8") as fp:
        try:
            raw = json.load(fp)
        except json.JSONDecodeError:
            raw = {}
    registry = {}
    for key, payload in raw.items():
        registry[key] = InterceptMeta(
            intercept_id=payload["intercept_id"],
            title=payload.get("title", payload["intercept_id"]),
            status=payload.get("status", "unknown"),
            created_at=_datetime_from_iso(payload["created_at"]),
            updated_at=_datetime_from_iso(payload["updated_at"]),
            duration_s=payload.get("duration_s"),
            priority=payload.get("priority"),
            source=payload.get("source"),
            tags=payload.get("tags"),
            executive_summary=payload.get("executive_summary"),
            audio_rel_path=payload.get("audio_rel_path"),
        )
    return registry


def _discover_existing() -> Dict[str, InterceptMeta]:
    asr_root = ROOT_DIR / "data" / "processed" / "asr"
    reports_root = ROOT_DIR / "data" / "processed" / "reports"
    raw_root = ROOT_DIR / "data" / "raw"
    bg_root = ROOT_DIR / "data" / "processed" / "bg"
    if not asr_root.exists():
        return {}

    discovered: Dict[str, InterceptMeta] = {}
    for clip_dir in asr_root.iterdir():
        if not clip_dir.is_dir():
            continue
        intercept_id = clip_dir.name
        report_path = reports_root / f"{intercept_id}.md"
        audio_path = raw_root / f"{intercept_id}.wav"
        events_path = bg_root / intercept_id / "events.json"
        events_payload: List[dict] = []
        if events_path.exists():
            try:
                events_payload = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
            except Exception:
                events_payload = []
        timestamp_source = (
            audio_path
            if audio_path.exists()
            else report_path
            if report_path.exists()
            else clip_dir
        )
        created_at = datetime.fromtimestamp(
            timestamp_source.stat().st_mtime, tz=timezone.utc
        )

        summary_line: Optional[str] = None
        report_text: Optional[str] = None
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
            for line in report_text.splitlines():
                if line.startswith("- "):
                    summary_line = line.removeprefix("- ").strip()
                    break

        duration_s = _compute_duration(audio_path) if audio_path.exists() else None
        audio_rel_path = (
            str(audio_path.relative_to(ROOT_DIR)) if audio_path.exists() else None
        )

        discovered[intercept_id] = InterceptMeta(
            intercept_id=intercept_id,
            title=intercept_id.replace("_", " ").title(),
            status="ready" if report_path.exists() else "processing",
            created_at=created_at,
            updated_at=created_at,
            duration_s=duration_s,
            tags=derive_tags(
                intercept_id,
                report_text=report_text,
                events=events_payload,
            ),
            executive_summary=summary_line,
            audio_rel_path=audio_rel_path,
        )
    return discovered


def load_registry() -> Dict[str, InterceptMeta]:
    ensure_registry_dir()
    registry = _load_from_disk()
    if registry:
        updated = False
        for meta in registry.values():
            if meta.status == "ready":
                if not meta.tags:
                    meta.tags = derive_tags(meta.intercept_id)
                    updated = True
                if meta.duration_s is None and meta.audio_rel_path:
                    duration = _compute_duration(ROOT_DIR / meta.audio_rel_path)
                    if duration is not None:
                        meta.duration_s = duration
                        updated = True
        if updated:
            save_registry(registry)
        return registry
    discovered = _discover_existing()
    if discovered:
        save_registry(discovered)
    return discovered


def save_registry(registry: Dict[str, InterceptMeta]) -> None:
    ensure_registry_dir()
    serialized = {}
    for key, meta in registry.items():
        payload = asdict(meta)
        payload["created_at"] = _datetime_to_iso(meta.created_at)
        payload["updated_at"] = _datetime_to_iso(meta.updated_at)
        serialized[key] = payload
    with REGISTRY_PATH.open("w", encoding="utf-8") as fp:
        json.dump(serialized, fp, indent=2)


def upsert_meta(meta: InterceptMeta) -> None:
    registry = load_registry()
    registry[meta.intercept_id] = meta
    save_registry(registry)


def list_summaries(base_url: Optional[str] = None) -> List[InterceptSummary]:
    registry = load_registry()
    summaries = [meta.to_summary(base_url=base_url) for meta in registry.values()]
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


def get_meta(intercept_id: str) -> Optional[InterceptMeta]:
    registry = load_registry()
    return registry.get(intercept_id)


def delete_meta(intercept_id: str) -> Optional[InterceptMeta]:
    registry = load_registry()
    meta = registry.pop(intercept_id, None)
    save_registry(registry)
    return meta
