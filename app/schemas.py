"""Pydantic models shared by the intercept API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class TranscriptSegment(BaseModel):
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    speaker: str = Field(default="UNKNOWN")
    text: str = Field(default="")


class BackgroundEvent(BaseModel):
    label: str
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    probability: Optional[float] = Field(default=None, ge=0, le=1)


class InterceptSummary(BaseModel):
    intercept_id: str = Field(..., description="Unique identifier for the intercept")
    title: str = Field(..., description="Human readable name for display")
    status: str = Field(..., description="processing | ready | failed | archived")
    created_at: datetime
    updated_at: datetime
    duration_s: Optional[float] = None
    priority: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    audio_url: Optional[HttpUrl] = None


class InterceptDetail(BaseModel):
    meta: InterceptSummary
    transcript: List[TranscriptSegment] = Field(default_factory=list)
    background_events: List[BackgroundEvent] = Field(default_factory=list)
    report_markdown: str = ""
