from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CategoryConfig(BaseModel):
    id: str
    name: str
    description: str = ""
    priority: int = 100


class AppConfig(BaseModel):
    raw_audio_dir: str = "raw_audio"
    clips_dir: str = "output/clips"
    processed_audio_dir: str = "workspaces/processed_input"
    original_backup_dir: str = "backups/originals"
    enable_original_backup: bool = True
    final_output_dir: str = "output/final"
    completed_output_dir: str = "completed_results"
    export_dir: str = "output/exports"
    downloads_dir: str = "~/Downloads"
    gemini_uploads_dir: str = "output/gemini_uploads"
    clip_format: str = "wav"
    gemini_model: str = "gemini-3.1-pro-preview"
    gemini_timeout_sec: int = 180
    gemini_retry_count: int = 1
    enable_prompt_cache: bool = True
    prompt_cache_ttl_sec: int = 86400
    categories: list[CategoryConfig] = Field(default_factory=list)


class GeminiSegment(BaseModel):
    section: Literal["verse", "chorus", "unknown"] = "unknown"
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 0.0
    needs_review: bool = False
    reason: str = ""


class GeminiTrackResult(BaseModel):
    track_id: str
    needs_review: bool = False
    segments: list[GeminiSegment] = Field(default_factory=list)
    song_level_label: str = ""
    song_level_reason: str = ""


class ClipRecord(BaseModel):
    clip_id: str
    track_id: str
    source_filename: str
    source_audio_path: str
    clip_path: str
    section: Literal["verse", "chorus", "unknown"] = "unknown"
    start_sec: float
    end_sec: float
    duration_sec: float
    model_label: str
    manual_label: str = ""
    final_label: str
    confidence: float = 0.0
    needs_review: bool = False
    reason: str = ""
    display_name: str
    export_filename: str
    status: str = "confirmed"

    def refresh_final_label(self) -> None:
        self.final_label = self.manual_label.strip() or self.model_label


class ResultsState(BaseModel):
    clips: list[ClipRecord] = Field(default_factory=list)


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser()
