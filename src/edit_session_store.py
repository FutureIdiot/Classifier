from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.models import ClipRecord, ResultsState


SourceLookup = Callable[[str, ClipRecord | None], tuple[Path, str, str] | None]


class EditSessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, state: ResultsState, source_lookup: SourceLookup | None = None) -> dict | None:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return self.rebuild_from_state(state, source_lookup)

    def save(self, session: dict | None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if session is None:
            if self.path.exists():
                self.path.unlink()
            return
        self.path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebuild_from_state(self, state: ResultsState, source_lookup: SourceLookup | None = None) -> dict | None:
        editing_clips = [clip for clip in state.clips if clip.status == "editing"]
        if not editing_clips:
            return None
        track_id = editing_clips[0].track_id
        track_clips = [clip for clip in editing_clips if clip.track_id == track_id]
        source_info = source_lookup(track_id, track_clips[0]) if source_lookup else None
        source_filename = source_info[1] if source_info else track_clips[0].source_filename
        source_audio_path = source_info[2] if source_info else track_clips[0].source_audio_path
        regions = [
            {
                "clip_id": item.clip_id,
                "display_name": item.display_name,
                "label": item.final_label,
                "section": item.section,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
            }
            for item in sorted(track_clips, key=lambda value: value.start_sec)
        ]
        return {
            "track_id": track_id,
            "source_filename": source_filename,
            "source_audio_path": source_audio_path,
            "old_status": {clip.clip_id: "confirmed" for clip in track_clips},
            "regions": regions,
        }
