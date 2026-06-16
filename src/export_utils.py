from __future__ import annotations

import os
import platform
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import resolve_project_path
from src.models import AppConfig, ClipRecord, ResultsState


CSV_COLUMNS = [
    "track_id",
    "source_filename",
    "display_name",
    "export_filename",
    "section",
    "start_sec",
    "end_sec",
    "duration_sec",
    "model_label",
    "manual_label",
    "final_label",
    "confidence",
    "needs_review",
    "reason",
    "clip_path",
    "source_audio_path",
]


def export_csv(state: ResultsState, config: AppConfig) -> Path:
    export_dir = resolve_project_path(config.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "classified_segments.csv"
    rows = [
        {column: getattr(clip, column) for column in CSV_COLUMNS}
        for clip in visible_clips(state)
    ]
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def build_classified_folders(state: ResultsState, config: AppConfig) -> Path:
    final_dir = resolve_project_path(config.final_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    used_names: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for clip in visible_clips(state):
        source = resolve_project_path(clip.clip_path)
        if not source.exists():
            continue
        label = safe_folder_name(clip.final_label or "待复核")
        target_dir = final_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = ensure_wav_suffix(clip.export_filename or clip.display_name)
        target_name = unique_filename(filename, used_names[label])
        shutil.copy2(source, target_dir / target_name)
    return final_dir


def build_zip(state: ResultsState, config: AppConfig) -> Path:
    final_dir = build_classified_folders(state, config)
    downloads_dir = resolve_project_path(config.downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = downloads_dir / "classified_clips.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in final_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(final_dir))
    return zip_path


def open_folder(path: Path) -> tuple[bool, str]:
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True, str(path)
    except Exception as exc:
        return False, f"{path}（自动打开失败：{exc}）"


def visible_clips(state: ResultsState) -> list[ClipRecord]:
    return [clip for clip in state.clips if clip.status not in {"hidden", "replaced"}]


def safe_folder_name(name: str) -> str:
    clean = "".join("_" if char in r'\/:*?"<>|' else char for char in name).strip()
    return clean or "待复核"


def ensure_wav_suffix(filename: str) -> str:
    path = Path(filename)
    return f"{path.stem}.wav"


def unique_filename(filename: str, counter: defaultdict[str, int]) -> str:
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix or ".wav"
    counter[filename] += 1
    count = counter[filename]
    if count == 1:
        return f"{stem}{suffix}"
    return f"{stem}_{count:02d}{suffix}"

