from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import sys
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import Header, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from nicegui import app, ui

from src.audio_utils import stable_track_id
from src.config import (
    ERRORS_PATH,
    PROMPT_CACHE_PATH,
    RAW_RESPONSES_PATH,
    ROOT,
    RESULTS_PATH,
    append_jsonl,
    get_gemini_api_key,
    load_app_config,
    load_completed_tracks,
    load_results,
    resolve_project_path,
    save_app_config,
    save_completed_tracks,
    save_gemini_api_key,
    save_results,
)
from src.edit_session_store import EditSessionStore
from src.export_utils import build_zip, export_csv, open_folder, sync_classified_folders, visible_clips
from src.gemini_client import list_available_gemini_models
from src.models import AppConfig, ResultsState
from src.pipeline import (
    add_category,
    archive_successful_audio_source,
    find_clip,
    prune_empty_categories,
    recut_clip,
    scan_tracks,
    update_clip_display_name,
    update_clip_label,
)
from src.runtime import configure_utf8_runtime
from src.ui_templates import board_html, escape_attr


configure_utf8_runtime()
app.add_static_files("/static", ROOT / "static")

config: AppConfig = load_app_config()
state: ResultsState = load_results()
progress_text = "就绪"
processing = False
analysis_process: asyncio.subprocess.Process | None = None
run_logs: deque[str] = deque(maxlen=300)
analysis_started_at: datetime | None = None
last_log_at: datetime | None = None
run_status_label = None
run_log_textarea = None
progress_label_widget = None
waveform_cache: dict[str, tuple[int, int, str]] = {}
EDIT_SESSION_PATH = ROOT / "data" / "edit_session.json"
edit_session_store = EditSessionStore(EDIT_SESSION_PATH)
edit_session: dict | None = None
result_filters = {
    "search": "",
    "label": "全部",
    "section": "全部",
    "status": "默认",
    "min_confidence": 0.0,
    "needs_review": False,
    "sort": "confidence_desc",
}


def add_log(message: str) -> None:
    global last_log_at
    last_log_at = datetime.now()
    line = f"{datetime.now().strftime('%H:%M:%S')}  {message}"
    run_logs.append(line)
    print(line, flush=True)
    update_run_widgets()


def reload_state() -> None:
    global config, edit_session, state
    config = load_app_config()
    state = load_results()
    edit_session = load_edit_session()


def load_edit_session() -> dict | None:
    source_lookup = find_existing_source_audio if "find_existing_source_audio" in globals() else None
    return edit_session_store.load(state, source_lookup)


def save_edit_session(session: dict | None) -> None:
    edit_session_store.save(session)


@app.get("/media/{clip_id}")
def media(clip_id: str, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    try:
        clip = find_clip(state, clip_id)
        path = find_existing_clip_audio(clip)
        return audio_response(path, clip.export_filename, range_header)
    except Exception as exc:
        return missing_media_response(f"片段音频加载失败：{clip_id}，{exc}")


@app.get("/api/debug_clip/{clip_id}")
def debug_clip(clip_id: str) -> JSONResponse:
    clip = find_clip(state, clip_id)
    path = resolve_project_path(clip.clip_path)
    fallback_path = find_existing_clip_audio(clip)
    source_info = find_existing_source_audio(clip.track_id, clip)
    return JSONResponse(
        {
            "clip_id": clip_id,
            "clip_path": clip.clip_path,
            "resolved_path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else None,
            "fallback_path": str(fallback_path),
            "fallback_exists": fallback_path.exists(),
            "source_path": str(source_info[0]) if source_info else None,
            "source_exists": source_info[0].exists() if source_info else False,
        }
    )


@app.get("/api/debug_visible_clips")
def debug_visible_clips() -> JSONResponse:
    records = []
    for clip in visible_clips(state):
        path = resolve_project_path(clip.clip_path)
        fallback_path = find_existing_clip_audio(clip)
        source_info = find_existing_source_audio(clip.track_id, clip)
        records.append(
            {
                "clip_id": clip.clip_id,
                "track_id": clip.track_id,
                "status": clip.status,
                "source_filename": clip.source_filename,
                "clip_path": clip.clip_path,
                "clip_exists": path.exists(),
                "clip_size": path.stat().st_size if path.exists() else None,
                "fallback_path": str(fallback_path),
                "fallback_exists": fallback_path.exists(),
                "source_path": str(source_info[0]) if source_info else None,
                "source_exists": source_info[0].exists() if source_info else False,
                "source_ref": source_info[2] if source_info else clip.source_audio_path,
            }
        )
    return JSONResponse({"count": len(records), "clips": records})


@app.get("/api/debug_state")
def debug_state() -> JSONResponse:
    disk_state = load_results()
    return JSONResponse(
        {
            "memory": debug_state_summary(state),
            "disk": debug_state_summary(disk_state),
            "edit_session_loaded": edit_session is not None,
            "results_path": str(RESULTS_PATH),
            "results_path_exists": RESULTS_PATH.exists(),
            "results_path_size": RESULTS_PATH.stat().st_size if RESULTS_PATH.exists() else None,
        }
    )


def debug_state_summary(results: ResultsState) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    clips = []
    for clip in results.clips:
        status_counts[clip.status] = status_counts.get(clip.status, 0) + 1
        path = resolve_project_path(clip.clip_path)
        source_path = resolve_project_path(clip.source_audio_path)
        clips.append(
            {
                "clip_id": clip.clip_id,
                "track_id": clip.track_id,
                "status": clip.status,
                "source_filename": clip.source_filename,
                "clip_path": clip.clip_path,
                "clip_exists": path.exists(),
                "clip_size": path.stat().st_size if path.exists() else None,
                "source_audio_path": clip.source_audio_path,
                "source_exists": source_path.exists(),
                "source_size": source_path.stat().st_size if source_path.exists() else None,
                "final_label": clip.final_label,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
            }
        )
    return {
        "clip_count": len(results.clips),
        "visible_count": len(visible_clips(results)),
        "status_counts": status_counts,
        "clips": clips,
    }


@app.get("/waveform/{clip_id}")
def waveform_preview(clip_id: str) -> Response:
    try:
        clip = find_clip(state, clip_id)
        path = find_existing_clip_audio(clip)
        svg = cached_waveform_svg(path)
    except Exception as exc:
        svg = error_waveform_svg(f"波形加载失败：{clip_id}，{exc}")
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "no-store",
        },
    )


@app.get("/source/{clip_id}")
def source_media(clip_id: str, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    try:
        clip = find_clip(state, clip_id)
        source_info = find_existing_source_audio(clip.track_id, clip)
        if source_info is None:
            return audio_response(resolve_project_path(clip.source_audio_path), clip.source_filename, range_header)
        source_path, source_filename, _ = source_info
        return audio_response(source_path, source_filename, range_header)
    except Exception as exc:
        return missing_media_response(f"原曲音频加载失败：{clip_id}，{exc}")


@app.get("/edit/source/{track_id}")
def edit_source_media(track_id: str, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    try:
        source_info = find_existing_source_audio(track_id)
        if source_info is None:
            return missing_media_response(f"找不到原曲：{track_id}", status_code=404)
        source_path, source_filename, _ = source_info
        return audio_response(source_path, source_filename, range_header)
    except Exception as exc:
        return missing_media_response(f"编辑区原曲加载失败：{track_id}，{exc}")


def find_existing_source_audio(track_id: str, preferred_clip: Any | None = None) -> tuple[Path, str, str] | None:
    candidates: list[tuple[str, str]] = []
    if preferred_clip is not None:
        candidates.append((preferred_clip.source_audio_path, preferred_clip.source_filename))
    if edit_session and edit_session.get("track_id") == track_id:
        candidates.append((edit_session.get("source_audio_path", ""), edit_session.get("source_filename", "")))

    track_clips = [clip for clip in state.clips if clip.track_id == track_id]
    track_clips.sort(key=lambda clip: (clip.status in {"replaced", "hidden"}, clip.start_sec))
    candidates.extend((clip.source_audio_path, clip.source_filename) for clip in track_clips)

    seen: set[str] = set()
    for path_value, filename in candidates:
        if not path_value or path_value in seen:
            continue
        seen.add(path_value)
        path = resolve_project_path(path_value)
        if path.exists():
            return path, filename or path.name, path_value
        found_path = find_audio_by_filename(filename)
        if found_path is not None:
            return found_path, filename or found_path.name, relative_or_absolute_app(found_path)
    return None


def find_existing_clip_audio(clip: Any) -> Path:
    candidates = [
        resolve_project_path(clip.clip_path),
        resolve_project_path(config.clips_dir) / Path(clip.clip_path).name,
        resolve_project_path(config.clips_dir) / ensure_wav_filename(clip.export_filename or clip.display_name),
        resolve_project_path(config.final_output_dir)
        / safe_output_folder_name(clip.final_label or "待复核")
        / ensure_wav_filename(clip.export_filename or clip.display_name),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def find_audio_by_filename(filename: str) -> Path | None:
    if not filename:
        return None
    roots = [
        resolve_project_path(config.raw_audio_dir),
        resolve_project_path(config.processed_audio_dir),
    ]
    for root in roots:
        direct = root / filename
        if direct.exists():
            return direct
        if root.exists():
            try:
                match = next((path for path in root.rglob(filename) if path.is_file()), None)
            except OSError:
                match = None
            if match is not None:
                return match
    return None


def ensure_wav_filename(filename: str) -> str:
    path = Path(filename)
    return f"{path.stem}.wav"


def safe_output_folder_name(name: str) -> str:
    clean = "".join("_" if char in r'\/:*?"<>|' else char for char in name).strip()
    return clean or "待复核"


def relative_or_absolute_app(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


edit_session = load_edit_session()


def audio_response(path: Path, filename: str, range_header: str | None = None) -> Response:
    path = path.expanduser()
    if not path.exists():
        return Response(
            content=f"Audio file not found: {path}",
            status_code=404,
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )
    size = path.stat().st_size
    media_type = audio_media_type(path)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename, safe='')}",
    }

    byte_range, range_unsatisfiable = parse_range_header(range_header, size)
    if range_unsatisfiable:
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})

    if byte_range is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(file_bytes(path, 0, size - 1), media_type=media_type, headers=headers)

    start, end = byte_range
    headers.update(
        {
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{size}",
        }
    )
    return StreamingResponse(file_bytes(path, start, end), status_code=206, media_type=media_type, headers=headers)


def missing_media_response(message: str, status_code: int = 404) -> Response:
    return Response(
        content=message,
        status_code=status_code,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


def audio_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/mp4"
    if suffix == ".aac":
        return "audio/aac"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".ogg":
        return "audio/ogg"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def parse_range_header(range_header: str | None, size: int) -> tuple[tuple[int, int] | None, bool]:
    if not range_header or not range_header.startswith("bytes=") or size <= 0:
        return None, False
    value = range_header.removeprefix("bytes=").strip()
    if "," in value or "-" not in value:
        return None, False
    start_text, end_text = value.split("-", 1)
    try:
        if start_text == "":
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None, False
            start = max(0, size - suffix_length)
            end = size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError:
        return None, False
    if start < 0 or end < start or start >= size:
        return None, True
    return (start, min(end, size - 1)), False


def file_bytes(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def build_waveform_svg(path: Path, width: int = 180, height: int = 18) -> str:
    peaks = waveform_peaks(path, width)
    if not peaks:
        mid = height // 2
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="none"/>'
            f'<path d="M0 {mid} H{width}" stroke="#cbd5e1" stroke-width="1"/>'
            "</svg>"
        )
    mid = height / 2
    scale = max(1.0, mid - 2)
    lines = []
    for index, peak in enumerate(peaks):
        amp = max(0.04, min(1.0, peak))
        y1 = round(mid - amp * scale, 2)
        y2 = round(mid + amp * scale, 2)
        lines.append(f'<line x1="{index}" y1="{y1}" x2="{index}" y2="{y2}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="none">'
        f'<rect width="{width}" height="{height}" rx="3" fill="#f8fafc"/>'
        f'<g stroke="#64748b" stroke-width="1">{"".join(lines)}</g>'
        "</svg>"
    )


def error_waveform_svg(message: str, width: int = 180, height: int = 18) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="none">'
        f'<rect width="{width}" height="{height}" rx="3" fill="#fff1f2"/>'
        f'<path d="M0 {height // 2} H{width}" stroke="#f43f5e" stroke-width="1"/>'
        f'<title>{html_escape(message)}</title>'
        "</svg>"
    )


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def waveform_peaks(path: Path, width: int) -> list[float]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            if frame_count <= 0 or sample_width not in {1, 2, 3, 4}:
                return []
            frames_per_bucket = max(1, frame_count // width)
            peaks: list[float] = []
            max_possible = float(1 << (8 * sample_width - 1))
            for _ in range(width):
                data = handle.readframes(frames_per_bucket)
                if not data:
                    break
                peak = pcm_peak(data, sample_width, channels) / max_possible
                peaks.append(min(1.0, peak))
            if not peaks:
                return []
            while len(peaks) < width:
                peaks.append(0.0)
            return normalize_peaks(peaks)
    except Exception:
        return []


def normalize_peaks(peaks: list[float]) -> list[float]:
    maximum = max(peaks) if peaks else 0.0
    if maximum <= 0:
        return peaks
    return [min(1.0, peak / maximum) for peak in peaks]


def cached_waveform_svg(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return build_waveform_svg(path)
    key = str(path.resolve())
    cached = waveform_cache.get(key)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached and cached[0] == signature[0] and cached[1] == signature[1]:
        return cached[2]
    svg = build_waveform_svg(path)
    waveform_cache[key] = (signature[0], signature[1], svg)
    return svg


def pcm_peak(data: bytes, sample_width: int, channels: int) -> int:
    frame_width = sample_width * max(1, channels)
    if frame_width <= 0:
        return 0
    peak = 0
    for frame_start in range(0, len(data) - frame_width + 1, frame_width):
        for channel in range(max(1, channels)):
            offset = frame_start + channel * sample_width
            sample = pcm_sample_value(data[offset : offset + sample_width], sample_width)
            peak = max(peak, abs(sample))
    return peak


def pcm_sample_value(sample: bytes, sample_width: int) -> int:
    if sample_width == 1:
        return sample[0] - 128
    if sample_width == 3:
        sign_byte = b"\xff" if sample[2] & 0x80 else b"\x00"
        sample = sample + sign_byte
        return int.from_bytes(sample, byteorder="little", signed=True)
    return int.from_bytes(sample, byteorder="little", signed=True)


@app.post("/api/move_clip")
async def move_clip(request: Request) -> JSONResponse:
    payload = await request.json()
    update_clip_label(state, payload["clip_id"], payload["label"])
    sync_outputs("move_clip")
    return JSONResponse({"ok": True})


@app.post("/api/rename_clip")
async def rename_clip(request: Request) -> JSONResponse:
    payload = await request.json()
    update_clip_display_name(state, payload["clip_id"], payload["display_name"])
    sync_outputs("rename_clip")
    return JSONResponse({"ok": True})


@app.post("/api/batch_update_clips")
async def batch_update_clips(request: Request) -> JSONResponse:
    global state, edit_session
    payload = await request.json()
    action = payload.get("action", "")
    if action == "confirm":
        clip_ids = {clip.clip_id for clip in visible_clips(state)}
    else:
        clip_ids = set(payload.get("clip_ids", []))
    updated = 0
    for clip in state.clips:
        if clip.clip_id not in clip_ids:
            continue
        if action == "confirm":
            clip.status = "auto_confirmed"
            clip.needs_review = False
        elif action == "review":
            clip.status = "pending_review"
            clip.needs_review = True
        elif action == "hide":
            clip.status = "hidden"
        elif action == "restore":
            clip.status = "confirmed"
        else:
            continue
        updated += 1
    save_results(state)
    finalized = False
    if action == "confirm" and updated:
        finalized = finalize_current_batch()
    else:
        sync_outputs("batch_update_clips")
    if finalized or updated:
        refresh_main_views(reset_runtime=finalized)
    return JSONResponse({"ok": True, "updated": updated, "finalized": finalized})


@app.post("/api/start_edit_clip")
async def start_edit_clip(request: Request) -> JSONResponse:
    global edit_session
    payload = await request.json()
    clip = find_clip(state, payload["clip_id"])
    source_info = find_existing_source_audio(clip.track_id, clip)
    if source_info is None:
        return JSONResponse(
            {"ok": False, "error": f"找不到原曲文件，无法进入编辑区：{clip.source_filename}"},
            status_code=400,
        )
    _, source_filename, source_audio_path = source_info
    for item in state.clips:
        if item.track_id == clip.track_id:
            item.source_audio_path = source_audio_path
            item.source_filename = source_filename
    track_clips = [
        item
        for item in state.clips
        if item.track_id == clip.track_id and item.status not in {"hidden", "replaced"}
    ]
    old_status = {item.clip_id: item.status for item in track_clips}
    regions = []
    for item in sorted(track_clips, key=lambda value: value.start_sec):
        regions.append(
            {
                "clip_id": item.clip_id,
                "display_name": item.display_name,
                "label": item.final_label,
                "section": item.section,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
            }
        )
        item.status = "editing"
    save_results(state)
    edit_session = {
        "track_id": clip.track_id,
        "source_filename": source_filename,
        "source_audio_path": source_audio_path,
        "old_status": old_status,
        "regions": regions,
    }
    save_edit_session(edit_session)
    return JSONResponse({"ok": True, "track_id": clip.track_id, "regions": regions})


@app.post("/api/cancel_edit")
def cancel_edit() -> JSONResponse:
    global edit_session
    if edit_session:
        old_status = edit_session.get("old_status", {})
        for clip in state.clips:
            if clip.clip_id in old_status and clip.status == "editing":
                clip.status = old_status[clip.clip_id]
        save_results(state)
        edit_session = None
        save_edit_session(None)
    return JSONResponse({"ok": True})


@app.post("/api/commit_edit_regions")
async def commit_edit_regions(request: Request) -> JSONResponse:
    global edit_session
    if not edit_session:
        return JSONResponse({"ok": False, "error": "没有正在编辑的歌曲。"}, status_code=400)
    payload = await request.json()
    regions = payload.get("regions", [])
    selected_clip_ids = {region.get("clip_id") for region in regions}
    created = 0
    for region in regions:
        clip_id = region.get("clip_id")
        start_sec = float(region.get("start_sec", 0))
        end_sec = float(region.get("end_sec", 0))
        final_label = region.get("label") or None
        if not clip_id or end_sec <= start_sec:
            continue
        new_clip = recut_clip(config, state, clip_id, start_sec, end_sec, final_label)
        new_clip.status = "manual_reviewed"
        new_clip.display_name = region.get("display_name") or new_clip.display_name
        new_clip.export_filename = f"{Path(new_clip.display_name).stem}.wav"
        created += 1
    for clip in state.clips:
        if clip.track_id == edit_session["track_id"] and clip.status == "editing" and clip.clip_id not in selected_clip_ids:
            clip.status = "replaced"
    save_results(state)
    sync_outputs("commit_edit_regions")
    edit_session = None
    save_edit_session(None)
    return JSONResponse({"ok": True, "created": created})


@app.post("/api/update_category")
async def api_update_category(request: Request) -> JSONResponse:
    payload = await request.json()
    category_id = payload["category_id"]
    for category in config.categories:
        if category.id == category_id:
            old_name = category.name
            category.name = payload.get("name", "").strip()
            category.description = payload.get("description", "").strip()
            if old_name and category.name and old_name != category.name:
                for clip in state.clips:
                    if clip.model_label == old_name:
                        clip.model_label = category.name
                    if clip.manual_label == old_name:
                        clip.manual_label = category.name
                    if clip.final_label == old_name:
                        clip.final_label = category.name
                save_results(state)
                sync_outputs("update_category")
            break
    prune_empty_categories(config, state)
    save_app_config(config)
    return JSONResponse({"ok": True})


@app.get("/download/zip")
def download_zip() -> FileResponse:
    zip_path = build_zip(state, config)
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


def visible_by_label() -> dict[str, list]:
    labels = [category.name for category in config.categories if category.name.strip()]
    grouped = {label: [] for label in labels}
    for clip in filtered_sorted_clips():
        grouped.setdefault(clip.final_label or "待复核", []).append(clip)
    return grouped


def filtered_sorted_clips() -> list:
    clips = state.clips
    if result_filters["status"] == "默认":
        clips = [clip for clip in clips if clip.status not in {"hidden", "replaced", "editing"}]
    elif result_filters["status"] != "全部":
        clips = [clip for clip in clips if clip.status == result_filters["status"]]

    search = str(result_filters["search"]).strip().lower()
    if search:
        clips = [
            clip
            for clip in clips
            if search in clip.display_name.lower()
            or search in clip.source_filename.lower()
            or search in clip.track_id.lower()
        ]
    if result_filters["label"] != "全部":
        clips = [clip for clip in clips if clip.final_label == result_filters["label"]]
    if result_filters["section"] != "全部":
        clips = [clip for clip in clips if clip.section == result_filters["section"]]
    if result_filters["needs_review"]:
        clips = [clip for clip in clips if clip.needs_review]
    min_confidence = float(result_filters["min_confidence"] or 0)
    clips = [clip for clip in clips if clip.confidence >= min_confidence]

    sort = result_filters["sort"]
    if sort == "confidence_asc":
        return sorted(clips, key=lambda clip: clip.confidence)
    if sort == "filename":
        return sorted(clips, key=lambda clip: (clip.source_filename, clip.start_sec))
    if sort == "time":
        return sorted(clips, key=lambda clip: (clip.track_id, clip.start_sec))
    if sort == "label":
        return sorted(clips, key=lambda clip: (clip.final_label, clip.source_filename, clip.start_sec))
    if sort == "status":
        return sorted(clips, key=lambda clip: (clip.status, clip.source_filename, clip.start_sec))
    return sorted(clips, key=lambda clip: clip.confidence, reverse=True)


@app.post("/api/add_category")
def api_add_category() -> JSONResponse:
    add_category(config)
    return JSONResponse({"ok": True})


def add_styles() -> None:
    ui.add_head_html('<link rel="stylesheet" href="/static/app.css?v=20260618d">')
    ui.add_body_html('<script src="/static/app.js?v=20260618d"></script>')

@ui.refreshable
def render_board() -> None:
    media_version = str(int(datetime.now().timestamp() * 1000))
    ui.html(board_html(config, visible_by_label(), media_version=media_version), sanitize=False).classes("w-full")


@ui.refreshable
def render_recut_area() -> None:
    global edit_session
    if edit_session is None:
        edit_session = load_edit_session()
    with ui.card().classes("w-full p-3").style("border-radius:8px"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("手动裁剪").classes("text-sm font-bold")
            if edit_session:
                with ui.row().classes("gap-2"):
                    ui.button("新增片段", on_click=lambda: ui.run_javascript("mcAddEditRegion()")).props("outline")
                    ui.button("取消编辑", on_click=cancel_edit_ui).props("outline")
                    ui.button("裁剪并确认", on_click=lambda: ui.run_javascript("mcCommitEditRegions()"))
        if not edit_session:
            ui.html(
                """
                <div class="edit-drop-zone"
                  ondragover="event.preventDefault()"
                  ondrop="mcDropToEditor(event)">
                  将下方任意片段拖到这里，进入整首歌重切模式
                </div>
                """,
                sanitize=False,
            )
            return

        track_id = edit_session["track_id"]
        regions_json = json.dumps(edit_session["regions"], ensure_ascii=False)
        categories_json = json.dumps([category.model_dump() for category in config.categories if category.name.strip()], ensure_ascii=False)
        base_clip_id = edit_session["regions"][0]["clip_id"] if edit_session["regions"] else ""
        encoded_track_id = quote(track_id, safe="")
        source_url = f"/edit/source/{encoded_track_id}?v={int(datetime.now().timestamp() * 1000)}"
        ui.label(f"{edit_session['source_filename']} · {track_id}").classes("text-xs text-gray-500")
        ui.html(
            f"""
            <div id="mc-editor-root" class="wave-editor"
              data-track-id="{escape_attr(track_id)}"
              data-base-clip-id="{escape_attr(base_clip_id)}"
              data-source-url="{escape_attr(source_url)}"
              data-regions="{escape_attr(regions_json)}"
              data-categories="{escape_attr(categories_json)}">
              <audio id="mc-edit-audio" controls src="{escape_attr(source_url)}"></audio>
              <div class="wave-stage">
                <canvas id="mc-wave-canvas"></canvas>
                <div id="mc-region-layer"></div>
              </div>
              <div id="mc-region-table" class="region-table"></div>
            </div>
            """,
            sanitize=False,
        ).classes("w-full")
        ui.timer(0.1, lambda: ui.run_javascript("mcInitWaveEditor()"), once=True)


def load_recut_values(clip_id: str, start_input, end_input, label_select) -> None:
    clip = find_clip(state, clip_id)
    start_input.value = clip.start_sec
    end_input.value = clip.end_sec
    label_select.value = clip.final_label
    ui.notify("已载入片段时间")


def do_recut(clip_id: str, start_sec: float, end_sec: float, final_label: str) -> None:
    try:
        recut_clip(config, state, clip_id, float(start_sec), float(end_sec), final_label)
        sync_outputs("recut")
        reload_state()
        render_board.refresh()
        render_recut_area.refresh()
        ui.notify("已重新裁剪")
    except Exception as exc:
        ui.notify(f"重切失败：{exc}", type="negative")


def cancel_edit_ui() -> None:
    global edit_session
    if edit_session:
        old_status = edit_session.get("old_status", {})
        for clip in state.clips:
            if clip.clip_id in old_status and clip.status == "editing":
                clip.status = old_status[clip.clip_id]
        save_results(state)
        edit_session = None
        save_edit_session(None)
        reload_state()
        render_recut_area.refresh()
        render_board.refresh()
        ui.notify("已取消编辑")


def render_run_panel() -> None:
    global run_log_textarea, run_status_label
    with ui.card().classes("w-full p-2").style("border-radius:8px"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("运行状态").classes("text-sm font-bold")
            run_status_label = ui.label(run_status_text()).classes("text-xs text-gray-500")
        log_text = "\n".join(run_logs) if run_logs else "暂无日志。"
        run_log_textarea = ui.textarea(value=log_text).props("readonly outlined dense").classes("w-full text-xs").style(
            "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; height: 120px; resize: none; overflow-y: auto;"
        )


def run_status_text() -> str:
    if processing and analysis_started_at:
        elapsed = int((datetime.now() - analysis_started_at).total_seconds())
        idle = int((datetime.now() - last_log_at).total_seconds()) if last_log_at else elapsed
        return f"{progress_text} · 已运行 {elapsed}s · 距最后日志 {idle}s"
    return progress_text


def last_run_had_failures() -> bool:
    for line in reversed(run_logs):
        if "处理结束：" in line:
            return "失败 0" not in line
    return False


def load_token_usage_stats() -> dict[str, Any]:
    stats: dict[str, Any] = {
        "records": 0,
        "with_usage": 0,
        "cached_requests": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0,
        "latest": None,
    }
    if not RAW_RESPONSES_PATH.exists():
        return stats
    with RAW_RESPONSES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["records"] += 1
            if record.get("cached_content_name"):
                stats["cached_requests"] += 1
            usage = record.get("usage_metadata") or {}
            if not isinstance(usage, dict) or not usage:
                continue
            stats["with_usage"] += 1
            stats["prompt_tokens"] += int(usage.get("prompt_token_count") or 0)
            stats["cached_tokens"] += int(usage.get("cached_content_token_count") or 0)
            stats["output_tokens"] += int(usage.get("candidates_token_count") or 0)
            stats["thoughts_tokens"] += int(usage.get("thoughts_token_count") or 0)
            stats["total_tokens"] += int(usage.get("total_token_count") or 0)
            stats["latest"] = {
                "track_id": record.get("track_id", ""),
                "cached": bool(record.get("cached_content_name")),
                "usage": usage,
            }
    return stats


def token_value(value: int) -> str:
    return f"{value:,}" if value else "-"


def load_failed_audio_records() -> list[dict[str, Any]]:
    if not ERRORS_PATH.exists():
        return []
    successful_tracks = {clip.track_id for clip in state.clips if clip.status not in {"hidden", "replaced", "editing"}}
    latest_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    abandoned_by_key: dict[tuple[str, str, str], str] = {}
    with ERRORS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(record.get("kind") or "")
            if kind == "failure_abandoned":
                failed_kind = str(record.get("failed_kind") or "track_failed")
                track_id = str(record.get("track_id") or "")
                source_path = str(record.get("source_audio_path") or "")
                abandoned_by_key[(failed_kind, track_id, source_path)] = str(record.get("time") or "")
                continue
            if kind not in {"track_failed", "segment_failed"}:
                continue
            track_id = str(record.get("track_id") or "")
            if kind == "track_failed" and track_id in successful_tracks:
                continue
            source_path = str(record.get("source_audio_path") or "")
            retryable = kind == "track_failed" and source_path and resolve_project_path(source_path).exists()
            key = (kind, track_id, source_path)
            latest_by_key[key] = {
                "time": str(record.get("time") or ""),
                "failure_kind": kind,
                "kind": "整首失败" if kind == "track_failed" else "片段失败",
                "track_id": track_id or "-",
                "source_audio_path": source_path,
                "source_name": Path(source_path).name if source_path else "-",
                "error": compact_error(str(record.get("error") or "")),
                "retryable": "是" if retryable else "否",
                "retryable_bool": retryable,
            }
    failures = [
        item
        for key, item in latest_by_key.items()
        if abandoned_by_key.get(key, "") < item["time"]
    ]
    return sorted(failures, key=lambda item: item["time"], reverse=True)


def compact_error(error: str, limit: int = 180) -> str:
    cleaned = " ".join(error.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "..."


@ui.refreshable
def render_failure_panel() -> None:
    failures = load_failed_audio_records()
    if not failures:
        return
    selectable: list[tuple[dict[str, Any], Any]] = []

    async def retry_selected() -> None:
        selected = [item for item, checkbox in selectable if checkbox.value and item["retryable_bool"]]
        if not selected:
            ui.notify("请先勾选可重试的整首失败音频。", type="warning")
            return
        await retry_failed_items(selected)

    def abandon_selected() -> None:
        selected = [item for item, checkbox in selectable if checkbox.value]
        if not selected:
            ui.notify("请先勾选要放弃的失败记录。", type="warning")
            return
        abandon_failed_items(selected)

    with ui.card().classes("w-full p-2").style("border-radius:8px"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"失败音频 {len(failures)}").classes("text-sm font-bold text-red-700")
            with ui.row().classes("items-center gap-2"):
                ui.label("勾选后只重试选中的整首失败；不需要的记录可放弃。").classes("text-xs text-gray-500")
                ui.button("重试选中", on_click=retry_selected).props("outline color=warning dense")
                ui.button("放弃选中", on_click=abandon_selected).props("outline dense")
        for item in failures[:12]:
            async def retry_one(item: dict[str, Any] = item) -> None:
                await retry_failed_items([item])

            def abandon_one(item: dict[str, Any] = item) -> None:
                abandon_failed_items([item])

            with ui.column().classes("w-full gap-0 border-t border-gray-200 py-2"):
                with ui.row().classes("items-center gap-2 w-full"):
                    checkbox = ui.checkbox(value=False).props("dense")
                    selectable.append((item, checkbox))
                    ui.label(item["source_name"]).classes("text-sm font-bold")
                    ui.label(item["kind"]).classes("text-xs text-red-700")
                    ui.label(f"track={item['track_id']}").classes("text-xs text-gray-500")
                    ui.label(f"可重试={item['retryable']}").classes("text-xs text-gray-500")
                    ui.label(item["time"]).classes("text-xs text-gray-500")
                    if item["retryable_bool"]:
                        ui.button("重试", on_click=retry_one).props("outline color=warning dense")
                    ui.button("放弃", on_click=abandon_one).props("outline dense")
                ui.label(item["error"] or "未知错误").classes("text-xs text-gray-700")


async def retry_failed_items(items: list[dict[str, Any]]) -> None:
    track_ids = {str(item["track_id"]) for item in items if item.get("retryable_bool") and item.get("track_id") != "-"}
    if not track_ids:
        ui.notify("没有可重试的失败音频。", type="warning")
        return
    add_log(f"准备重试选中的失败音频：{', '.join(sorted(track_ids))}")
    await do_analyze(progress_label_widget, retry_track_ids=track_ids)


def abandon_failed_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        append_jsonl(
            ERRORS_PATH,
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "kind": "failure_abandoned",
                "failed_kind": item.get("failure_kind") or "track_failed",
                "track_id": item.get("track_id") if item.get("track_id") != "-" else "",
                "source_audio_path": item.get("source_audio_path") or "",
                "source_name": item.get("source_name") or "",
            },
        )
    add_log(f"已放弃 {len(items)} 条失败记录。")
    render_failure_panel.refresh()
    ui.notify("已放弃所选失败记录。")


@ui.refreshable
def render_token_usage_panel() -> None:
    stats = load_token_usage_stats()
    latest = stats["latest"] or {}
    latest_usage = latest.get("usage") or {}
    latest_text = "暂无 token usage。"
    if latest_usage:
        latest_parts = [
            f"track={latest.get('track_id') or '-'}",
            f"cache={'是' if latest.get('cached') else '否'}",
            f"prompt={latest_usage.get('prompt_token_count', '-')}",
            f"cached={latest_usage.get('cached_content_token_count', '-')}",
            f"output={latest_usage.get('candidates_token_count', '-')}",
            f"thoughts={latest_usage.get('thoughts_token_count', '-')}",
            f"total={latest_usage.get('total_token_count', '-')}",
        ]
        latest_text = " · ".join(latest_parts)

    with ui.card().classes("w-full p-2").style("border-radius:8px"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Token 统计").classes("text-sm font-bold")
            ui.label(f"响应 {stats['records']} 条 / 有 usage {stats['with_usage']} 条 / cache 请求 {stats['cached_requests']} 条").classes(
                "text-xs text-gray-500"
            )
        with ui.row().classes("gap-3 text-xs text-gray-700"):
            ui.label(f"prompt: {token_value(stats['prompt_tokens'])}")
            ui.label(f"cached: {token_value(stats['cached_tokens'])}")
            ui.label(f"output: {token_value(stats['output_tokens'])}")
            ui.label(f"thoughts: {token_value(stats['thoughts_tokens'])}")
            ui.label(f"total: {token_value(stats['total_tokens'])}")
        ui.label(f"最近一次：{latest_text}").classes("text-xs text-gray-500")
        if stats["records"] and not stats["with_usage"]:
            ui.label("当前 Gemini API/模型没有在响应里返回 token usage；日志和 raw_responses 会在可用时自动记录。").classes(
                "text-xs text-orange-600"
            )


def update_run_widgets() -> None:
    if run_status_label is not None:
        run_status_label.set_text(run_status_text())
    if run_log_textarea is not None:
        run_log_textarea.value = "\n".join(run_logs) if run_logs else "暂无日志。"


def sync_outputs(reason: str = "") -> Path | None:
    try:
        return sync_classified_folders(state, config)
    except Exception as exc:
        suffix = f"（{reason}）" if reason else ""
        add_log(f"分类结果同步失败{suffix}：{exc}")
        return None


def refresh_main_views(reset_runtime: bool = False) -> None:
    render_board.refresh()
    render_recut_area.refresh()
    render_result_controls.refresh()
    render_token_usage_panel.refresh()
    render_failure_panel.refresh()
    if reset_runtime:
        ui.run_javascript("if (window.mcResetRuntimeState) window.mcResetRuntimeState();")


def finalize_current_batch() -> bool:
    global edit_session, state
    clips = visible_clips(state)
    if not clips:
        return False
    if sync_outputs("finalize") is None:
        add_log("完成失败：分类结果文件夹同步失败，页面记录未清空。")
        return False
    missing_outputs = missing_classified_outputs(clips)
    if missing_outputs:
        preview = "、".join(missing_outputs[:6])
        suffix = f" 等 {len(missing_outputs)} 个片段" if len(missing_outputs) > 6 else ""
        add_log(f"完成失败：仍有片段未写入结果文件夹，页面记录未清空：{preview}{suffix}")
        return False
    record_completed_tracks(clips)
    archive_completed_input_files({clip.track_id for clip in clips})
    state = ResultsState()
    save_results(state)
    edit_session = None
    save_edit_session(None)
    waveform_cache.clear()
    add_log(f"已完成并清空页面记录：{len(clips)} 个片段。分类结果文件夹已保留。")
    return True


def missing_classified_outputs(clips) -> list[str]:
    final_dir = resolve_project_path(config.final_output_dir)
    manifest_path = final_dir / ".classified_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        manifest = {}
    missing = []
    for clip in clips:
        entry = manifest.get(clip.clip_id)
        if not isinstance(entry, dict) or not entry.get("target_path"):
            missing.append(clip.display_name or clip.export_filename or clip.clip_id)
            continue
        output_path = final_dir / entry["target_path"]
        if not output_path.exists():
            missing.append(clip.display_name or clip.export_filename or clip.clip_id)
    return missing


def record_completed_tracks(clips) -> None:
    payload = load_completed_tracks()
    tracks = payload.setdefault("tracks", {})
    completed_at = datetime.now().isoformat(timespec="seconds")
    for track_id in sorted({clip.track_id for clip in clips}):
        track_clips = [clip for clip in clips if clip.track_id == track_id]
        source_info = find_existing_source_audio(track_id, track_clips[0])
        source_filename = source_info[1] if source_info else track_clips[0].source_filename
        source_audio_path = source_info[2] if source_info else track_clips[0].source_audio_path
        tracks[track_id] = {
            "track_id": track_id,
            "source_filename": source_filename,
            "source_audio_path": source_audio_path,
            "completed_at": completed_at,
            "clip_count": len(track_clips),
            "labels": sorted({clip.final_label for clip in track_clips if clip.final_label}),
        }
    save_completed_tracks(payload)


def archive_completed_input_files(track_ids: set[str]) -> None:
    raw_root = resolve_project_path(config.raw_audio_dir)
    for audio_path in scan_tracks(config):
        track_id = stable_track_id(audio_path, raw_root)
        if track_id not in track_ids:
            continue
        archive_successful_audio_source(audio_path, raw_root, track_id, config, state, add_log)


def completed_input_track_matches() -> list[dict[str, str]]:
    completed = load_completed_tracks().get("tracks", {})
    if not completed:
        return []
    raw_root = resolve_project_path(config.raw_audio_dir)
    matches = []
    for audio_path in scan_tracks(config):
        track_id = stable_track_id(audio_path, raw_root)
        record = completed.get(track_id)
        if not record:
            continue
        matches.append(
            {
                "track_id": track_id,
                "filename": audio_path.name,
                "completed_at": str(record.get("completed_at") or ""),
                "clip_count": str(record.get("clip_count") or ""),
            }
        )
    return matches


def do_scan() -> None:
    try:
        reload_state()
        refresh_main_views(reset_runtime=True)
        tracks = scan_tracks(config)
        message = f"扫描完成：发现 {len(tracks)} 个音频文件"
        add_log(message)
        for path in tracks[:20]:
            add_log(f"  - {path.name}")
        if len(tracks) > 20:
            add_log(f"  ... 还有 {len(tracks) - 20} 个")
        ui.notify(message)
    except Exception as exc:
        add_log(f"扫描失败：{exc}")
        ui.notify(f"扫描失败：{exc}", type="negative")


@ui.refreshable
def render_result_controls() -> None:
    label_options = ["全部"] + [category.name for category in config.categories if category.name.strip()]
    section_options = ["全部", "verse", "chorus", "unknown"]
    status_options = [
        "默认",
        "全部",
        "confirmed",
        "auto_confirmed",
        "pending_review",
        "hidden",
        "replaced",
    ]
    sort_options = {
        "confidence_desc": "confidence 高到低",
        "confidence_asc": "confidence 低到高",
        "filename": "文件名",
        "time": "片段时间",
        "label": "分类",
        "status": "状态",
    }
    with ui.card().classes("w-full p-2").style("border-radius:8px"):
        with ui.row().classes("items-end gap-2 w-full"):
            search = ui.input("搜索", value=result_filters["search"]).props("outlined dense clearable").classes("min-w-[220px]")
            label_select = ui.select(label_options, label="分类", value=result_filters["label"]).props("outlined dense").classes("min-w-[140px]")
            section_select = ui.select(section_options, label="段落", value=result_filters["section"]).props("outlined dense").classes("min-w-[120px]")
            status_select = ui.select(status_options, label="状态", value=result_filters["status"]).props("outlined dense").classes("min-w-[140px]")
            min_confidence = ui.number("最低分", value=result_filters["min_confidence"], min=0, max=1, step=0.05, format="%.2f").props("outlined dense").classes("w-[110px]")
            needs_review = ui.checkbox("只看待复核", value=result_filters["needs_review"])
            sort_select = ui.select(sort_options, label="排序", value=result_filters["sort"]).props("outlined dense").classes("min-w-[170px]")

            def apply_filters() -> None:
                result_filters["search"] = search.value or ""
                result_filters["label"] = label_select.value or "全部"
                result_filters["section"] = section_select.value or "全部"
                result_filters["status"] = status_select.value or "默认"
                result_filters["min_confidence"] = float(min_confidence.value or 0)
                result_filters["needs_review"] = bool(needs_review.value)
                result_filters["sort"] = sort_select.value or "confidence_desc"
                render_board.refresh()
                ui.notify(f"当前显示 {len(filtered_sorted_clips())} 个片段")

            def reset_filters() -> None:
                result_filters.update(
                    {
                        "search": "",
                        "label": "全部",
                        "section": "全部",
                        "status": "默认",
                        "min_confidence": 0.0,
                        "needs_review": False,
                        "sort": "confidence_desc",
                    }
                )
                render_result_controls.refresh()
                render_board.refresh()

            ui.button("应用", on_click=apply_filters)
            ui.button("重置", on_click=reset_filters).props("outline")
        ui.label(f"当前匹配 {len(filtered_sorted_clips())} 个片段；默认状态隐藏 hidden/replaced。").classes("text-xs text-gray-500")


def clear_data_dialog():
    with ui.dialog() as dialog, ui.card().classes("w-[560px] max-w-full"):
        ui.label("清除分析结果").classes("text-lg font-bold")
        ui.label("只清空当前页面记录和分析日志，不会删除 input、切片、分类结果、导出包或 Gemini 上传代理文件。").classes("text-sm text-gray-600")
        clear_results = ui.checkbox("清空 data/results.json", value=True)
        clear_logs = ui.checkbox("清空 data/raw_responses.jsonl 和 data/errors.jsonl", value=True)
        clear_prompt_cache = ui.checkbox("清除 prompt cache 记录", value=False)
        confirm_text = ui.input("输入 CLEAR 确认").props("outlined dense").classes("w-full")

        def do_clear() -> None:
            if confirm_text.value != "CLEAR":
                ui.notify("请输入 CLEAR 确认清除。", type="warning")
                return
            try:
                if clear_results.value:
                    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    RESULTS_PATH.write_text('{\n  "clips": []\n}\n', encoding="utf-8")
                    save_edit_session(None)
                if clear_logs.value:
                    for path in [RAW_RESPONSES_PATH, ERRORS_PATH]:
                        if path.exists():
                            path.unlink()
                if clear_prompt_cache.value and PROMPT_CACHE_PATH.exists():
                    PROMPT_CACHE_PATH.unlink()
                reload_state()
                run_logs.clear()
                add_log("已清除分析结果，输出文件夹未删除。")
                render_board.refresh()
                render_recut_area.refresh()
                render_result_controls.refresh()
                render_token_usage_panel.refresh()
                render_failure_panel.refresh()
                dialog.close()
                ui.notify("清除完成")
            except Exception as exc:
                ui.notify(f"清除失败：{exc}", type="negative")

        with ui.row().classes("justify-end w-full"):
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("确认清除", on_click=do_clear).props("color=negative")
    return dialog


def settings_dialog():
    with ui.dialog() as dialog, ui.card().classes("w-[760px] max-w-full"):
        ui.label("配置").classes("text-lg font-bold")
        raw_audio = ui.input("原始音频目录", value=config.raw_audio_dir).props("outlined dense").classes("w-full")
        clips_dir = ui.input("片段输出目录", value=config.clips_dir).props("outlined dense").classes("w-full")
        processed_audio_dir = ui.input("已处理原曲归档目录", value=config.processed_audio_dir).props("outlined dense").classes("w-full")
        final_dir = ui.input("最终分类结果目录", value=config.final_output_dir).props("outlined dense").classes("w-full")
        export_dir = ui.input("导出目录", value=config.export_dir).props("outlined dense").classes("w-full")
        downloads_dir = ui.input("ZIP 下载目录", value=config.downloads_dir).props("outlined dense").classes("w-full")
        gemini_uploads_dir = ui.input("Gemini 上传代理目录", value=config.gemini_uploads_dir).props("outlined dense").classes("w-full")
        model = ui.input("Gemini 模型名", value=config.gemini_model).props("outlined dense").classes("w-full")
        timeout = ui.number("Gemini 超时秒数", value=config.gemini_timeout_sec, min=30, step=30, format="%d").props("outlined dense").classes("w-full")
        retry_count = ui.number("Gemini 单曲失败重试次数", value=config.gemini_retry_count, min=0, max=5, step=1, format="%d").props("outlined dense").classes("w-full")
        enable_prompt_cache = ui.checkbox("启用 prompt cache（缓存固定分类规则）", value=config.enable_prompt_cache)
        prompt_cache_ttl = ui.number("Prompt cache TTL 秒", value=config.prompt_cache_ttl_sec, min=300, step=3600, format="%d").props("outlined dense").classes("w-full")
        api_key = ui.input("Gemini API Key", value=get_gemini_api_key(), password=True, password_toggle_button=True).props("outlined dense").classes("w-full")
        model_list = ui.textarea("当前 API Key 可用模型", value="").props("readonly outlined dense autogrow").classes("w-full")
        ui.label("API Key 会保存到本地 .env，不会写入 results.json。分类列可直接在主看板顶部修改。").classes("text-xs text-gray-500")

        async def list_models_click() -> None:
            try:
                models = await asyncio.to_thread(list_available_gemini_models, api_key.value or "", 30)
                model_list.value = "\n".join(models) if models else "没有找到支持 generateContent 的模型。"
                ui.notify(f"找到 {len(models)} 个可用模型")
            except Exception as exc:
                model_list.value = f"获取失败：{exc}"
                ui.notify(f"获取模型失败：{exc}", type="negative")

        def save_settings() -> None:
            config.raw_audio_dir = raw_audio.value
            config.clips_dir = clips_dir.value
            config.processed_audio_dir = processed_audio_dir.value
            config.final_output_dir = final_dir.value
            config.export_dir = export_dir.value
            config.downloads_dir = downloads_dir.value
            config.gemini_uploads_dir = gemini_uploads_dir.value
            config.gemini_model = model.value
            config.gemini_timeout_sec = int(timeout.value or 180)
            config.gemini_retry_count = int(retry_count.value or 0)
            config.enable_prompt_cache = bool(enable_prompt_cache.value)
            config.prompt_cache_ttl_sec = int(prompt_cache_ttl.value or 86400)
            save_gemini_api_key(api_key.value or "")
            save_app_config(config)
            sync_outputs("save_settings")
            dialog.close()
            ui.notify("配置已保存")

        with ui.row().classes("justify-end w-full"):
            ui.button("列出可用模型", on_click=list_models_click).props("outline")
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("保存", on_click=save_settings)
    return dialog


def prepare_analysis_run() -> None:
    global edit_session
    if edit_session:
        old_status = edit_session.get("old_status", {})
        for clip in state.clips:
            if clip.clip_id in old_status and clip.status == "editing":
                clip.status = old_status[clip.clip_id]
        save_results(state)
        edit_session = None
        save_edit_session(None)
    waveform_cache.clear()


async def do_analyze(progress_label=None, force_reanalyze: bool = False, retry_track_ids: set[str] | None = None) -> None:
    global analysis_process, analysis_started_at, processing, progress_text
    if processing:
        return
    processing = True
    prepare_analysis_run()
    refresh_main_views(reset_runtime=True)
    analysis_started_at = datetime.now()
    progress_text = "准备分析..."
    if retry_track_ids:
        add_log(f"开始 Gemini 分析（只重试 {len(retry_track_ids)} 首失败音频）")
    else:
        add_log("开始 Gemini 分析" + ("（重新分析已完成文件）" if force_reanalyze else ""))
    try:
        analysis_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "src.worker",
            cwd=str(ROOT),
            env={
                **os.environ,
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONLEGACYWINDOWSSTDIO": "0",
                "MUSIC_CLASSIFIER_FORCE_REANALYZE": "1" if force_reanalyze else "0",
                "MUSIC_CLASSIFIER_RETRY_TRACK_IDS": json.dumps(sorted(retry_track_ids), ensure_ascii=False) if retry_track_ids else "",
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert analysis_process.stdout is not None
        while True:
            line = await analysis_process.stdout.readline()
            if not line:
                break
            message = line.decode("utf-8", errors="replace").rstrip()
            if message:
                progress_text = message
                add_log(message)
        return_code = await analysis_process.wait()
        reload_state()
        render_board.refresh()
        render_recut_area.refresh()
        render_token_usage_panel.refresh()
        render_failure_panel.refresh()
        if return_code in {-15, 1, 130} and "正在终止分析子进程" in "\n".join(run_logs):
            add_log("分析已中断，Gemini 请求已随子进程终止。")
            ui.notify("分析已中断")
        elif return_code != 0:
            if "Gemini 请求超时" in "\n".join(run_logs):
                add_log("分析失败：Gemini 请求已超时断开。")
                ui.notify("Gemini 请求已超时断开", type="negative")
            else:
                add_log(f"分析子进程异常退出，退出码 {return_code}")
                ui.notify(f"分析失败，退出码 {return_code}", type="negative")
        else:
            if last_run_had_failures():
                add_log("Gemini 分析完成，但有歌曲失败；请查看失败音频面板，可点击“重试失败音频”。")
                ui.notify("分析完成，但有歌曲失败", type="warning")
            else:
                add_log("Gemini 分析完成")
                ui.notify("Gemini 分析完成")
        ui.run_javascript("setTimeout(() => window.location.reload(), 500)")
    except Exception as exc:
        add_log(f"分析失败：{exc}")
        ui.notify(f"分析失败：{exc}", type="negative")
        ui.run_javascript("setTimeout(() => window.location.reload(), 800)")
    finally:
        analysis_process = None
        analysis_started_at = None
        processing = False
        progress_text = "就绪"
        if progress_label is not None:
            progress_label.text = progress_text


async def prompt_or_start_analyze(progress_label) -> None:
    matches = completed_input_track_matches()
    if not matches:
        await do_analyze(progress_label)
        return
    names = "\n".join(
        f"- {item['filename']}（上次完成 {item['completed_at'] or '-'}，片段 {item['clip_count'] or '-'}）"
        for item in matches[:12]
    )
    if len(matches) > 12:
        names += f"\n... 还有 {len(matches) - 12} 个"

    with ui.dialog() as dialog, ui.card().classes("w-[560px] max-w-full"):
        ui.label("发现已完成文件").classes("text-lg font-bold")
        ui.label("input 里有已经完成过的音频。可以跳过并清理 input，也可以重新分析。").classes("text-sm text-gray-600")
        ui.textarea(value=names).props("readonly outlined dense").classes("w-full text-xs").style(
            "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; height: 140px;"
        )

        async def skip_completed() -> None:
            dialog.close()
            await do_analyze(progress_label, force_reanalyze=False)

        async def reanalyze_completed() -> None:
            dialog.close()
            await do_analyze(progress_label, force_reanalyze=True)

        with ui.row().classes("justify-end w-full"):
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("跳过并清理 input", on_click=skip_completed).props("outline")
            ui.button("重新分析", on_click=reanalyze_completed)
    dialog.open()


async def request_stop() -> None:
    global analysis_process
    if not processing:
        ui.notify("当前没有正在运行的分析")
        return
    if analysis_process is None or analysis_process.returncode is not None:
        ui.notify("分析进程已经结束")
        return
    add_log("正在终止分析子进程，Gemini 请求会一起断开。")
    ui.notify("正在中断分析...", type="warning")
    analysis_process.terminate()
    try:
        await asyncio.wait_for(analysis_process.wait(), timeout=5)
    except asyncio.TimeoutError:
        add_log("子进程未及时退出，强制结束。")
        analysis_process.kill()
        await analysis_process.wait()


def main() -> None:
    global progress_label_widget
    add_styles()
    sync_outputs("startup")
    dialog = settings_dialog()
    clear_dialog = clear_data_dialog()
    with ui.header().classes("items-center justify-between bg-white text-gray-900 border-b"):
        ui.label("本地音频分类工作台").classes("text-base font-bold")
        with ui.row().classes("items-center gap-2"):
            progress_label = ui.label(progress_text).classes("text-xs text-gray-500")
            progress_label_widget = progress_label
            ui.timer(0.5, lambda: (progress_label.set_text(progress_text), update_run_widgets()))
            ui.timer(3.0, render_token_usage_panel.refresh)
            ui.timer(3.0, lambda: None if processing else render_failure_panel.refresh())
            async def start_analyze_click() -> None:
                await prompt_or_start_analyze(progress_label)

            ui.button("扫描音频", on_click=do_scan).props("outline")
            ui.button("开始 Gemini 分析", on_click=start_analyze_click)
            ui.button("重试失败音频", on_click=start_analyze_click).props("outline color=warning")
            ui.button("中断分析", on_click=request_stop).props("outline color=negative")
            ui.button("保存当前结果", on_click=lambda: (save_results(state), ui.notify("结果已保存"))).props("outline")
            ui.button("导出 CSV", on_click=do_export_csv).props("outline")
            ui.button("生成 ZIP", on_click=do_zip).props("outline")
            ui.button("清除分析结果", on_click=clear_dialog.open).props("outline color=negative")
            ui.button(icon="settings", on_click=dialog.open).props("flat round")

    with ui.column().classes("w-full p-4 gap-4"):
        render_run_panel()
        render_failure_panel()
        render_token_usage_panel()
        render_recut_area()
        render_result_controls()
        render_board()
        with ui.row().classes("w-full justify-end"):
            ui.button("打开结果位置", on_click=do_open_final).props("outline")


def do_export_csv() -> None:
    try:
        path = export_csv(state, config)
        ui.notify(f"CSV 已导出：{path}")
    except Exception as exc:
        ui.notify(f"导出失败：{exc}", type="negative")


def do_zip() -> None:
    try:
        path = build_zip(state, config)
        ui.notify(f"ZIP 已生成：{path}")
        ui.download(str(path))
    except Exception as exc:
        ui.notify(f"生成 ZIP 失败：{exc}", type="negative")


def do_open_final() -> None:
    path = resolve_project_path(config.final_output_dir)
    ok, message = open_folder(path)
    ui.notify(f"已打开：{message}" if ok else f"请手动打开：{message}", type="positive" if ok else "warning")


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title="本地音频分类工作台", reload=False)
