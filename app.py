from __future__ import annotations

import asyncio
import html
import json
import shutil
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Header, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from nicegui import app, ui

from src.audio_utils import seconds_to_mmss
from src.config import (
    ERRORS_PATH,
    PROMPT_CACHE_PATH,
    RAW_RESPONSES_PATH,
    ROOT,
    RESULTS_PATH,
    get_gemini_api_key,
    load_app_config,
    load_results,
    resolve_project_path,
    save_app_config,
    save_gemini_api_key,
    save_results,
)
from src.export_utils import build_classified_folders, build_zip, export_csv, open_folder, visible_clips
from src.gemini_client import list_available_gemini_models
from src.models import AppConfig, ResultsState
from src.pipeline import (
    add_category,
    find_clip,
    prune_empty_categories,
    recut_clip,
    scan_tracks,
    update_clip_display_name,
    update_clip_label,
)


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
EDIT_SESSION_PATH = ROOT / "data" / "edit_session.json"
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
    if EDIT_SESSION_PATH.exists():
        try:
            return json.loads(EDIT_SESSION_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return rebuild_edit_session_from_state()


def save_edit_session(session: dict | None) -> None:
    EDIT_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    if session is None:
        if EDIT_SESSION_PATH.exists():
            EDIT_SESSION_PATH.unlink()
        return
    EDIT_SESSION_PATH.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_edit_session_from_state() -> dict | None:
    editing_clips = [clip for clip in state.clips if clip.status == "editing"]
    if not editing_clips:
        return None
    track_id = editing_clips[0].track_id
    track_clips = [clip for clip in editing_clips if clip.track_id == track_id]
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
        "source_filename": track_clips[0].source_filename,
        "source_audio_path": track_clips[0].source_audio_path,
        "old_status": {clip.clip_id: "confirmed" for clip in track_clips},
        "regions": regions,
    }


edit_session = load_edit_session()


@app.get("/media/{clip_id}")
def media(clip_id: str, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    clip = find_clip(state, clip_id)
    return audio_response(resolve_project_path(clip.clip_path), clip.export_filename, range_header)


@app.get("/api/debug_clip/{clip_id}")
def debug_clip(clip_id: str) -> JSONResponse:
    clip = find_clip(state, clip_id)
    path = resolve_project_path(clip.clip_path)
    return JSONResponse(
        {
            "clip_id": clip_id,
            "clip_path": clip.clip_path,
            "resolved_path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else None,
        }
    )


@app.get("/source/{clip_id}")
def source_media(clip_id: str, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    clip = find_clip(state, clip_id)
    return audio_response(resolve_project_path(clip.source_audio_path), clip.source_filename, range_header)


@app.get("/edit/source/{track_id}")
def edit_source_media(track_id: str, range_header: str | None = Header(default=None, alias="Range")) -> Response:
    source_clip = next((clip for clip in state.clips if clip.track_id == track_id), None)
    if source_clip is None:
        raise KeyError(f"找不到原曲：{track_id}")
    return audio_response(resolve_project_path(source_clip.source_audio_path), source_clip.source_filename, range_header)


def audio_response(path: Path, filename: str, range_header: str | None = None) -> Response:
    path = path.expanduser()
    file_size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
    }
    if not range_header:
        with path.open("rb") as handle:
            data = handle.read()
        headers["Content-Length"] = str(file_size)
        return Response(content=data, status_code=200, media_type="audio/wav", headers=headers)

    start = 0
    end = file_size - 1
    range_value = range_header.replace("bytes=", "").strip()
    try:
        if "-" in range_value:
            start_text, end_text = range_value.split("-", 1)
            if start_text:
                start = int(start_text)
            if end_text:
                end = int(end_text)
    except ValueError:
        return Response(status_code=416, headers=headers)
    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    length = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(length)
    headers.update(
        {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        }
    )
    return Response(content=data, status_code=206, media_type="audio/wav", headers=headers)


@app.post("/api/move_clip")
async def move_clip(request: Request) -> JSONResponse:
    payload = await request.json()
    update_clip_label(state, payload["clip_id"], payload["label"])
    return JSONResponse({"ok": True})


@app.post("/api/rename_clip")
async def rename_clip(request: Request) -> JSONResponse:
    payload = await request.json()
    update_clip_display_name(state, payload["clip_id"], payload["display_name"])
    return JSONResponse({"ok": True})


@app.post("/api/batch_update_clips")
async def batch_update_clips(request: Request) -> JSONResponse:
    payload = await request.json()
    clip_ids = set(payload.get("clip_ids", []))
    action = payload.get("action", "")
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
    return JSONResponse({"ok": True, "updated": updated})


@app.post("/api/start_edit_clip")
async def start_edit_clip(request: Request) -> JSONResponse:
    global edit_session
    payload = await request.json()
    clip = find_clip(state, payload["clip_id"])
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
        "source_filename": clip.source_filename,
        "source_audio_path": clip.source_audio_path,
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


def board_html() -> str:
    grouped = visible_by_label()
    columns = []
    for category in config.categories:
        if not category.name.strip() and not category.description.strip():
            title = ""
        else:
            title = category.name
        clips = grouped.get(category.name, []) if category.name else []
        card_rows = "\n".join(card_html(clip) for clip in clips)
        columns.append(
            f"""
            <section class="kanban-column" data-label="{escape_attr(category.name)}"
              ondragover="event.preventDefault()"
              ondrop="mcDropClip(event, {js_str(category.name)})">
              <div class="category-head">
                <input class="category-name" value="{escape_attr(title)}"
                  placeholder="分类名"
                  onchange="mcUpdateCategory({js_str(category.id)}, this.value, this.parentElement.querySelector('textarea').value)">
                <textarea class="category-desc" placeholder="分类描述"
                  onchange="mcUpdateCategory({js_str(category.id)}, this.parentElement.querySelector('input').value, this.value)">{html.escape(category.description)}</textarea>
              </div>
              <div class="clip-list">{card_rows}</div>
            </section>
            """
        )

    for label, clips in grouped.items():
        if label in {category.name for category in config.categories}:
            continue
        card_rows = "\n".join(card_html(clip) for clip in clips)
        columns.append(
            f"""
            <section class="kanban-column" data-label="{escape_attr(label)}"
              ondragover="event.preventDefault()"
              ondrop="mcDropClip(event, {js_str(label)})">
              <div class="category-head readonly">
                <input class="category-name" value="{escape_attr(label)}" readonly>
                <textarea class="category-desc" readonly>当前配置外的分类，建议在配置中确认。</textarea>
              </div>
              <div class="clip-list">{card_rows}</div>
            </section>
            """
        )

    column_count = max(1, len(columns))
    visible_columns = min(5, column_count)
    return f"""
    <div class="board-shell">
      <div class="board-actions">
        <button class="batch-button" onclick="mcSelectVisible(true)">全选可见</button>
        <button class="batch-button" onclick="mcSelectVisible(false)">取消选择</button>
        <button class="batch-button" onclick="mcBatchUpdate('confirm')">批量确认</button>
        <button class="batch-button" onclick="mcBatchUpdate('review')">标记待复核</button>
        <button class="batch-button danger" onclick="mcBatchUpdate('hide')">隐藏</button>
        <button class="add-column" onclick="mcAddColumn()" title="添加分类列">+</button>
      </div>
      <div class="kanban-wrap" style="--visible-columns: {visible_columns};">
        {"".join(columns)}
      </div>
    </div>
    """


def card_html(clip) -> str:
    duration = seconds_to_mmss(clip.duration_sec)
    original = js_str(clip.display_name)
    confidence_text = f"{clip.confidence:.2f}"
    return f"""
    <div class="clip-row" data-clip-id="{escape_attr(clip.clip_id)}" draggable="true" ondragstart="mcDragClip(event, {js_str(clip.clip_id)})">
      <input class="clip-check" type="checkbox" data-clip-id="{escape_attr(clip.clip_id)}" onclick="event.stopPropagation()">
      <button class="play-btn" onclick="mcPlay({js_str(clip.clip_id)})">▶</button>
      <button class="edit-btn" onclick="mcStartEditClip({js_str(clip.clip_id)})" title="送到上方编辑区">✎</button>
      <input class="clip-name" value="{escape_attr(clip.display_name)}"
        onkeydown="mcInputKey(event, {original})"
        onblur="mcRenameClip({js_str(clip.clip_id)}, this.value)">
      <span class="clip-duration">{duration}</span>
      <span class="confidence" style="{confidence_style(clip.confidence)}" title="Gemini confidence">{confidence_text}</span>
    </div>
    """


def confidence_style(value: float) -> str:
    value = max(0.0, min(1.0, value))
    red = (220, 38, 38)
    blue = (37, 99, 235)
    rgb = tuple(round(red[index] + (blue[index] - red[index]) * value) for index in range(3))
    return f"color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); font-weight: 700;"


def escape_attr(value: str) -> str:
    return html.escape(value, quote=True)


def js_str(value: str) -> str:
    return repr(value)


@app.post("/api/add_category")
def api_add_category() -> JSONResponse:
    add_category(config)
    return JSONResponse({"ok": True})


def add_styles() -> None:
    ui.add_head_html(
        """
        <style>
          body { background: #f6f7f8; }
          .board-shell {
            display: grid;
            gap: 8px;
          }
          .board-actions {
            display: flex;
            gap: 8px;
            justify-content: flex-end;
            flex-wrap: wrap;
            padding-right: 2px;
          }
          .kanban-wrap {
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding: 4px 2px 16px;
            min-height: 58vh;
            scrollbar-gutter: stable;
          }
          .kanban-column {
            flex: 0 0 calc((100% - (var(--visible-columns) - 1) * 12px) / var(--visible-columns));
            min-width: 220px;
            background: #ffffff;
            border: 1px solid #dde1e5;
            border-radius: 8px;
            padding: 10px;
            min-height: 360px;
          }
          .category-head {
            display: grid;
            gap: 6px;
            margin-bottom: 10px;
          }
          .category-name, .category-desc, .clip-name {
            width: 100%;
            border: 1px solid #d8dde3;
            border-radius: 6px;
            color: #1f2933;
            background: #fff;
            outline: none;
          }
          .category-name { height: 32px; padding: 0 8px; font-weight: 700; }
          .category-desc { min-height: 76px; padding: 7px 8px; resize: vertical; font-size: 12px; line-height: 1.35; }
          .clip-list { display: grid; gap: 6px; }
          .clip-row {
            display: grid;
            grid-template-columns: 18px 28px 28px minmax(0, 1fr) 48px 44px;
            align-items: center;
            gap: 8px;
            height: 38px;
            padding: 0 8px;
            background: #fafafa;
            border: 1px solid #e4e7eb;
            border-radius: 6px;
            cursor: grab;
          }
          .clip-row:active { cursor: grabbing; }
          .clip-check {
            width: 14px;
            height: 14px;
            accent-color: #2563eb;
          }
          .clip-row.is-playing {
            background: #e8f1ff;
            border-color: #2563eb;
            box-shadow: inset 3px 0 0 #2563eb;
          }
          .clip-row.is-playing .clip-name,
          .clip-row.is-playing .clip-duration,
          .clip-row.is-playing .confidence {
            color: #123c7c;
          }
          .play-btn {
            width: 26px;
            height: 26px;
            border: 1px solid #cbd2d9;
            border-radius: 50%;
            background: #fff;
            color: #243b53;
            line-height: 1;
          }
          .edit-btn {
            width: 26px;
            height: 26px;
            border: 1px solid #cbd2d9;
            border-radius: 50%;
            background: #fff;
            color: #52606d;
            line-height: 1;
            font-size: 13px;
          }
          .edit-btn:hover {
            border-color: #2563eb;
            color: #2563eb;
          }
          .clip-row.is-playing .play-btn {
            border-color: #2563eb;
            background: #2563eb;
            color: #fff;
          }
          .clip-name {
            border: none;
            background: transparent;
            height: 28px;
            padding: 0;
            font-size: 13px;
          }
          .clip-name:focus {
            background: #fff;
            border: 1px solid #b8c4d0;
            padding: 0 6px;
          }
          .clip-duration, .confidence {
            font-size: 12px;
            color: #52606d;
            white-space: nowrap;
          }
          .add-column {
            height: 44px;
            width: 44px;
            border: 1px solid #cbd2d9;
            border-radius: 8px;
            background: #fff;
            font-size: 26px;
            color: #334e68;
          }
          .batch-button {
            height: 32px;
            border: 1px solid #cbd2d9;
            border-radius: 6px;
            background: #fff;
            color: #334e68;
            padding: 0 10px;
            font-size: 12px;
          }
          .batch-button.danger {
            color: #b42318;
            border-color: #f2b8b5;
          }
          .edit-drop-zone {
            height: 96px;
            border: 1px dashed #9aa5b1;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: #52606d;
            background: #fbfcfd;
            font-size: 13px;
          }
          .wave-editor {
            display: grid;
            gap: 10px;
            width: 100%;
          }
          #mc-edit-audio {
            width: 100%;
            height: 36px;
          }
          .wave-stage {
            position: relative;
            width: 100%;
            height: 220px;
            border: 1px solid #d8dde3;
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
          }
          #mc-wave-canvas {
            width: 100%;
            height: 100%;
            display: block;
          }
          #mc-region-layer {
            position: absolute;
            inset: 0;
          }
          .edit-region {
            position: absolute;
            top: 16px;
            height: 188px;
            border: 1px solid rgba(37, 99, 235, 0.9);
            background: rgba(37, 99, 235, 0.16);
            border-radius: 6px;
            cursor: move;
          }
          .edit-region .region-handle {
            position: absolute;
            top: 0;
            width: 8px;
            height: 100%;
            background: rgba(37, 99, 235, 0.75);
            cursor: ew-resize;
          }
          .edit-region .region-handle.left { left: 0; border-radius: 5px 0 0 5px; }
          .edit-region .region-handle.right { right: 0; border-radius: 0 5px 5px 0; }
          .edit-region .region-label {
            position: absolute;
            left: 10px;
            top: 8px;
            right: 10px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            color: #123c7c;
            font-size: 12px;
            font-weight: 700;
            pointer-events: none;
          }
          .region-table {
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 4px;
          }
          .region-card {
            flex: 1 0 180px;
            min-width: 180px;
            border: 1px solid #d8dde3;
            border-radius: 8px;
            background: #fff;
            padding: 8px;
            display: grid;
            gap: 6px;
            font-size: 12px;
          }
          .region-card.is-muted {
            opacity: 0.45;
          }
          .region-card-head {
            display: grid;
            grid-template-columns: 18px minmax(0, 1fr);
            gap: 6px;
            align-items: center;
          }
          .region-card input,
          .region-card select {
            height: 28px;
            border: 1px solid #d8dde3;
            border-radius: 6px;
            padding: 0 6px;
            min-width: 0;
          }
          .region-time-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
          }
          .region-category-row {
            display: grid;
            grid-template-columns: 12px minmax(0, 1fr);
            gap: 6px;
            align-items: center;
          }
          .region-color-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
          }
          @media (max-width: 900px) {
            .kanban-column {
              flex-basis: min(86vw, 360px);
            }
          }
        </style>
        """
    )
    ui.add_body_html(
        """
        <script>
          function mcDropClip(event, label) {
            event.preventDefault();
            const clipId = event.dataTransfer.getData('text/plain');
            fetch('/api/move_clip', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({clip_id: clipId, label})
            }).then(() => window.location.reload());
          }
          function mcDragClip(event, clipId) {
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/plain', clipId);
            event.dataTransfer.setData('application/x-clip-id', clipId);
          }
          function mcRenameClip(clipId, value) {
            fetch('/api/rename_clip', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({clip_id: clipId, display_name: value})
            });
          }
          function mcUpdateCategory(categoryId, name, description) {
            fetch('/api/update_category', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({category_id: categoryId, name, description})
            }).then(() => setTimeout(() => window.location.reload(), 120));
          }
          function mcPlay(clipId) {
            const url = '/media/' + encodeURIComponent(clipId);
            let audio = document.getElementById('mc-global-audio');
            if (!audio) {
              audio = document.createElement('audio');
              audio.id = 'mc-global-audio';
              audio.style.display = 'none';
              audio.addEventListener('ended', mcClearPlaying);
              audio.addEventListener('pause', () => {
                if (audio.dataset.manualPause === 'true') mcClearPlaying();
              });
              document.body.appendChild(audio);
            }
            const escapedClipId = window.CSS && CSS.escape ? CSS.escape(clipId) : clipId.replace(/"/g, '\\"');
            const row = document.querySelector(`.clip-row[data-clip-id="${escapedClipId}"]`);
            if (audio.dataset.clipId === clipId && !audio.paused) {
              audio.dataset.manualPause = 'true';
              audio.pause();
              return;
            }
            mcClearPlaying();
            audio.dataset.manualPause = 'false';
            audio.dataset.clipId = clipId;
            audio.onerror = () => alert('音频加载失败，请直接打开 ' + url + ' 检查接口。');
            audio.src = url + '?t=' + Date.now();
            audio.play()
              .then(() => {
                if (row) {
                  row.classList.add('is-playing');
                  const button = row.querySelector('.play-btn');
                  if (button) button.textContent = '❚❚';
                }
              })
              .catch(error => alert('播放失败：' + error.message));
          }
          function mcClearPlaying() {
            document.querySelectorAll('.clip-row.is-playing').forEach(row => {
              row.classList.remove('is-playing');
              const button = row.querySelector('.play-btn');
              if (button) button.textContent = '▶';
            });
            const audio = document.getElementById('mc-global-audio');
            if (audio) audio.dataset.clipId = '';
          }
          function mcAddColumn() {
            fetch('/api/add_category', {method: 'POST'}).then(() => window.location.reload());
          }
          function mcInputKey(event, original) {
            if (event.key === 'Enter') event.target.blur();
            if (event.key === 'Escape') {
              event.target.value = original;
              event.target.blur();
            }
          }
          function mcSelectedClipIds() {
            return Array.from(document.querySelectorAll('.clip-check:checked')).map(input => input.dataset.clipId);
          }
          function mcSelectVisible(checked) {
            document.querySelectorAll('.clip-check').forEach(input => input.checked = checked);
          }
          function mcBatchUpdate(action) {
            const clipIds = mcSelectedClipIds();
            if (clipIds.length === 0) {
              alert('请先勾选片段。');
              return;
            }
            fetch('/api/batch_update_clips', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({clip_ids: clipIds, action})
            }).then(() => window.location.reload());
          }
          function mcDropToEditor(event) {
            event.preventDefault();
            const clipId = event.dataTransfer.getData('application/x-clip-id') || event.dataTransfer.getData('text/plain');
            if (!clipId) return;
            mcStartEditClip(clipId);
          }
          function mcStartEditClip(clipId) {
            fetch('/api/start_edit_clip', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({clip_id: clipId})
            }).then(response => {
              if (!response.ok) throw new Error('HTTP ' + response.status);
              return response.json();
            }).then(() => window.location.reload())
              .catch(error => alert('进入编辑区失败：' + error.message));
          }
          async function mcInitWaveEditor() {
            const root = document.getElementById('mc-editor-root');
            if (!root || root.dataset.ready === 'true') return;
            root.dataset.ready = 'true';
            const audio = document.getElementById('mc-edit-audio');
            const canvas = document.getElementById('mc-wave-canvas');
            const layer = document.getElementById('mc-region-layer');
            const table = document.getElementById('mc-region-table');
            const regions = JSON.parse(root.dataset.regions || '[]');
            const categories = JSON.parse(root.dataset.categories || '[]');
            window.mcEditRegions = regions;
            window.mcEditCategories = categories;

            const syncDuration = () => {
              const duration = audio.duration || Math.max(...regions.map(r => Number(r.end_sec || 0)), 1);
              root.dataset.duration = String(duration);
              mcRenderRegions();
              mcRenderRegionTable();
            };
            audio.addEventListener('loadedmetadata', syncDuration);
            if (audio.readyState >= 1) syncDuration();

            try {
              const response = await fetch(root.dataset.sourceUrl);
              const buffer = await response.arrayBuffer();
              const audioContext = new (window.AudioContext || window.webkitAudioContext)();
              const audioBuffer = await audioContext.decodeAudioData(buffer.slice(0));
              mcDrawWaveform(canvas, audioBuffer);
              if (!audio.duration) {
                root.dataset.duration = String(audioBuffer.duration);
                mcRenderRegions();
                mcRenderRegionTable();
              }
              audioContext.close();
            } catch (error) {
              const ctx = canvas.getContext('2d');
              ctx.font = '13px sans-serif';
              ctx.fillStyle = '#b42318';
              ctx.fillText('波形加载失败：' + error.message, 16, 32);
            }

            window.addEventListener('resize', () => {
              if (window.mcEditRegions) mcRenderRegions();
            });
          }
          function mcDrawWaveform(canvas, audioBuffer) {
            const rect = canvas.getBoundingClientRect();
            const width = Math.max(600, Math.floor(rect.width * window.devicePixelRatio));
            const height = Math.max(120, Math.floor(rect.height * window.devicePixelRatio));
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);
            ctx.strokeStyle = '#d8dde3';
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.stroke();
            const data = audioBuffer.getChannelData(0);
            const step = Math.ceil(data.length / width);
            ctx.strokeStyle = '#52606d';
            ctx.beginPath();
            for (let x = 0; x < width; x++) {
              let min = 1;
              let max = -1;
              const start = x * step;
              const end = Math.min(start + step, data.length);
              for (let i = start; i < end; i++) {
                const value = data[i];
                if (value < min) min = value;
                if (value > max) max = value;
              }
              ctx.moveTo(x, (1 + min) * height / 2);
              ctx.lineTo(x, (1 + max) * height / 2);
            }
            ctx.stroke();
          }
          function mcRenderRegions() {
            const root = document.getElementById('mc-editor-root');
            const layer = document.getElementById('mc-region-layer');
            if (!root || !layer || !window.mcEditRegions) return;
            const duration = Number(root.dataset.duration || 1);
            layer.innerHTML = '';
            window.mcEditRegions.forEach((region, index) => {
              const left = Math.max(0, Number(region.start_sec || 0) / duration * 100);
              const right = Math.min(100, Number(region.end_sec || 0) / duration * 100);
              const el = document.createElement('div');
              el.className = 'edit-region';
              el.dataset.index = index;
              el.style.left = left + '%';
              el.style.width = Math.max(0.5, right - left) + '%';
              const color = mcRegionColor(region.label, 0.9);
              el.style.borderColor = color;
              el.style.background = mcRegionColor(region.label, 0.18);
              el.innerHTML = `<div class="region-handle left"></div><div class="region-label">${region.display_name || region.clip_id}</div><div class="region-handle right"></div>`;
              el.querySelectorAll('.region-handle').forEach(handle => handle.style.background = color);
              mcAttachRegionDrag(el, region);
              layer.appendChild(el);
            });
          }
          function mcRegionColor(label, alpha) {
            const palette = ['37,99,235', '14,165,163', '124,58,237', '217,119,6', '5,150,105', '219,39,119'];
            const categories = window.mcEditCategories || [];
            const index = Math.max(0, categories.findIndex(category => category.name === label));
            const rgb = palette[index % palette.length];
            return `rgba(${rgb}, ${alpha})`;
          }
          function mcAttachRegionDrag(el, region) {
            let mode = 'move';
            let startX = 0;
            let originalStart = 0;
            let originalEnd = 0;
            el.querySelector('.left').addEventListener('pointerdown', event => { mode = 'left'; begin(event); });
            el.querySelector('.right').addEventListener('pointerdown', event => { mode = 'right'; begin(event); });
            el.addEventListener('pointerdown', event => { if (!event.target.classList.contains('region-handle')) { mode = 'move'; begin(event); } });
            function begin(event) {
              event.preventDefault();
              startX = event.clientX;
              originalStart = Number(region.start_sec);
              originalEnd = Number(region.end_sec);
              el.setPointerCapture(event.pointerId);
              el.addEventListener('pointermove', move);
              el.addEventListener('pointerup', end);
            }
            function move(event) {
              const root = document.getElementById('mc-editor-root');
              const duration = Number(root.dataset.duration || 1);
              const stage = document.querySelector('.wave-stage');
              const delta = (event.clientX - startX) / stage.clientWidth * duration;
              if (mode === 'left') {
                region.start_sec = Math.max(0, Math.min(originalEnd - 1, originalStart + delta));
              } else if (mode === 'right') {
                region.end_sec = Math.min(duration, Math.max(originalStart + 1, originalEnd + delta));
              } else {
                const length = originalEnd - originalStart;
                const nextStart = Math.max(0, Math.min(duration - length, originalStart + delta));
                region.start_sec = nextStart;
                region.end_sec = nextStart + length;
              }
              region.start_sec = Math.round(region.start_sec * 10) / 10;
              region.end_sec = Math.round(region.end_sec * 10) / 10;
              mcRenderRegions();
              mcRenderRegionTable();
            }
            function end(event) {
              el.releasePointerCapture(event.pointerId);
              el.removeEventListener('pointermove', move);
              el.removeEventListener('pointerup', end);
            }
          }
          function mcRenderRegionTable() {
            const table = document.getElementById('mc-region-table');
            if (!table || !window.mcEditRegions) return;
            table.innerHTML = '';
            window.mcEditRegions.forEach((region, index) => {
              if (region.selected === undefined) region.selected = true;
              const card = document.createElement('div');
              card.className = 'region-card' + (region.selected ? '' : ' is-muted');
              card.style.borderColor = mcRegionColor(region.label, 0.55);
              const options = (window.mcEditCategories || []).map(category => {
                const selected = category.name === region.label ? 'selected' : '';
                return `<option value="${category.name}" ${selected}>${category.name}</option>`;
              }).join('');
              card.innerHTML = `
                <div class="region-card-head">
                  <input type="checkbox" ${region.selected ? 'checked' : ''} onchange="window.mcEditRegions[${index}].selected=this.checked; mcRenderRegionTable();">
                  <input value="${region.display_name || region.clip_id}" onchange="window.mcEditRegions[${index}].display_name=this.value; mcRenderRegions();">
                </div>
                <div class="region-time-row">
                  <input type="number" step="0.1" value="${Number(region.start_sec).toFixed(1)}" onchange="window.mcEditRegions[${index}].start_sec=Number(this.value); mcRenderRegions();">
                  <input type="number" step="0.1" value="${Number(region.end_sec).toFixed(1)}" onchange="window.mcEditRegions[${index}].end_sec=Number(this.value); mcRenderRegions();">
                </div>
                <div class="region-category-row">
                  <span class="region-color-dot" style="background:${mcRegionColor(region.label, 0.9)}"></span>
                  <select onchange="window.mcEditRegions[${index}].label=this.value; mcRenderRegions(); mcRenderRegionTable();">${options}</select>
                </div>
              `;
              table.appendChild(card);
            });
          }
          function mcCommitEditRegions() {
            if (!window.mcEditRegions || window.mcEditRegions.length === 0) {
              alert('没有可裁剪的片段。');
              return;
            }
            const selectedRegions = window.mcEditRegions.filter(region => region.selected !== false);
            if (selectedRegions.length === 0) {
              alert('请至少勾选一个片段。');
              return;
            }
            fetch('/api/commit_edit_regions', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({regions: selectedRegions})
            }).then(response => {
              if (!response.ok) throw new Error('HTTP ' + response.status);
              return response.json();
            }).then(data => {
              alert('已生成 ' + data.created + ' 个新片段。');
              window.location.reload();
            }).catch(error => alert('重切失败：' + error.message));
          }
        </script>
        """
    )


@ui.refreshable
def render_board() -> None:
    ui.html(board_html(), sanitize=False).classes("w-full")


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
        source_url = f"/edit/source/{track_id}"
        ui.label(f"{edit_session['source_filename']} · {track_id}").classes("text-xs text-gray-500")
        ui.html(
            f"""
            <div id="mc-editor-root" class="wave-editor"
              data-track-id="{escape_attr(track_id)}"
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


def do_scan() -> None:
    try:
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
        ui.label("清除测试结果").classes("text-lg font-bold")
        ui.label("默认只清空结果和日志，不会删除原始音频。勾选输出目录后，会删除已生成的切片、导出文件和 Gemini 上传代理。").classes("text-sm text-gray-600")
        clear_results = ui.checkbox("清空 data/results.json", value=True)
        clear_logs = ui.checkbox("清空 data/raw_responses.jsonl 和 data/errors.jsonl", value=True)
        clear_prompt_cache = ui.checkbox("清除 prompt cache 记录", value=False)
        clear_outputs = ui.checkbox("删除 clips/final/export/gemini_uploads 目录内容", value=False)
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
                if clear_outputs.value:
                    for path_value in [
                        config.clips_dir,
                        config.final_output_dir,
                        config.export_dir,
                        config.gemini_uploads_dir,
                    ]:
                        directory = resolve_project_path(path_value)
                        if directory.exists():
                            shutil.rmtree(directory)
                        directory.mkdir(parents=True, exist_ok=True)
                reload_state()
                run_logs.clear()
                add_log("已清除测试结果。")
                render_board.refresh()
                render_recut_area.refresh()
                render_result_controls.refresh()
                render_token_usage_panel.refresh()
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
        final_dir = ui.input("最终分类结果目录", value=config.final_output_dir).props("outlined dense").classes("w-full")
        export_dir = ui.input("导出目录", value=config.export_dir).props("outlined dense").classes("w-full")
        downloads_dir = ui.input("ZIP 下载目录", value=config.downloads_dir).props("outlined dense").classes("w-full")
        gemini_uploads_dir = ui.input("Gemini 上传代理目录", value=config.gemini_uploads_dir).props("outlined dense").classes("w-full")
        model = ui.input("Gemini 模型名", value=config.gemini_model).props("outlined dense").classes("w-full")
        timeout = ui.number("Gemini 超时秒数", value=config.gemini_timeout_sec, min=30, step=30, format="%d").props("outlined dense").classes("w-full")
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
            config.final_output_dir = final_dir.value
            config.export_dir = export_dir.value
            config.downloads_dir = downloads_dir.value
            config.gemini_uploads_dir = gemini_uploads_dir.value
            config.gemini_model = model.value
            config.gemini_timeout_sec = int(timeout.value or 180)
            config.enable_prompt_cache = bool(enable_prompt_cache.value)
            config.prompt_cache_ttl_sec = int(prompt_cache_ttl.value or 86400)
            save_gemini_api_key(api_key.value or "")
            save_app_config(config)
            dialog.close()
            ui.notify("配置已保存")

        with ui.row().classes("justify-end w-full"):
            ui.button("列出可用模型", on_click=list_models_click).props("outline")
            ui.button("取消", on_click=dialog.close).props("flat")
            ui.button("保存", on_click=save_settings)
    return dialog


async def do_analyze(progress_label) -> None:
    global analysis_process, analysis_started_at, processing, progress_text
    if processing:
        return
    processing = True
    analysis_started_at = datetime.now()
    progress_text = "准备分析..."
    add_log("开始 Gemini 分析")
    try:
        analysis_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "src.worker",
            cwd=str(ROOT),
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
            add_log("Gemini 分析完成")
            ui.notify("Gemini 分析完成")
    except Exception as exc:
        add_log(f"分析失败：{exc}")
        ui.notify(f"分析失败：{exc}", type="negative")
    finally:
        analysis_process = None
        analysis_started_at = None
        processing = False
        progress_text = "就绪"
        progress_label.text = progress_text


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
    add_styles()
    dialog = settings_dialog()
    clear_dialog = clear_data_dialog()
    with ui.header().classes("items-center justify-between bg-white text-gray-900 border-b"):
        ui.label("本地音频分类工作台").classes("text-base font-bold")
        with ui.row().classes("items-center gap-2"):
            progress_label = ui.label(progress_text).classes("text-xs text-gray-500")
            ui.timer(0.5, lambda: (progress_label.set_text(progress_text), update_run_widgets()))
            ui.timer(3.0, render_token_usage_panel.refresh)
            async def start_analyze_click() -> None:
                await do_analyze(progress_label)

            ui.button("扫描音频", on_click=do_scan).props("outline")
            ui.button("开始 Gemini 分析", on_click=start_analyze_click)
            ui.button("中断分析", on_click=request_stop).props("outline color=negative")
            ui.button("保存当前结果", on_click=lambda: (save_results(state), ui.notify("结果已保存"))).props("outline")
            ui.button("导出 CSV", on_click=do_export_csv).props("outline")
            ui.button("生成 ZIP", on_click=do_zip).props("outline")
            ui.button("清除测试结果", on_click=clear_dialog.open).props("outline color=negative")
            ui.button(icon="settings", on_click=dialog.open).props("flat round")

    with ui.column().classes("w-full p-4 gap-4"):
        render_run_panel()
        render_token_usage_panel()
        render_recut_area()
        render_result_controls()
        render_board()
        with ui.row().classes("w-full justify-end"):
            ui.button("生成分类文件夹", on_click=do_classified_folders).props("outline")
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


def do_classified_folders() -> None:
    try:
        path = build_classified_folders(state, config)
        ui.notify(f"分类文件夹已生成：{path}")
    except Exception as exc:
        ui.notify(f"生成失败：{exc}", type="negative")


def do_open_final() -> None:
    path = resolve_project_path(config.final_output_dir)
    ok, message = open_folder(path)
    ui.notify(f"已打开：{message}" if ok else f"请手动打开：{message}", type="positive" if ok else "warning")


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title="本地音频分类工作台", reload=False)
