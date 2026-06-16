from __future__ import annotations

import asyncio
import html
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import Header, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from nicegui import app, ui

from src.audio_utils import seconds_to_mmss
from src.config import (
    ROOT,
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


def add_log(message: str) -> None:
    global last_log_at
    last_log_at = datetime.now()
    line = f"{datetime.now().strftime('%H:%M:%S')}  {message}"
    run_logs.append(line)
    print(line, flush=True)
    update_run_widgets()


def reload_state() -> None:
    global config, state
    config = load_app_config()
    state = load_results()


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


def confidence_meter(value: float) -> str:
    if value >= 0.9:
        return "▂▄▆█"
    if value >= 0.75:
        return "▂▄▆"
    if value >= 0.6:
        return "▂▄"
    return "▂"


def visible_by_label() -> dict[str, list]:
    labels = [category.name for category in config.categories if category.name.strip()]
    grouped = {label: [] for label in labels}
    for clip in visible_clips(state):
        grouped.setdefault(clip.final_label or "待复核", []).append(clip)
    return grouped


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
        <button class="add-column" onclick="mcAddColumn()" title="添加分类列">+</button>
      </div>
      <div class="kanban-wrap" style="--visible-columns: {visible_columns};">
        {"".join(columns)}
      </div>
    </div>
    """


def card_html(clip) -> str:
    duration = seconds_to_mmss(clip.duration_sec)
    name = html.escape(clip.display_name)
    original = js_str(clip.display_name)
    return f"""
    <div class="clip-row" data-clip-id="{escape_attr(clip.clip_id)}" draggable="true" ondragstart="event.dataTransfer.setData('text/plain', {js_str(clip.clip_id)})">
      <button class="play-btn" onclick="mcPlay({js_str(clip.clip_id)})">▶</button>
      <input class="clip-name" value="{escape_attr(clip.display_name)}"
        onkeydown="mcInputKey(event, {original})"
        onblur="mcRenameClip({js_str(clip.clip_id)}, this.value)">
      <span class="clip-duration">{duration}</span>
      <span class="confidence" title="{clip.confidence:.2f}">{confidence_meter(clip.confidence)}</span>
    </div>
    """


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
            justify-content: flex-end;
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
            grid-template-columns: 28px minmax(0, 1fr) 48px 54px;
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
        </script>
        """
    )


@ui.refreshable
def render_board() -> None:
    ui.html(board_html(), sanitize=False).classes("w-full")


@ui.refreshable
def render_recut_area() -> None:
    clips = visible_clips(state)
    options = {clip.clip_id: f"{clip.display_name} · {clip.source_filename}" for clip in clips}
    with ui.card().classes("w-full p-3").style("border-radius:8px"):
        ui.label("手动裁剪").classes("text-sm font-bold")
        if not clips:
            ui.label("暂无片段。先分析音频后，可以在这里重新裁剪。").classes("text-xs text-gray-500")
            return
        selected = ui.select(options=options, label="片段", value=clips[0].clip_id).props("dense outlined").classes("w-full")
        with ui.row().classes("items-end gap-3 w-full"):
            start_input = ui.number("start_sec", value=clips[0].start_sec, step=0.1, format="%.1f").props("dense outlined")
            end_input = ui.number("end_sec", value=clips[0].end_sec, step=0.1, format="%.1f").props("dense outlined")
            label_options = [category.name for category in config.categories if category.name.strip()]
            label_select = ui.select(label_options, label="分类", value=clips[0].final_label).props("dense outlined")
            ui.button("载入片段", on_click=lambda: load_recut_values(selected.value, start_input, end_input, label_select)).props("outline")
            ui.button("重新裁剪", on_click=lambda: do_recut(selected.value, start_input.value, end_input.value, label_select.value))
        ui.html(f'<audio controls style="width:100%;height:38px" src="/source/{clips[0].clip_id}"></audio>', sanitize=False)


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
    with ui.header().classes("items-center justify-between bg-white text-gray-900 border-b"):
        ui.label("本地音频分类工作台").classes("text-base font-bold")
        with ui.row().classes("items-center gap-2"):
            progress_label = ui.label(progress_text).classes("text-xs text-gray-500")
            ui.timer(0.5, lambda: (progress_label.set_text(progress_text), update_run_widgets()))
            async def start_analyze_click() -> None:
                await do_analyze(progress_label)

            ui.button("扫描音频", on_click=do_scan).props("outline")
            ui.button("开始 Gemini 分析", on_click=start_analyze_click)
            ui.button("中断分析", on_click=request_stop).props("outline color=negative")
            ui.button("保存当前结果", on_click=lambda: (save_results(state), ui.notify("结果已保存"))).props("outline")
            ui.button("导出 CSV", on_click=do_export_csv).props("outline")
            ui.button("生成 ZIP", on_click=do_zip).props("outline")
            ui.button(icon="settings", on_click=dialog.open).props("flat round")

    with ui.column().classes("w-full p-4 gap-4"):
        render_run_panel()
        render_recut_area()
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
