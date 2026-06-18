from __future__ import annotations

import html
from urllib.parse import quote

from src.audio_utils import seconds_to_mmss
from src.models import AppConfig, ClipRecord


def board_html(config: AppConfig, grouped: dict[str, list[ClipRecord]]) -> str:
    columns = []
    for category in config.categories:
        title = "" if not category.name.strip() and not category.description.strip() else category.name
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

    known_labels = {category.name for category in config.categories}
    for label, clips in grouped.items():
        if label in known_labels:
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
        <button class="batch-button" onclick="mcBatchUpdate('confirm')">完成任务并清空</button>
        <button class="batch-button danger" onclick="mcBatchUpdate('hide')">隐藏</button>
        <button class="add-column" onclick="mcAddColumn()" title="添加分类列">+</button>
      </div>
      <div class="kanban-wrap" style="--visible-columns: {visible_columns};">
        {"".join(columns)}
      </div>
    </div>
    """


def card_html(clip: ClipRecord) -> str:
    duration = seconds_to_mmss(clip.duration_sec)
    original = js_str(clip.display_name)
    confidence_text = f"{clip.confidence:.2f}"
    waveform_url = f"/waveform/{quote(clip.clip_id, safe='')}"
    return f"""
    <div class="clip-row" data-clip-id="{escape_attr(clip.clip_id)}" draggable="true" ondragstart="mcDragClip(event, {js_str(clip.clip_id)})">
      <input class="clip-check" type="checkbox" data-clip-id="{escape_attr(clip.clip_id)}" onclick="event.stopPropagation()">
      <button class="play-btn" onclick="mcPlay({js_str(clip.clip_id)})">▶</button>
      <button class="edit-btn" onclick="mcStartEditClip({js_str(clip.clip_id)})" title="送到上方编辑区">✎</button>
      <div class="clip-name-wrap">
        <img class="clip-waveform" src="{escape_attr(waveform_url)}" alt="">
        <input class="clip-name" value="{escape_attr(clip.display_name)}"
          onkeydown="mcInputKey(event, {original})"
          onblur="mcRenameClip({js_str(clip.clip_id)}, this.value)">
      </div>
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
