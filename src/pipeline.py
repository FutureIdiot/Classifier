from __future__ import annotations

import subprocess
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from src.audio_utils import (
    clip_audio,
    create_gemini_upload_audio,
    file_size_mb,
    get_audio_duration,
    scan_audio_files,
    stable_track_id,
)
from src.config import ERRORS_PATH, append_jsonl, resolve_project_path, save_app_config, save_results
from src.gemini_client import GeminiClient, GeminiRetryableError
from src.models import AppConfig, CategoryConfig, ClipRecord, GeminiSegment, ResultsState
from src.prompt_builder import build_prompt, build_static_prompt_context, build_track_prompt, prompt_cache_hash


ProgressCallback = Callable[[str], None]
StopCallback = Callable[[], bool]


def scan_tracks(config: AppConfig) -> list[Path]:
    return scan_audio_files(resolve_project_path(config.raw_audio_dir))


def analyze_tracks(
    config: AppConfig,
    state: ResultsState,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> ResultsState:
    raw_root = resolve_project_path(config.raw_audio_dir)
    tracks = scan_tracks(config)
    emit(progress, f"开始分析队列：{len(tracks)} 个音频文件，目录 {raw_root}")
    client = GeminiClient(config.gemini_model, config.gemini_timeout_sec)
    cached_content_name = None
    if config.enable_prompt_cache:
        static_prompt = build_static_prompt_context(config.categories)
        cache_hash = prompt_cache_hash(config.gemini_model, config.categories)
        cached_content_name = client.get_or_create_prompt_cache(
            cache_hash=cache_hash,
            static_prompt=static_prompt,
            ttl_sec=config.prompt_cache_ttl_sec,
            log=progress,
        )
    else:
        emit(progress, "prompt cache 已关闭，使用完整 prompt。")
    known_track_ids = {clip.track_id for clip in state.clips if clip.status != "replaced"}
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for index, audio_path in enumerate(tracks, start=1):
        if should_stop and should_stop():
            emit(progress, "收到中断请求，已停止后续歌曲处理。")
            break
        track_id = stable_track_id(audio_path, raw_root)
        emit(progress, f"正在处理第 {index} / {len(tracks)} 首：{audio_path.name}")
        try:
            if track_id in known_track_ids:
                emit(progress, f"跳过已分析歌曲：{audio_path.name}")
                skipped_count += 1
                continue
            emit(progress, f"读取音频时长：{audio_path.name}")
            duration = get_audio_duration(audio_path)
            emit(progress, f"音频时长 {duration:.1f}s，准备调用 Gemini：{audio_path.name}")
            upload_audio_path = prepare_gemini_upload_audio(audio_path, track_id, config, progress)
            fallback_prompt = build_prompt(track_id, config.categories)
            prompt = build_track_prompt(track_id, config.categories) if cached_content_name else fallback_prompt
            gemini_result = analyze_audio_with_retries(
                client=client,
                audio_path=upload_audio_path,
                prompt=prompt,
                track_id=track_id,
                progress=progress,
                cached_content_name=cached_content_name,
                fallback_prompt=fallback_prompt if cached_content_name else None,
                retry_count=config.gemini_retry_count,
            )
            emit(progress, f"Gemini 返回 {len(gemini_result.segments)} 个片段：{audio_path.name}")
            for seg_index, segment in enumerate(gemini_result.segments, start=1):
                try:
                    emit(
                        progress,
                        f"裁剪片段 {seg_index}/{len(gemini_result.segments)}："
                        f"{segment.section} {segment.start_sec:.1f}-{segment.end_sec:.1f}s -> {segment.label}",
                    )
                    record = segment_to_clip_record(
                        segment=segment,
                        segment_index=seg_index,
                        track_id=track_id,
                        source_path=audio_path,
                        raw_root=raw_root,
                        source_duration_sec=duration,
                        config=config,
                    )
                    state.clips.append(record)
                    emit(progress, f"已生成片段：{record.clip_path}")
                except Exception as exc:
                    emit(progress, f"片段失败：{audio_path.name} #{seg_index}，{exc}")
                    log_error(
                        "segment_failed",
                        track_id=track_id,
                        source_audio_path=str(audio_path),
                        error=str(exc),
                        segment=segment.model_dump(),
                    )
        except Exception as exc:
            failed_count += 1
            emit(progress, f"歌曲失败：{audio_path.name}，{exc}")
            log_error(
                "track_failed",
                track_id=track_id,
                source_audio_path=str(audio_path),
                error=str(exc),
            )
            continue
        finally:
            save_results(state)
        success_count += 1
    emit(progress, f"处理结束：共扫描 {len(tracks)} 首，成功 {success_count}，跳过 {skipped_count}，失败 {failed_count}")
    return state


def analyze_audio_with_retries(
    client: GeminiClient,
    audio_path: Path,
    prompt: str,
    track_id: str,
    progress: ProgressCallback | None,
    cached_content_name: str | None,
    fallback_prompt: str | None,
    retry_count: int,
):
    max_attempts = max(1, int(retry_count or 0) + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                emit(progress, f"重试 Gemini 分析：第 {attempt} / {max_attempts} 次")
            return client.analyze_audio(
                audio_path,
                prompt,
                track_id,
                progress,
                cached_content_name=cached_content_name,
                fallback_prompt=fallback_prompt,
            )
        except GeminiRetryableError as exc:
            if attempt >= max_attempts:
                raise
            emit(progress, f"Gemini 可重试错误：{exc}")
            delay_sec = retry_delay_seconds(attempt)
            emit(progress, f"等待 {delay_sec} 秒后重试当前歌曲。")
            import time

            time.sleep(delay_sec)
    raise RuntimeError("Gemini 分析重试流程异常结束。")


def retry_delay_seconds(failed_attempt: int) -> int:
    delays = [3, 10, 30, 60]
    index = max(0, min(failed_attempt - 1, len(delays) - 1))
    return delays[index]


def prepare_gemini_upload_audio(
    source_path: Path,
    track_id: str,
    config: AppConfig,
    progress: ProgressCallback | None = None,
) -> Path:
    upload_dir = resolve_project_path(config.gemini_uploads_dir)
    upload_path = upload_dir / f"{track_id}_gemini.m4a"
    source_mb = file_size_mb(source_path)
    if upload_path.exists() and upload_path.stat().st_mtime >= source_path.stat().st_mtime:
        emit(progress, f"复用 Gemini 上传代理音频：{upload_path.name}（{file_size_mb(upload_path)} MB，原文件 {source_mb} MB）")
        return upload_path
    emit(progress, f"生成 Gemini 上传代理音频：原文件 {source_mb} MB -> m4a 96kbps")
    create_gemini_upload_audio(source_path, upload_path)
    emit(progress, f"代理音频生成完成：{upload_path.name}（{file_size_mb(upload_path)} MB）")
    return upload_path


def emit(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)
    else:
        print(message, flush=True)


def segment_to_clip_record(
    segment: GeminiSegment,
    segment_index: int,
    track_id: str,
    source_path: Path,
    raw_root: Path,
    source_duration_sec: float,
    config: AppConfig,
) -> ClipRecord:
    if segment.end_sec <= segment.start_sec:
        raise ValueError(f"无效时间戳 start={segment.start_sec}, end={segment.end_sec}")
    if segment.start_sec < 0 or segment.start_sec >= source_duration_sec:
        raise ValueError(f"start_sec 超出原曲范围：{segment.start_sec}")

    label = normalize_label(segment.label, config)
    section = segment.section if segment.section in {"verse", "chorus"} else "unknown"
    clip_id = f"{track_id}_{section}_{segment_index:02d}"
    clip_path = resolve_project_path(config.clips_dir) / f"{clip_id}.wav"
    actual_duration = clip_audio(
        source_path=source_path,
        clip_path=clip_path,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        source_duration_sec=source_duration_sec,
    )
    display_name = clip_id
    return ClipRecord(
        clip_id=clip_id,
        track_id=track_id,
        source_filename=source_path.name,
        source_audio_path=relative_or_absolute(source_path),
        clip_path=relative_or_absolute(clip_path),
        section=section,
        start_sec=round(segment.start_sec, 1),
        end_sec=round(segment.end_sec, 1),
        duration_sec=round(segment.end_sec - segment.start_sec, 1),
        model_label=label,
        manual_label="",
        final_label=label,
        confidence=max(0.0, min(1.0, float(segment.confidence))),
        needs_review=segment.needs_review or label == "待复核",
        reason=segment.reason,
        display_name=display_name,
        export_filename=f"{display_name}.wav",
        status="confirmed" if actual_duration > 0 else "needs_review",
    )


def recut_clip(
    config: AppConfig,
    state: ResultsState,
    source_clip_id: str,
    start_sec: float,
    end_sec: float,
    final_label: str | None = None,
) -> ClipRecord:
    source_clip = find_clip(state, source_clip_id)
    source_path = resolve_project_path(source_clip.source_audio_path)
    source_duration = get_audio_duration(source_path)
    if end_sec <= start_sec:
        raise ValueError("结束时间必须大于开始时间。")
    new_clip_id = f"{source_clip.clip_id}_recut_{uuid.uuid4().hex[:6]}"
    clip_path = resolve_project_path(config.clips_dir) / f"{new_clip_id}.wav"
    actual_duration = clip_audio(source_path, clip_path, start_sec, end_sec, source_duration)
    label = final_label or source_clip.final_label
    if label not in {category.name for category in config.categories if category.name.strip()}:
        label = normalize_label(label, config)
    source_clip.status = "replaced"
    new_clip = source_clip.model_copy(
        update={
            "clip_id": new_clip_id,
            "clip_path": relative_or_absolute(clip_path),
            "start_sec": round(start_sec, 1),
            "end_sec": round(end_sec, 1),
            "duration_sec": round(end_sec - start_sec, 1),
            "manual_label": label if label != source_clip.model_label else source_clip.manual_label,
            "final_label": label,
            "display_name": new_clip_id,
            "export_filename": f"{new_clip_id}.wav",
            "status": "confirmed" if actual_duration > 0 else "needs_review",
        }
    )
    state.clips.append(new_clip)
    save_results(state)
    return new_clip


def update_clip_label(state: ResultsState, clip_id: str, label: str) -> None:
    clip = find_clip(state, clip_id)
    clip.manual_label = label
    clip.refresh_final_label()
    save_results(state)


def update_clip_display_name(state: ResultsState, clip_id: str, display_name: str) -> None:
    clip = find_clip(state, clip_id)
    clean = display_name.strip() or clip.clip_id
    clip.display_name = clean
    clip.export_filename = f"{Path(clean).stem}.wav"
    save_results(state)


def update_category(config: AppConfig, category_id: str, name: str, description: str) -> None:
    for category in config.categories:
        if category.id == category_id:
            old_name = category.name
            category.name = name.strip()
            category.description = description.strip()
            if old_name and category.name and old_name != category.name:
                # Keep existing board positions coherent after a rename.
                state = None
            break
    save_app_config(config)


def prune_empty_categories(config: AppConfig, state: ResultsState) -> None:
    occupied = {clip.final_label for clip in state.clips if clip.status not in {"hidden", "replaced"}}
    config.categories = [
        category
        for category in config.categories
        if category.name.strip() or category.description.strip() or category.name in occupied
    ]
    save_app_config(config)


def add_category(config: AppConfig) -> CategoryConfig:
    category = CategoryConfig(
        id=f"cat_{uuid.uuid4().hex[:8]}",
        name="",
        description="",
        priority=max([item.priority for item in config.categories], default=0) + 1,
    )
    config.categories.append(category)
    save_app_config(config)
    return category


def ensure_review_category(config: AppConfig) -> None:
    if any(category.name == "待复核" for category in config.categories):
        return
    config.categories.append(
        CategoryConfig(
            id="review",
            name="待复核",
            description="Gemini 返回的分类不在当前分类列表中，或模型不确定，需要人工检查。",
            priority=999,
        )
    )
    save_app_config(config)


def normalize_label(label: str, config: AppConfig) -> str:
    valid = {category.name for category in config.categories if category.name.strip()}
    if label in valid:
        return label
    ensure_review_category(config)
    return "待复核"


def find_clip(state: ResultsState, clip_id: str) -> ClipRecord:
    for clip in state.clips:
        if clip.clip_id == clip_id:
            return clip
    raise KeyError(f"找不到片段：{clip_id}")


def log_error(kind: str, **payload: object) -> None:
    append_jsonl(
        ERRORS_PATH,
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            **payload,
        },
    )


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
