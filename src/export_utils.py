from __future__ import annotations

import filecmp
import json
import os
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.config import resolve_project_path
from src.models import AppConfig, ClipRecord, ResultsState


MANIFEST_NAME = ".classified_manifest.json"

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
    return sync_classified_folders(state, config)


def sync_classified_folders(state: ResultsState, config: AppConfig) -> Path:
    final_dir = resolve_project_path(config.final_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(final_dir)
    used_names: dict[Path, set[str]] = {}
    replaceable_targets = replaceable_output_paths(state, final_dir, manifest, config.clip_format)
    visible_ids = {clip.clip_id for clip in visible_clips(state)}

    for clip in visible_clips(state):
        sync_classified_clip(
            clip,
            config,
            manifest=manifest,
            used_names=used_names,
            replaceable_targets=replaceable_targets,
            save_manifest_file=False,
        )
    prune_inactive_manifest_outputs(final_dir, manifest, visible_ids)
    save_manifest(final_dir, manifest)
    return final_dir


def sync_track_classified_folders(state: ResultsState, config: AppConfig, track_id: str) -> Path:
    final_dir = resolve_project_path(config.final_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(final_dir)
    used_names: dict[Path, set[str]] = {}
    replaceable_targets = replaceable_output_paths(state, final_dir, manifest, config.clip_format)
    visible_ids = {clip.clip_id for clip in visible_clips(state)}
    for clip in visible_clips(state):
        if clip.track_id == track_id:
            sync_classified_clip(
                clip,
                config,
                manifest=manifest,
                used_names=used_names,
                replaceable_targets=replaceable_targets,
                save_manifest_file=False,
            )
    prune_inactive_manifest_outputs(final_dir, manifest, visible_ids)
    save_manifest(final_dir, manifest)
    return final_dir


def sync_classified_clip(
    clip: ClipRecord,
    config: AppConfig,
    manifest: dict[str, dict[str, str]] | None = None,
    used_names: dict[Path, set[str]] | None = None,
    replaceable_targets: set[str] | None = None,
    save_manifest_file: bool = True,
) -> Path | None:
    if clip.status in {"hidden", "replaced", "editing"}:
        return None

    source = resolve_project_path(clip.clip_path)
    if not source.exists():
        return None

    final_dir = resolve_project_path(config.final_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest if manifest is not None else load_manifest(final_dir)
    used_names = used_names if used_names is not None else {}

    label = safe_folder_name(clip.final_label or "待复核")
    target_dir = final_dir / label
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = ensure_clip_suffix(clip.export_filename or clip.display_name, config.clip_format)
    desired_path = target_dir / filename
    desired_rel = desired_path.relative_to(final_dir).as_posix()
    prior_manifest_path: Path | None = None

    entry = manifest.get(clip.clip_id)
    if entry:
        manifest_path = final_dir / entry.get("target_path", "")
        prior_manifest_path = manifest_path
        if (
            entry.get("label") == label
            and entry.get("export_filename") == filename
            and manifest_path.exists()
            and same_file_content(source, manifest_path)
            and (
                manifest_path == desired_path
                or desired_rel not in (replaceable_targets or set())
                or not desired_path.exists()
            )
        ):
            return manifest_path

    if desired_path.exists() and same_file_content(source, desired_path):
        target_path = desired_path
        remove_stale_owned_output(prior_manifest_path, target_path)
    elif desired_path.exists() and replaceable_targets is not None and desired_rel in replaceable_targets:
        try:
            shutil.copy2(source, desired_path)
        except OSError:
            return None
        remove_manifest_entries_for_target(manifest, desired_rel, keep_clip_id=clip.clip_id)
        remove_stale_owned_output(prior_manifest_path, desired_path)
        target_path = desired_path
    else:
        target_name = unique_filename(filename, used_names.setdefault(target_dir, existing_file_names(target_dir)))
        target_path = target_dir / target_name
        try:
            shutil.copy2(source, target_path)
        except OSError:
            return None
        remove_stale_owned_output(prior_manifest_path, target_path)

    manifest[clip.clip_id] = {
        "target_path": target_path.relative_to(final_dir).as_posix(),
        "label": label,
        "export_filename": filename,
        "clip_path": clip.clip_path,
    }
    if save_manifest_file:
        save_manifest(final_dir, manifest)
    return target_path


def append_completed_folders(config: AppConfig) -> Path:
    workspace_dir = resolve_project_path(config.final_output_dir)
    completed_dir = resolve_project_path(config.completed_output_dir)
    completed_dir.mkdir(parents=True, exist_ok=True)
    used_names: dict[Path, set[str]] = {}
    if not workspace_dir.exists():
        return completed_dir
    for source in workspace_dir.rglob("*"):
        if not source.is_file() or source.name == MANIFEST_NAME:
            continue
        try:
            relative = source.relative_to(workspace_dir)
        except ValueError:
            relative = Path(source.name)
        target_dir = completed_dir / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = ensure_clip_suffix(source.name, config.clip_format)
        target_name = unique_filename(filename, used_names.setdefault(target_dir, existing_file_names(target_dir)))
        shutil.copy2(source, target_dir / target_name)
    return completed_dir


def clear_classified_workspace(config: AppConfig) -> Path:
    final_dir = resolve_project_path(config.final_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    for child in final_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass
    return final_dir


def clear_clips_workspace(config: AppConfig) -> Path:
    """finalize 后清空切片工作目录。

    此时所有可见切片已复制进 completed_results，state 即将清空，
    slices/ 不再被引用，删除可消除与归档目录的永久双份冗余。
    """
    clips_dir = resolve_project_path(config.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    for child in clips_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass
    return clips_dir


def replaceable_output_paths(
    state: ResultsState,
    final_dir: Path,
    manifest: dict[str, dict[str, str]],
    clip_format: str = "wav",
) -> set[str]:
    status_by_id = {clip.clip_id: clip.status for clip in state.clips}
    replaceable: set[str] = {
        str(entry.get("target_path") or "")
        for clip_id, entry in manifest.items()
        if status_by_id.get(clip_id) == "replaced" and entry.get("target_path")
    }
    for clip in state.clips:
        if clip.status != "replaced":
            continue
        label = safe_folder_name(clip.final_label or "待复核")
        filename = ensure_clip_suffix(clip.export_filename or clip.display_name, clip_format)
        replaceable.add((final_dir / label / filename).relative_to(final_dir).as_posix())
    return replaceable


def remove_manifest_entries_for_target(
    manifest: dict[str, dict[str, str]],
    target_path: str,
    keep_clip_id: str,
) -> None:
    for clip_id in [
        clip_id
        for clip_id, entry in manifest.items()
        if clip_id != keep_clip_id and entry.get("target_path") == target_path
    ]:
        manifest.pop(clip_id, None)


def remove_stale_owned_output(old_path: Path | None, new_path: Path) -> None:
    if old_path is None or old_path == new_path or not old_path.exists():
        return
    try:
        old_path.unlink()
    except OSError:
        pass


def prune_inactive_manifest_outputs(final_dir: Path, manifest: dict[str, dict[str, str]], visible_clip_ids: set[str]) -> None:
    for clip_id in list(manifest):
        if clip_id in visible_clip_ids:
            continue
        target = final_dir / str(manifest.get(clip_id, {}).get("target_path") or "")
        remove_stale_owned_output(target, final_dir / MANIFEST_NAME)
        manifest.pop(clip_id, None)


def build_zip(state: ResultsState, config: AppConfig) -> Path:
    final_dir = build_classified_folders(state, config)
    downloads_dir = resolve_project_path(config.downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    zip_path = downloads_dir / "classified_clips.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in final_dir.rglob("*"):
            if file_path.is_file() and file_path.name != MANIFEST_NAME:
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
    return [clip for clip in state.clips if clip.status not in {"hidden", "replaced", "editing"}]


def safe_folder_name(name: str) -> str:
    clean = "".join("_" if char in r'\/:*?"<>|' else char for char in name).strip()
    return clean or "待复核"


def ensure_clip_suffix(filename: str, clip_format: str = "wav") -> str:
    path = Path(filename)
    suffix = ".flac" if (clip_format or "wav").lower() == "flac" else ".wav"
    return f"{path.stem}{suffix}"


def existing_file_names(target_dir: Path) -> set[str]:
    return {path.name.lower() for path in target_dir.iterdir() if path.is_file() and path.name != MANIFEST_NAME}


def unique_filename(filename: str, used_names: set[str]) -> str:
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix or ".wav"
    index = 1
    candidate = f"{stem}{suffix}"
    while candidate.lower() in used_names:
        index += 1
        candidate = f"{stem}_{index:02d}{suffix}"
    used_names.add(candidate.lower())
    return candidate


def same_file_content(source: Path, target: Path) -> bool:
    try:
        if source.stat().st_size != target.stat().st_size:
            return False
        return filecmp.cmp(source, target, shallow=False)
    except OSError:
        return False


def load_manifest(final_dir: Path) -> dict[str, dict[str, str]]:
    path = final_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def save_manifest(final_dir: Path, manifest: dict[str, dict[str, str]]) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
