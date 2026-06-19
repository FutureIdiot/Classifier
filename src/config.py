from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.models import AppConfig, CategoryConfig, ResultsState


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "app_config.json"
PROMPT_CONFIG_PATH = ROOT / "config" / "prompt_config.json"
RESULTS_PATH = ROOT / "data" / "results.json"
RAW_RESPONSES_PATH = ROOT / "data" / "raw_responses.jsonl"
ERRORS_PATH = ROOT / "data" / "errors.jsonl"
COMPLETED_TRACKS_PATH = ROOT / "data" / "completed_tracks.json"
ENV_PATH = ROOT / ".env"
PROMPT_CACHE_PATH = ROOT / "data" / "prompt_cache.json"


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_environment() -> None:
    load_dotenv(ENV_PATH, override=True)


def get_gemini_api_key() -> str:
    load_environment()
    return os.getenv("GEMINI_API_KEY", "").strip()


def save_gemini_api_key(api_key: str) -> None:
    values: dict[str, str] = {}
    if ENV_PATH.exists():
        with ENV_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                values[key.strip()] = value.strip()
    values["GEMINI_API_KEY"] = api_key.strip()
    with ENV_PATH.open("w", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")
    os.environ["GEMINI_API_KEY"] = api_key.strip()


def ensure_directories(config: AppConfig) -> None:
    for path_value in [
        config.raw_audio_dir,
        config.clips_dir,
        config.processed_audio_dir,
        config.original_backup_dir,
        config.final_output_dir,
        config.completed_output_dir,
        config.export_dir,
        config.downloads_dir,
        config.gemini_uploads_dir,
    ]:
        try:
            resolve_project_path(path_value).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    try:
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def load_app_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        config = AppConfig(categories=default_categories())
        save_app_config(config)
        return config
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = AppConfig.model_validate(json.load(handle))
    ensure_directories(config)
    return config


def save_app_config(config: AppConfig) -> None:
    normalize_categories(config)
    atomic_write_json(CONFIG_PATH, config.model_dump())
    ensure_directories(config)


def load_results() -> ResultsState:
    if not RESULTS_PATH.exists():
        return ResultsState()
    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ResultsState.model_validate(payload)


def save_results(state: ResultsState) -> None:
    atomic_write_json(RESULTS_PATH, state.model_dump())


def load_completed_tracks() -> dict:
    if not COMPLETED_TRACKS_PATH.exists():
        return {"tracks": {}}
    try:
        with COMPLETED_TRACKS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"tracks": {}}
    if not isinstance(payload, dict):
        return {"tracks": {}}
    tracks = payload.get("tracks")
    if not isinstance(tracks, dict):
        payload["tracks"] = {}
    return payload


def save_completed_tracks(payload: dict) -> None:
    payload.setdefault("tracks", {})
    atomic_write_json(COMPLETED_TRACKS_PATH, payload)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, payload: object) -> None:
    """写入 JSON：先写临时文件再原子替换，并保留上一版本 .bak。

    崩溃时目标文件要么是完整的旧内容、要么是完整的新内容，不会出现截断。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        try:
            backup_path = path.with_suffix(path.suffix + ".bak")
            os.replace(path, backup_path)
        except OSError:
            pass
    os.replace(tmp_path, path)


def normalize_categories(config: AppConfig) -> None:
    seen: set[str] = set()
    normalized: list[CategoryConfig] = []
    for category in config.categories:
        if not category.id:
            category.id = f"cat_{uuid.uuid4().hex[:8]}"
        category.id = category.id.strip() or f"cat_{uuid.uuid4().hex[:8]}"
        while category.id in seen:
            category.id = f"{category.id}_{uuid.uuid4().hex[:4]}"
        seen.add(category.id)
        normalized.append(category)
    config.categories = normalized


def default_categories() -> list[CategoryConfig]:
    return [
        CategoryConfig(
            id="lyrical",
            name="抒情",
            description=(
                "以人声情绪表达、叙事、柔和旋律和长句铺垫为主。语速不急，咬字密度不高，"
                "听感偏舒缓、深情、慢歌、抒发型。只有在人声高音爆发不构成主要记忆点、"
                "短句推进和咬字节奏不明显时，才归为抒情。"
                "如果片段虽然速度慢但副歌高音爆发明显，应优先归为高音。"
                "如果人声短句连续、咬字密、重音和换气形成明显推进感，应优先归为中速快歌。"
            ),
            priority=3,
        ),
        CategoryConfig(
            id="high_note",
            name="高音",
            description=(
                "片段的主要记忆点是人声往上顶、强声、高音区、爆发、飙唱、高潮感。"
                "优先级最高。即使整体速度慢、没有伴奏或歌曲风格偏抒情，只要当前片段的核心听感是高音爆发、"
                "强声持续、音区明显上升或情绪强度突然拉高，就归为高音。不要因为歌曲整体是慢歌就归为抒情。"
                "如果只是轻微抬高、柔和假声、没有爆发记忆点，不归为高音。"
            ),
            priority=1,
        ),
        CategoryConfig(
            id="mid_fast",
            name="中速快歌",
            description=(
                "干声场景下按人声本身判断：咬字密度高、短句连续、语速和重音有明显推进感，"
                "换气频率较高，听感不沉，有流行快歌或律动型唱法的感觉。"
                "不要求有鼓点、伴奏或真实 BPM 很快；只要人声主要记忆点是推进感和节奏感，就归为中速快歌。"
                "如果人声核心记忆点是高音爆发，优先归为高音。"
                "如果只有轻微节奏但整体仍是慢速叙事、情绪铺垫，没有明显短句推进，归为抒情。"
            ),
            priority=2,
        ),
    ]
