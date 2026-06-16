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
ENV_PATH = ROOT / ".env"


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
        config.final_output_dir,
        config.export_dir,
        config.downloads_dir,
        config.gemini_uploads_dir,
    ]:
        resolve_project_path(path_value).mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)


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
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(), handle, ensure_ascii=False, indent=2)
    ensure_directories(config)


def load_results() -> ResultsState:
    if not RESULTS_PATH.exists():
        return ResultsState()
    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ResultsState.model_validate(payload)


def save_results(state: ResultsState) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(state.model_dump(), handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
            description="整体以情绪表达、叙事、柔和旋律为主。节奏不抢，律动不强，人声不以炫技高音为核心，听感偏舒缓、深情、慢歌、抒发型。",
            priority=3,
        ),
        CategoryConfig(
            id="high_note",
            name="高音",
            description="片段的主要记忆点是人声往上顶、强声、高音区、爆发、飙唱、高潮感。即使整体是抒情歌，只要这个片段的核心听感是高音爆发，也归为高音。",
            priority=1,
        ),
        CategoryConfig(
            id="mid_fast",
            name="中速快歌",
            description="节奏、律动、鼓点、推进感明显。听感不沉，有稳定动感、速度感、流行快歌感。未必特别快，但明显比抒情歌更有推动力。",
            priority=2,
        ),
    ]
