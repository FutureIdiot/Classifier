from __future__ import annotations

import sys
import os
import json
from src.config import load_app_config, load_results
from src.pipeline import analyze_tracks
from src.runtime import configure_utf8_runtime


configure_utf8_runtime()


def main() -> int:
    try:
        config = load_app_config()
        state = load_results()
        force_reanalyze = os.getenv("MUSIC_CLASSIFIER_FORCE_REANALYZE", "") == "1"
        retry_track_ids = parse_retry_track_ids(os.getenv("MUSIC_CLASSIFIER_RETRY_TRACK_IDS", ""))
        analyze_tracks(
            config,
            state,
            lambda message: print(message, flush=True),
            force_reanalyze=force_reanalyze,
            retry_track_ids=retry_track_ids,
        )
        return 0
    except KeyboardInterrupt:
        print("分析子进程收到中断信号。", flush=True)
        return 130
    except Exception as exc:
        print(f"分析子进程异常：{exc}", flush=True)
        return 1


def parse_retry_track_ids(value: str) -> set[str] | None:
    if not value.strip():
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        payload = value.split(",")
    if not isinstance(payload, list):
        return None
    track_ids = {str(item).strip() for item in payload if str(item).strip()}
    return track_ids or None


if __name__ == "__main__":
    raise SystemExit(main())
