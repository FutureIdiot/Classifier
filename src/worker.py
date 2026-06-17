from __future__ import annotations

import sys
from src.config import load_app_config, load_results
from src.pipeline import analyze_tracks
from src.runtime import configure_utf8_runtime


configure_utf8_runtime()


def main() -> int:
    try:
        config = load_app_config()
        state = load_results()
        analyze_tracks(config, state, lambda message: print(message, flush=True))
        return 0
    except KeyboardInterrupt:
        print("分析子进程收到中断信号。", flush=True)
        return 130
    except Exception as exc:
        print(f"分析子进程异常：{exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
