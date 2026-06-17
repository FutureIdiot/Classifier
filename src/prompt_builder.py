from __future__ import annotations

import json

from src.models import CategoryConfig


def category_payload(categories: list[CategoryConfig]) -> list[dict]:
    return [
        {
            "name": category.name,
            "description": category.description,
            "priority": category.priority,
        }
        for category in sorted(categories, key=lambda item: item.priority)
        if category.name.strip()
    ]


def build_static_prompt_context(categories: list[CategoryConfig]) -> str:
    payload = category_payload(categories)
    category_names = [item["name"] for item in payload]
    priority_lines = "\n".join(
        f"{index}. {item['name']}：priority={item['priority']}" for index, item in enumerate(payload, start=1)
    )
    return f"""
你是一个面向音乐素材管理工作流的音频听感分类助手。

固定业务规则：
- 请完整聆听音频，并一次性完成主歌/副歌片段识别与业务听感分类。
- 这是业务听感分类，不要求计算精确 BPM、真实音高或音乐理论分析。
- 只能根据音频内容判断，不要根据文件名、歌名、路径或元数据判断。
- 需要识别主歌和副歌；section 只能是 "verse"、"chorus"、"unknown"。
- 分类对象是当前片段，不是整首歌风格；同一首歌的主歌和副歌可以属于不同分类。
- 判断时先找当前片段的主要记忆点：人声高音爆发、节奏律动推进、还是情绪叙事铺垫。
- 不要因为整首歌是慢歌就自动归为抒情；也不要因为伴奏有轻微鼓点就自动归为快歌。
- 每个片段的 label 只能来自当前分类名：{json.dumps(category_names, ensure_ascii=False)}。
- 分类配置如下，priority 数值越小优先级越高；当片段同时符合多个类别时，优先选择 priority 更小的分类。
{json.dumps(payload, ensure_ascii=False, indent=2)}
- 当前优先级顺序：
{priority_lines}
- 常见冲突处理：
  1. 慢速、柔和伴奏，但副歌人声强声上行、高音持续或高潮爆发明显，优先归为高音。
  2. 有节奏但人声核心记忆点是高音爆发，优先归为高音。
  3. 有鼓点但整体仍是慢速叙事、情绪铺垫、没有明显推进感，归为抒情。
  4. 节奏推进、律动、鼓点和速度感构成主要记忆点时，归为中速快歌。
- 如果不确定，仍选择最接近的分类，但 needs_review 必须为 true。
- confidence 表示你对分类结果的把握。不要机械给满分；边界模糊、片段过短、结构不清、多个分类都像时，应降低 confidence 并设置 needs_review=true。
- 时间单位为秒，保留一位小数。
- 输出必须是纯 JSON，不要 markdown，不要代码围栏，不要额外解释。
""".strip()


def build_track_prompt(track_id: str, categories: list[CategoryConfig]) -> str:
    payload = category_payload(categories)
    fallback_label = payload[0]["name"] if payload else "待复核"
    return f"""
请分析本次上传的这一首音频。

输出 JSON 格式必须符合：
{{
  "track_id": "{track_id}",
  "needs_review": false,
  "segments": [
    {{
      "section": "verse",
      "start_sec": 18.2,
      "end_sec": 51.6,
      "label": "{fallback_label}",
      "confidence": 0.84,
      "needs_review": false,
      "reason": "简短中文原因"
    }}
  ],
  "song_level_label": "{fallback_label}",
  "song_level_reason": "简短中文原因"
}}
""".strip()


def build_prompt(track_id: str, categories: list[CategoryConfig]) -> str:
    static_context = build_static_prompt_context(categories)
    track_prompt = build_track_prompt(track_id, categories)
    return f"{static_context}\n\n{track_prompt}"


def prompt_cache_hash(model: str, categories: list[CategoryConfig]) -> str:
    import hashlib

    payload = {
        "model": model,
        "static_prompt": build_static_prompt_context(categories),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
