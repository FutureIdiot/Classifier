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
- 输入通常是干声或人声音频，可能没有伴奏、鼓点、和声或完整编曲；所有判断优先基于人声本身。
- 请完整聆听音频，并一次性完成可用人声片段识别、主歌/副歌段落判断与业务听感分类。
- 这是业务听感分类，不要求计算精确 BPM、真实音高或音乐理论分析。
- 只能根据音频内容判断，不要根据文件名、歌名、路径或元数据判断。
- 需要识别主歌和副歌；section 只能是 "verse"、"chorus"、"unknown"。
- 分类对象是当前片段，不是整首歌风格；同一首歌的主歌和副歌可以属于不同分类。
- 判断时先找当前片段的人声主要记忆点：高音/强声爆发、咬字节奏和短句推进、还是舒缓叙事和情绪铺垫。
- 不要依据不存在的伴奏、鼓点、BPM 或编曲来判断；如果是干声，按人声的语速、咬字密度、换气频率、旋律推进和情绪强度判断。
- 不要因为整首歌是慢歌就自动归为抒情；也不要因为人声有轻微节奏感就自动归为中速快歌。
- 每个片段的 label 只能来自当前分类名：{json.dumps(category_names, ensure_ascii=False)}。
- 分类配置如下，priority 数值越小优先级越高；当片段同时符合多个类别时，优先选择 priority 更小的分类。
{json.dumps(payload, ensure_ascii=False, indent=2)}
- 当前优先级顺序：
{priority_lines}
- 切段硬性要求：
  1. 每个 segment 必须主要包含清晰可听的人声演唱或说唱；不要输出纯静音、长空白、只有呼吸/噪声、或明显不含有效人声的区间。
  2. start_sec 应落在该段第一句有效人声开始附近；end_sec 应落在该段最后一句有效人声结束附近。不要在片段前后保留超过约 1 秒的静音或空白。
  3. 如果某个候选区间中间有很长空白，应拆开或只保留有连续有效人声的一段。
  4. 不要只返回第一组主歌/副歌；歌曲中每次明显重复出现的主歌、预副歌、hook/副歌都应尽量单独返回。目标是覆盖所有有用的人声素材，而不是只给 1-2 个示例。
  5. 一般优先返回 2-8 个可用片段；只有确实没有更多有效人声段落时才少于 2 个。不要为了凑数量切空白。
- 主歌/副歌判断：
  1. chorus 通常是更像 hook 的段落：旋律/歌词重复度更高、记忆点更强、音区或情绪更集中，常在歌曲中重复出现。
  2. verse 通常是叙事或推进段落：歌词信息量更大、旋律重复度较低、情绪和音区相对铺垫。
  3. 干声没有伴奏时，不要用鼓点/编曲进入来判断主副歌；用人声旋律重复、歌词重复、强弱变化、音区变化和句式结构判断。
  4. 如果无法可靠区分主副歌，section 使用 "unknown"，但仍要保证片段是有效人声。
- 常见冲突处理：
  1. 即使没有伴奏，只要人声强声上行、高音持续、高潮爆发或飙唱明显，优先归为高音。
  2. 人声有节奏推进但核心记忆点是高音爆发，优先归为高音。
  3. 人声语速较慢、长音多、叙事感和情绪铺垫为主，没有明显短句推进，归为抒情。
  4. 人声咬字密、短句连续、换气和重音形成明显推进感，听感不沉，归为中速快歌；不需要有鼓点或伴奏。
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

请先在内部完成两步判断，但最终只输出 JSON：
1. 找出所有有清晰人声的连续可用区间，排除静音、长空白和无效噪声。
2. 再对每个可用区间判断 section 和 label。

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
