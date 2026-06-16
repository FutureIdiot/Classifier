from __future__ import annotations

import json

from src.models import CategoryConfig


def build_prompt(track_id: str, categories: list[CategoryConfig]) -> str:
    category_payload = [
        {
            "name": category.name,
            "description": category.description,
            "priority": category.priority,
        }
        for category in sorted(categories, key=lambda item: item.priority)
        if category.name.strip()
    ]
    category_names = [item["name"] for item in category_payload]
    return f"""
请完整聆听这首音频，并一次性完成主歌/副歌片段识别与业务听感分类。

重要规则：
- 这是业务听感分类，不要求计算精确 BPM、真实音高或音乐理论分析。
- 只能根据音频内容判断，不要根据文件名、歌名、路径或元数据判断。
- 需要识别主歌和副歌；section 只能是 "verse"、"chorus"、"unknown"。
- 每个片段的 label 只能来自当前分类名：{json.dumps(category_names, ensure_ascii=False)}。
- 分类配置如下，priority 数值越小优先级越高；当片段同时符合多个类别时，优先选择 priority 更小的分类。
{json.dumps(category_payload, ensure_ascii=False, indent=2)}
- 如果不确定，仍选择最接近的分类，但 needs_review 必须为 true。
- 时间单位为秒，保留一位小数。
- 输出必须是纯 JSON，不要 markdown，不要代码围栏，不要额外解释。

输出 JSON 格式必须符合：
{{
  "track_id": "{track_id}",
  "needs_review": false,
  "segments": [
    {{
      "section": "verse",
      "start_sec": 18.2,
      "end_sec": 51.6,
      "label": "{category_names[0] if category_names else "待复核"}",
      "confidence": 0.84,
      "needs_review": false,
      "reason": "简短中文原因"
    }}
  ],
  "song_level_label": "{category_names[0] if category_names else "待复核"}",
  "song_level_reason": "简短中文原因"
}}
""".strip()

