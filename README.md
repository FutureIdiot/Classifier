# 本地音频分类工作台 MVP

一个本地运行的 NiceGUI 音频分类 Web UI。它会扫描 `raw_audio/`，用 Gemini 听整首歌并一次性返回主歌/副歌时间戳和分类，再用 ffmpeg 裁剪 `.wav` 片段，最后在多列看板里试听、改名、拖拽调整分类，并导出 CSV 或分类 ZIP。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python app.py
```

打开 `http://localhost:8080`。

Gemini API Key 可以在右上角齿轮配置里填写，也可以直接编辑 `.env`。

## 使用流程

1. 把 `.wav` 音频放进 `raw_audio/`。
2. 点右上角齿轮，确认路径、ZIP 下载目录和 Gemini 模型。
3. 点“扫描音频”检查数量。
4. 点“开始 Gemini 分析”。
5. 在看板中试听、编辑显示名、拖拽片段到其他分类列。
6. 使用“导出 CSV”或“生成 ZIP”。

## 依赖

需要本机安装并能从 PATH 调用：

- `ffmpeg`
- `ffprobe`

状态保存在 `data/results.json`，原始 Gemini 响应保存在 `data/raw_responses.jsonl`，错误记录保存在 `data/errors.jsonl`。
