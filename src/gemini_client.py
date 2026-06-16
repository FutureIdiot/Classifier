from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from google import genai
from google.genai import types

from src.config import RAW_RESPONSES_PATH, append_jsonl, load_environment
from src.models import GeminiTrackResult


LogCallback = Callable[[str], None]


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped.strip(), flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


class GeminiClient:
    def __init__(self, model: str, timeout_sec: int = 180) -> None:
        load_environment()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("缺少 GEMINI_API_KEY，请在 .env 中配置。")
        self.model = model
        self.timeout_sec = max(30, int(timeout_sec or 180))
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=self.timeout_sec * 1000),
        )

    def analyze_audio(
        self,
        audio_path: Path,
        prompt: str,
        track_id: str,
        log: LogCallback | None = None,
    ) -> GeminiTrackResult:
        should_cleanup_upload = True
        self._log(log, f"上传音频到 Gemini：{audio_path.name}")
        try:
            uploaded_file = self.client.files.upload(file=str(audio_path))
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"Gemini 上传音频超时（{self.timeout_sec}s），请检查网络或调大超时时间。") from exc
        try:
            file_name = getattr(uploaded_file, "name", "")
            file_uri = getattr(uploaded_file, "uri", "")
            mime_type = getattr(uploaded_file, "mime_type", "")
            state = getattr(uploaded_file, "state", "")
            self._log(log, f"上传完成：name={file_name} mime={mime_type} state={state}")
            if file_uri:
                self._log(log, "Gemini 已返回文件 URI，说明音频已被服务端接收。")
            self._verify_uploaded_file(file_name, log)
            self._log(log, f"发送流式生成请求：{self.model}，等待上限 {self.timeout_sec}s")
            text_parts: list[str] = []
            first_chunk_at: float | None = None
            chunk_count = 0
            used_fallback = False
            started_at = time.monotonic()
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=[
                        uploaded_file,
                        types.Part.from_text(text=prompt),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                for chunk in stream:
                    chunk_text = chunk.text or ""
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                        self._log(log, f"收到 Gemini 首个响应 chunk，用时 {first_chunk_at - started_at:.1f}s")
                    if chunk_text:
                        text_parts.append(chunk_text)
                        chunk_count += 1
                        if chunk_count == 1 or chunk_count % 10 == 0:
                            self._log(log, f"Gemini 正在流式返回：已收到 {sum(len(part) for part in text_parts)} 字符")
                text = "".join(text_parts)
            except httpx.TimeoutException as exc:
                elapsed = time.monotonic() - started_at
                should_cleanup_upload = False
                self._log(
                    log,
                    f"Gemini 请求超时：实际等待 {elapsed:.1f}s，异常类型 {type(exc).__name__}；"
                    "本地连接已断开，跳过远端文件清理以便立即结束。",
                )
                raise RuntimeError(
                    f"Gemini 请求超时（配置 {self.timeout_sec}s，实际等待 {elapsed:.1f}s），"
                    "可以在配置里调大超时时间或换更快模型。"
                ) from exc
            except httpx.RemoteProtocolError as exc:
                elapsed = time.monotonic() - started_at
                self._log(
                    log,
                    f"Gemini 流式连接被服务端断开：实际等待 {elapsed:.1f}s，异常类型 {type(exc).__name__}，{exc}",
                )
                if first_chunk_at is not None:
                    raise
                self._log(log, "流式请求未收到任何 chunk，切换普通生成请求重试一次。")
                used_fallback = True
                retry_started_at = time.monotonic()
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[
                            uploaded_file,
                            types.Part.from_text(text=prompt),
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    text = response.text or ""
                    self._log(log, f"普通生成请求返回，用时 {time.monotonic() - retry_started_at:.1f}s，收到 {len(text)} 字符")
                except httpx.TimeoutException as retry_exc:
                    retry_elapsed = time.monotonic() - retry_started_at
                    should_cleanup_upload = False
                    self._log(
                        log,
                        f"普通生成请求超时：实际等待 {retry_elapsed:.1f}s，异常类型 {type(retry_exc).__name__}",
                    )
                    raise RuntimeError(
                        f"Gemini 普通生成请求超时（配置 {self.timeout_sec}s，实际等待 {retry_elapsed:.1f}s）。"
                    ) from retry_exc
                except Exception as retry_exc:
                    retry_elapsed = time.monotonic() - retry_started_at
                    self._log(
                        log,
                        f"普通生成请求也失败：实际等待 {retry_elapsed:.1f}s，异常类型 {type(retry_exc).__name__}，{retry_exc}",
                    )
                    raise
            except Exception as exc:
                elapsed = time.monotonic() - started_at
                self._log(log, f"Gemini 流式请求异常：实际等待 {elapsed:.1f}s，异常类型 {type(exc).__name__}，{exc}")
                raise
            if first_chunk_at is None and not used_fallback:
                self._log(log, "Gemini 流式请求结束，但没有收到任何文本 chunk。")
            response_mode = "普通生成 fallback" if used_fallback else "流式响应"
            self._log(log, f"Gemini {response_mode}结束，共收到 {len(text)} 字符，准备解析 JSON")
            append_jsonl(
                RAW_RESPONSES_PATH,
                {
                    "track_id": track_id,
                    "source_audio_path": str(audio_path),
                    "response": text,
                },
            )
            payload = extract_json(text)
            payload["track_id"] = payload.get("track_id") or track_id
            return GeminiTrackResult.model_validate(payload)
        finally:
            if should_cleanup_upload:
                try:
                    self._log(log, "清理 Gemini 上传文件")
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

    def _log(self, log: LogCallback | None, message: str) -> None:
        if log:
            log(message)

    def _verify_uploaded_file(self, file_name: str, log: LogCallback | None) -> None:
        if not file_name:
            return
        try:
            file_info = self.client.files.get(name=file_name)
            state = getattr(file_info, "state", "")
            size_bytes = getattr(file_info, "size_bytes", None)
            self._log(log, f"Gemini 文件状态确认：state={state} size_bytes={size_bytes}")
        except Exception as exc:
            self._log(log, f"Gemini 文件状态确认失败：{exc}")


def list_available_gemini_models(api_key: str, timeout_sec: int = 30) -> list[str]:
    if not api_key.strip():
        raise RuntimeError("缺少 GEMINI_API_KEY。")
    client = genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=max(10, int(timeout_sec or 30)) * 1000),
    )
    names: list[str] = []
    for model in client.models.list():
        supported_actions = set(getattr(model, "supported_actions", []) or [])
        name = getattr(model, "name", "")
        if not name or "generateContent" not in supported_actions:
            continue
        names.append(name.replace("models/", ""))
    return sorted(names)
