from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import PROMPT_CACHE_PATH, RAW_RESPONSES_PATH, append_jsonl, load_environment
from src.models import GeminiTrackResult


LogCallback = Callable[[str], None]


class GeminiRetryableError(RuntimeError):
    pass


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
PROMPT_CACHE_DISABLED_STATUSES = {"too_small", "generation_failed"}


def retryable_api_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in RETRYABLE_STATUS_CODES:
        return True
    text = str(exc)
    return any(
        marker in text
        for marker in [
            "429",
            "500",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
        ]
    )


def prompt_cache_too_small_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return all(marker in text for marker in ["cached", "content", "small"]) or all(
        marker in text for marker in ["cache", "minimum"]
    )


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


def usage_metadata_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if isinstance(usage, dict):
        return {key: value for key, value in usage.items() if value is not None}
    return None


def usage_summary(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "usage metadata 未返回"
    fields = [
        ("prompt", "prompt_token_count"),
        ("cached", "cached_content_token_count"),
        ("output", "candidates_token_count"),
        ("thoughts", "thoughts_token_count"),
        ("total", "total_token_count"),
    ]
    parts = [f"{label}={usage.get(key)}" for label, key in fields if usage.get(key) is not None]
    return "tokens: " + ", ".join(parts) if parts else "usage metadata 未包含 token 字段"


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
        self.disabled_prompt_caches: set[str] = set()

    def analyze_audio(
        self,
        audio_path: Path,
        prompt: str,
        track_id: str,
        log: LogCallback | None = None,
        cached_content_name: str | None = None,
        fallback_prompt: str | None = None,
    ) -> GeminiTrackResult:
        if cached_content_name and fallback_prompt and cached_content_name in self.disabled_prompt_caches:
            self._log(log, "本轮已禁用 prompt cache，直接使用完整 prompt。")
            return self.analyze_audio(audio_path, fallback_prompt, track_id, log, None, None)
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
            cache_note = f"，使用 prompt cache {cached_content_name}" if cached_content_name else ""
            self._log(log, f"发送流式生成请求：{self.model}，等待上限 {self.timeout_sec}s{cache_note}")
            text_parts: list[str] = []
            first_chunk_at: float | None = None
            chunk_count = 0
            used_fallback = False
            usage_metadata: dict[str, Any] | None = None
            last_usage_line = ""
            started_at = time.monotonic()
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=[
                        uploaded_file,
                        types.Part.from_text(text=prompt),
                    ],
                    config=self._generation_config(cached_content_name),
                )
                for chunk in stream:
                    chunk_usage = usage_metadata_to_dict(getattr(chunk, "usage_metadata", None))
                    if chunk_usage:
                        usage_metadata = chunk_usage
                        current_usage_line = usage_summary(usage_metadata)
                        if current_usage_line != last_usage_line:
                            self._log(log, current_usage_line)
                            last_usage_line = current_usage_line
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
                if cached_content_name and fallback_prompt:
                    return self._retry_without_prompt_cache(
                        audio_path,
                        fallback_prompt,
                        track_id,
                        log,
                        cached_content_name,
                        f"流式请求超时（{elapsed:.1f}s）",
                    )
                raise GeminiRetryableError(
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
                    raise GeminiRetryableError(f"Gemini 流式连接中断：{exc}") from exc
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
                        config=self._generation_config(cached_content_name),
                    )
                    text = response.text or ""
                    usage_metadata = usage_metadata_to_dict(getattr(response, "usage_metadata", None))
                    self._log(log, f"普通生成请求返回，用时 {time.monotonic() - retry_started_at:.1f}s，收到 {len(text)} 字符")
                except httpx.TimeoutException as retry_exc:
                    retry_elapsed = time.monotonic() - retry_started_at
                    should_cleanup_upload = False
                    self._log(
                        log,
                        f"普通生成请求超时：实际等待 {retry_elapsed:.1f}s，异常类型 {type(retry_exc).__name__}",
                    )
                    if cached_content_name and fallback_prompt:
                        return self._retry_without_prompt_cache(
                            audio_path,
                            fallback_prompt,
                            track_id,
                            log,
                            cached_content_name,
                            f"普通生成请求超时（{retry_elapsed:.1f}s）",
                        )
                    raise GeminiRetryableError(
                        f"Gemini 普通生成请求超时（配置 {self.timeout_sec}s，实际等待 {retry_elapsed:.1f}s）。"
                    ) from retry_exc
                except Exception as retry_exc:
                    retry_elapsed = time.monotonic() - retry_started_at
                    self._log(
                        log,
                        f"普通生成请求也失败：实际等待 {retry_elapsed:.1f}s，异常类型 {type(retry_exc).__name__}，{retry_exc}",
                    )
                    if retryable_api_error(retry_exc):
                        raise GeminiRetryableError(f"Gemini 普通生成请求可重试错误：{retry_exc}") from retry_exc
                    raise
            except (genai_errors.APIError, genai_errors.ClientError, genai_errors.ServerError) as exc:
                if cached_content_name and fallback_prompt:
                    return self._retry_without_prompt_cache(
                        audio_path,
                        fallback_prompt,
                        track_id,
                        log,
                        cached_content_name,
                        f"{type(exc).__name__}，{exc}",
                    )
                elapsed = time.monotonic() - started_at
                self._log(log, f"Gemini API 请求异常：实际等待 {elapsed:.1f}s，异常类型 {type(exc).__name__}，{exc}")
                if retryable_api_error(exc):
                    raise GeminiRetryableError(f"Gemini API 可重试错误：{exc}") from exc
                raise
            except Exception as exc:
                if cached_content_name and fallback_prompt:
                    return self._retry_without_prompt_cache(
                        audio_path,
                        fallback_prompt,
                        track_id,
                        log,
                        cached_content_name,
                        f"{type(exc).__name__}，{exc}",
                    )
                elapsed = time.monotonic() - started_at
                self._log(log, f"Gemini 流式请求异常：实际等待 {elapsed:.1f}s，异常类型 {type(exc).__name__}，{exc}")
                raise
            if first_chunk_at is None and not used_fallback:
                self._log(log, "Gemini 流式请求结束，但没有收到任何文本 chunk。")
            response_mode = "普通生成 fallback" if used_fallback else "流式响应"
            self._log(log, f"Gemini {response_mode}结束，共收到 {len(text)} 字符，准备解析 JSON")
            self._log(log, usage_summary(usage_metadata))
            append_jsonl(
                RAW_RESPONSES_PATH,
                {
                    "track_id": track_id,
                    "source_audio_path": str(audio_path),
                    "response": text,
                    "usage_metadata": usage_metadata,
                    "cached_content_name": cached_content_name,
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

    def _retry_without_prompt_cache(
        self,
        audio_path: Path,
        fallback_prompt: str,
        track_id: str,
        log: LogCallback | None,
        cached_content_name: str,
        reason: str,
    ) -> GeminiTrackResult:
        self.disabled_prompt_caches.add(cached_content_name)
        self._mark_prompt_cache_generation_failed(cached_content_name, reason)
        self._log(log, f"使用 prompt cache 请求失败，已禁用该 cache 并切换完整 prompt 重试：{reason}")
        return self.analyze_audio(audio_path, fallback_prompt, track_id, log, None, None)

    def _generation_config(self, cached_content_name: str | None = None) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            cached_content=cached_content_name,
        )

    def get_or_create_prompt_cache(
        self,
        cache_hash: str,
        static_prompt: str,
        ttl_sec: int,
        log: LogCallback | None = None,
    ) -> str | None:
        cached = self._load_prompt_cache_record()
        if (
            cached
            and cached.get("hash") == cache_hash
            and cached.get("model") == self.model
            and cached.get("status") in PROMPT_CACHE_DISABLED_STATUSES
        ):
            reason = cached.get("error") or cached.get("status")
            self._log(log, f"prompt cache 上次不可用，跳过创建并使用普通 prompt：{reason}")
            return None
        if (
            cached
            and cached.get("hash") == cache_hash
            and cached.get("model") == self.model
            and cached.get("name")
        ):
            try:
                self.client.caches.get(name=cached["name"])
                self._log(log, f"复用 prompt cache：{cached['name']}")
                return cached["name"]
            except Exception as exc:
                self._log(log, f"已有 prompt cache 不可用，将重建：{exc}")

        try:
            self._log(log, "创建 prompt cache：固定分类规则和系统提示")
            cache = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    display_name=f"music-classifier-{cache_hash[:12]}",
                    contents=static_prompt,
                    ttl=f"{max(300, int(ttl_sec or 86400))}s",
                ),
            )
            name = getattr(cache, "name", "")
            if not name:
                self._log(log, "prompt cache 创建成功但没有返回 name，改用普通 prompt。")
                return None
            self._save_prompt_cache_record({"hash": cache_hash, "model": self.model, "name": name})
            self._log(log, f"prompt cache 创建成功：{name}")
            return name
        except Exception as exc:
            self._log(log, f"prompt cache 创建失败，改用普通 prompt：{type(exc).__name__}，{exc}")
            if prompt_cache_too_small_error(exc):
                self._log(log, "prompt cache 内容低于 Gemini 最小长度，已记录；后续同配置会直接使用普通 prompt。")
                self._save_prompt_cache_record(
                    {
                        "hash": cache_hash,
                        "model": self.model,
                        "status": "too_small",
                        "error": str(exc),
                    }
                )
            return None

    def _load_prompt_cache_record(self) -> dict | None:
        if not PROMPT_CACHE_PATH.exists():
            return None
        try:
            return json.loads(PROMPT_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_prompt_cache_record(self, payload: dict) -> None:
        PROMPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROMPT_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _mark_prompt_cache_generation_failed(self, cached_content_name: str, reason: str) -> None:
        record = self._load_prompt_cache_record() or {}
        if record.get("name") != cached_content_name:
            return
        record["status"] = "generation_failed"
        record["error"] = reason
        self._save_prompt_cache_record(record)

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
