from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}


def clip_extension(clip_format: str) -> str:
    """切片输出扩展名，默认 wav（无损，供训练），flac 作为兼容选项。"""
    return ".flac" if (clip_format or "wav").lower() == "flac" else ".wav"


def clip_codec(clip_format: str) -> str:
    """切片输出 ffmpeg 编码器，与 clip_extension 对应。"""
    return "flac" if (clip_format or "wav").lower() == "flac" else "pcm_s16le"


def scan_audio_files(raw_audio_dir: Path) -> list[Path]:
    if not raw_audio_dir.exists():
        raw_audio_dir.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(
        path
        for path in raw_audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"找不到 {name}，请先安装并加入 PATH。")


def run_audio_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def get_audio_duration(audio_path: Path) -> float:
    require_binary("ffprobe")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = run_audio_command(command)
    return round(float(result.stdout.strip()), 3)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def create_gemini_upload_audio(source_path: Path, target_path: Path) -> Path:
    require_binary("ffmpeg")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "32000",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(target_path),
    ]
    run_audio_command(command)
    return target_path


def clip_audio(
    source_path: Path,
    clip_path: Path,
    start_sec: float,
    end_sec: float,
    source_duration_sec: float,
    padding_sec: float = 0.0,
    audio_codec: str = "pcm_s16le",
) -> float:
    require_binary("ffmpeg")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError(f"无效时间戳 start={start_sec}, end={end_sec}")
    padded_start = max(0.0, start_sec - padding_sec)
    padded_end = min(source_duration_sec, end_sec + padding_sec)
    duration = max(0.0, padded_end - padded_start)
    if duration <= 0:
        raise ValueError(f"裁剪时长无效 start={padded_start}, end={padded_end}")

    clip_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{padded_start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-acodec",
        audio_codec,
        str(clip_path),
    ]
    run_audio_command(command)
    return round(duration, 1)


def stable_track_id(audio_path: Path, raw_root: Path) -> str:
    try:
        relative = audio_path.relative_to(raw_root)
    except ValueError:
        relative = audio_path.name
    stem = Path(relative).with_suffix("").as_posix()
    digest = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:6]
    safe_stem = "".join(char if char.isalnum() else "_" for char in stem).strip("_")
    return safe_stem or digest


def seconds_to_mmss(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"
