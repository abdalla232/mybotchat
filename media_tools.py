"""Blocking media helpers. Call these with asyncio.to_thread()."""
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def is_youtube_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return host in {"youtube.com", "youtu.be"} or host.endswith(".youtube.com")


def convert_to_mp3(source: Path, target: Path, timeout: int = 180) -> Path:
    """Convert Telegram voice/audio to a small format accepted by transcription."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "64k", str(target),
        ],
        check=True,
        timeout=timeout,
        capture_output=True,
    )
    return target


def download_youtube_audio(url: str, directory: Path, max_duration: int) -> Path:
    """Download one public YouTube video's audio and convert it to 64 kbps MP3."""
    if not is_youtube_url(url):
        raise ValueError("invalid_youtube_url")

    from yt_dlp import YoutubeDL

    common = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "retries": 2,
    }
    with YoutubeDL(common) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = int(info.get("duration") or 0)
    if info.get("is_live"):
        raise ValueError("live_video")
    if not duration or duration > max_duration:
        raise ValueError("video_too_long")

    options = {
        **common,
        "format": "bestaudio/best",
        "outtmpl": str(directory / "youtube.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    with YoutubeDL(options) as ydl:
        ydl.download([url])

    output = directory / "youtube.mp3"
    if not output.exists():
        raise RuntimeError("youtube_download_failed")
    return output
