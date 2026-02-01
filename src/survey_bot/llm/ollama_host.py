"""Auto-detect Ollama host: local first, then remote fallback."""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

LOCAL_OLLAMA = "http://localhost:11434"
_cached_host: str | None = None


def detect_ollama_host() -> str:
    """Return reachable Ollama host URL.

    Tries local Ollama first. If unreachable, falls back to
    OLLAMA_REMOTE_HOST env var. If OLLAMA_HOST is explicitly set,
    uses that directly without autodetection.
    """
    global _cached_host
    if _cached_host is not None:
        return _cached_host

    # If user explicitly set OLLAMA_HOST, respect it
    explicit = os.environ.get("OLLAMA_HOST")
    if explicit:
        _cached_host = explicit.rstrip("/")
        logger.info("Using explicit OLLAMA_HOST: %s", _cached_host)
        return _cached_host

    # Try local
    try:
        resp = httpx.get(f"{LOCAL_OLLAMA}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            _cached_host = LOCAL_OLLAMA
            logger.info("Local Ollama detected at %s", LOCAL_OLLAMA)
            return _cached_host
    except Exception:
        pass

    # Fallback to remote
    remote = os.environ.get("OLLAMA_REMOTE_HOST", "").rstrip("/")
    if remote:
        _cached_host = remote
        logger.info("Using remote Ollama at %s", remote)
        return _cached_host

    # Nothing available, return local as default (will fail later with clear error)
    _cached_host = LOCAL_OLLAMA
    logger.warning("No Ollama server found. Defaulting to %s", LOCAL_OLLAMA)
    return _cached_host


async def detect_ollama_host_async() -> str:
    """Async version of detect_ollama_host."""
    global _cached_host
    if _cached_host is not None:
        return _cached_host

    explicit = os.environ.get("OLLAMA_HOST")
    if explicit:
        _cached_host = explicit.rstrip("/")
        logger.info("Using explicit OLLAMA_HOST: %s", _cached_host)
        return _cached_host

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{LOCAL_OLLAMA}/api/tags")
            if resp.status_code == 200:
                _cached_host = LOCAL_OLLAMA
                logger.info("Local Ollama detected at %s", LOCAL_OLLAMA)
                return _cached_host
    except Exception:
        pass

    remote = os.environ.get("OLLAMA_REMOTE_HOST", "").rstrip("/")
    if remote:
        _cached_host = remote
        logger.info("Using remote Ollama at %s", remote)
        return _cached_host

    _cached_host = LOCAL_OLLAMA
    logger.warning("No Ollama server found. Defaulting to %s", LOCAL_OLLAMA)
    return _cached_host
