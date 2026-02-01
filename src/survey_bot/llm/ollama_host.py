"""Auto-detect vision LLM backend: local Ollama or DashScope cloud."""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

LOCAL_OLLAMA = "http://localhost:11434"
_cached_host: Optional[str] = None
_cached_backend: Optional[str] = None  # "ollama" or "dashscope"


def is_running_on_server() -> bool:
    """Check if running on a cloud server (Render, Docker, etc.) vs local machine."""
    # Render sets RENDER=true automatically
    if os.environ.get("RENDER"):
        return True
    # Running inside Docker
    if os.path.exists("/.dockerenv"):
        return True
    # Common cloud env vars
    if any(os.environ.get(v) for v in ("DYNO", "FLY_APP_NAME", "RAILWAY_ENVIRONMENT")):
        return True
    return False


def _dashscope_configured() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def detect_ollama_host() -> str:
    """Return reachable Ollama host URL.

    Tries local Ollama first. If unreachable, falls back to
    OLLAMA_REMOTE_HOST env var. If OLLAMA_HOST is explicitly set,
    uses that directly without autodetection.
    """
    global _cached_host
    if _cached_host is not None:
        return _cached_host

    explicit = os.environ.get("OLLAMA_HOST")
    if explicit:
        _cached_host = explicit.rstrip("/")
        logger.info("Using explicit OLLAMA_HOST: %s", _cached_host)
        return _cached_host

    try:
        resp = httpx.get(f"{LOCAL_OLLAMA}/api/tags", timeout=2.0)
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


def detect_backend() -> str:
    """Detect which vision backend to use: 'ollama' or 'dashscope'.

    Priority:
    1. VISION_BACKEND env var (explicit override)
    2. Running on server (Render/Docker) + DASHSCOPE_API_KEY -> 'dashscope'
    3. Local Ollama reachable -> 'ollama'
    4. DASHSCOPE_API_KEY set -> 'dashscope'
    5. OLLAMA_REMOTE_HOST set -> 'ollama'
    6. Default to 'ollama'
    """
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend

    on_server = is_running_on_server()
    env_label = "server" if on_server else "local"
    logger.info("Environment detected: %s", env_label)

    explicit = os.environ.get("VISION_BACKEND", "").lower()
    if explicit in ("ollama", "dashscope"):
        _cached_backend = explicit
        logger.info("[%s] Using explicit VISION_BACKEND: %s", env_label, explicit)
        return _cached_backend

    # On server, prefer DashScope (no local Ollama available)
    if on_server and _dashscope_configured():
        _cached_backend = "dashscope"
        logger.info("[%s] Using DashScope backend (cloud deployment)", env_label)
        return _cached_backend

    # Try local Ollama
    if not on_server:
        try:
            resp = httpx.get(f"{LOCAL_OLLAMA}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                _cached_backend = "ollama"
                logger.info("[%s] Local Ollama detected, using ollama backend", env_label)
                return _cached_backend
        except Exception:
            pass

    # Check DashScope fallback
    if _dashscope_configured():
        _cached_backend = "dashscope"
        logger.info("[%s] Using DashScope backend (DASHSCOPE_API_KEY set)", env_label)
        return _cached_backend

    # Check remote Ollama
    if os.environ.get("OLLAMA_REMOTE_HOST"):
        _cached_backend = "ollama"
        logger.info("[%s] Using remote Ollama backend", env_label)
        return _cached_backend

    _cached_backend = "ollama"
    logger.warning("[%s] No vision backend detected. Defaulting to ollama.", env_label)
    return _cached_backend


async def detect_backend_async() -> str:
    """Async version of detect_backend."""
    global _cached_backend
    if _cached_backend is not None:
        return _cached_backend

    on_server = is_running_on_server()
    env_label = "server" if on_server else "local"
    logger.info("Environment detected: %s", env_label)

    explicit = os.environ.get("VISION_BACKEND", "").lower()
    if explicit in ("ollama", "dashscope"):
        _cached_backend = explicit
        logger.info("[%s] Using explicit VISION_BACKEND: %s", env_label, explicit)
        return _cached_backend

    # On server, prefer DashScope
    if on_server and _dashscope_configured():
        _cached_backend = "dashscope"
        logger.info("[%s] Using DashScope backend (cloud deployment)", env_label)
        return _cached_backend

    # Try local Ollama
    if not on_server:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{LOCAL_OLLAMA}/api/tags")
                if resp.status_code == 200:
                    _cached_backend = "ollama"
                    logger.info("[%s] Local Ollama detected, using ollama backend", env_label)
                    return _cached_backend
        except Exception:
            pass

    # Check DashScope fallback
    if _dashscope_configured():
        _cached_backend = "dashscope"
        logger.info("[%s] Using DashScope backend (DASHSCOPE_API_KEY set)", env_label)
        return _cached_backend

    # Check remote Ollama
    if os.environ.get("OLLAMA_REMOTE_HOST"):
        _cached_backend = "ollama"
        logger.info("[%s] Using remote Ollama backend", env_label)
        return _cached_backend

    _cached_backend = "ollama"
    logger.warning("[%s] No vision backend detected. Defaulting to ollama.", env_label)
    return _cached_backend


DASHSCOPE_MODELS = [
    "qwen3-vl-235b-a22b-thinking",  # Free tier, try first
    "qwen-vl-max-latest",
    "qwen-vl-plus-latest",
]


async def call_vision_llm(
    prompt: str,
    image_b64: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """Unified vision LLM call. Routes to Ollama or DashScope automatically.

    Args:
        prompt: Text prompt for the model.
        image_b64: Base64-encoded image.
        model: Model name override. For Ollama: e.g. 'qwen3-vl'. For DashScope: e.g. 'qwen-vl-max-latest'.
        host: Ollama host override (ignored for DashScope).
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.

    Returns:
        Model response text.
    """
    backend = await detect_backend_async()

    if backend == "dashscope":
        return await _call_dashscope(prompt, image_b64, model, temperature, max_tokens)
    else:
        return await _call_ollama(prompt, image_b64, model, host, temperature, max_tokens)


async def _call_ollama(
    prompt: str,
    image_b64: str,
    model: Optional[str] = None,
    host: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """Call Ollama /api/generate endpoint."""
    if host is None:
        host = await detect_ollama_host_async()
    if model is None:
        model = os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl")

    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
        )

        if response.status_code == 404:
            raise Exception(f"Model '{model}' not found. Run: ollama pull {model}")
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        result = response.json()
        text = result.get("response", "")

        # Some models output to 'thinking' field instead of 'response'
        if not text or not text.strip():
            thinking = result.get("thinking", "")
            if thinking and thinking.strip():
                logger.warning("Ollama response empty, using 'thinking' field (%d chars)", len(thinking))
                text = thinking

        return text


async def _call_dashscope(
    prompt: str,
    image_b64: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """Call DashScope vision API with automatic model fallback.

    Tries the preferred model first. If it fails (quota exceeded,
    model unavailable, rate limited), falls back to the next model
    in the list.
    """
    import asyncio
    import dashscope

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise Exception("DASHSCOPE_API_KEY not set. Required for DashScope backend.")

    # Use US region endpoint
    dashscope.base_http_api_url = "https://dashscope-us.aliyuncs.com/api/v1"

    # Build model list: explicit model first, then fallbacks
    if model is not None:
        models_to_try = [model] + [m for m in DASHSCOPE_MODELS if m != model]
    else:
        env_model = os.environ.get("DASHSCOPE_MODEL")
        if env_model:
            models_to_try = [env_model] + [m for m in DASHSCOPE_MODELS if m != env_model]
        else:
            models_to_try = list(DASHSCOPE_MODELS)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{image_b64}"},
                {"text": prompt},
            ],
        }
    ]

    last_error = None
    loop = asyncio.get_event_loop()

    for current_model in models_to_try:
        try:
            logger.info("Trying DashScope model: %s", current_model)

            response = await loop.run_in_executor(
                None,
                lambda m=current_model: dashscope.MultiModalConversation.call(
                    api_key=api_key,
                    model=m,
                    messages=messages,
                ),
            )

            if response.status_code != 200:
                error_msg = f"{response.status_code} - {response.code} - {response.message}"
                logger.warning("DashScope model %s failed: %s", current_model, error_msg)
                last_error = error_msg
                continue

            text = _extract_dashscope_text(response)
            if text:
                logger.info("DashScope model %s returned %d chars", current_model, len(text))
                return text

            logger.warning("DashScope model %s returned empty response", current_model)
            last_error = "empty response"
            continue

        except Exception as e:
            logger.warning("DashScope model %s error: %s", current_model, e)
            last_error = str(e)
            continue

    raise Exception(f"All DashScope models failed. Last error: {last_error}")


def _extract_dashscope_text(response: object) -> str:
    """Extract text from a DashScope MultiModalConversation response."""
    message = response["output"]["choices"][0]["message"]  # type: ignore[index]
    content = message.content

    # Extract text from content (list of dicts with "text" keys)
    text = ""
    if isinstance(content, list):
        text = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    else:
        text = str(content)

    # Thinking models may return empty content with reasoning in thinking_content
    if not text.strip():
        thinking = getattr(message, "thinking_content", None)
        if thinking:
            if isinstance(thinking, list):
                text = "".join(item.get("text", "") for item in thinking if isinstance(item, dict))
            else:
                text = str(thinking)
            logger.warning("DashScope response empty, using thinking_content (%d chars)", len(text))

    return text
