"""Async wrapper around the Groq chat API.

Why this file exists:

1. The original code called ``groq.chat.completions.create`` (a *blocking*
   network call) from inside ``async def`` functions. That blocks FastAPI's
   event loop, so while one user's LLM call is in flight, every other
   request is frozen. Here we use Groq's native async client
   (``AsyncGroq``), so the event loop stays free.

2. External APIs fail transiently (rate limits, 5xx, dropped connections).
   We retry with exponential backoff instead of crashing the whole run.

The rest of the codebase talks to the LLM only through ``LLMClient`` and
never imports ``groq`` directly, so it can be swapped or mocked easily.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from groq import AsyncGroq
from groq import APIError, APIConnectionError, RateLimitError

from config import get_settings

logger = logging.getLogger(__name__)

# Errors that are worth retrying (transient). A bad request (400) is not here
# on purpose: retrying it would just fail again and waste time/quota.
_RETRYABLE = (APIConnectionError, RateLimitError)


class LLMError(RuntimeError):
    """Raised when the LLM call fails after all retries."""


class LLMClient:
    def __init__(self, api_key: str | None = None, client: AsyncGroq | None = None):
        settings = get_settings()
        # ``client`` injection is what lets tests pass a fake with no network.
        self._client = client or AsyncGroq(api_key=api_key or settings.groq_api_key)
        self._model = settings.llm_model
        self._max_retries = settings.llm_max_retries
        self._base_delay = settings.retry_base_delay
        self._timeout = settings.request_timeout

    async def complete(self, prompt: str, temperature: float) -> str:
        """One-shot completion. Returns the full text."""
        resp = await self._call(prompt, temperature, stream=False)
        return resp.choices[0].message.content or ""

    async def stream(self, prompt: str, temperature: float) -> AsyncIterator[str]:
        """Yield the completion token-by-token as it is generated."""
        stream = await self._call(prompt, temperature, stream=True)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def _call(self, prompt: str, temperature: float, stream: bool):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=stream,
                    timeout=self._timeout,
                )
            except _RETRYABLE as exc:
                last_exc = exc
                delay = self._base_delay * (2 ** attempt)  # 0.5, 1.0, 2.0 ...
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1, self._max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
            except APIError as exc:
                # Non-retryable API error (e.g. malformed request). Fail fast.
                raise LLMError(f"LLM request failed: {exc}") from exc
        raise LLMError(f"LLM call failed after {self._max_retries} retries") from last_exc
