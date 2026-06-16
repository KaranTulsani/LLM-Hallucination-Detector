"""
Model-agnostic LLM client wrapper.

Every LLM call in the project goes through this class.
To swap providers (Groq → OpenAI → Anthropic), change only this file.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMClient:
    """Thin abstraction over any chat-completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        raw_keys = api_key or os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY", "")
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        
        self._key_index = 0
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.api_keys:
            self._client = Groq(api_key="")
            return
        current_key = self.api_keys[self._key_index]
        self._client = Groq(api_key=current_key)

    def _rotate_key(self):
        if len(self.api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            self._init_client()

    # ── core completions ────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant's text reply."""
        max_attempts = max(3, len(self.api_keys))
        for attempt in range(max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    # If we have multiple keys, rotate first
                    if len(self.api_keys) > 1 and attempt < len(self.api_keys) - 1:
                        self._rotate_key()
                        continue
                    
                    # Otherwise, backoff and retry
                    if attempt < max_attempts - 1:
                        retry_seconds = 5.0
                        match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e))
                        if match:
                            retry_seconds = float(match.group(1)) + 0.5
                        retry_seconds = min(15.0, retry_seconds)
                        time.sleep(retry_seconds)
                        continue
                raise e
        return ""

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """Yield text tokens one-by-one using Groq's streaming mode.

        Each yielded value is a raw string token (may be empty string).
        Caller is responsible for concatenating tokens into the full response.
        """
        max_attempts = max(3, len(self.api_keys))
        for attempt in range(max_attempts):
            try:
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    token = delta.content or ""
                    yield token
                return
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if len(self.api_keys) > 1 and attempt < len(self.api_keys) - 1:
                        self._rotate_key()
                        continue
                    if attempt < max_attempts - 1:
                        retry_seconds = 5.0
                        match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e))
                        if match:
                            retry_seconds = float(match.group(1)) + 0.5
                        retry_seconds = min(15.0, retry_seconds)
                        time.sleep(retry_seconds)
                        continue
                raise e

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Return parsed JSON from the assistant's reply.

        Attempts to extract a JSON object even if the model wraps it
        in markdown fences or extra prose.
        """
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self._extract_json(raw)

    # ── convenience helpers ─────────────────────────────────────────

    def generate_response(self, query: str, temperature: float = 0.0) -> str:
        """Simple single-turn generation."""
        return self.chat(
            [{"role": "user", "content": query}],
            temperature=temperature,
        )

    def generate_response_stream(self, query: str, temperature: float = 0.0):
        """Stream tokens for a single-turn generation."""
        yield from self.chat_stream(
            [{"role": "user", "content": query}],
            temperature=temperature,
        )

    def generate_multiple(
        self, query: str, n: int = 5, temperature: float = 0.7
    ) -> list[str]:
        """Generate *n* independent responses for the same query."""
        return [
            self.generate_response(query, temperature=temperature) for _ in range(n)
        ]

    # ── internal ────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Best-effort JSON extraction from potentially messy LLM output."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Find first { … } block
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass

        return {"raw": text, "parse_error": True}
