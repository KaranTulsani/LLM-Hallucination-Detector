"""
Model-agnostic LLM client wrapper.

Every LLM call in the project goes through this class.
To swap providers (Groq → OpenAI → Anthropic), change only this file.
"""

from __future__ import annotations

import json
import os
import re
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
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self._client = Groq(api_key=self.api_key)

    # ── core completions ────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Return the assistant's text reply."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

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
