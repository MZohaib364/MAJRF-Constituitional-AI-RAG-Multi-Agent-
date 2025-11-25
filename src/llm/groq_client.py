from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from groq import Groq


class GroqLLM:
    """Lightweight wrapper around the Groq Chat Completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        timeout: int = 30,
    ) -> None:
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Please export it before using GroqLLM.")
        self.client = Groq(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]], response_format: Optional[Dict[str, Any]] = None) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format=response_format or {"type": "json_object"},
        )
        return completion.choices[0].message.content

    def structured_response(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.chat(messages)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"analysis": raw}

