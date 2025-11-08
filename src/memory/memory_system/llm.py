import asyncio

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Protocol
from openai import OpenAI

class LLMClient(Protocol):
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        ) -> str:
        ...

class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini", client: Optional[OpenAI] = None) -> None:
        self._client = client or OpenAI()
        self._model = model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None

        try:
            response = await asyncio.to_thread(
                self._client.responses.create,
                model=self._model,
                input=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            if hasattr(response, "output_text"):
                return response.output_text
        except (AttributeError, TypeError) as exc:
            last_error = exc

        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            message = response.choices[0].message
            return message["content"] if isinstance(message, dict) else message.content
        except AttributeError as exc:
            last_error = last_error or exc

        try:
            response = await asyncio.to_thread(
                self._client.ChatCompletion.create,
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message["content"]
        except Exception as exc:
            raise last_error or exc