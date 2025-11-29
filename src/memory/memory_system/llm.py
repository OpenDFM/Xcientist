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
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_retries: int = 5,
        retry_delay: float = 1.0,
    ) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await self._complete_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    raise last_error
                # simple exponential backoff between attempts
                delay = retry_delay * (2 ** attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError("LLM completion failed.")

    async def _complete_once(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
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
