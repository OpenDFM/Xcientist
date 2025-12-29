import asyncio
import json
import requests
from typing import Any, Dict, List, Optional, Protocol

from openai import OpenAI


JsonSchema = Dict[str, Any]


class LLMClient(Protocol):
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        *,
        json_schema: Optional[JsonSchema] = None,
        schema_name: str = "working-slots",
        strict: bool = True,
        force_json_object: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str:
        ...


class OpenAIClient:
    """
    backend:
      - "openai": uses OpenAI SDK (Responses preferred, then ChatCompletions fallback)
      - "vllm":  sends HTTP to vLLM OpenAI-compatible server (/v1/chat/completions)

    Structured outputs:
      - If json_schema is provided:
          OpenAI Responses: text.format = {"type":"json_schema", ...}
          ChatCompletions: response_format = {"type":"json_schema", "json_schema": {...}}
          vLLM:            response_format first; fallback to structured_outputs if needed
      - Else if force_json_object=True:
          OpenAI Responses: text.format = {"type":"json_object"}
          ChatCompletions: response_format = {"type":"json_object"}
          vLLM:            response_format = {"type":"json_object"}
      - Else:
          normal text generation (no format constraints)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client: Optional[OpenAI] = None,
        *,
        backend: str = "openai",
        vllm_url: str = "http://localhost:8014",
        vllm_model: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self._backend = backend.lower().strip()
        self._model = model
        self._timeout = timeout

        if self._backend == "openai":
            self._client = client or OpenAI()
        else:
            self._client = client

        self._vllm_url = vllm_url.rstrip("/") if vllm_url else None
        self._vllm_model = vllm_model

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.01,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        *,
        json_schema: Optional[JsonSchema] = None,
        schema_name: str = "working-slots",
        strict: bool = True,
        force_json_object: bool = False,
        stop: Optional[List[str]] = None,
    ) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await self._complete_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    strict=strict,
                    force_json_object=force_json_object,
                    stop=stop,
                )
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    raise
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
        *,
        json_schema: Optional[JsonSchema],
        schema_name: str,
        strict: bool,
        force_json_object: bool,
        stop: Optional[List[str]],
    ) -> str:
        if self._backend == "vllm":
            return await self._complete_once_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=json_schema,
                schema_name=schema_name,
                strict=strict,
                force_json_object=force_json_object,
                stop=stop,
            )
        else:
            return await self._complete_once_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=json_schema,
                schema_name=schema_name,
                strict=strict,
                force_json_object=force_json_object,
            )

    async def _complete_once_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        *,
        json_schema: Optional[JsonSchema],
        schema_name: str,
        strict: bool,
        force_json_object: bool,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None

        # ---------- 1) Try Responses API ----------
        try:
            resp_kwargs: Dict[str, Any] = dict(
                model=self._model,
                input=messages,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            # Responses uses text.format
            if json_schema is not None:
                resp_kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": strict,
                    }
                }
            elif force_json_object:
                resp_kwargs["text"] = {"format": {"type": "json_object"}}

            response = await asyncio.to_thread(self._client.responses.create, **resp_kwargs)
            if hasattr(response, "output_text"):
                return response.output_text
        except (AttributeError, TypeError) as exc:
            last_error = exc
        except Exception as exc:
            # Responses exists but request failed for other reasons; keep error and continue fallback
            last_error = exc

        # ---------- 2) Fallback: Chat Completions ----------
        try:
            chat_kwargs: Dict[str, Any] = dict(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # ChatCompletions uses response_format
            if json_schema is not None:
                chat_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": strict,
                    },
                }
            elif force_json_object:
                chat_kwargs["response_format"] = {"type": "json_object"}

            response = await asyncio.to_thread(self._client.chat.completions.create, **chat_kwargs)
            message = response.choices[0].message
            return message["content"] if isinstance(message, dict) else message.content
        except Exception as exc:
            last_error = last_error or exc

        # ---------- 3) Last resort: legacy ChatCompletion.create ----------
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

    async def _complete_once_vllm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        *,
        json_schema: Optional[JsonSchema],
        schema_name: str,
        strict: bool,
        force_json_object: bool,
        stop: Optional[List[str]],
    ) -> str:
        if not self._vllm_url:
            raise RuntimeError("vLLM backend selected but vllm_url is not set.")

        model_name = self._vllm_model or self._model

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "enable_thinking": False,
        }
        if stop:
            payload["stop"] = stop

        # Prefer OpenAI-style response_format when possible
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": strict,
                },
            }
        elif force_json_object:
            payload["response_format"] = {"type": "json_object"}

        def _post(p: Dict[str, Any]):
            return requests.post(
                f"{self._vllm_url}/v1/chat/completions",
                json=p,
                timeout=self._timeout,
            )

        response = await asyncio.to_thread(_post, payload)

        # If vLLM rejects response_format, fallback to structured_outputs (some versions prefer this)
        if response.status_code != 200 and json_schema is not None:
            payload2 = dict(payload)
            payload2.pop("response_format", None)
            payload2["structured_outputs"] = {"json": json_schema}
            response = await asyncio.to_thread(_post, payload2)

        status = response.status_code
        try:
            data = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse JSON from vLLM (status={status}): {response.text[:500]}"
            ) from e

        if status != 200:
            raise RuntimeError(f"vLLM returned error (status={status}): {data}")

        if "choices" not in data:
            raise RuntimeError(f"vLLM response missing 'choices': {data}")

        choice = data["choices"][0]
        message = choice.get("message", {})
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
        else:
            content = (getattr(message, "content", "") or "").strip()

        return content
