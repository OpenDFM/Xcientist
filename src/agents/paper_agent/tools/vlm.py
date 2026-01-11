import base64
import json
import os
import time
from typing import Any, Dict, List, Optional
import httpx
import httpcore
import openai

from agents import function_tool
from openai import OpenAI


def _read_bytes(path: str, max_bytes: int = 20_000_000) -> bytes:
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    return data


def _b64_png_data_url(png_path: str) -> str:
    b = _read_bytes(png_path)
    s = base64.b64encode(b).decode("utf-8")
    return "data:image/png;base64," + s


def _try_parse_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    
    # Try to clean markdown code blocks
    if "```" in s:
        # Extract content between first ```json (or just ```) and the next ```
        import re
        # Pattern to match ```json ... ``` or ``` ... ```
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, s)
        if match:
            s = match.group(1).strip()
            
    try:
        return json.loads(s)
    except Exception:
        return None


def vlm_layout_review_from_pages(
    page_png_paths: List[str],
    model: str,
    api_key: str,
    base_url: str = "",
    extra_body: Optional[dict] = None,
    max_pages: int = 3,
) -> Dict[str, Any]:
    model = str(model or "").strip()
    api_key = str(api_key or "").strip()
    base_url = str(base_url or "").strip()
    if not model or not api_key:
        return {
            "success": False,
            "error": "VLM model and API key are required for VLM review",
        }

    pages = [p for p in (page_png_paths or []) if p and os.path.exists(p)]
    pages = pages[: int(max_pages)]
    if not pages:
        return {"success": False, "error": "no page PNGs to review"}

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    prompt = (
        "You are a LaTeX paper layout reviewer for a top-tier ML conference (ICML).\n"
        "Your job is to enforce strict aesthetic and structural guidelines defined in the Constitution.\n"
        "\n"
        "Check these specific points:\n"
        "1. **Page Limit Visual Check**: The paper main body should ideally fill 8 pages. If it looks like 7 pages or less, flag it as a MAJOR issue.\n"
        "2. **Visual Hierarchy**: Are section headers clear? Are bold terms used effectively without overusing?\n"
        "3. **Figures**: Are they vector graphics (sharp) or blurry? Are fonts in figures legible (comparable to caption size)?\n"
        "4. **Tables**: Do they use booktabs (no vertical lines)? Are best results bolded?\n"
        "5. **Overall Density**: Is the text too dense (wall of text) or too sparse?\n"
        "\n"
        "Focus ONLY on aesthetics, layout, and visual structure. Do NOT critique technical content text.\n"
        "Give actionable suggestions tied to page numbers.\n"
        "\n"
        "Return JSON with keys:\n"
        "- reviewed_pages: [int]\n"
        "- findings: list of {page:int, severity:string(critical|major|minor), category:string, problem:string, suggestion:string}\n"
    )

    content: List[dict] = [{"type": "text", "text": prompt}]
    reviewed_pages = []
    for i, p in enumerate(pages, 1):
        reviewed_pages.append(i)
        content.append(
            {"type": "image_url", "image_url": {"url": _b64_png_data_url(p)}}
        )

    try:
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
        }
        if isinstance(extra_body, dict) and extra_body:
            create_kwargs["extra_body"] = extra_body
        
        resp = None
        max_retries = 10
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    **create_kwargs,
                )
                break
            except (
                httpx.RemoteProtocolError, 
                httpcore.RemoteProtocolError, 
                openai.APIConnectionError, 
                openai.InternalServerError, 
                openai.RateLimitError
            ) as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2.0 * (1.5 ** attempt)
                time.sleep(wait_time)
        
        text = ""
        try:
            text = resp.choices[0].message.content or ""
        except Exception:
            text = str(resp)
        parsed = _try_parse_json(text)
        if parsed is None:
            return {
                "success": True,
                "reviewed_pages": reviewed_pages,
                "findings": [],
                "raw": text,
                "warning": "model did not return JSON",
            }
        return {"success": True, **parsed}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


@function_tool
def vlm_layout_review(pages_dir: str, max_pages: int = 3) -> dict:
    """
    Run VLM-based layout review from PNG pages in pages_dir.

    Model selection priority:
    1) paper_agent run config models["vlm"] (if PAPER_AGENT_RUN_CONFIG_PATH is set)
    2) PAPER_VLM_MODEL / PAPER_AGENT_VLM_MODEL env var
    """
    pages_dir = os.path.abspath(str(pages_dir or ""))
    model = ""
    try:
        from src.agents.paper_agent.utils.config import get_runtime_config

        cfg = get_runtime_config()
        cfg_models = getattr(cfg, "models", None) or {}
        model = str(cfg_models.get("vlm") or "").strip()
    except Exception:
        model = ""
    if not model:
        from src.agents.paper_agent.utils.config import PAPER_VLM_MODEL

        model = str(PAPER_VLM_MODEL or "").strip()

    from src.agents.paper_agent.utils.config import get_openai_config

    provider_cfg = get_openai_config(model=model)
    api_key = str(provider_cfg.get("api_key", "") or "").strip()
    base_url = str(provider_cfg.get("base_url", "") or "").strip()
    extra_body = provider_cfg.get("extra_body", None)

    if not os.path.isdir(pages_dir):
        return {"success": False, "error": f"pages_dir not found: {pages_dir}"}

    pngs = [
        os.path.join(pages_dir, x)
        for x in sorted(os.listdir(pages_dir))
        if x.lower().endswith(".png")
    ]
    res = vlm_layout_review_from_pages(
        pngs,
        model=model,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        max_pages=int(max_pages),
    )
    if isinstance(res, dict) and res.get("success") is True:
        reviewed = res.get("reviewed_pages", None)
        findings = res.get("findings", None)
        reviewed_n = len(reviewed) if isinstance(reviewed, list) else 0
        findings_n = len(findings) if isinstance(findings, list) else 0
        res["message"] = f"VLM reviewed {reviewed_n} page(s), findings={findings_n}"
    return res
