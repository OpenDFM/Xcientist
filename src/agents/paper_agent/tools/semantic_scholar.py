import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from agents import function_tool


from src.agents.paper_agent.utils.config import get_semantic_scholar_config


S2_API_BASE, S2_API_KEY = get_semantic_scholar_config()


def _http_get_json(
    url: str, headers: Optional[dict] = None, timeout_sec: int = 30
) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=int(timeout_sec)) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _default_headers() -> dict:
    h = {"User-Agent": "paper_agent/1.0"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


def _fmt_paper_brief(p: Dict[str, Any]) -> str:
    title = str((p or {}).get("title", "") or "").strip()
    year = (p or {}).get("year", None)
    venue = str((p or {}).get("venue", "") or "").strip()
    cc = (p or {}).get("citationCount", None)
    pid = str((p or {}).get("paperId", "") or "").strip()
    parts = []
    if year:
        parts.append(str(year))
    if title:
        parts.append(title)
    if venue:
        parts.append(f"({venue})")
    if isinstance(cc, int):
        parts.append(f"citations={cc}")
    if pid:
        parts.append(f"paperId={pid}")
    return " | ".join(parts) if parts else "(empty record)"


def _summarize_search(data: Dict[str, Any], top_k: int = 3) -> str:
    papers = (data or {}).get("data", None)
    if not isinstance(papers, list) or not papers:
        return "Semantic Scholar search returned 0 results."
    lines = []
    for i, p in enumerate(papers[: max(1, int(top_k))], 1):
        if isinstance(p, dict):
            lines.append(f"{i}. {_fmt_paper_brief(p)}")
    total = (data or {}).get("total", None)
    total_str = f"total={total}" if isinstance(total, int) else f"shown={len(lines)}"
    return "Top results (" + total_str + "): " + " || ".join(lines)


@function_tool
def semantic_scholar_search(query: str, limit: int = 10, fields: str = "") -> dict:
    """
    Search papers via Semantic Scholar Graph API.

    Args:
        query: Search query (title/keywords).
        limit: Max results.
        fields: Comma-separated fields (optional).
    """
    q = str(query or "").strip()
    if not q:
        return {"success": False, "error": "query is empty"}

    if not fields:
        fields = "title,abstract,year,authors,url,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,paperId"

    url = f"{S2_API_BASE}/graph/v1/paper/search?query={urllib.parse.quote(q)}&limit={int(limit)}&fields={urllib.parse.quote(fields)}"
    try:
        data = _http_get_json(url, headers=_default_headers(), timeout_sec=60)
        return {
            "success": True,
            "query": q,
            "message": _summarize_search(data=data, top_k=3),
            "data": data,
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "url": url}


@function_tool
def semantic_scholar_paper(paper_id: str, fields: str = "") -> dict:
    """
    Fetch a paper record by Semantic Scholar paperId / DOI / arXiv id.

    Args:
        paper_id: e.g., "CorpusId:xxxxx", "DOI:...", "ARXIV:...", or raw paperId.
        fields: Comma-separated fields (optional).
    """
    pid = str(paper_id or "").strip()
    if not pid:
        return {"success": False, "error": "paper_id is empty"}

    if not fields:
        fields = "title,abstract,year,authors,url,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,referenceCount,references.paperId,references.title"

    safe_pid = urllib.parse.quote(pid, safe="")
    url = f"{S2_API_BASE}/graph/v1/paper/{safe_pid}?fields={urllib.parse.quote(fields)}"
    try:
        data = _http_get_json(url, headers=_default_headers(), timeout_sec=60)
        brief = _fmt_paper_brief(data if isinstance(data, dict) else {})
        return {
            "success": True,
            "paper_id": pid,
            "message": f"Fetched paper: {brief}",
            "data": data,
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "url": url}


@function_tool
def semantic_scholar_download_openaccess_pdf(
    paper_id: str, output_pdf_path: str
) -> dict:
    """
    Download OpenAccess PDF if available (openAccessPdf.url).

    Args:
        paper_id: Semantic Scholar paperId/DOI/arXiv id.
        output_pdf_path: Where to save the PDF.
    """
    pid = str(paper_id or "").strip()
    out_path = str(output_pdf_path or "").strip()
    if not pid or not out_path:
        return {"success": False, "error": "paper_id/output_pdf_path required"}

    meta = semantic_scholar_paper(pid, fields="title,isOpenAccess,openAccessPdf")
    if not meta.get("success"):
        return {
            "success": False,
            "error": f"failed to fetch paper meta: {meta.get('error')}",
        }

    data = meta.get("data") or {}
    oa = data.get("openAccessPdf") or {}
    url = oa.get("url") or ""
    if not url:
        return {
            "success": False,
            "error": "openAccessPdf.url not available",
            "paper": {"title": data.get("title"), "paperId": pid},
        }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        req = urllib.request.Request(url, headers=_default_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        with open(out_path, "wb") as f:
            f.write(raw)
        return {
            "success": True,
            "paper_id": pid,
            "pdf_url": url,
            "output_pdf_path": out_path,
            "bytes": len(raw),
            "message": f"Downloaded OpenAccess PDF ({len(raw)} bytes) to {out_path}",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "pdf_url": url}
