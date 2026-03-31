import os
import re
import sys
from typing import List, Dict, Optional

import torch
import hydra
from sentence_transformers import SentenceTransformer, util
import json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.rich_logger import get_logger
from utils.config_utils import merge_with_default_survey_config
from modules.work_collector import WorkCollector
class OutcomeRAG:
    """Subsection retriever for DeepSurvey markdown output."""

    def __init__(self, config, work_collector):
        self.config = config
        self.json_path = config.BasicInfo.save_json_path
        self.md_path = config.BasicInfo.save_path
        self.model = SentenceTransformer(config.ModuleInfo.WorkCollector.sentence_transformer_model)
        self.work_collector = work_collector
        self.logger = get_logger("OutcomeRAG")
        try:
            self.model = self.model.cuda()
            self.device = "cuda"
        except Exception:
            self.device = "cpu"

        self.subsections: List[str] = []
        self.subsection_titles: List[str] = []
        self.emb_subsections: List[str] = []
        self.emb_subsection_titles: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.title_embeddings: Optional[torch.Tensor] = None

        with open(self.json_path, "r", encoding="utf-8") as f:
            try:
                self.json_data = json.load(f)
            except Exception:
                self.json_data = None

    def _read_md(self) -> str:
        if not self.md_path:
            raise ValueError("md_path is not set")
        if not os.path.exists(self.md_path):
            raise FileNotFoundError(f"Markdown not found: {self.md_path}")
        with open(self.md_path, "r", encoding="utf-8") as f:
            return f.read()

    def slice_text(self, md_text: str) -> List[str]:
        lines = md_text.splitlines()
        pattern = re.compile(r"^(#{2,6})\s+(.*)")
        parts: List[str] = []
        titles: List[str] = []
        current: List[str] = []

        def flush():
            if current:
                block = "\n".join(current).strip()
                if block:
                    parts.append(block)
                    # title is the first heading line without hashes
                    heading = current[0]
                    m = pattern.match(heading)
                    title = m.group(2).strip() if m else heading.strip()
                    titles.append(title)

        for line in lines:
            if pattern.match(line):
                flush()
                current = [line]
            else:
                current.append(line)
        flush()
        self.subsection_titles = titles
        subsections = [p for p in parts if p]
        self.logger.debug(f"Sliced {len(subsections)} subsections")
        return subsections

    def build_embeddings(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
            device=self.device,
            show_progress_bar=False,
        )

    def build_index(self, md_text: Optional[str] = None):
        if md_text is None:
            md_text = self._read_md()
        md_text = self._strip_references(md_text)
        self.subsections = self.slice_text(md_text)
        # clean content and titles: remove citation tags like [12], strip heading hashes
        self.emb_subsections = [self._clean_text(t) for t in self.subsections]
        if self.subsection_titles:
            self.emb_subsection_titles = [self._clean_text(t) for t in self.subsection_titles]
        if not self.subsections:
            raise ValueError("No subsections found to index")
        self.embeddings = self.build_embeddings(self.emb_subsections)
        if self.emb_subsection_titles:
            self.title_embeddings = self.build_embeddings(self.emb_subsection_titles)

    def _strip_references(self, md_text: str) -> str:
        """Drop the trailing References section if present."""
        lines = md_text.splitlines()
        lines = lines[1:] # delete survey title line
        cut_idx = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("references:"):
                cut_idx = i
        if cut_idx is not None:
            return "\n".join(lines[:cut_idx])
        return md_text

    def _clean_text(self, text: str) -> str:
        # remove citation markers like [12]
        text = re.sub(r"\[\d+\]", "", text)
        # remove leading heading hashes/spaces
        text = re.sub(r"^#+\s*", "", text)
        return text.strip()

    def retrieve(self, query: str, top_k: int = 5, mode: str = "content", alpha: float = 0.5, cite_top_k: Optional[int] = None) -> List[Dict]:
        if self.embeddings is None or not self.subsections:
            raise ValueError("Index not built. Call build_index first.")
        query_emb = self.model.encode([query], convert_to_tensor=True, device=self.device)

        if mode == "title":
            if self.title_embeddings is None:
                raise ValueError("Title embeddings not built.")
            scores = util.pytorch_cos_sim(query_emb, self.title_embeddings)[0]
        elif mode == "hybrid":
            if self.title_embeddings is None:
                raise ValueError("Title embeddings not built.")
            scores_content = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
            scores_title = util.pytorch_cos_sim(query_emb, self.title_embeddings)[0]
            alpha = max(0.0, min(1.0, alpha))
            scores = alpha * scores_content + (1 - alpha) * scores_title
        else:
            scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]

        top_k = min(top_k, len(self.subsections))
        values, indices = torch.topk(scores, k=top_k)
        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            subsection_text = self.subsections[idx]
            citation_entries = self._collect_citations(subsection_text, top_k=cite_top_k)
            citations = [c.get("title") for c in citation_entries if c.get("title")]
            # self.logger.info(f"Retrieved subsection citations: {citations}")
            results.append({
                "subsection": subsection_text,
                "title": self.subsection_titles[idx] if idx < len(self.subsection_titles) else None,
                "score": float(score),
                "citations": citations,
                "citation_entries": citation_entries,
            })
        return results
    

    def _collect_citations(self, text: str, top_k: Optional[int] = None) -> List[Dict]:
        ids, counts = self._extract_citation_ids(text)
        paired = list(zip(ids, counts))
        paired.sort(key=lambda x: (-x[1], x[0]))
        out = []
        for cid, cnt in paired:
            entry = self._lookup_citation(cid)
            out.append({"id": cid, "count": cnt, **entry})
        if top_k is not None:
            out = out[:top_k]
        return out

    def _extract_citation_ids(self, text: str) -> (List[int], List[int]):
        ids = re.findall(r"\[(\d+)\]", text)
        uniq: List[int] = []
        occurs: List[int] = []
        for x in ids:
            i = int(x)
            if i not in uniq:
                uniq.append(i)
                occurs.append(1)
            else:
                occurs[uniq.index(i)] += 1
        return uniq, occurs

    def _lookup_citation(self, cid: int) -> Dict[str, str]:
        paper_ids = self.json_data.get("references") if self.json_data else None
        if isinstance(paper_ids, list) and 1 <= cid <= len(paper_ids):
            paper_id = paper_ids[cid - 1]  # assuming cid starts from 1
            return {
                "paper_id": paper_id,
                "title": self.work_collector.get_paper_title(paper_id),
            }
        else:
            self.logger.warning(f"Invalid citation id {cid} or references missing; skipping title lookup")
            return {"paper_id": "", "title": ""}

    def log_hits(self, hits: List[Dict], label: str = "Hits") -> None:
        self.logger.info(f"{label}: {len(hits)} result(s)")
        for i, h in enumerate(hits):
            text = h.get("subsection", "").strip().replace("\n", " ")
            score = h.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            cites = h.get("citations") or []
            cite_str = "; ".join([f"{c.get('id')}(x{c.get('count')}) {c.get('title','')}" if isinstance(c, dict) else str(c) for c in cites]) or "-"
            self.logger.info(f"[{i}] score={score_str} text={text}")
            if cites:
                self.logger.info(f"    citations: {cite_str}")


@hydra.main(config_path="../config", config_name="outcomeRAG", version_base=None)
def main(cfg):
    cfg = merge_with_default_survey_config(cfg)
    logger = get_logger("OutcomeRAGMain")
    work_collector = WorkCollector(cfg)

    rag = OutcomeRAG(cfg, work_collector)
    rag.build_index()
    # content match
    hits_content = rag.retrieve("application of Multi Modal Large Language Model", top_k=2, mode="content")
    # title-only match
    hits_title = rag.retrieve("Core Capabilities", top_k=2, mode="title")
    # hybrid
    hits_hybrid = rag.retrieve("Core Capabilities", top_k=2, mode="hybrid", alpha=0.9)
    logger.info("---------[Content]----------")
    rag.log_hits(hits_content, label="Content")
    logger.info("---------[Title]----------")
    rag.log_hits(hits_title, label="Title")
    logger.info("---------[Hybrid]----------")
    rag.log_hits(hits_hybrid, label="Hybrid")

if __name__ == "__main__":
    main()
