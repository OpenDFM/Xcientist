"""Ablation-native symbolic memory indexed by component and component family.

Each record mirrors one component entry from
``src/agents/idea_agent/ablation_results.json`` and preserves the raw
ablation evidence directly:

- ``component``: component name from the ablation report
- ``component_family``: derived retrieval key for hierarchical lookup
- ``result``: positive / negative / inconclusive
- ``metric`` / ``value`` / ``analysis``: raw evidence fields
- ``method_context``: idea introduction with this component removed

Retrieval remains hierarchical:
1. exact ``component`` match
2. exact ``component_family`` match
3. exact ``macro_role`` match

When one tier returns multiple records, rerank them by comparing the current
idea abstract against each record's ``method_context``.
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from memory.api.base_symbolic_memory_system_api import (
    SymbolicMemorySystem as BaseSymbolicMemorySystem,
    SymbolicMemorySystemConfig,
    SymbolicRecord,
    SymbolicRecordPayload,
)
from memory.api.component_taxonomy import parse_component_family
from memory.memory_system.utils import (
    _evidence_strength,
    _normalize_component_key,
    _reliability_score,
    compute_overlap_score,
    new_id,
)


class SymbolicMemorySystem(BaseSymbolicMemorySystem):
    def __init__(self, **kwargs):
        self.cfg = SymbolicMemorySystemConfig(**kwargs)
        self._records: Dict[int, SymbolicRecord] = {}
        self.fidmap2mid: Dict[int, str] = {}
        self.midmap2fid: Dict[str, int] = {}
        self._next_id = 0

        self._component_index: Dict[str, List[int]] = {}
        self._family_index: Dict[str, List[int]] = {}
        self._role_index: Dict[str, List[int]] = {}

    def _index_record(self, fid: int, record: SymbolicRecord) -> None:
        component = _normalize_component_key(record.component)
        if component:
            self._component_index.setdefault(component, [])
            if fid not in self._component_index[component]:
                self._component_index[component].append(fid)

        family = (record.component_family or "").strip().lower()
        if family:
            self._family_index.setdefault(family, [])
            if fid not in self._family_index[family]:
                self._family_index[family].append(fid)
            role, _ = parse_component_family(family)
            self._role_index.setdefault(role, [])
            if fid not in self._role_index[role]:
                self._role_index[role].append(fid)

    def _unindex_record(self, fid: int, record: SymbolicRecord) -> None:
        component = _normalize_component_key(record.component)
        if component and component in self._component_index:
            self._component_index[component] = [
                item for item in self._component_index[component] if item != fid
            ]
            if not self._component_index[component]:
                del self._component_index[component]

        family = (record.component_family or "").strip().lower()
        if family and family in self._family_index:
            self._family_index[family] = [item for item in self._family_index[family] if item != fid]
            if not self._family_index[family]:
                del self._family_index[family]
            role, _ = parse_component_family(family)
            if role in self._role_index:
                self._role_index[role] = [item for item in self._role_index[role] if item != fid]
                if not self._role_index[role]:
                    del self._role_index[role]

    def _rebuild_indexes(self) -> None:
        self._component_index.clear()
        self._family_index.clear()
        self._role_index.clear()
        for fid, record in self._records.items():
            self._index_record(fid, record)

    def instantiate_symbolic_record(self, **kwargs) -> SymbolicRecord:
        payload = SymbolicRecordPayload(**kwargs)
        return SymbolicRecord(
            id=new_id("sym"),
            component=payload.component,
            component_family=payload.component_family,
            result=payload.result,
            metric=payload.metric,
            value=payload.value,
            analysis=payload.analysis,
            method_context=payload.method_context,
            confidence=payload.confidence,
        )

    @property
    def size(self) -> int:
        return len(self._records)

    def get_records_by_ids(self, mids: List[str]) -> List[SymbolicRecord]:
        records: List[SymbolicRecord] = []
        for mid in mids:
            fid = self.midmap2fid.get(mid)
            if fid is None:
                continue
            record = self._records.get(fid)
            if record is not None:
                records.append(record)
        return records

    def get_last_k_records(self, k: int) -> Tuple[List[Dict[str, Any]], int]:
        if self.size == 0:
            return [], 0
        if k >= self.size:
            payload = [record.to_dict() for record in self._records.values()]
            return payload, len(payload)
        sorted_fids = sorted(self.fidmap2mid.keys(), reverse=True)
        latest = [self._records[fid].to_dict() for fid in sorted_fids[:k] if fid in self._records]
        return latest, len(latest)

    def is_exists(self, mids: List[str]) -> List[bool]:
        return [mid in self.midmap2fid and self.midmap2fid[mid] in self._records for mid in mids]

    def add(self, memories: List[SymbolicRecord] = None, agent_id: str = "") -> bool:
        memories = memories or []
        try:
            for memory in memories:
                if not isinstance(memory, SymbolicRecord):
                    continue
                record = memory
                if agent_id and not record.id.endswith(f"_{agent_id}"):
                    record.id = f"{record.id}_{agent_id}"
                if record.id in self.midmap2fid:
                    fid = self.midmap2fid[record.id]
                    old_record = self._records.get(fid)
                    if old_record is not None:
                        self._unindex_record(fid, old_record)
                    self._records[fid] = record
                    self._index_record(fid, record)
                    continue

                fid = self._next_id
                self._next_id += 1
                self._records[fid] = record
                self.fidmap2mid[fid] = record.id
                self.midmap2fid[record.id] = fid
                self._index_record(fid, record)
            return True
        except Exception as exc:
            print(f"Error adding symbolic memories: {exc}")
            return False

    def update(self, memories: List[SymbolicRecord] = None) -> bool:
        memories = memories or []
        try:
            for memory in memories:
                if not isinstance(memory, SymbolicRecord):
                    continue
                fid = self.midmap2fid.get(memory.id)
                if fid is None:
                    continue
                old_record = self._records.get(fid)
                if old_record is not None:
                    self._unindex_record(fid, old_record)
                self._records[fid] = memory
                self._index_record(fid, memory)
            return True
        except Exception as exc:
            print(f"Error updating symbolic memories: {exc}")
            return False

    def delete(self, mids: List[str]) -> bool:
        try:
            for mid in mids:
                fid = self.midmap2fid.pop(mid, None)
                if fid is None:
                    continue
                self.fidmap2mid.pop(fid, None)
                old_record = self._records.pop(fid, None)
                if old_record is not None:
                    self._unindex_record(fid, old_record)
            return True
        except Exception as exc:
            print(f"Error deleting symbolic memories: {exc}")
            return False

    def _record_identity_key(self, record: SymbolicRecord) -> str:
        return "|".join(
            [
                str(record.component or "").strip().lower(),
                str(record.component_family or "").strip().lower(),
                str(record.metric or "").strip().lower(),
            ]
        )

    def upsert_normal_records(
        self,
        records: List[SymbolicRecord],
        agent_id: str = "",
    ) -> None:
        for record in records or []:
            if not isinstance(record, SymbolicRecord):
                continue

            target: Optional[SymbolicRecord] = None
            record_key = self._record_identity_key(record)
            for existing in self._records.values():
                if agent_id and not str(existing.id).endswith(f"_{agent_id}"):
                    continue
                if self._record_identity_key(existing) == record_key:
                    target = existing
                    break

            if target is None:
                self.add([record], agent_id=agent_id)
                continue

            target_result = str(target.result or "").strip().lower()
            record_result = str(record.result or "").strip().lower()
            merged_result = target_result
            if target_result != record_result:
                if "inconclusive" in {target_result, record_result}:
                    merged_result = record_result if target_result == "inconclusive" else target_result
                else:
                    merged_result = "inconclusive"
            merged_confidence = (float(target.confidence) + float(record.confidence)) / 2.0

            target.update(
                component=record.component,
                component_family=record.component_family,
                result=merged_result,
                metric=record.metric or target.metric,
                value=record.value or target.value,
                analysis=record.analysis or target.analysis,
                method_context=record.method_context or target.method_context,
                confidence=merged_confidence,
            )
            self.update([target])

    def query(
        self,
        query_text: str,
        method: str = "lexical",
        limit: int = 5,
        filters: Optional[Dict] = None,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        if self.size == 0:
            return []

        method = (method or "lexical").lower()
        if method != "lexical":
            raise ValueError(
                f"Unsupported symbolic query method: {method}. Only 'lexical' is supported."
            )

        ranked: List[Tuple[float, SymbolicRecord]] = []
        for record in self._records.values():
            if agent_id and not str(record.id).endswith(f"_{agent_id}"):
                continue
            if not self._match_filters(record, filters):
                continue

            lexical_score = self._lexical_score(query_text, record)
            score = lexical_score
            score *= max(_reliability_score(record), 0.05) * (0.4 + 0.6 * _evidence_strength(record))

            if score >= threshold:
                ranked.append((float(score), record))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[: max(0, min(limit, len(ranked)))]

    def get_nearest_k_records(
        self,
        record: SymbolicRecord,
        method: str = "lexical",
        k: int = 5,
        filters: Optional[Dict] = None,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        return self.query(
            query_text=self._record_text(record),
            method=method,
            limit=k,
            filters=filters,
            agent_id=agent_id,
        )

    def retrieve_hierarchical(
        self,
        target_component: str = "",
        target_family: str = "",
        limit: int = 2,
        threshold: float = 0.25,
        agent_id: str = "",
        query_context: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        if self.size == 0 or limit <= 0:
            return []

        target_component_l = _normalize_component_key(target_component)
        target_family_l = (target_family or "").strip().lower()
        target_role, _ = parse_component_family(target_family_l) if target_family_l else ("", "")
        def _context_match_scores(
            context_query: str,
            records: List[SymbolicRecord],
        ) -> List[float]:
            if not context_query or len(records) <= 1:
                return [0.0] * len(records)

            docs = [str(record.method_context or "") for record in records]
            if not any(doc.strip() for doc in docs):
                return [0.0] * len(records)

            tokenized = [self._context_tokens(context_query)] + [
                self._context_tokens(doc) for doc in docs
            ]
            if not tokenized[0]:
                return [0.0] * len(records)

            doc_freq: Dict[str, int] = {}
            for tokens in tokenized:
                for token in set(tokens):
                    doc_freq[token] = doc_freq.get(token, 0) + 1

            vectors: List[Dict[str, float]] = []
            doc_count = len(tokenized)
            for tokens in tokenized:
                term_freq: Dict[str, int] = {}
                for token in tokens:
                    term_freq[token] = term_freq.get(token, 0) + 1

                token_count = sum(term_freq.values()) or 1
                vector: Dict[str, float] = {}
                for token, count in term_freq.items():
                    tf = count / token_count
                    idf = math.log((1.0 + doc_count) / (1.0 + doc_freq.get(token, 0))) + 1.0
                    vector[token] = tf * idf
                vectors.append(vector)

            query_vector = vectors[0]
            scores: List[float] = []
            for idx, doc in enumerate(docs, start=1):
                tfidf_score = self._cosine_similarity(query_vector, vectors[idx])
                lexical_score = compute_overlap_score(doc, context_query)
                scores.append(min(1.0, 0.8 * tfidf_score + 0.2 * lexical_score))
            return scores

        def _filter_exact_candidates(fids: List[int]) -> List[Tuple[int, SymbolicRecord]]:
            candidates: List[Tuple[int, SymbolicRecord]] = []
            for fid in fids:
                record = self._records.get(fid)
                if record is None:
                    continue
                if agent_id and not str(record.id).endswith(f"_{agent_id}"):
                    continue
                candidates.append((fid, record))
            return candidates

        def _rank_exact_candidates(
            candidates: List[Tuple[int, SymbolicRecord]],
            *,
            tier_base: float,
        ) -> List[Tuple[float, SymbolicRecord]]:
            if not candidates:
                return []

            if query_context and len(candidates) > 1:
                context_scores = _context_match_scores(
                    query_context,
                    [record for _, record in candidates],
                )
            else:
                context_scores = [0.0] * len(candidates)

            ranked_candidates: List[Tuple[float, float, int, SymbolicRecord]] = []
            for (fid, record), context_score in zip(candidates, context_scores):
                final_score = min(1.0, tier_base + 0.1 * context_score)
                if final_score < threshold:
                    continue
                ranked_candidates.append((final_score, context_score, fid, record))

            ranked_candidates.sort(
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            return [
                (final_score, record)
                for final_score, _, _, record in ranked_candidates[: max(0, min(limit, len(ranked_candidates)))]
            ]

        component_candidates = _filter_exact_candidates(
            self._component_index.get(target_component_l, []) if target_component_l else []
        )
        if component_candidates:
            return _rank_exact_candidates(component_candidates, tier_base=0.9)

        family_candidates = _filter_exact_candidates(
            self._family_index.get(target_family_l, []) if target_family_l else []
        )
        if family_candidates:
            return _rank_exact_candidates(family_candidates, tier_base=0.8)

        role_candidates = _filter_exact_candidates(
            self._role_index.get(target_role, []) if target_role else []
        )
        if role_candidates:
            return _rank_exact_candidates(role_candidates, tier_base=0.7)

        return []

    def save(self, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            payload = {
                "next_id": self._next_id,
                "fidmap2mid": self.fidmap2mid,
                "records": {str(fid): record.to_dict() for fid, record in self._records.items()},
                "config": self.cfg.model_dump(),
            }
            with open(os.path.join(path, "symbolic_memory.json"), "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            print(f"Error saving symbolic memories: {exc}")
            return False

    def load(self, path: str) -> bool:
        try:
            file_path = os.path.join(path, "symbolic_memory.json")
            with open(file_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self._records = {}
            self.fidmap2mid = {int(fid): mid for fid, mid in (payload.get("fidmap2mid") or {}).items()}
            self.midmap2fid = {mid: fid for fid, mid in self.fidmap2mid.items()}
            self._next_id = int(payload.get("next_id", 0))

            for fid_str, record_payload in (payload.get("records") or {}).items():
                fid = int(fid_str)
                self._records[fid] = SymbolicRecord.from_dict(record_payload)
                if fid not in self.fidmap2mid:
                    self.fidmap2mid[fid] = self._records[fid].id
                    self.midmap2fid[self._records[fid].id] = fid

            if self._next_id <= 0 and self._records:
                self._next_id = max(self._records.keys()) + 1

            self._rebuild_indexes()
            return True
        except Exception as exc:
            print(f"Error loading symbolic memories: {exc}")
            return False

    def _record_text(self, record: SymbolicRecord) -> str:
        blocks = [
            record.component,
            record.component_family,
            record.result,
            record.metric,
            record.value,
            record.analysis,
            record.method_context,
        ]
        return "\n".join(block for block in blocks if block)

    def _lexical_score(self, query_text: str, record: SymbolicRecord) -> float:
        return compute_overlap_score(self._record_text(record), query_text)

    def _match_filters(self, record: SymbolicRecord, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            actual = getattr(record, key, None)

            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    @staticmethod
    def _context_tokens(text: str) -> List[str]:
        stopwords = {
            "a",
            "an",
            "the",
            "of",
            "and",
            "or",
            "to",
            "in",
            "on",
            "for",
            "with",
            "at",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "this",
            "that",
            "these",
            "those",
            "it",
            "as",
            "into",
            "up",
            "down",
            "we",
            "our",
            "their",
            "via",
            "using",
            "use",
            "uses",
            "based",
            "approach",
            "method",
            "model",
            "module",
            "system",
        }
        return [
            token
            for token in re.findall(r"\w+", (text or "").lower())
            if len(token) > 1 and token not in stopwords and not token.isdigit()
        ]

    @staticmethod
    def _cosine_similarity(
        lhs: Dict[str, float],
        rhs: Dict[str, float],
    ) -> float:
        if not lhs or not rhs:
            return 0.0
        lhs_norm = math.sqrt(sum(value * value for value in lhs.values()))
        rhs_norm = math.sqrt(sum(value * value for value in rhs.values()))
        if lhs_norm <= 0.0 or rhs_norm <= 0.0:
            return 0.0
        dot = 0.0
        for token in set(lhs).intersection(rhs):
            dot += lhs[token] * rhs[token]
        return max(0.0, min(1.0, dot / (lhs_norm * rhs_norm)))
