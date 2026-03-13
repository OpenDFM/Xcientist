"""Ablation-native symbolic memory indexed by component family.

Each record mirrors one component entry from
``src/agents/idea_agent/ablation_results.json`` and preserves the raw
ablation evidence directly:

- ``component``: component name from the ablation report
- ``component_family``: derived retrieval key for hierarchical lookup
- ``op``: operation represented by the evidence, usually ``remove``
- ``result``: positive / negative / inconclusive
- ``metric`` / ``value`` / ``analysis``: raw evidence fields
- ``method_context``: idea introduction with this component removed
- ``run_summary``: optional structured context

Retrieval remains hierarchical:
1. exact ``component_family`` match
2. macro-role fallback
3. global fallback with structural + lexical ranking
"""

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from memory.api.base_symbolic_memory_system_api import (
    SymbolicMemorySystem as BaseSymbolicMemorySystem,
    SymbolicMemorySystemConfig,
    SymbolicRecord,
    SymbolicRecordPayload,
)
from memory.api.component_taxonomy import ContextSignature, parse_component_family
from memory.memory_system.utils import _safe_dump_str, compute_overlap_score


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


def _reliability_score(record: SymbolicRecord) -> float:
    support = max(record.support_count, 1)
    support_factor = 1.0 - 1.0 / (1.0 + math.log2(support))
    return record.confidence * max(0.0, support_factor)


def _evidence_strength(record: SymbolicRecord) -> float:
    if str(record.result or "").strip().lower() == "inconclusive":
        return 0.15
    return max(0.15, min(1.0, float(record.confidence or 0.0)))


def _result_effect(record: SymbolicRecord) -> float:
    result = str(record.result or "").strip().lower()
    confidence = max(0.0, min(1.0, float(record.confidence or 0.0)))
    if result == "positive":
        return confidence
    if result == "negative":
        return -confidence
    return 0.0


def _context_match_score(query_sig: ContextSignature, record: SymbolicRecord) -> float:
    text_pool = "\n".join(
        [
            record.analysis or "",
            record.method_context or "",
            record.metric or "",
            record.value or "",
            _safe_dump_str(record.run_summary),
        ]
    ).lower()
    if not text_pool.strip():
        return 0.5

    tokens: List[str] = []
    tokens.extend(query_sig.macro_roles_present)
    tokens.extend(query_sig.defect_profile)
    tokens.append(query_sig.coverage_bucket)
    tokens.append(query_sig.stability_bucket)
    tokens.append(query_sig.cost_bucket)
    tokens.append(query_sig.budget_pressure)
    if query_sig.last_main_op:
        tokens.append(query_sig.last_main_op)
    lowered = [token.lower() for token in tokens if token]
    if not lowered:
        return 0.5
    hits = sum(1 for token in lowered if token in text_pool)
    return max(0.1, hits / len(lowered))


class SymbolicMemorySystem(BaseSymbolicMemorySystem):
    def __init__(self, **kwargs):
        self.cfg = SymbolicMemorySystemConfig(**kwargs)
        self._records: Dict[int, SymbolicRecord] = {}
        self.fidmap2mid: Dict[int, str] = {}
        self.midmap2fid: Dict[str, int] = {}
        self._next_id = 0

        self._family_index: Dict[str, List[int]] = {}
        self._role_index: Dict[str, List[int]] = {}
        self._op_index: Dict[str, List[int]] = {}

    def _index_record(self, fid: int, record: SymbolicRecord) -> None:
        family = (record.component_family or "").strip().lower()
        if family:
            self._family_index.setdefault(family, [])
            if fid not in self._family_index[family]:
                self._family_index[family].append(fid)
            role, _ = parse_component_family(family)
            self._role_index.setdefault(role, [])
            if fid not in self._role_index[role]:
                self._role_index[role].append(fid)

        op = (record.op or "").strip().lower()
        if op:
            self._op_index.setdefault(op, [])
            if fid not in self._op_index[op]:
                self._op_index[op].append(fid)

    def _unindex_record(self, fid: int, record: SymbolicRecord) -> None:
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

        op = (record.op or "").strip().lower()
        if op and op in self._op_index:
            self._op_index[op] = [item for item in self._op_index[op] if item != fid]
            if not self._op_index[op]:
                del self._op_index[op]

    def _rebuild_indexes(self) -> None:
        self._family_index.clear()
        self._role_index.clear()
        self._op_index.clear()
        for fid, record in self._records.items():
            self._index_record(fid, record)

    def instantiate_symbolic_record(self, **kwargs) -> SymbolicRecord:
        payload = SymbolicRecordPayload(**kwargs)
        timestamp = _now_iso()
        return SymbolicRecord(
            id=_new_id("sym"),
            component=payload.component,
            component_family=payload.component_family,
            op=payload.op,
            result=payload.result,
            metric=payload.metric,
            value=payload.value,
            analysis=payload.analysis,
            method_context=payload.method_context,
            confidence=payload.confidence,
            run_summary=dict(payload.run_summary or {}),
            metadata=dict(payload.metadata or {}),
            support_count=payload.support_count,
            created_at=timestamp,
            updated_at=timestamp,
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
                memory.updated_at = _now_iso()
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
                str(record.op or "").strip().lower(),
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

            total_support = max(1, int(target.support_count)) + max(1, int(record.support_count))
            target_result = str(target.result or "").strip().lower()
            record_result = str(record.result or "").strip().lower()
            merged_result = target_result
            if target_result != record_result:
                if "inconclusive" in {target_result, record_result}:
                    merged_result = record_result if target_result == "inconclusive" else target_result
                else:
                    merged_result = "inconclusive"
            merged_confidence = (
                float(target.confidence) * max(1, int(target.support_count))
                + float(record.confidence) * max(1, int(record.support_count))
            ) / total_support

            target.update(
                component=record.component,
                component_family=record.component_family,
                op=record.op,
                result=merged_result,
                metric=record.metric or target.metric,
                value=record.value or target.value,
                analysis=record.analysis or target.analysis,
                method_context=record.method_context or target.method_context,
                confidence=merged_confidence,
                run_summary={**target.run_summary, **record.run_summary},
                metadata={**target.metadata, **record.metadata},
                support_count=total_support,
            )
            self.update([target])

    def query(
        self,
        query_text: str,
        method: Literal["hybrid", "rule", "overlapping"] = "hybrid",
        limit: int = 5,
        filters: Optional[Dict] = None,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        if self.size == 0:
            return []

        method = (method or "hybrid").lower()
        if method not in {"hybrid", "rule", "overlapping"}:
            raise ValueError(f"Unsupported symbolic query method: {method}")

        ranked: List[Tuple[float, SymbolicRecord]] = []
        for record in self._records.values():
            if agent_id and not str(record.id).endswith(f"_{agent_id}"):
                continue
            if not self._match_filters(record, filters):
                continue

            lexical_score = self._lexical_score(query_text, record)
            rule_score = self._rule_score(query_text, record)
            recency_score = self._recency_score(record)

            if method == "overlapping":
                score = lexical_score
            elif method == "rule":
                score = rule_score
            else:
                score = (
                    self.cfg.lexical_weight * lexical_score
                    + self.cfg.rule_weight * rule_score
                    + self.cfg.recency_weight * recency_score
                )
            score *= max(_reliability_score(record), 0.05) * (0.4 + 0.6 * _evidence_strength(record))

            if score >= threshold:
                ranked.append((float(score), record))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[: max(0, min(limit, len(ranked)))]

    def get_nearest_k_records(
        self,
        record: SymbolicRecord,
        method: Literal["hybrid", "rule", "overlapping"] = "hybrid",
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
        target_family: str = "",
        context_sig: Optional[ContextSignature] = None,
        op: str = "",
        limit: int = 5,
        threshold: float = 0.25,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        if self.size == 0:
            return []

        context_sig = context_sig or ContextSignature()
        target_family_l = (target_family or "").strip().lower()
        target_role, _ = parse_component_family(target_family_l) if target_family_l else ("", "")
        op_l = (op or "").strip().lower()
        scored: Dict[int, Tuple[float, SymbolicRecord]] = {}

        def _score_candidate(fid: int, tier_bonus: float) -> None:
            if fid in scored:
                return
            record = self._records.get(fid)
            if record is None:
                return
            if agent_id and not str(record.id).endswith(f"_{agent_id}"):
                return
            if op_l and (record.op or "").strip().lower() != op_l:
                return

            struct_match = _context_match_score(context_sig, record)
            lexical_match = self._lexical_score(self._context_query_text(context_sig), record)
            reliability = _reliability_score(record)
            evidence = _evidence_strength(record)

            raw_score = struct_match * max(lexical_match, 0.1) * max(reliability, 0.05) * evidence
            final_score = min(1.0, raw_score + tier_bonus)
            if final_score >= threshold:
                scored[fid] = (float(final_score), record)

        if target_family_l and target_family_l in self._family_index:
            for fid in self._family_index[target_family_l]:
                _score_candidate(fid, tier_bonus=0.15)

        if len(scored) < limit and target_role and target_role in self._role_index:
            for fid in self._role_index[target_role]:
                _score_candidate(fid, tier_bonus=0.05)

        if len(scored) < limit:
            for fid in self._records:
                _score_candidate(fid, tier_bonus=0.0)

        ranked = sorted(scored.values(), key=lambda item: item[0], reverse=True)
        return ranked[: max(0, min(limit, len(ranked)))]

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
            record.op,
            record.result,
            record.metric,
            record.value,
            record.analysis,
            record.method_context,
            _safe_dump_str(record.run_summary),
            _safe_dump_str(record.metadata),
        ]
        return "\n".join(block for block in blocks if block)

    def _lexical_score(self, query_text: str, record: SymbolicRecord) -> float:
        return compute_overlap_score(self._record_text(record), query_text)

    def _rule_score(self, query_text: str, record: SymbolicRecord) -> float:
        tokens = self._tokenize(query_text)
        if not tokens:
            return 0.0

        sections = {
            "component": str(record.component or "").lower(),
            "family": str(record.component_family or "").lower(),
            "op": str(record.op or "").lower(),
            "result": str(record.result or "").lower(),
            "metric": str(record.metric or "").lower(),
            "analysis": str(record.analysis or "").lower(),
            "method_context": str(record.method_context or "").lower(),
        }
        weight_map = {
            "component": 1.4,
            "family": 1.6,
            "op": 1.2,
            "result": 0.7,
            "metric": 1.0,
            "analysis": 1.0,
            "method_context": 0.9,
        }
        max_per_token = sum(weight_map.values())
        total = 0.0
        for token in tokens:
            for key, section in sections.items():
                if token in section:
                    total += weight_map[key]
        score = total / (len(tokens) * max_per_token)
        return max(0.0, min(1.0, score))

    def _recency_score(self, record: SymbolicRecord) -> float:
        ts = record.updated_at or record.created_at
        if not ts:
            return 0.3
        try:
            dt = datetime.fromisoformat(ts)
            age_days = max(0.0, (datetime.now(dt.tzinfo) - dt).total_seconds() / 86400.0)
            return float(1.0 / (1.0 + age_days / 30.0))
        except ValueError:
            return 0.3

    def _match_filters(self, record: SymbolicRecord, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if key.startswith("metadata."):
                actual = record.metadata.get(key.split(".", 1)[1])
            elif key.startswith("run_summary."):
                actual = record.run_summary.get(key.split(".", 1)[1])
            else:
                actual = getattr(record, key, None)

            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())
