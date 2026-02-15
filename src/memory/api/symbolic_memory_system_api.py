import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from uuid import uuid4

from memory.api.base_symbolic_memory_system_api import (
    SymbolicMemorySystem as BaseSymbolicMemorySystem,
    SymbolicMemorySystemConfig,
    SymbolicRecordPayload,
    SymbolicRecord,
)
from memory.memory_system.utils import (
    _safe_dump_str,
    compute_overlap_score,
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class SymbolicMemorySystem(BaseSymbolicMemorySystem):
    def __init__(self, **kwargs):
        cfg = SymbolicMemorySystemConfig(**kwargs)
        self.cfg = cfg

        self._records: Dict[int, SymbolicRecord] = {}
        self.fidmap2mid: Dict[int, str] = {}
        self.midmap2fid: Dict[str, int] = {}
        self._next_id = 0

    def instantiate_symbolic_record(self, **kwargs) -> SymbolicRecord:
        payload = SymbolicRecordPayload(**kwargs)
        timestamp = _now_iso()
        return SymbolicRecord(
            id=_new_id("sym"),
            summary=payload.summary,
            pattern=payload.pattern,
            conditions=list(payload.conditions or []),
            actions=list(payload.actions or []),
            rationale=payload.rationale,
            expected_outcomes=list(payload.expected_outcomes or []),
            anti_patterns=list(payload.anti_patterns or []),
            tags=list(payload.tags or []),
            priority=payload.priority,
            confidence=payload.confidence,
            source=payload.source,
            support_count=payload.support_count,
            metadata=dict(payload.metadata or {}),
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
            if record:
                records.append(record)
        return records

    def get_last_k_records(self, k: int) -> Tuple[List[Dict[str, Any]], int]:
        if self.size == 0:
            return [], 0
        if k >= self.size:
            return ([record.to_dict() for record in self._records.values()], self.size)
        sorted_fids = sorted(self.fidmap2mid.keys(), reverse=True)
        latest = [self._records[fid].to_dict() for fid in sorted_fids[:k] if fid in self._records]
        return latest, len(latest)

    def is_exists(self, mids: List[str]) -> List[bool]:
        return [mid in self.midmap2fid and self.midmap2fid[mid] in self._records for mid in mids]

    def add(self, memories: List[SymbolicRecord] = None, agent_id: str = "") -> bool:
        if memories is None:
            memories = []
        try:
            for memory in memories:
                if not isinstance(memory, SymbolicRecord):
                    continue
                record = memory
                if agent_id and not record.id.endswith(f"_{agent_id}"):
                    record.id = f"{record.id}_{agent_id}"
                if record.id in self.midmap2fid:
                    fid = self.midmap2fid[record.id]
                    self._records[fid] = record
                    continue

                fid = self._next_id
                self._next_id += 1
                self._records[fid] = record
                self.fidmap2mid[fid] = record.id
                self.midmap2fid[record.id] = fid
            return True
        except Exception as exc:
            print(f"Error adding symbolic memories: {exc}")
            return False

    def update(self, memories: List[SymbolicRecord] = None) -> bool:
        if memories is None:
            memories = []
        try:
            for memory in memories:
                if not isinstance(memory, SymbolicRecord):
                    continue
                fid = self.midmap2fid.get(memory.id)
                if fid is None:
                    continue
                memory.updated_at = _now_iso()
                self._records[fid] = memory
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
                self._records.pop(fid, None)
            return True
        except Exception as exc:
            print(f"Error deleting symbolic memories: {exc}")
            return False

    def upsert_normal_records(
        self,
        records: List[SymbolicRecord],
        agent_id: str = "",
    ) -> None:
        if not records:
            return

        for record in records:
            nearest = self.get_nearest_k_records(
                record=record,
                method="hybrid",
                k=1,
                agent_id=agent_id,
            )
            if nearest and nearest[0][0] >= self.cfg.upsert_threshold:
                target = nearest[0][1]
                target.update(
                    summary=record.summary,
                    pattern=record.pattern,
                    conditions=record.conditions,
                    actions=record.actions,
                    rationale=record.rationale,
                    expected_outcomes=record.expected_outcomes,
                    anti_patterns=record.anti_patterns,
                    tags=record.tags,
                    priority=record.priority,
                    confidence=max(target.confidence, record.confidence),
                    source=record.source or target.source,
                    support_count=target.support_count + 1,
                    metadata={**target.metadata, **record.metadata},
                )
                self.update([target])
            else:
                self.add([record], agent_id=agent_id)

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
            score *= (0.4 + 0.6 * record.confidence) * (0.5 + 0.5 * record.priority)

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
        query_text = self._record_text(record)
        return self.query(
            query_text=query_text,
            method=method,
            limit=k,
            filters=filters,
            agent_id=agent_id,
        )

    def retrieve_priors_for_expand(
        self,
        topic: str,
        operator: str = "",
        defects: Optional[List[str]] = None,
        limit: int = 5,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        defects = defects or []
        query_text = "\n".join([topic, operator, *defects]).strip()
        candidates = self.query(
            query_text=query_text,
            method="hybrid",
            limit=max(limit * 3, limit),
            threshold=threshold,
            agent_id=agent_id,
        )

        boosted: List[Tuple[float, SymbolicRecord]] = []
        operator_l = operator.lower().strip()
        defect_tokens = [d.lower().strip() for d in defects if d and d.strip()]

        for score, record in candidates:
            boost = 0.0
            record_actions = " ".join(record.actions).lower()
            record_tags = " ".join(record.tags).lower()
            record_text = self._record_text(record).lower()

            if operator_l:
                if operator_l in record_actions:
                    boost += 0.20
                elif operator_l in record_tags or operator_l in record_text:
                    boost += 0.10

            if defect_tokens:
                hit = 0
                anti_text = " ".join(record.anti_patterns).lower()
                cond_text = " ".join(record.conditions).lower()
                for defect in defect_tokens:
                    if defect in anti_text:
                        hit += 1
                    elif defect in cond_text or defect in record_text:
                        hit += 1
                boost += min(0.30, 0.08 * hit)

            final_score = min(1.0, score + boost)
            boosted.append((final_score, record))

        boosted.sort(key=lambda item: item[0], reverse=True)
        return boosted[: max(0, min(limit, len(boosted)))]

    def save(self, path: str) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            payload = {
                "next_id": self._next_id,
                "fidmap2mid": self.fidmap2mid,
                "records": {str(fid): record.to_dict() for fid, record in self._records.items()},
                "config": self.cfg.model_dump(),
            }
            with open(os.path.join(path, "symbolic_memory.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return True
        except Exception as exc:
            print(f"Error saving symbolic memories: {exc}")
            return False

    def load(self, path: str) -> bool:
        try:
            file_path = os.path.join(path, "symbolic_memory.json")
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

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
            return True
        except Exception as exc:
            print(f"Error loading symbolic memories: {exc}")
            return False

    def _record_text(self, record: SymbolicRecord) -> str:
        blocks = [
            record.summary,
            record.pattern,
            record.rationale,
            " ".join(record.conditions),
            " ".join(record.actions),
            " ".join(record.expected_outcomes),
            " ".join(record.anti_patterns),
            " ".join(record.tags),
            _safe_dump_str(record.metadata),
        ]
        return "\n".join(text for text in blocks if text)

    def _lexical_score(self, query_text: str, record: SymbolicRecord) -> float:
        return compute_overlap_score(self._record_text(record), query_text)

    def _rule_score(self, query_text: str, record: SymbolicRecord) -> float:
        tokens = self._tokenize(query_text)
        if not tokens:
            return 0.0

        sections = {
            "pattern": (record.pattern or "").lower(),
            "conditions": " ".join(record.conditions).lower(),
            "actions": " ".join(record.actions).lower(),
            "anti_patterns": " ".join(record.anti_patterns).lower(),
            "outcomes": " ".join(record.expected_outcomes).lower(),
            "tags": " ".join(record.tags).lower(),
        }
        weight_map = {
            "pattern": 1.2,
            "conditions": 1.0,
            "actions": 1.2,
            "anti_patterns": 1.1,
            "outcomes": 1.0,
            "tags": 1.2,
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
                meta_key = key.split(".", 1)[1]
                actual = record.metadata.get(meta_key)
            else:
                actual = getattr(record, key, None)
            if isinstance(expected, (list, tuple, set)):
                if isinstance(actual, list):
                    if not any(item in actual for item in expected):
                        return False
                else:
                    if actual not in expected:
                        return False
            else:
                if isinstance(actual, list):
                    if expected not in actual:
                        return False
                elif actual != expected:
                    return False
        return True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())
