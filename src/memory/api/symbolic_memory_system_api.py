"""Symbolic memory system indexed by component_family.

Each memory record contains:
- **component_family**: the target component family (``{macro_role}.{sub_type}``)
- **family_pair**: optional secondary family (forming a family pair)
- **main_op**: the component-level primary operation type
- **context_signature**: structured query features compressed from the parent
  node state (structural features, discretised signals, trajectory features,
  budget / contract context)
- **delta_score**: the observed composite-score improvement from applying this
  component-level action in the recorded context

Retrieval is executed hierarchically during the *expand* phase:
1. **Exact family match** -- records whose ``component_family`` exactly
   matches the query target.
2. **Macro-role fallback** -- if insufficient results, broaden to all records
   sharing the same ``macro_role``.
3. **Defect / bucket fallback** -- if still insufficient, scan all records and
   rank by defect-profile and bucket overlap.

Candidates are ranked by:
    structural_match x condition_match x reliability x local_priority

The ranked results are converted to per-``main_op`` prior gains (estimated
delta-score under the current parent state), which serve as *action priors*
in PUCT rather than replacing Q-values.
"""
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal, Sequence
from uuid import uuid4

from memory.api.base_symbolic_memory_system_api import (
    SymbolicMemorySystem as BaseSymbolicMemorySystem,
    SymbolicMemorySystemConfig,
    SymbolicRecordPayload,
    SymbolicRecord,
)
from memory.api.component_taxonomy import (
    MACRO_ROLE_NAMES,
    ContextSignature,
    parse_component_family,
)
from memory.memory_system.utils import (
    _safe_dump_str,
    compute_overlap_score,
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
#  Scoring helpers
# ---------------------------------------------------------------------------

def _reliability_score(record: SymbolicRecord) -> float:
    """Combine confidence and support_count into a [0, 1] reliability score.

    reliability = confidence * (1 - 1 / (1 + log2(support_count)))
    A single observation yields ~0; many observations converge to confidence.
    """
    sc = max(record.support_count, 1)
    support_factor = 1.0 - 1.0 / (1.0 + math.log2(sc))
    return record.confidence * max(0.0, support_factor)


def _condition_match_score(
    query_sig: ContextSignature,
    record: SymbolicRecord,
) -> float:
    """Score how well the record's stored conditions match *query_sig*.

    The record's ``context_signature`` (stored at recording time) is compared
    against the current parent-node context via
    ``ContextSignature.structural_match_score``.  If the record has no stored
    signature, a fallback heuristic based on textual conditions is used.
    """
    stored_raw = record.context_signature
    if stored_raw:
        stored_sig = (
            stored_raw
            if isinstance(stored_raw, ContextSignature)
            else ContextSignature.from_dict(stored_raw)
        )
        return query_sig.structural_match_score(stored_sig)

    # Fallback: lightweight text overlap between conditions and query fields
    cond_text = " ".join(record.conditions).lower()
    if not cond_text:
        return 0.5  # neutral when no conditions recorded

    query_tokens = set()
    query_tokens.update(query_sig.macro_roles_present)
    query_tokens.update(query_sig.defect_profile)
    query_tokens.add(query_sig.coverage_bucket)
    query_tokens.add(query_sig.stability_bucket)
    query_tokens.add(query_sig.cost_bucket)
    query_tokens.add(query_sig.budget_pressure)
    if query_sig.last_main_op:
        query_tokens.add(query_sig.last_main_op)
    query_tokens.discard("")

    if not query_tokens:
        return 0.5
    hits = sum(1 for t in query_tokens if t in cond_text)
    return hits / len(query_tokens)


# ============================================================================
#  Main class
# ============================================================================

class SymbolicMemorySystem(BaseSymbolicMemorySystem):
    """Component-family indexed symbolic memory system.

    Each memory record contains:
    - **component_family**: the target component family
    - **family_pair**: optional secondary family (family pair)
    - **main_op**: the component-level primary operation type
    - **context_signature**: structured query features compressed from the
      parent-node state
    - **delta_score**: expected delta-score of executing the action under the
      recorded context

    Retrieval is executed hierarchically:
    1. exact family match
    2. macro_role fallback
    3. defect / bucket-level fallback
    4. lexical / embedding fine-ranking
    """

    def __init__(self, **kwargs):
        cfg = SymbolicMemorySystemConfig(**kwargs)
        self.cfg = cfg

        self._records: Dict[int, SymbolicRecord] = {}
        self.fidmap2mid: Dict[int, str] = {}
        self.midmap2fid: Dict[str, int] = {}
        self._next_id = 0

        # -- secondary indexes (accelerate hierarchical retrieval) --
        self._family_index: Dict[str, List[int]] = {}   # family -> [fid, ...]
        self._role_index: Dict[str, List[int]] = {}     # macro_role -> [fid, ...]
        self._mainop_index: Dict[str, List[int]] = {}   # main_op -> [fid, ...]

    # ──────────────────────────────────────────────────────────────────────────
    #  Index maintenance
    # ──────────────────────────────────────────────────────────────────────────

    def _index_record(self, fid: int, record: SymbolicRecord) -> None:
        """Add a record to the secondary indexes."""
        family = (record.component_family or "").strip().lower()
        if family:
            self._family_index.setdefault(family, [])
            if fid not in self._family_index[family]:
                self._family_index[family].append(fid)
            role, _ = parse_component_family(family)
            self._role_index.setdefault(role, [])
            if fid not in self._role_index[role]:
                self._role_index[role].append(fid)
        main_op = (record.main_op or "").strip().lower()
        if main_op:
            self._mainop_index.setdefault(main_op, [])
            if fid not in self._mainop_index[main_op]:
                self._mainop_index[main_op].append(fid)

    def _unindex_record(self, fid: int, record: SymbolicRecord) -> None:
        """Remove a record from the secondary indexes."""
        family = (record.component_family or "").strip().lower()
        if family and family in self._family_index:
            self._family_index[family] = [f for f in self._family_index[family] if f != fid]
            if not self._family_index[family]:
                del self._family_index[family]
            role, _ = parse_component_family(family)
            if role in self._role_index:
                self._role_index[role] = [f for f in self._role_index[role] if f != fid]
                if not self._role_index[role]:
                    del self._role_index[role]
        main_op = (record.main_op or "").strip().lower()
        if main_op and main_op in self._mainop_index:
            self._mainop_index[main_op] = [f for f in self._mainop_index[main_op] if f != fid]
            if not self._mainop_index[main_op]:
                del self._mainop_index[main_op]

    def _rebuild_indexes(self) -> None:
        """Fully rebuild secondary indexes from ``_records``."""
        self._family_index.clear()
        self._role_index.clear()
        self._mainop_index.clear()
        for fid, record in self._records.items():
            self._index_record(fid, record)

    # ──────────────────────────────────────────────────────────────────────────
    #  Record instantiation
    # ──────────────────────────────────────────────────────────────────────────

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
            component_family=payload.component_family,
            family_pair=payload.family_pair,
            main_op=payload.main_op,
            context_signature=dict(payload.context_signature or {}),
            delta_score=payload.delta_score,
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
                    old_record = self._records.get(fid)
                    if old_record:
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
        if memories is None:
            memories = []
        try:
            for memory in memories:
                if not isinstance(memory, SymbolicRecord):
                    continue
                fid = self.midmap2fid.get(memory.id)
                if fid is None:
                    continue
                old_record = self._records.get(fid)
                if old_record:
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
                if old_record:
                    self._unindex_record(fid, old_record)
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

    # ──────────────────────────────────────────────────────────────────────────
    #  Generic query (text-based, backward-compatible)
    # ──────────────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────────────
    #  Hierarchical retrieval for the expand phase
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve_hierarchical(
        self,
        target_family: str = "",
        context_sig: Optional[ContextSignature] = None,
        main_op: str = "",
        limit: int = 5,
        threshold: float = 0.25,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        """Three-level hierarchical retrieval.

        Cascade:
        1. **Exact family** -- records whose ``component_family`` matches
           *target_family* exactly.
        2. **Macro-role fallback** -- all records sharing the same macro_role
           as *target_family*.
        3. **Defect / bucket fallback** -- remaining records ranked by
           defect-profile and bucket overlap with *context_sig*.

        Within each level, candidates are scored by::

            structural_match * condition_match * reliability * local_priority

        Records with ``score < threshold`` are discarded.

        Args:
            target_family: component_family to query (e.g.
                ``"constraint.uncertainty_weighted"``).
            context_sig: ContextSignature extracted from the parent-node state.
            main_op: optional filter to restrict results to a specific main_op.
            limit: maximum number of results to return.
            threshold: minimum score to include a candidate.
            agent_id: restrict to records tagged with this agent.

        Returns:
            Sorted list of ``(score, SymbolicRecord)`` tuples, highest first.
        """
        if self.size == 0:
            return []

        context_sig = context_sig or ContextSignature()
        target_family_l = (target_family or "").strip().lower()
        target_role, _ = parse_component_family(target_family_l) if target_family_l else ("", "")
        main_op_l = (main_op or "").strip().lower()

        scored: Dict[int, Tuple[float, SymbolicRecord]] = {}

        def _score_candidate(fid: int, tier_bonus: float) -> None:
            """Score a single candidate and store if above threshold."""
            if fid in scored:
                return
            record = self._records.get(fid)
            if record is None:
                return
            if agent_id and not str(record.id).endswith(f"_{agent_id}"):
                return
            if main_op_l and (record.main_op or "").strip().lower() != main_op_l:
                return

            struct_match = _condition_match_score(context_sig, record)
            cond_match = self._condition_text_relevance(context_sig, record)
            reliability = _reliability_score(record)
            local_priority = record.priority

            raw = struct_match * cond_match * max(reliability, 0.05) * (0.4 + 0.6 * local_priority)
            # Apply tier bonus (exact family > macro_role > global fallback)
            final = min(1.0, raw + tier_bonus)
            if final >= threshold:
                scored[fid] = (float(final), record)

        # --- Level 1: exact family match ---
        if target_family_l and target_family_l in self._family_index:
            for fid in self._family_index[target_family_l]:
                _score_candidate(fid, tier_bonus=0.15)

        # --- Level 2: macro_role fallback ---
        if len(scored) < limit and target_role and target_role in self._role_index:
            for fid in self._role_index[target_role]:
                _score_candidate(fid, tier_bonus=0.05)

        # --- Level 3: defect / bucket global fallback ---
        if len(scored) < limit:
            for fid in self._records:
                _score_candidate(fid, tier_bonus=0.0)

        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        return ranked[: max(0, min(limit, len(ranked)))]

    def retrieve_priors_for_expand(
        self,
        topic: str,
        operator: str = "",
        defects: Optional[List[str]] = None,
        limit: int = 5,
        threshold: float = 0.0,
        agent_id: str = "",
        *,
        target_family: str = "",
        context_sig: Optional[ContextSignature] = None,
    ) -> List[Tuple[float, SymbolicRecord]]:
        """Retrieve priors for the expand phase.

        When *target_family* and *context_sig* are provided, uses the
        three-level hierarchical retrieval (``retrieve_hierarchical``).
        Otherwise falls back to the legacy text-based retrieval with
        operator / defect boosting.

        Args:
            topic: free text describing the current expansion target.
            operator: optional operator / skill name to boost.
            defects: list of current defect labels.
            limit: maximum number of results.
            threshold: minimum score cutoff.
            agent_id: restrict to records from this agent.
            target_family: (keyword-only) component_family for hierarchical
                retrieval.
            context_sig: (keyword-only) ContextSignature from the parent node
                for structural matching.

        Returns:
            Sorted ``[(score, SymbolicRecord), ...]``, highest first.
        """
        # ---- hierarchical path (preferred) ----
        if target_family or context_sig:
            return self.retrieve_hierarchical(
                target_family=target_family,
                context_sig=context_sig,
                main_op=operator,
                limit=limit,
                threshold=threshold,
                agent_id=agent_id,
            )

        # ---- legacy text-based path ----
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

    # ──────────────────────────────────────────────────────────────────────────
    #  Action-prior computation for PUCT
    # ──────────────────────────────────────────────────────────────────────────

    def compute_action_priors(
        self,
        target_family: str,
        context_sig: ContextSignature,
        limit: int = 20,
        threshold: float = 0.25,
        agent_id: str = "",
    ) -> Dict[str, float]:

        candidates = self.retrieve_hierarchical(
            target_family=target_family,
            context_sig=context_sig,
            limit=limit,
            threshold=threshold,
            agent_id=agent_id,
        )

        if not candidates:
            return {}

        # Accumulate weighted delta_score per main_op
        op_weight_sum: Dict[str, float] = defaultdict(float)
        op_delta_sum: Dict[str, float] = defaultdict(float)

        for score, record in candidates:
            op = (record.main_op or "").strip().lower()
            if not op:
                continue
            weight = score  # score already encodes struct*cond*reliability*priority
            op_weight_sum[op] += weight
            op_delta_sum[op] += weight * record.delta_score

        priors: Dict[str, float] = {}
        for op in op_weight_sum:
            w = op_weight_sum[op]
            if w > 1e-9:
                priors[op] = op_delta_sum[op] / w

        return priors



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

            # Rebuild secondary indexes after loading
            self._rebuild_indexes()
            return True
        except Exception as exc:
            print(f"Error loading symbolic memories: {exc}")
            return False

    def _record_text(self, record: SymbolicRecord) -> str:
        """Concatenate all textual fields of a record into a single string."""
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

    def _condition_text_relevance(
        self,
        context_sig: ContextSignature,
        record: SymbolicRecord,
    ) -> float:
        """Lightweight relevance between context_sig fields and record's
        textual conditions / anti_patterns / tags.

        Returns a score in [0, 1].
        """
        text_pool = " ".join([
            " ".join(record.conditions),
            " ".join(record.anti_patterns),
            " ".join(record.tags),
            record.pattern or "",
        ]).lower()

        if not text_pool.strip():
            return 0.5  # neutral when record has no textual signals

        tokens: List[str] = []
        tokens.extend(context_sig.macro_roles_present)
        tokens.extend(context_sig.defect_profile)
        tokens.append(context_sig.coverage_bucket)
        tokens.append(context_sig.stability_bucket)
        tokens.append(context_sig.cost_bucket)
        tokens.append(context_sig.budget_pressure)
        if context_sig.last_main_op:
            tokens.append(context_sig.last_main_op)
        tokens = [t.lower() for t in tokens if t]

        if not tokens:
            return 0.5

        hits = sum(1 for t in tokens if t in text_pool)
        return max(0.1, hits / len(tokens))

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
