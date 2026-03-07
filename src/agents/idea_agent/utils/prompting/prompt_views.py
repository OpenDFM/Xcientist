from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_TEXT_LIMIT = 2048


def _clip_text(value: Any, limit: int = DEFAULT_TEXT_LIMIT) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _extract_named_items(
    value: Any,
    *,
    max_items: int = 6,
) -> List[str]:
    items: List[str] = []
    for raw in _normalize_list(value):
        if len(items) >= max_items:
            break
        if isinstance(raw, Mapping):
            text = (
                raw.get("title")
                or raw.get("name")
                or raw.get("label")
                or raw.get("gap")
                or raw.get("hypothesis")
                or raw.get("summary")
                or raw.get("text")
            )
        else:
            text = raw
        clipped = _clip_text(text)
        if clipped:
            items.append(clipped)
    return _dedupe_keep_order(items)


def _format_kv_line(label: str, value: Any, limit: int = DEFAULT_TEXT_LIMIT) -> str:
    clipped = _clip_text(value)
    return f"{label}: {clipped or 'None'}"


def _format_inline_list(
    value: Any,
    *,
    max_items: int = 6,
    item_limit: int = 80,
    empty: str = "None",
) -> str:
    items = _extract_named_items(value, max_items=max_items)
    return ", ".join(items) if items else empty


def _format_section(title: str, lines: Sequence[str]) -> str:
    filtered = [line for line in lines if line and line.strip()]
    if not filtered:
        return ""
    return f"== {title} ==\n" + "\n".join(filtered)


def _join_sections(sections: Sequence[str], empty: str) -> str:
    filtered = [section for section in sections if section]
    if not filtered:
        return empty
    return "\n\n".join(filtered)


def _extract_paper_summary(paper: Mapping[str, Any]) -> str:
    for key in ("summary", "tldr", "abstract", "snippet", "excerpt"):
        value = paper.get(key)
        if value:
            return _clip_text(value)
    keynote = paper.get("keynote")
    if isinstance(keynote, Mapping):
        for key in ("tldr", "summary", "abstract", "insight", "result", "background"):
            value = keynote.get(key)
            if value:
                return _clip_text(value)
    if keynote:
        return _clip_text(keynote)
    return "No summary available."


def _format_authors(authors: Any, max_items: int = 4) -> str:
    names: List[str] = []
    for raw in _normalize_list(authors):
        if len(names) >= max_items:
            break
        if isinstance(raw, Mapping):
            candidate = raw.get("name") or raw.get("author") or raw.get("full_name")
        else:
            candidate = raw
        clipped = _clip_text(candidate)
        if clipped:
            names.append(clipped)
    return ", ".join(names) if names else ""


def _format_budget_delta(delta: Any) -> str:
    if not isinstance(delta, Mapping) or not delta:
        return "None"
    parts: List[str] = []
    for key in sorted(delta):
        value = delta.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key}={value:+.2f}")
        else:
            parts.append(f"{key}={_clip_text(value)}")
    return "; ".join(parts) if parts else "None"


def format_analysis_prompt_view(analysis: Any) -> str:
    if isinstance(analysis, list):
        entries = analysis
        latest = analysis[-1] if analysis else None
    else:
        entries = [analysis] if analysis else []
        latest = analysis

    if latest is None:
        return "No prior analysis."

    if not isinstance(latest, Mapping):
        return _clip_text(latest) or "No prior analysis."

    evaluation_gap_lines: List[str] = []
    for idx, gap in enumerate(_normalize_list(latest.get("evaluation_gaps")), start=1):
        if idx > 4:
            break
        if isinstance(gap, Mapping):
            evaluation_gap_lines.append(
                f"{idx}. Gap={_clip_text(gap.get('gap')) or 'Unspecified'}"
            )
            why = _clip_text(gap.get("why_it_matters"))
            if why:
                evaluation_gap_lines.append(f"   Why it matters: {why}")
            bar = _clip_text(gap.get("icml_expectation"))
            if bar:
                evaluation_gap_lines.append(f"   ICML bar: {bar}")
        else:
            evaluation_gap_lines.append(f"{idx}. {_clip_text(gap)}")

    idea_seed_lines: List[str] = []
    for idx, seed in enumerate(_normalize_list(latest.get("divergent_idea_seeds")), start=1):
        if idx > 3:
            break
        if isinstance(seed, Mapping):
            title = _clip_text(seed.get("title") or seed.get("hypothesis")) or f"Seed {idx}"
            hypothesis = _clip_text(seed.get("hypothesis"))
            mechanism = _clip_text(seed.get("method_sketch"))
            evaluation = _clip_text(seed.get("evaluation_plan"))
            risk = _clip_text(seed.get("risk"))
            idea_seed_lines.append(f"{idx}. {title}")
            if hypothesis:
                idea_seed_lines.append(f"   Hypothesis: {hypothesis}")
            if mechanism:
                idea_seed_lines.append(f"   Mechanism: {mechanism}")
            if evaluation:
                idea_seed_lines.append(f"   Validation: {evaluation}")
            if risk:
                idea_seed_lines.append(f"   Risk: {risk}")
        else:
            idea_seed_lines.append(f"{idx}. {_clip_text(seed)}")

    inspiration_lines: List[str] = []
    for idx, item in enumerate(_normalize_list(latest.get("cross_domain_inspiration")), start=1):
        if idx > 3:
            break
        if isinstance(item, Mapping):
            source_field = _clip_text(item.get("source_field")) or "unknown field"
            mechanism = _clip_text(item.get("transferable_mechanism"))
            hook = _clip_text(item.get("application_hook"))
            inspiration_lines.append(
                f"{idx}. {source_field}: mechanism={mechanism or 'None'} | hook={hook or 'None'}"
            )
        else:
            inspiration_lines.append(f"{idx}. {_clip_text(item)}")

    sections = [
        _format_section(
            "Analysis Snapshot",
            [
                f"History entries retained: {len(entries)}",
                _format_kv_line("TLDR", latest.get("tldr"), 220),
            ],
        ),
        _format_section(
            "Method Clusters",
            [f"- {item}" for item in _extract_named_items(latest.get("key_methods"), max_items=6)],
        ),
        _format_section(
            "Existing Problems",
            [f"- {item}" for item in _extract_named_items(latest.get("existing_problems"), max_items=6)],
        ),
        _format_section("Evaluation Gaps", evaluation_gap_lines),
        _format_section(
            "Future Directions",
            [f"- {item}" for item in _extract_named_items(latest.get("future_directions"), max_items=5)],
        ),
        _format_section("Divergent Idea Seeds", idea_seed_lines),
        _format_section("Cross-Domain Inspiration", inspiration_lines),
    ]
    return _join_sections(sections, empty="No prior analysis.")


def format_idea_pool_prompt_view(idea_pool: Any, *, max_ideas: int = 3) -> str:
    ideas = [item for item in _normalize_list(idea_pool) if item is not None]
    if not ideas:
        return "No prior ideas in the current run."

    latest_items = list(reversed(ideas[-max_ideas:]))
    blocks: List[str] = [
        _format_section(
            "Idea Pool Snapshot",
            [f"Retained ideas in run: {len(ideas)}", f"Showing latest: {len(latest_items)}"],
        )
    ]

    for idx, raw in enumerate(latest_items, start=1):
        if not isinstance(raw, Mapping):
            blocks.append(
                _format_section(
                    f"Idea {idx}",
                    [_format_kv_line("Raw", raw, 400)],
                )
            )
            continue

        evaluation = raw.get("evaluation") if isinstance(raw.get("evaluation"), Mapping) else {}
        metrics: List[str] = []
        for key in ("novelty", "feasibility", "impact", "risk", "confidence"):
            value = evaluation.get(key)
            if isinstance(value, (int, float)):
                metrics.append(f"{key}={value:.2f}")
        score = raw.get("search_score")
        if isinstance(score, (int, float)):
            metrics.insert(0, f"search_score={score:.2f}")

        blocks.append(
            _format_section(
                f"Idea {idx}",
                [
                    _format_kv_line("Title", raw.get("title"), 120),
                    _format_kv_line(
                        "Core",
                        raw.get("core_contribution") or raw.get("core_contribute"),
                        180,
                    ),
                    _format_kv_line(
                        "Method",
                        raw.get("method") or raw.get("methodology"),
                        200,
                    ),
                    _format_kv_line(
                        "Experiments",
                        raw.get("experiments") or raw.get("experiment_design"),
                        180,
                    ),
                    _format_kv_line("Risks", raw.get("risks"), 180),
                    f"Components: {_format_inline_list(raw.get('components'), max_items=6, item_limit=50)}",
                    f"Target defects: {_format_inline_list(raw.get('target_defects'), max_items=6, item_limit=50)}",
                    f"Evaluation: {'; '.join(metrics) if metrics else 'None'}",
                ],
            )
        )

    return _join_sections(blocks, empty="No prior ideas in the current run.")


def format_paper_capsules_prompt_view(papers: Any, *, max_papers: int = 8) -> str:
    items = [item for item in _normalize_list(papers) if item is not None]
    if not items:
        return "No curated paper capsules."

    sections = [
        _format_section(
            "Curated Paper Capsules",
            [f"Available capsules: {len(items)}", f"Showing: {min(len(items), max_papers)}"],
        )
    ]
    for idx, raw in enumerate(items[:max_papers], start=1):
        if not isinstance(raw, Mapping):
            sections.append(_format_section(f"Paper {idx}", [_format_kv_line("Raw", raw, 300)]))
            continue

        authors = _format_authors(raw.get("authors"))
        meta_parts: List[str] = []
        if raw.get("year"):
            meta_parts.append(f"year={raw.get('year')}")
        if raw.get("venue"):
            meta_parts.append(f"venue={_clip_text(raw.get('venue'))}")
        if authors:
            meta_parts.append(f"authors={authors}")
        if raw.get("paper_id"):
            meta_parts.append(f"id={_clip_text(raw.get('paper_id'))}")

        sections.append(
            _format_section(
                f"Paper {idx}",
                [
                    _format_kv_line("Title", raw.get("title") or raw.get("paper_title"), 140),
                    f"Metadata: {' | '.join(meta_parts) if meta_parts else 'None'}",
                    _format_kv_line("Summary", _extract_paper_summary(raw), 260),
                ],
            )
        )

    return _join_sections(sections, empty="No curated paper capsules.")


def format_paper_context_prompt_view(
    entries: Sequence[Dict[str, Any]],
    artifact: Optional[Dict[str, Any]] = None,
    *,
    max_papers: int = 8,
    max_rag_hits: int = 4,
    max_survey_items: int = 4,
) -> str:
    artifact = artifact or {}
    paper_lines: List[str] = []
    for idx, entry in enumerate(list(entries or [])[:max_papers], start=1):
        if not isinstance(entry, Mapping):
            paper_lines.append(f"[P{idx}] {_clip_text(entry)}")
            continue
        header_parts = [f"[P{idx}] {_clip_text(entry.get('title') or entry.get('paper_id'))}"]
        source = _clip_text(entry.get("source"))
        if source:
            header_parts.append(f"source={source}")
        authors = _format_authors(entry.get("authors"))
        if authors:
            header_parts.append(f"authors={authors}")
        paper_lines.append(" | ".join(header_parts))
        paper_lines.append(f"  Summary: {_clip_text(entry.get('summary')) or 'No summary available.'}")

    rag_lines: List[str] = []
    rag_entries = artifact.get("rag_hits", [])
    latest_rag = rag_entries[-1] if isinstance(rag_entries, list) and rag_entries else rag_entries
    hits = []
    if isinstance(latest_rag, Mapping):
        hits = latest_rag.get("hits") or []
    elif isinstance(latest_rag, list):
        hits = latest_rag
    for idx, hit in enumerate(hits[:max_rag_hits], start=1):
        if not isinstance(hit, Mapping):
            rag_lines.append(f"[R{idx}] {_clip_text(hit)}")
            continue
        rag_lines.append(f"[R{idx}] {_clip_text(hit.get('title') or f'RAG hit {idx}')}")
        rag_lines.append(f"  Signal: {_clip_text(hit.get('subsection')) or 'None'}")
        citations = _format_inline_list(hit.get("citations"), max_items=4, item_limit=50)
        rag_lines.append(f"  Citations: {citations}")

    survey_lines: List[str] = []
    rag_contents = artifact.get("rag_contents", [])
    latest_sections = rag_contents[-1] if isinstance(rag_contents, list) and rag_contents else rag_contents
    if isinstance(latest_sections, str):
        latest_sections = [latest_sections]
    for idx, section in enumerate(_normalize_list(latest_sections)[:max_survey_items], start=1):
        survey_lines.append(f"[S{idx}] {_clip_text(section)}")

    sections = [
        _format_section(
            "Paper Context Snapshot",
            [
                f"Curated papers: {len(entries or [])}",
                f"RAG hits: {len(hits)}",
                f"Survey excerpts: {len(_normalize_list(latest_sections))}",
            ],
        ),
        _format_section("Curated Papers", paper_lines),
        _format_section("RAG Evidence", rag_lines),
        _format_section("Survey Excerpts", survey_lines),
    ]
    return _join_sections(sections, empty="No curated papers available yet.")


def format_idea_prompt_view(idea: Any, *, heading: str = "Idea Snapshot") -> str:
    payload = idea.to_payload() if hasattr(idea, "to_payload") and callable(idea.to_payload) else idea
    if not isinstance(payload, Mapping):
        return _join_sections(
            [_format_section(heading, [_format_kv_line("Raw", payload, 1200)])],
            empty="No idea available.",
        )

    component_lines: List[str] = []
    components = _normalize_list(payload.get("components"))
    explanations = payload.get("component_explanations")
    explanation_map = explanations if isinstance(explanations, Mapping) else {}
    for component in components[:8]:
        name = _clip_text(component)
        if not name:
            continue
        explanation = _clip_text(explanation_map.get(component)) if explanation_map else ""
        if explanation:
            component_lines.append(f"- {name}: {explanation}")
        else:
            component_lines.append(f"- {name}")

    skill_metrics = payload.get("skill_metrics") if isinstance(payload.get("skill_metrics"), Mapping) else {}
    metric_parts: List[str] = []
    for key in sorted(skill_metrics):
        if key in {"skill_prior_before", "skill_prior_after"}:
            continue
        value = skill_metrics.get(key)
        if isinstance(value, (int, float)):
            metric_parts.append(f"{key}={value:.2f}")
        else:
            metric_parts.append(f"{key}={_clip_text(value)}")

    budget = payload.get("budget")
    if isinstance(budget, Mapping):
        budget_text = "; ".join(
            f"{key}={budget[key]:.2f}" if isinstance(budget[key], (int, float)) else f"{key}={_clip_text(budget[key])}"
            for key in sorted(budget)
        )
    else:
        budget_text = "None"

    sections = [
        _format_section(
            heading,
            [
                _format_kv_line("Title", payload.get("title"), 140),
                _format_kv_line("Abstract", payload.get("abstract"), 220),
                _format_kv_line(
                    "Core Contribution",
                    payload.get("core_contribution") or payload.get("core_contribute"),
                    220,
                ),
                _format_kv_line(
                    "Method",
                    payload.get("method") or payload.get("methodology"),
                    260,
                ),
                _format_kv_line(
                    "Experiments",
                    payload.get("experiments") or payload.get("experiment_design"),
                    240,
                ),
                _format_kv_line("Risks", payload.get("risks"), 220),
                f"Operator: {_clip_text(payload.get('operator')) or 'None'}",
                f"Target defects: {_format_inline_list(payload.get('target_defects'), max_items=6, item_limit=50)}",
                f"Budget: {budget_text}",
                f"Memory refs: {_format_inline_list(payload.get('memory_refs'), max_items=6, item_limit=60)}",
                _format_kv_line("Rationale", payload.get("rationale"), 220),
                f"Skill metrics: {'; '.join(metric_parts) if metric_parts else 'None'}",
            ],
        ),
        _format_section("Components", component_lines),
    ]
    return _join_sections(sections, empty="No idea available.")


def format_edit_plan_prompt_view(plan: Any, *, heading: str = "Compiled Edit Plan") -> str:
    payload = plan.to_dict() if hasattr(plan, "to_dict") and callable(plan.to_dict) else plan
    if not isinstance(payload, Mapping):
        return _join_sections(
            [_format_section(heading, [_format_kv_line("Raw", payload, 1200)])],
            empty="No edit plan available.",
        )

    component_lines: List[str] = []
    for idx, edit in enumerate(_normalize_list(payload.get("component_edits")), start=1):
        if not isinstance(edit, Mapping):
            component_lines.append(f"{idx}. {_clip_text(edit)}")
            continue
        component_lines.append(
            f"{idx}. op={_clip_text(edit.get('op')) or 'unknown'} | "
            f"component={_clip_text(edit.get('component')) or 'None'} | "
            f"target={_clip_text(edit.get('target')) or 'None'}"
        )
        condition = _clip_text(edit.get("condition"))
        details = _clip_text(edit.get("details"))
        reason = _clip_text(edit.get("reason"))
        if condition:
            component_lines.append(f"   condition: {condition}")
        if details:
            component_lines.append(f"   details: {details}")
        if reason:
            component_lines.append(f"   reason: {reason}")

    validation = payload.get("validation") if isinstance(payload.get("validation"), Mapping) else {}
    validation_lines: List[str] = []
    for label, key in (
        ("Regression", "regression_tests"),
        ("Ablation", "ablation_tests"),
        ("Stress", "stress_tests"),
    ):
        items = _extract_named_items(validation.get(key), max_items=4)
        if not items:
            continue
        validation_lines.append(f"{label}:")
        validation_lines.extend(f"- {item}" for item in items)

    sections = [
        _format_section(
            heading,
            [
                _format_kv_line("Skill", payload.get("skill_name"), 100),
                _format_kv_line("Objective", payload.get("objective"), 180),
                f"Target defects: {_format_inline_list(payload.get('target_defects'), max_items=6, item_limit=50)}",
                f"Guardrails: {_format_inline_list(payload.get('guardrails'), max_items=6, item_limit=70)}",
                f"Budget delta: {_format_budget_delta(payload.get('estimated_budget_delta'))}",
                f"Memory refs: {_format_inline_list(payload.get('memory_refs'), max_items=6, item_limit=60)}",
                _format_kv_line("Compile notes", payload.get("compile_notes"), 180),
            ],
        ),
        _format_section("Component Edits", component_lines),
        _format_section("Validation Protocol", validation_lines),
    ]
    return _join_sections(sections, empty="No edit plan available.")
