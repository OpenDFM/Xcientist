import pytest

from memory.api.symbolic_memory_system_api import SymbolicMemorySystem


IDEA_ABSTRACT = (
    "An adaptive confidence gating mechanism routes uncertain samples to a "
    "stronger expert branch while easy cases use a cheap path."
)


def _add_record(
    store: SymbolicMemorySystem,
    *,
    component: str,
    family: str,
    method_context: str,
    confidence: float,
) -> str:
    record = store.instantiate_symbolic_record(
        component=component,
        component_family=family,
        result="positive",
        metric="accuracy",
        value="+1.2",
        analysis="Helpful in ablation.",
        method_context=method_context,
        confidence=confidence,
    )
    assert store.add([record], agent_id="idea_agent")
    return record.id


def test_component_level_reranks_with_idea_abstract() -> None:
    store = SymbolicMemorySystem()
    matched_id = _add_record(
        store,
        component="Confidence Gate",
        family="controller.gating_module",
        method_context=(
            "A confidence-aware gate routes uncertain samples to an expert path "
            "and keeps easy cases on a cheap branch."
        ),
        confidence=0.25,
    )
    distractor_id = _add_record(
        store,
        component="Confidence Gate",
        family="controller.gating_module",
        method_context=(
            "A curriculum controller stages training over multiple phases "
            "without dynamic expert routing at inference."
        ),
        confidence=0.95,
    )

    baseline = store.retrieve_hierarchical(
        target_component="Confidence Gate",
        target_family="controller.gating_module",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
    )
    reranked = store.retrieve_hierarchical(
        target_component="Confidence Gate",
        target_family="controller.gating_module",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
        query_context=IDEA_ABSTRACT,
    )

    assert baseline[0][1].id == distractor_id
    assert reranked[0][1].id == matched_id


def test_family_level_reranks_with_idea_abstract() -> None:
    store = SymbolicMemorySystem()
    matched_id = _add_record(
        store,
        component="Confidence Gate",
        family="controller.gating_module",
        method_context=(
            "The confidence gate escalates uncertain inputs to a stronger "
            "expert branch and preserves cheap execution for easy cases."
        ),
        confidence=0.35,
    )
    distractor_id = _add_record(
        store,
        component="Budget Gate",
        family="controller.gating_module",
        method_context=(
            "A latency budget gate disables expensive modules once the runtime "
            "cost exceeds a fixed threshold."
        ),
        confidence=0.95,
    )

    baseline = store.retrieve_hierarchical(
        target_family="controller.gating_module",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
    )
    reranked = store.retrieve_hierarchical(
        target_family="controller.gating_module",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
        query_context=IDEA_ABSTRACT,
    )

    assert baseline[0][1].id == distractor_id
    assert reranked[0][1].id == matched_id


def test_macro_role_level_reranks_with_idea_abstract() -> None:
    store = SymbolicMemorySystem()
    matched_id = _add_record(
        store,
        component="Confidence Gate",
        family="controller.gating_module",
        method_context=(
            "A confidence-aware routing gate sends uncertain examples to a "
            "stronger path and keeps easy examples on a cheap route."
        ),
        confidence=0.30,
    )
    distractor_id = _add_record(
        store,
        component="Curriculum Scheduler",
        family="controller.scheduler_module",
        method_context=(
            "A scheduler increases task difficulty over time and allocates "
            "training phases with a fixed curriculum."
        ),
        confidence=0.98,
    )

    baseline = store.retrieve_hierarchical(
        target_family="controller.unknown_module",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
    )
    reranked = store.retrieve_hierarchical(
        target_family="controller.unknown_module",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
        query_context=IDEA_ABSTRACT,
    )

    assert baseline[0][1].id == distractor_id
    assert reranked[0][1].id == matched_id


def test_query_only_supports_lexical_mode() -> None:
    store = SymbolicMemorySystem()
    matched_id = _add_record(
        store,
        component="Confidence Gate",
        family="controller.gating_module",
        method_context=(
            "A confidence-aware gate routes uncertain samples to an expert path "
            "and keeps easy cases on a cheap branch."
        ),
        confidence=0.85,
    )
    _add_record(
        store,
        component="Budget Gate",
        family="controller.gating_module",
        method_context=(
            "A latency budget gate disables expensive modules once the runtime "
            "cost exceeds a fixed threshold."
        ),
        confidence=0.85,
    )

    lexical = store.query(
        "confidence gate expert branch uncertain samples",
        method="lexical",
        limit=1,
        threshold=0.05,
        agent_id="idea_agent",
    )
    assert lexical[0][1].id == matched_id

    with pytest.raises(ValueError, match="Only 'lexical' is supported"):
        store.query(
            "confidence gate expert branch uncertain samples",
            method="rule",
            limit=1,
            threshold=0.05,
            agent_id="idea_agent",
        )
