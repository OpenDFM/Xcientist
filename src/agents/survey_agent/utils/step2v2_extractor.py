"""
Step2v2 Extractor Module - Refactored for compatibility with ChatAgent

This module provides functions to extract structured information from academic papers.
It can be used with the project's ChatAgent via batch_remote_chat_with_retry.

Usage:
    1. Call extract_paper_info_batch() to extract info for multiple papers
    2. The function handles prompt building, API calls, and result aggregation internally
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any

from utils.utils import extract_json

# ================= CONSTANTS =================

ARXIV_DOMAINS = [
    "cs.CV", "cs.CL", "cs.LG", "cs.AI", "cs.RO", "cs.NE", "stat.ML", "cs.MA",
    "cs.DC", "cs.OS", "cs.AR", "cs.NI", "cs.PL", "cs.SE", "cs.DB", "cs.PF",
    "cs.CC", "cs.DS", "cs.LO", "cs.IT", "cs.CR", "cs.GT",
    "cs.HC", "cs.SI", "cs.CY", "cs.IR", "cs.MM", "cs.DL",
    "cs.CE", "cs.CG", "cs.DM", "cs.ET", "cs.FL", "cs.GL", 
    "cs.GR", "cs.MS", "cs.NA", "cs.OH", "cs.SC", "cs.SD", "cs.SY",
    "Hybrid", "Others"
]

PAPER_TYPES = [
    "Methodology", "System Design", "Benchmark/Dataset", 
    "Theoretical Proof", "Empirical Study", "Survey/Review", 
    "Application", "Position Paper", "Hybrid", 
    "Resource Paper", "Reproducibility Study", "Negative Results", 
    "Tutorial/Educational", "Others"
]

METRIC_BLACKLIST = {
    "accuracy", "precision", "recall", "f1", "auc", "roc", "map", "iou", "psnr", "ssim", 
    "fid", "inception score", "bleu", "rouge", "meteor", "cider", "fps", "latency", 
    "throughput", "flops", "parameters", "training time", "convergence", "error rate",
    "speed", "cost", "memory", "loss", "perplexity", "top-1", "top-5", "mae", "rmse",
    "validity", "novelty", "uniqueness", "diversity", "baseline", "sota", "method", 
    "model", "algorithm", "previous work", "prior art", "ablation", "variant", "proposed"
}


# ================= UTILITY FUNCTIONS =================

def clean_title_latex(title: str) -> str:
    """Clean LaTeX special characters from title."""
    if not title: return "Unknown Title"
    t = re.sub(r'^\d+(\.\d+)*\s+', '', title) 
    t = re.sub(r'[$\\{}"]', '', t) 
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_regex_candidates(full_text: str) -> Dict[str, List[str]]:
    """Extract candidate methods and baselines using regex patterns."""
    candidates = {"methods": set(), "baselines": set()}
    
    method_pattern = re.compile(
        r"(?:proposed|our|new|introduce|present)\s+(?:method|framework|algorithm|network|model)?\s*([A-Z][A-Za-z0-9\-]{2,15})", 
        re.MULTILINE
    )
    
    baseline_pattern = re.compile(
        r"(?:compare(?:s|d)?\s+(?:with|to|against)|outperform(?:s|ed)|surpass(?:es|ed)|superior\s+to|better\s+than|following|adopted\s+from|baseline|built\s+upon|extends?|evaluates?|analyzes?|prior\s+work|existing\s+method)\s+([A-Z][A-Za-z0-9\-]{2,15})", 
        re.IGNORECASE | re.MULTILINE
    )
    
    for m in method_pattern.findall(full_text):
        if len(m) > 2 and m.lower() not in METRIC_BLACKLIST:
            candidates["methods"].add(m)
    for b in baseline_pattern.findall(full_text):
        if len(b) > 2 and b.lower() not in METRIC_BLACKLIST:
            candidates["baselines"].add(b)
            
    return {"methods": list(candidates["methods"]), "baselines": list(candidates["baselines"])}

def mark_entities_in_text(full_text: str, entity_list: List[str], marker_prefix: str) -> str:
    """Mark entities in text with special markers."""
    marked_text = full_text
    for entity in entity_list:
        if not entity or len(entity) < 3:
            continue
        escaped = re.escape(entity)
        pattern = r'\b' + escaped + r'\b'
        replacement = f"[{marker_prefix}:{entity}]"
        marked_text = re.sub(pattern, replacement, marked_text, count=1, flags=re.IGNORECASE)
    return marked_text


# ================= PROMPT BUILDERS =================

def build_main_extraction_prompt(markdown_text: str, title: str, regex_candidates: dict) -> Tuple[str, str]:
    """
    Build main extraction prompt for a single paper.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    domain_str = ", ".join(ARXIV_DOMAINS)
    type_str = ", ".join(PAPER_TYPES)
    
    marked_text = mark_entities_in_text(markdown_text, regex_candidates['methods'], "HINT_METHOD")
    
    system = """You are a Senior Research Analyst extracting structured metadata from academic papers.

CRITICAL GROUNDING RULES:
1. Insight: Must provide INDEPENDENT ANALYSIS beyond summary. Describe prerequisites, trade-offs, design choices, or potential defects. DO NOT copy summary or quote.
2. Quote: Must be VERBATIM text from the original paper. Use ... for exact excerpts. DO NOT paraphrase or fabricate.
3. Summary: Concise factual description in natural sentence form.
4. Keywords: Concept words used for COARSE-GRAINED RETRIEVAL and filtering.

PRIORITY: RECALL - Extract ALL relevant entities (cores, components, problems, innovations, limitations, future work).

Core vs Component Granularity Rules:
- Core Contribution: The LARGEST, TOP-LEVEL contribution that is NOT contained by any other entity.
- Component: ALL sub-elements that are PART OF a Core Contribution.

Text Marking System:
Markers like [HINT_METHOD:MethodName] are SUGGESTIONS only. Use them to guide attention but read the FULL paper independently.

Output JSON Schema:
{
  "metadata": {
    "domain": "str (from allowed domains)",
    "paper_type": "str (from allowed types)",
    "structured_summary": {"background": "str", "method": "str", "result": "str"},
    "code_url": "str or null"
  },
  "problems": [{"keywords": ["str"], "summary": "str", "insight": "str", "quote": "str", "related_to_core": "str"}],
  "core_contributions": [{"name": "str", "acronym": "str or null", "type": "str", "keywords": ["str"], "summary": "str", "insight": "str", "quote": "str"}],
  "core_relations": [{"source": "str", "target": "str", "keywords": ["str"], "summary": "str", "insight": "str", "quote": "str"}],
  "components": [{"name": "str", "acronym": "str or null", "related_to_core": "str", "keywords": ["str"], "summary": "str", "insight": "str", "quote": "str"}],
  "innovations": [{"keywords": ["str"], "summary": "str", "insight": "str", "quote": "str", "related_to_core": "str"}],
  "limitations": [{"keywords": ["str"], "summary": "str", "insight": "str", "quote": "str", "related_to_core": "str"}],
  "future_work": [{"keywords": ["str"], "summary": "str", "insight": "str", "quote": "str", "related_to_core": "str"}]
}

CRITICAL: Follow exact attribute order. Ignore [HINT_METHOD:...] markers in output."""

    user = f"""Domain options: {domain_str}
Type options: {type_str}

FULL PAPER TEXT (marked suggestions):
{marked_text}

Task: Extract structured information for paper titled "{title}"

Return JSON only:"""
    
    return system, user


def build_baseline_extraction_prompt(markdown_text: str, core_names: List[str], regex_candidates: dict) -> Tuple[str, str]:
    """
    Build baseline/dataset extraction prompt for a single paper.
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    marked_text = mark_entities_in_text(markdown_text, regex_candidates['baselines'], "HINT_BASELINE")
    
    system = """You are a Research Graph Builder focusing on relationship extraction.

CRITICAL: summary/insight/quote describe THE RELATIONSHIP/COMPARISON, NOT the entity itself.

PRIORITY: RECALL > PRECISION. Extract ALL baselines and datasets from:
- Experimental results, Related Work, Appendix, Theoretical comparisons

What is a Baseline:
1. Competitors in result tables
2. Predecessors this work builds upon
3. Related work discussed as alternatives
4. Ablation variants (e.g., Ours w/o attention)
5. For Dataset papers: Previous datasets compared

Output JSON Schema:
{
  "baselines": [{"name": "str", "acronym": "str or null", "related_to_core": "str", "keywords": ["str"], "summary": "str", "metrics": ["str"] or null, "insight": "str", "quote": "str"}],
  "datasets": [{"name": "str", "acronym": "str or null", "related_to_core": "str", "keywords": ["str"], "summary": "str", "metrics": ["str"] or null, "insight": "str", "quote": "str"}]
}

Never include Core Contributions from THIS paper as baselines."""

    user = f"""Core Methods in THIS Paper (DO NOT include as baselines): {json.dumps(core_names)}

FULL PAPER TEXT (focus especially on Experiments, Related Work, Appendix):
{marked_text}

Task: Extract ALL baselines and datasets.

Return JSON only:"""
    
    return system, user


# ================= VALIDATION & PARSING (COMBINED) =================

def validate_and_parse_main(response: str, info_dict: dict = None) -> Tuple[bool, dict]:
    """
    Validate and parse main extraction response.
    
    This function combines validation and parsing - the second return value
    is the parsed result for use in aggregation.
    
    Args:
        response: LLM response string
        info_dict: Optional dict with 'paper_id' and 'regex_candidates'
        
    Returns:
        Tuple of (is_valid, parsed_result_or_response)
        - If valid: (True, parsed dict with paper_id and result)
        - If invalid: raises ValueError
    """
    if not response:
        raise ValueError("Empty response")
    
    parsed = extract_json(response)
    
    if not isinstance(parsed, dict):
        raise ValueError("Response is not a JSON object")
    
    # Check required top-level keys
    if 'metadata' not in parsed:
        raise ValueError("Missing required key: metadata")
    
    # Extract core names for baseline extraction
    core_contributions = parsed.get("core_contributions", [])
    core_names = [c.get("name") for c in core_contributions if c.get("name")]
    
    result = {
        'node_id': info_dict.get('node_id') if info_dict else None,
        'result': parsed,
        'core_names': core_names
    }
    
    return True, result


def validate_and_parse_baseline(response: str, info_dict: dict = None) -> Tuple[bool, dict]:
    """
    Validate and parse baseline extraction response.
    
    This function combines validation and parsing - the second return value
    is the parsed result for use in aggregation.
    
    Args:
        response: LLM response string
        info_dict: Optional dict with 'node_id' and 'core_names'
        
    Returns:
        Tuple of (is_valid, parsed_result_or_response)
        - If valid: (True, parsed dict with node_id and result)
        - If invalid: raises ValueError
    """
    if not response:
        raise ValueError("Empty response")
    
    parsed = extract_json(response)
    
    if not isinstance(parsed, dict):
        raise ValueError("Response is not a JSON object")
    
    # Check required top-level keys
    if 'baselines' not in parsed and 'datasets' not in parsed:
        raise ValueError("Missing required keys: baselines or datasets")
    
    result = {
        'node_id': info_dict.get('node_id') if info_dict else None,
        'result': parsed,
        'core_names': info_dict.get('core_names', []) if info_dict else []
    }
    
    return True, result


# ================= RESULT AGGREGATOR =================

def aggregate_results(
    main_results: List[dict],
    baseline_results: List[dict],
    source_info: Dict[str, dict]
) -> List[dict]:
    """Aggregate main and baseline results into final output."""
    main_by_paper = {r['node_id']: r for r in main_results if r}
    baseline_by_paper = {r['node_id']: r for r in baseline_results if r}
    
    aggregated = []
    for node_id in main_by_paper:
        main_res = main_by_paper[node_id]
        baseline_res = baseline_by_paper.get(node_id, {})
        info = source_info.get(node_id, {})
        
        result = main_res['result']
        core_contributions = result.get("core_contributions", [])
        core_names = main_res.get('core_names', [])
        
        meta_info = result.get("metadata", {})
        official_title = meta_info.get('title', 'Unknown')
        
        # Build self-reference set
        self_names = {official_title.lower(), "ours", "proposed", "method", "model", "framework", "this work"}
        for c in core_contributions:
            if c.get("name"): self_names.add(c.get("name").lower())
            if c.get("acronym"): self_names.add(c.get("acronym").lower())
        
        # Filter components
        valid_components = []
        for comp in result.get("components", []):
            c_name = comp.get("name", "").strip()
            if not c_name or c_name.lower() in self_names:
                continue
            if any(c_name.lower() == cn.lower() for cn in core_names):
                continue
            related = comp.get("related_to_core")
            if related not in core_names and len(core_names) == 1:
                comp["related_to_core"] = core_names[0]
            valid_components.append(comp)
        
        # Filter baselines
        valid_baselines = []
        graph_data = baseline_res.get('result', {})
        for item in graph_data.get("baselines", []):
            name = item.get("name", "").strip()
            if not name or name.lower() in self_names:
                continue
            related = item.get("related_to_core")
            if related not in core_names and len(core_names) >= 1:
                item["related_to_core"] = core_names[0]
            if "keywords" in item:
                item["keywords"] = [k for k in item["keywords"] if k.lower() not in METRIC_BLACKLIST]
            valid_baselines.append(item)
        
        # Filter datasets
        valid_datasets = []
        for ds in graph_data.get("datasets", []):
            name = ds.get("name", "").strip()
            if not name:
                continue
            related = ds.get("related_to_core")
            if related not in core_names and len(core_names) >= 1:
                ds["related_to_core"] = core_names[0]
            valid_datasets.append(ds)
        
        final_output = {
            "node_id": node_id,
            "source_venue": info.get("source_venue", "Unknown"),
            "pub_year": info.get("pub_year", "2024"),
            "metadata": {
                "title": official_title,
                "domain": meta_info.get("domain", "Others"),
                "paper_type": meta_info.get("paper_type", "Others"),
                "structured_summary": meta_info.get("structured_summary", {}),
                "code_url": meta_info.get("code_url")
            },
            "ideation_resource": {
                "problems": result.get("problems", []),
                "core_contributions": core_contributions,
                "core_relations": result.get("core_relations", []),
                "components": valid_components,
                "innovations": result.get("innovations", []),
                "limitations": result.get("limitations", []),
                "future_work": result.get("future_work", [])
            },
            "graph_data": {
                "baselines": valid_baselines,
                "datasets": valid_datasets
            }
        }
        aggregated.append(final_output)
    
    return aggregated


# ================= MAIN PIPELINE FUNCTION =================

def extract_paper_info_batch(
    papers: Dict[str, str],
    chat_agent,
    source_info: Dict[str, str] = None,
    temperature: float = 0.05,
    max_retry: int = 3,
) -> List[dict]:
    """
    Extract paper information using ChatAgent with retry.
    
    Args:
        papers: Dict mapping paper_id -> markdown_text
        chat_agent: ChatAgent instance with batch_remote_chat_with_retry method
        source_info: Optional dict mapping paper_id -> {source_venue, pub_year}
        temperature: LLM temperature
        max_retry: Max retry attempts per prompt
        
    Returns:
        List of extraction results
    """
    if source_info is None:
        source_info = {}
    
    paper_ids = list(papers.keys())
    markdowns = list(papers.values())
    
    # ===== Step 1: Main Extraction =====
    main_prompts = []
    main_metadata = []
    
    for paper_id, markdown in zip(paper_ids, markdowns):
        # Extract title from markdown
        title = "Unknown Title"
        for line in markdown.split('\n')[:20]:
            match = re.match(r'^#\s+(.+)$', line.strip())
            if match:
                title = clean_title_latex(match.group(1))
                break
        
        regex_candidates = extract_regex_candidates(markdown)
        system, user = build_main_extraction_prompt(markdown, title, regex_candidates)
        
        # Combine system + user for the prompt
        full_prompt = f"{system}\n\n{user}"
        main_prompts.append(full_prompt)
        main_metadata.append({
            'paper_id': paper_id,
            'title': title,
            'regex_candidates': regex_candidates
        })
    
    # Call with retry and validation - info_dict contains paper_id for later aggregation
    main_responses = chat_agent.batch_remote_chat_with_retry(
        main_prompts,
        validate_fn=validate_and_parse_main,
        max_retry=max_retry,
        desc="Main extraction",
        temperature=temperature,
        info_dict={'metadata': main_metadata}  # Pass metadata for aggregation
    )
    
    # main_responses are already parsed results from validate_and_parse_main
    main_results = main_responses
    
    # Build core names map for baseline extraction
    core_names_map = {}
    for res in main_results:
        if res:
            core_names_map[res['node_id']] = res.get('core_names', [])
    
    # ===== Step 2: Baseline Extraction =====
    baseline_prompts = []
    baseline_metadata = []
    
    for paper_id, markdown in zip(paper_ids, markdowns):
        core_names = core_names_map.get(paper_id, [])
        regex_candidates = extract_regex_candidates(markdown)
        system, user = build_baseline_extraction_prompt(markdown, core_names, regex_candidates)
        
        full_prompt = f"{system}\n\n{user}"
        baseline_prompts.append(full_prompt)
        baseline_metadata.append({
            'paper_id': paper_id,
            'core_names': core_names
        })
    
    # Call with retry and validation
    baseline_responses = chat_agent.batch_remote_chat_with_retry(
        baseline_prompts,
        validate_fn=validate_and_parse_baseline,
        max_retry=max_retry,
        desc="Baseline extraction",
        temperature=temperature,
        info_dict={'metadata': baseline_metadata}
    )
    
    # baseline_responses are already parsed results from validate_and_parse_baseline
    baseline_results = baseline_responses
    
    # ===== Step 3: Aggregate =====
    return aggregate_results(main_results, baseline_results, source_info)


def format_extraction_result(result: dict) -> str:
    """Format extraction result into readable keynote string."""
    if not result:
        return "No details found."
    
    meta = result.get('metadata', {})
    structured_summary = meta.get('structured_summary', {})
    
    formatted = f"Paper Title: {meta.get('title', 'N/A')}\n"
    formatted += f"Paper Type: {meta.get('paper_type', 'N/A')}\n"
    formatted += f"Domain: {meta.get('domain', 'N/A')}\n"
    formatted += f"Summary (Background): {structured_summary.get('background', 'N/A')}\n"
    formatted += f"Summary (Method): {structured_summary.get('method', 'N/A')}\n"
    formatted += f"Summary (Result): {structured_summary.get('result', 'N/A')}\n"
    
    ideation = result.get('ideation_resource', {})
    cores = ideation.get('core_contributions', [])
    if cores:
        formatted += f"\nCore Contributions:\n"
        for core in cores:
            formatted += f"  - {core.get('name', 'N/A')}: {core.get('summary', 'N/A')}\n"
            formatted += f"    Insight: {core.get('insight', 'N/A')}\n"
    
    all_keywords = []
    for core in cores:
        all_keywords.extend(core.get('keywords', []))
    if all_keywords:
        formatted += f"Keywords: {', '.join(all_keywords[:10])}\n"
    
    return formatted


# ================= WRAPPER FUNCTIONS FOR COMPATIBILITY =================

def build_main_extraction_prompts(papers: Dict[str, str]) -> List[tuple]:
    """
    Build main extraction prompts for multiple papers.
    Returns list of (full_prompt, paper_id, metadata_dict)
    """
    results = []
    for paper_id, markdown in papers.items():
        # Extract title from markdown
        title = "Unknown Title"
        for line in markdown.split('\n')[:20]:
            match = re.match(r'^#\s+(.+)$', line.strip())
            if match:
                title = clean_title_latex(match.group(1))
                break
        
        regex_candidates = extract_regex_candidates(markdown)
        system, user = build_main_extraction_prompt(markdown, title, regex_candidates)
        full_prompt = f"{system}\n\n{user}"
        
        results.append((full_prompt, paper_id, {'title': title, 'regex_candidates': regex_candidates}))
    
    return results


def build_baseline_extraction_prompts(papers: Dict[str, str], core_names_map: Dict[str, List[str]]) -> List[tuple]:
    """
    Build baseline extraction prompts for multiple papers.
    Returns list of (full_prompt, paper_id, metadata_dict)
    """
    results = []
    for paper_id, markdown in papers.items():
        core_names = core_names_map.get(paper_id, [])
        regex_candidates = extract_regex_candidates(markdown)
        system, user = build_baseline_extraction_prompt(markdown, core_names, regex_candidates)
        full_prompt = f"{system}\n\n{user}"
        
        results.append((full_prompt, paper_id, {'core_names': core_names}))
    
    return results


def aggregate_extraction_results(
    main_results: List[dict],
    baseline_results: List[dict],
    source_info: Dict[str, dict]
) -> List[dict]:
    """Wrapper for aggregate_results for compatibility."""
    return aggregate_results(main_results, baseline_results, source_info)
