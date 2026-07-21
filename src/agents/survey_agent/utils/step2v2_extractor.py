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
    
    core_type_reference = """
Core Contribution Type Reference (Choose ONE single word):
1. Methodological: Algorithm, Architecture, Framework, TrainingMethod, InferenceMethod, LossFunction, OptimizationTechnique
2. System/Tool: System, Infrastructure, Tool, Library, Platform
3. Data/Evaluation: Dataset, Benchmark, DataAugmentation, AnnotationScheme, EvaluationMetric, EvaluationProtocol
4. Theoretical: TheoreticalFramework, TheoreticalProof, MathematicalFormulation, ComplexityAnalysis
5. Analytical: EmpiricalAnalysis, ComparativeAnalysis, AblationAnalysis, ErrorAnalysis, Survey, Taxonomy
6. Application: Application, CaseStudy, DomainAdaptation
7. Others: Hybrid, Others

OUTPUT FORMAT: Single word only (e.g., Algorithm, NOT Type: Algorithm)
"""
    
    system = f"""You are a Senior Research Analyst extracting structured metadata from academic papers.

CRITICAL GROUNDING RULES:
1. Insight: Must provide INDEPENDENT ANALYSIS beyond summary. Describe prerequisites, trade-offs, design choices, or potential defects. DO NOT copy summary or quote.
2. Quote: Must be VERBATIM text from the original paper. Use ... for exact excerpts. DO NOT paraphrase or fabricate.
3. Summary: Concise factual description in natural sentence form.
4. Keywords: Concept words used for COARSE-GRAINED RETRIEVAL and filtering. Choose representative terms that capture the essence for search/matching.

PRIORITY: RECALL - Extract ALL relevant entities (cores, components, problems, innovations, limitations, future work). Do not omit items due to uncertainty.

Core vs Component Granularity Rules:
- Core Contribution: The LARGEST, TOP-LEVEL contribution that is NOT contained by any other entity. This is the root method/system/dataset proposed.
  Example: Swin Transformer (the overall architecture)
  Characteristics: Has unique name, explicitly claimed as main contribution, can stand alone
  
- Component: ALL sub-elements, modules, techniques, or building blocks that are PART OF a Core Contribution.
  Example: Shifted Window Mechanism, Patch Merging Layer (parts of Swin Transformer)
  Characteristics: Cannot exist independently without the Core, implements specific functionality

If unsure: Ask Can this exist independently, or is it always part of something bigger?
- Independent -> Core
- Part of bigger system -> Component

Text Marking System:
You will see markers like [HINT_METHOD:MethodName] in the text. These are SUGGESTIONS from pattern matching.
- They may contain false positives or miss important entities
- Use them to guide attention but read the FULL paper independently
- Extract ALL relevant entities regardless of whether they have markers

Few-Shot Examples:

Example 1: Clear Core-Component Hierarchy
Paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

Core Contribution:
  name: Swin Transformer
  acronym: SwinT
  type: Architecture
  keywords: [hierarchical transformer, shifted windows, vision backbone]
  summary: Hierarchical vision transformer using shifted windows for efficient attention
  insight: Trade-off: Achieves O(n) complexity via window-based computation but sacrifices some long-range modeling capability compared to full self-attention
  quote: We propose Swin Transformer, a hierarchical Transformer whose representation is computed with shifted windows

Components (ALL parts of Swin Transformer):
  1. name: Shifted Window Mechanism
     acronym: null
     related_to_core: Swin Transformer
     keywords: [window attention, cross-window connection]
     summary: Window-based self-attention with shifting for cross-window connections
     insight: Design choice: 7x7 window balances efficiency and receptive field. Potential defect: Window boundaries may cause artifacts for objects spanning multiple windows
     quote: The shifted windowing scheme bridges the windows of the preceding layer
  
  2. name: Patch Merging Layer
     acronym: null
     related_to_core: Swin Transformer
     keywords: [downsampling, hierarchical features]
     summary: Downsampling layer that concatenates neighboring patch features
     insight: Design choice: 2x2 merging reduces resolution while doubling channels. Potential defect: Loses fine-grained spatial information needed for dense prediction
     quote: The patch merging layer concatenates the features of each group of 2x2 neighboring patches

Example 2: Multiple Cores
Paper proposes BOTH a new method AND a new dataset:

Core 1:
  name: RainNet
  acronym: null
  type: Algorithm
  keywords: [rain removal, weather robustness]
  
Core 2:
  name: RainCityscapes
  acronym: null
  type: Dataset
  keywords: [rainy driving, synthetic weather]

Component of Core 1:
  name: Weather Augmentation Module
  acronym: WAM
  related_to_core: RainNet
  keywords: [data augmentation, rain synthesis]
"""

    user = f"""
{core_type_reference}

FULL PAPER TEXT (Some entities marked with [HINT_METHOD:...] as suggestions only):
{marked_text}

Task: Extract structured information for paper titled "{title}"

Output JSON Schema (FOLLOW THIS EXACT ATTRIBUTE ORDER):
{{
  "metadata": {{
    "title": "str (paper title, use EXACT title provided in the task)",
    "domain": "str (from: {domain_str})",
    "paper_type": "str (from: {type_str})",
    "structured_summary": {{
      "background": "str",
      "method": "str",
      "result": "str"
    }},
    "code_url": "str or null"
  }},
  "problems": [
    {{
      "keywords": ["str"],
      "summary": "str",
      "insight": "str (MUST be independent analysis)",
      "quote": "str (MUST be verbatim)",
      "related_to_core": "str"
    }}
  ],
  "core_contributions": [
    {{
      "name": "str (TOP-LEVEL entity)",
      "acronym": "str or null",
      "type": "str (SINGLE WORD)",
      "keywords": ["str"],
      "summary": "str",
      "insight": "str (prerequisites, trade-offs, design rationale)",
      "quote": "str"
    }}
  ],
  "core_relations": [
    {{
      "source": "str (core name)",
      "target": "str (core name)",
      "keywords": ["str"],
      "summary": "str (natural sentence)",
      "insight": "str (why relationship exists)",
      "quote": "str"
    }}
  ],
  "components": [
    {{
      "name": "str (MUST NOT be same as core name)",
      "acronym": "str or null",
      "related_to_core": "str (MUST link to parent Core)",
      "keywords": ["str"],
      "summary": "str",
      "insight": "str (MUST include design choices AND potential defects)",
      "quote": "str"
    }}
  ],
  "innovations": [
    {{
      "keywords": ["str"],
      "summary": "str",
      "insight": "str",
      "quote": "str",
      "related_to_core": "str"
    }}
  ],
  "limitations": [
    {{
      "keywords": ["str"],
      "summary": "str",
      "insight": "str",
      "quote": "str",
      "related_to_core": "str"
    }}
  ],
  "future_work": [
    {{
      "keywords": ["str"],
      "summary": "str",
      "insight": "str",
      "quote": "str",
      "related_to_core": "str"
    }}
  ]
}}

CRITICAL REMINDERS:
1. PRIORITY: RECALL - Extract ALL entities, even if uncertain
2. Core vs Component: Use hierarchy rules above
3. Type field: Single word only
4. Insight: Independent analysis, not copy of summary/quote
5. Quote: Verbatim text with ... markers
6. Components: MUST discuss potential defects
7. Keywords: Choose terms for RETRIEVAL and FILTERING purposes
8. Attribute order: MUST follow schema order exactly (keywords first, then summary, insight, quote)
9. Ignore [HINT_METHOD:...] markers in your output

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

CRITICAL: For Baselines and Datasets, the summary/insight/quote fields describe THE RELATIONSHIP/COMPARISON, NOT the entity itself.

PRIORITY: RECALL > PRECISION
Your PRIMARY GOAL is to extract ALL baselines and datasets mentioned, especially from:
- Experimental results (tables, figures)
- Related Work section
- Appendix (supplementary experiments)
- Theoretical comparisons (complexity analysis)

It is BETTER to include a questionable baseline than to miss a real one.

Text Marking System:
You will see markers like [HINT_BASELINE:MethodName] - these are SUGGESTIONS only.
Read Experiments, Related Work, and Appendix thoroughly to find ALL baselines.

Keywords Usage:
Keywords should capture key concepts for COARSE-GRAINED RETRIEVAL. Choose representative terms for filtering and search.

Example Baseline Entry:
{
  "name": "ResNet-50",
  "acronym": null,
  "related_to_core": "Swin Transformer",
  "keywords": ["residual network", "image classification"],
  "summary": "Our SwinT outperforms this baseline by 3.2 percent on ImageNet top-1 accuracy",
  "metrics": ["Top-1 Accuracy", "Top-5 Accuracy"],
  "insight": "Performance gap stems from ResNet lack of long-range modeling vs attention mechanisms",
  "quote": "Our method achieves 82.1 percent top-1 accuracy, surpassing ResNet-50 78.9 percent"
}

Example Dataset Entry:
{
  "name": "ImageNet-1K",
  "acronym": "IN-1K",
  "related_to_core": "Swin Transformer",
  "keywords": ["image classification", "large-scale benchmark"],
  "summary": "Primary benchmark used to evaluate classification performance",
  "metrics": ["Top-1 Accuracy", "Top-5 Accuracy"],
  "insight": "ImageNet diversity makes it the de facto standard, though class distribution may not reflect real-world scenarios",
  "quote": "We evaluate on ImageNet-1K ILSVRC2012 validation set"
}"""

    user = f"""
Core Methods Proposed in THIS Paper (DO NOT include as baselines): {json.dumps(core_names)}

FULL PAPER TEXT (Some baselines marked with [HINT_BASELINE:...] as suggestions):
Focus especially on:
- Experimental Results sections
- Related Work section
- Appendix / Supplementary Material
- Theoretical Comparisons

{marked_text}

Task: Extract ALL baselines and datasets with MAXIMUM RECALL.

PRIORITY: RECALL - When in doubt, INCLUDE the baseline. Missing a baseline is worse than false positive.

What is a Baseline (Expanded):
1. Competitors: Methods explicitly compared in result tables
2. Predecessors: Methods this work builds upon, improves, or fixes
3. Related Work: Methods discussed as alternatives or inspiration
4. Theoretical Comparisons: Methods compared in complexity analysis
5. Appendix Comparisons: Additional baselines in supplementary experiments
6. For Dataset Papers: Previous datasets compared OR models evaluated on new benchmark
7. For Analysis Papers: Systems/methods being analyzed
8. Ablation Variants: Simplified versions of proposed method (e.g., Ours w/o attention)

Naming Convention:
- Prioritize Method Name (e.g., SimCLR, Faster R-CNN, ImageNet-1K)
- Only use Author Name (e.g., He et al.) IF method has no specific name
- For ablations: Use descriptive name (e.g., SwinT w/o shifted windows)

CONSTRAINT: 
Baselines must ONLY contain PREVIOUS EXISTING ENTITIES or ABLATION VARIANTS.
NEVER include full Core Contributions from THIS paper ({core_names}).

Output JSON Schema (FOLLOW THIS EXACT ATTRIBUTE ORDER):
{{
  "baselines": [
    {{
      "name": "str",
      "acronym": "str or null",
      "related_to_core": "str (which core compared to)",
      "keywords": ["str"],
      "summary": "str (natural sentence about relationship)",
      "metrics": ["str"] or null,
      "insight": "str (WHY performance gap or relationship exists)",
      "quote": "str (evidence)"
    }}
  ],
  "datasets": [
    {{
      "name": "str",
      "acronym": "str or null",
      "related_to_core": "str",
      "keywords": ["str"],
      "summary": "str (how dataset is used)",
      "metrics": ["str"] or null,
      "insight": "str (why chosen, what reveals)",
      "quote": "str"
    }}
  ]
}}

CRITICAL REMINDERS:
1. PRIORITY: RECALL - Extract ALL from experiments, related work, appendix, theory
2. summary/insight/quote: Describe THE EDGE (relationship), not node (entity)
3. metrics: List of metric NAMES only (e.g., [Accuracy, F1]), OR null if not applicable
4. Attribute order: MUST follow schema (name, acronym, related_to_core, keywords, summary, metrics, insight, quote)
5. Keywords: Choose terms for RETRIEVAL purposes
6. Verify entities appear in paper (no hallucination)
7. When uncertain, INCLUDE IT
8. Ignore [HINT_BASELINE:...] markers in output

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
        info_dict: Optional dict with 'node_id' and 'title' (original title from markdown)
        
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
    if 'metadata' not in parsed:
        raise ValueError("Missing required key: metadata")
    
    # Extract core names for baseline extraction
    core_contributions = parsed.get("core_contributions", [])
    core_names = [c.get("name") for c in core_contributions if c.get("name")]
    
    # Extract original title from info_dict for fallback (info_dict contains the extracted title from markdown)
    original_title = None
    if info_dict:
        original_title = info_dict.get('title')
    
    result = {
        'node_id': info_dict.get('node_id') if info_dict else None,
        'result': parsed,
        'core_names': core_names,
        'original_title': original_title
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
    baseline_by_paper = {r['node_id']: r for r in (baseline_results or []) if r}
    
    aggregated = []
    for node_id in main_by_paper:
        main_res = main_by_paper[node_id]
        baseline_res = baseline_by_paper.get(node_id, {})
        info = source_info.get(node_id, {})
        
        result = main_res['result']
        core_contributions = result.get("core_contributions", [])
        core_names = main_res.get('core_names', [])
        
        meta_info = result.get("metadata", {})
        # Use LLM's title if available, otherwise use original_title from extraction
        official_title = meta_info.get('title', 'Unknown')
        if official_title == 'Unknown' or not official_title:
            # Fallback to original title extracted from markdown
            original_title = main_res.get('original_title')
            if original_title and original_title != 'Unknown Title':
                official_title = original_title
        
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
    
    # Add Components section
    components = ideation.get('components', [])
    if components:
        formatted += f"\nComponents:\n"
        for comp in components:
            related_to_core = comp.get('related_to_core', 'N/A')
            formatted += f"  - {comp.get('name', 'N/A')} (Part of: {related_to_core})\n"
            formatted += f"    Summary: {comp.get('summary', 'N/A')}\n"
            formatted += f"    Insight: {comp.get('insight', 'N/A')}\n"
    
    all_keywords = []
    for core in cores:
        all_keywords.extend(core.get('keywords', []))
    for comp in components:
        all_keywords.extend(comp.get('keywords', []))
    if all_keywords:
        formatted += f"\nKeywords: {', '.join(all_keywords[:15])}\n"
    
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
