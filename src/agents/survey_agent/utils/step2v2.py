import os
import json
import re
import time
import requests
import random
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIGURATION =================

API_KEY = "sk-8b3f9c6d7a1e4f2c9d5b6a7e3c1f8d2a"
API_URL = "http://122.193.22.114:8889/v1/chat/completions"
MODEL_NAME = "glm-5"

BASE_DATA_DIR = "/public/home/sjtu_huzijian/mreg/data"
STEP1_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "step1v2")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "step2v2_glm") 

MAX_WORKERS = 8       
RETRY_DELAY = 2

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

# ================= UTILS (Unchanged) =================

def safe_read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def clean_json_text(text):
    """
    Clean LLM output to valid JSON.
    Handles markdown blocks and invalid escape sequences.
    """
    # Remove markdown
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = text.strip()
     
    def fix_escapes(text):
        result = []
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                next_char = text[i + 1]
                # Valid JSON escape chars: " \ / b f n r t u
                if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                    result.append(text[i:i+2])
                    i += 2
                else:
                    # Invalid escape, add extra backslash
                    result.append('\\\\')
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)
    
    return fix_escapes(text)

def extract_year_from_path(file_path):
    try:
        parent_dir = os.path.basename(os.path.dirname(file_path))
        match = re.search(r'(20[0-2][0-9]|199[0-9])', parent_dir)
        if match:
            return match.group(1)
        match_file = re.search(r'(20[0-2][0-9]|199[0-9])', os.path.basename(file_path))
        if match_file:
            return match_file.group(1)
    except:
        pass
    return "2024"

def clean_title_latex(title):
    if not title: return "Unknown Title"
    t = re.sub(r'^\d+(\.\d+)*\s+', '', title) 
    t = re.sub(r'[$\\{}"]', '', t) 
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_title_robust(data):
    raw_title = "Unknown Title"
    if data.get("title") and len(data.get("title")) > 5:
        raw_title = data.get("title")
    else:
        struct = data.get("structure") or data.get("structured_body") or []
        if struct and len(struct) > 0:
            candidate = struct[0].get("title", "")
            if candidate and len(candidate) < 300:
                raw_title = candidate
    return clean_title_latex(raw_title)

def clean_html_table(html_str):
    if not html_str: return ""
    cleaner = re.compile('<.*?>')
    text = re.sub(cleaner, ' | ', html_str)
    text = re.sub(r'\s+\|\s+', ' | ', text)
    text = re.sub(r'\|\s*\|', '|', text)
    return text.strip(' |')

def get_text_content(content_item, next_item=None):
    c_type = content_item.get("type", "text")
    res = []
    
    if c_type == "text":
        data = content_item.get("data") or content_item.get("value") or ""
        res.append(str(data))
        
    elif c_type == "table":
        captions = content_item.get("table_caption", [])
        if isinstance(captions, list):
            captions = " ".join(captions)
        if captions:
            res.append(f"\n[TABLE CAPTION]: {captions}")
        
        t_body = content_item.get("table_body", "")
        if t_body:
            res.append(f"[TABLE DATA]: {clean_html_table(t_body)}\n")
        else:
            raw_data = content_item.get("data") or content_item.get("value")
            if isinstance(raw_data, list):
                row_strs = []
                for r in raw_data:
                    if isinstance(r, list):
                        row_strs.append(" | ".join([str(x).replace("\n", " ") for x in r]))
                    else:
                        row_strs.append(str(r))
                res.append(f"[TABLE DATA]: {' '.join(row_strs)}\n")

    elif c_type == "image":
        caption = content_item.get("caption") or ""
        if isinstance(caption, list):
            caption = " ".join(caption)
        
        if not caption and next_item and next_item.get("type") == "text":
            text_preview = next_item.get("data", "")[:30].lower()
            if "figure" in text_preview or "fig." in text_preview:
                caption = next_item.get("data", "") 
        
        if caption:
            res.append(f"\n[FIGURE CAPTION]: {caption}\n")
            
    return "\n".join(res)

def get_full_text_linear(structure_list):
    text_list = []
    for sec in structure_list:
        title = sec.get("title", "")
        text_list.append(f"\n### Section: {title}\n")
        
        content_items = sec.get("content", [])
        for i, item in enumerate(content_items):
            next_item = content_items[i+1] if i + 1 < len(content_items) else None
            text_list.append(get_text_content(item, next_item))
            
    return "\n".join(text_list)

def mark_entities_in_text(full_text, entity_list, marker_prefix):
    marked_text = full_text
    
    for entity in entity_list:
        if not entity or len(entity) < 3:
            continue
        
        escaped = re.escape(entity)
        pattern = r'\b' + escaped + r'\b'
        replacement = f"[{marker_prefix}:{entity}]"
        marked_text = re.sub(pattern, replacement, marked_text, count=1, flags=re.IGNORECASE)
    
    return marked_text

def extract_regex_candidates(full_text):
    candidates = {
        "methods": set(),
        "baselines": set()
    }
    
    method_pattern = re.compile(
        r"(?:proposed|our|new|introduce|present)\s+(?:method|framework|algorithm|network|model)?\s*([A-Z][A-Za-z0-9\-]{2,15})", 
        re.MULTILINE
    )
    
    baseline_pattern = re.compile(
        r"(?:compare(?:s|d)?\s+(?:with|to|against)|outperform(?:s|ed)|surpass(?:es|ed)|superior\s+to|better\s+than|following|adopted\s+from|baseline|built\s+upon|extends?|evaluates?|analyzes?|prior\s+work|existing\s+method)\s+([A-Z][A-Za-z0-9\-]{2,15})", 
        re.IGNORECASE | re.MULTILINE
    )
    
    found_m = method_pattern.findall(full_text)
    found_b = baseline_pattern.findall(full_text)
    
    for m in found_m:
        if len(m) > 2 and m.lower() not in METRIC_BLACKLIST:
            candidates["methods"].add(m)
    for b in found_b:
        if len(b) > 2 and b.lower() not in METRIC_BLACKLIST:
            candidates["baselines"].add(b)
            
    return {
        "methods": list(candidates["methods"]),
        "baselines": list(candidates["baselines"])
    }

def call_llm(system_prompt, user_prompt, max_tokens=16000):
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.05, 
        "stream": False,
        "max_tokens": max_tokens, 
        "response_format": {"type": "json_object"}
    }
    
    # ISSUE 4 FIX: Add detailed error logging
    for attempt in range(3):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=300)
            print(f"API Response Preview: {resp.text[:500]}")
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                parsed = json.loads(clean_json_text(raw))
                return parsed
            else:
                # Log HTTP errors
                print(f"[LLM Error] HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(RETRY_DELAY)
        except json.JSONDecodeError as e:
            # Log JSON parsing errors
            print(f"[LLM Error] JSON Decode Failed (attempt {attempt+1}): {str(e)[:200]}")
            time.sleep(RETRY_DELAY)
        except requests.exceptions.Timeout:
            print(f"[LLM Error] Request Timeout (attempt {attempt+1})")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            # Log other exceptions
            print(f"[LLM Error] Unexpected error (attempt {attempt+1}): {str(e)[:200]}")
            time.sleep(RETRY_DELAY)
    
    return None

def verify_grounding(entity_list, full_text_lower, self_names=[]):
    valid = []
    dropped = []
    self_tokens = set()
    for n in self_names:
        if n: self_tokens.add(re.sub(r'[^a-z0-9]', '', n.lower()))

    if not entity_list: return valid, dropped

    for entity in entity_list:
        if not isinstance(entity, str): continue
        e_clean = entity.strip()
        if len(e_clean) < 2: continue
        
        if re.match(r'^\[?\d+(,\s*\d+)*\]?$', e_clean):
            dropped.append(f"{e_clean} (Citation)")
            continue
            
        e_norm = re.sub(r'[^a-z0-9]', '', e_clean.lower())
        
        if e_clean.lower() in METRIC_BLACKLIST:
            dropped.append(f"{e_clean} (Metric)")
            continue

        if e_norm in self_tokens or e_clean.lower() in ["ours", "proposed", "method", "model", "framework", "approach", "this work"]:
            dropped.append(f"{e_clean} (Self/Generic)")
            continue

        if e_clean.lower() in full_text_lower:
            valid.append(e_clean)
        elif len(e_norm) > 4 and e_norm in re.sub(r'[^a-z0-9]', '', full_text_lower):
            valid.append(e_clean)
        else:
            dropped.append(f"{e_clean} (Not Found)")
            
    return list(set(valid)), list(set(dropped))

# ================= PROMPTS (UPDATED) =================

def task_main_extraction(full_text, original_title, regex_candidates):
    domain_str = ", ".join(ARXIV_DOMAINS)
    type_str = ", ".join(PAPER_TYPES)
    
    marked_text = mark_entities_in_text(full_text, regex_candidates['methods'], "HINT_METHOD")
    
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
    
    system = """You are a Senior Research Analyst extracting structured metadata from academic papers.

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

    # ISSUE 2 FIX: Unified attribute order in schema
    prompt = f"""
{core_type_reference}

FULL PAPER TEXT (Some entities marked with [HINT_METHOD:...] as suggestions only):
{marked_text}

Task: Extract structured information for paper titled {original_title}

Output JSON Schema (FOLLOW THIS EXACT ATTRIBUTE ORDER):
{{
  "metadata": {{
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
"""
    
    return call_llm(system, prompt, max_tokens=16000)

def task_baseline_extraction(full_text, core_names, regex_candidates):
    marked_text = mark_entities_in_text(full_text, regex_candidates['baselines'], "HINT_BASELINE")
    
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
}
"""

    # ISSUE 1 FIX: metrics can be null
    # ISSUE 2 FIX: Unified attribute order
    prompt = f"""
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
"""
    
    return call_llm(system, prompt, max_tokens=16000)

# ================= WORKER (UPDATED) =================

def process_paper(args):
    json_path, out_dir = args
    data = safe_read_json(json_path)
    if not data: 
        return f"[Err] Read: {json_path}"
    
    paper_id = data.get("paper_id")
    target_file = os.path.join(out_dir, f"{paper_id}.json")
    if os.path.exists(target_file): 
        return f"[Skip] {paper_id}"
    
    struct_body = data.get("structure") or data.get("structured_body") or []
    if not struct_body: 
        return f"[Err] No Body: {paper_id}"

    full_text = get_full_text_linear(struct_body)
    full_text_lower = full_text.lower()
    
    pub_year = extract_year_from_path(json_path)
    official_title = extract_title_robust(data)

    regex_candidates = extract_regex_candidates(full_text)

    # Step 1: Main Extraction
    # ISSUE 4 FIX: Better error handling with detailed logging
    main_res = task_main_extraction(full_text, official_title, regex_candidates)
    if not main_res:
        # Log detailed error for debugging
        with open(os.path.join(out_dir, f"_error_{paper_id}.log"), 'w', encoding='utf-8') as log_f:
            log_f.write(f"Paper ID: {paper_id}\n")
            log_f.write(f"Title: {official_title}\n")
            log_f.write(f"Full text length: {len(full_text)}\n")
            log_f.write(f"Regex candidates: {json.dumps(regex_candidates, indent=2)}\n")
            log_f.write(f"Error: Main extraction returned None\n")
        return f"[Err] Main Extraction failed for {paper_id} (see log)"

    core_contributions = main_res.get("core_contributions", [])
    core_names = [c.get("name") for c in core_contributions if c.get("name")]
    if not core_names: 
        core_names = [official_title]
    
    # Step 2: Baseline/Dataset Extraction
    graph_res = task_baseline_extraction(full_text, core_names, regex_candidates)
    if not graph_res:
        with open(os.path.join(out_dir, f"_error_{paper_id}.log"), 'w', encoding='utf-8') as log_f:
            log_f.write(f"Paper ID: {paper_id}\n")
            log_f.write(f"Core names: {json.dumps(core_names)}\n")
            log_f.write(f"Error: Baseline extraction returned None\n")
        return f"[Err] Baseline Extraction failed for {paper_id} (see log)"

    # Build self-reference set
    self_names = {official_title.lower(), "ours", "proposed", "method", "model", "framework", "this work"}
    for c in core_contributions:
        if c.get("name"): 
            self_names.add(c.get("name").lower())
        if c.get("acronym"): 
            self_names.add(c.get("acronym").lower())

    # Component validation
    valid_components = []
    for comp in main_res.get("components", []):
        c_name = comp.get("name", "").strip()
        if not c_name or c_name.lower() in self_names:
            continue
        
        if any(c_name.lower() == cn.lower() for cn in core_names):
            continue
        
        related = comp.get("related_to_core")
        if related not in core_names and len(core_names) == 1:
            comp["related_to_core"] = core_names[0]
        
        insight = comp.get("insight", "")
        if not any(kw in insight.lower() for kw in ["defect", "limitation", "drawback", "weakness", "issue", "problem", "challenge"]):
            comp["_warning"] = "Insight may be missing potential defects discussion"
        
        valid_components.append(comp)
    
    main_res['components'] = valid_components

    # Baseline filtering
    valid_baselines = []
    for item in graph_res.get("baselines", []):
        name = item.get("name", "").strip()
        if not name or name.lower() in self_names:
            continue
        
        related = item.get("related_to_core")
        if related not in core_names:
            if len(core_names) == 1:
                item["related_to_core"] = core_names[0]
            else:
                item["related_to_core"] = core_names[0]
        
        v_names, _ = verify_grounding([name], full_text_lower, list(self_names))
        if not v_names:
            if len(name) > 3 and name.lower() in full_text_lower:
                pass
            else:
                continue
        
        if "keywords" in item:
            item["keywords"] = [k for k in item["keywords"] if k.lower() not in METRIC_BLACKLIST]
        
        # ISSUE 1 FIX: Handle metrics being null
        if "metrics" in item:
            if item["metrics"] is None:
                pass  # Keep as null
            elif not isinstance(item["metrics"], list):
                item["metrics"] = [item["metrics"]] if item["metrics"] else None
        else:
            item["metrics"] = None
        
        valid_baselines.append(item)

    # Dataset filtering
    valid_datasets = []
    for ds in graph_res.get("datasets", []):
        name = ds.get("name", "").strip()
        if not name:
            continue
        
        v_names, _ = verify_grounding([name], full_text_lower)
        if not v_names and len(name) > 3 and name.lower() not in full_text_lower:
            continue
        
        related = ds.get("related_to_core")
        if related not in core_names and len(core_names) >= 1:
            ds["related_to_core"] = core_names[0]
        
        # ISSUE 1 FIX: Handle metrics being null
        if "metrics" in ds:
            if ds["metrics"] is None:
                pass
            elif not isinstance(ds["metrics"], list):
                ds["metrics"] = [ds["metrics"]] if ds["metrics"] else None
        else:
            ds["metrics"] = None
        
        valid_datasets.append(ds)

    # Assemble output
    meta_info = main_res.get("metadata", {})
    
    final_output = {
        "paper_id": paper_id,
        "source_venue": data.get("source", "Unknown"),
        "pub_year": pub_year,
        "metadata": {
            "title": official_title,
            "domain": meta_info.get("domain", "Others"),
            "paper_type": meta_info.get("paper_type", "Others"),
            "structured_summary": meta_info.get("structured_summary", {}),
            "code_url": meta_info.get("code_url")
        },
        "ideation_resource": {
            "problems": main_res.get("problems", []),
            "core_contributions": core_contributions,
            "core_relations": main_res.get("core_relations", []),
            "components": valid_components,
            "innovations": main_res.get("innovations", []),
            "limitations": main_res.get("limitations", []),
            "future_work": main_res.get("future_work", [])
        },
        "graph_data": {
            "baselines": valid_baselines,
            "datasets": valid_datasets
        }
    }
    
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    return f"[OK] {paper_id} | Cores: {len(core_contributions)} | Baselines: {len(valid_baselines)} | Datasets: {len(valid_datasets)} | Components: {len(valid_components)}"

# ================= MAIN (Unchanged) =================

def main():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    tasks = []
    if os.path.exists(STEP1_OUTPUT_DIR):
        for venue_folder in os.listdir(STEP1_OUTPUT_DIR):
            venue_path = os.path.join(STEP1_OUTPUT_DIR, venue_folder)
            if not os.path.isdir(venue_path): 
                continue
            
            out_venue = os.path.join(OUTPUT_DIR, venue_folder)
            os.makedirs(out_venue, exist_ok=True)
            
            for f in os.listdir(venue_path):
                if f.endswith(".json"):
                    tasks.append((os.path.join(venue_path, f), out_venue))
    else:
        print(f"Directory not found: {STEP1_OUTPUT_DIR}")
        return
    total_found = len(tasks)
    SAMPLE_SIZE = 1000
    
    if total_found > SAMPLE_SIZE:
        print(f"Found {total_found} papers. Randomly sampling {SAMPLE_SIZE} for this run...")
        random.shuffle(tasks)       
        tasks = tasks[:SAMPLE_SIZE]
    else:
        print(f"Found {total_found} papers (less than limit {SAMPLE_SIZE}), processing all.")
    print(f"Batch Processing {len(tasks)} papers [HIGH RECALL MODE - v2]...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(process_paper, tasks):
            print(result)

if __name__ == "__main__":
    main()
