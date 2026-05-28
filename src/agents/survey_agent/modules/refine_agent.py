"""
Agentic Survey Revisor: An agent-based approach to review and revise survey sections
using a central planner with multiple skills.

This module provides two classes:
- AgenticRevisor: For reviewing and revising individual sections
- AgenticSurveyRevisor: For reviewing and revising the entire survey

Both inherit from BaseAgenticRevisor which contains common functionality.
"""

import json
import os
import random
import tiktoken
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
from utils.utils import extract_json
from modules.pe import SECTION_REVIEW, SECTION_REVISE, CODE_REPORT_PROMPT, CODE_REPORT_PROMPT_FOR_SECTION_REVISER, CODE_REPORT_PROMPT_FOR_SECTION_REVIEWER


# ============================================================================
# Utility Functions (standalone helpers)
# ============================================================================

def truncate_text(text: str, max_length: int, suffix: str = "(truncated)...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if not text:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# Prompts for Section Revision (AgenticRevisor)
# ============================================================================

AGENT_OPERATE_PROMPT = """
You are an intelligent agent that reviews and revises academic survey sections.

[Survey Information]
- Topic: {topic}
- Survey Title: {survey_title}

[Current Section Being Reviewed]
- Section Index: {section_index}/{total_sections}
- Section Title: {section_title}
- Section Description: {section_description}

[Current Section Text]
{current_section_text}

[Previous Section Text]
{previous_section_text}

[Next Section Text]
{next_section_text}

[Section Outline]
{section_outline}

[Memory - Operation History and Results]
{memory}

[Current Context Summary]
{context_summary}

---

Your task is to plan a sequence of operations to review and revise the current section.

Available operations (skills):
1. "review": Analyze the section and provide scores and suggestions for improvement in three dimensions:
   - Readability (clarity, logical flow, formatting)
   - Depth (comprehensiveness, insightfulness, citation coverage)
   - Framework (alignment with section outline, structural coherence)
   
2. "revise": Apply specific revision suggestions to improve the section text.
   - Input: A single suggestion from the review
   - Output: Modified section text with the suggestion applied

3. "get_pseudocode": Retrieve pseudocode for a specific repository
   - Input: repo_name
   - Context to include: what specific aspect to focus on

4. "get_code_report": Include code report content in context
   - Input: specific code report or reference to include

5. "get_full_survey": Include the complete current survey content in context
   - Useful for cross-section coherence checking

6. "finish": Complete the revision for this section when satisfactory
   - CRITICAL: You can only call "finish" when ALL THREE review scores are >= 8:
     * Readability >= 8
     * Depth >= 8
     * Framework >= 8
   - If any score is below 8, you must continue to revise or gather more context first

[WORKFLOW - CRITICAL - FOLLOW THIS ORDER]
You MUST follow this workflow strictly:
1. First, if you have NOT yet reviewed the section, call "review" to analyze and get scores + suggestions
2. If you have already reviewed, you MUST use the EXISTING suggestions from memory to call "revise" - DO NOT call "review" again!
3. When all current suggestions are met, you may call "review" again if needed
4. Call "finish" ONLY when ALL THREE scores are >= 8, or when you have addressed all suggestions

[Priority Guidelines]
- Focus on the most impactful suggestions first
- Focus on the Memory content to learn the actions you have taken, and plan future actions with a reasonable workflow
- Prioritize suggestions that address the LOWEST-scoring dimensions
- Use external knowledge (keynotes, pseudocode) when the section needs more depth
- Ensure logical flow between sections using previous/next section context
---

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "input": "...", "reason": "..."}},
        {{"operation": "...", "input": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "revise", "input" should contain the EXACT suggestion text to apply
- For "get_pseudocode", "input" should be the EXACT repo_name
- For "get_keynote", "input" should be the EXACT paper_title
- For "get_full_survey", no input needed
- For "finish", no input needed
- Include 1 to {max_plan_steps} operations per plan to complete the revision efficiently

Generate JSON directly without any other things.
"""

AGENT_OPERATE_PROMPT_NO_CODE = """
You are an intelligent agent that reviews and revises academic survey sections.

[Survey Information]
- Topic: {topic}
- Survey Title: {survey_title}

[Current Section Being Reviewed]
- Section Index: {section_index}/{total_sections}
- Section Title: {section_title}
- Section Description: {section_description}

[Current Section Text]
{current_section_text}

[Previous Section Text]
{previous_section_text}

[Next Section Text]
{next_section_text}

[Section Outline]
{section_outline}

[Memory - Operation History and Results]
{memory}

[Current Context Summary]
{context_summary}

---

Your task is to plan a sequence of operations to review and revise the current section.

Available operations (skills):
1. "review": Analyze the section and provide scores and suggestions for improvement in three dimensions:
   - Readability (clarity, logical flow, formatting)
   - Depth (comprehensiveness, insightfulness, citation coverage)
   - Framework (alignment with section outline, structural coherence)
   
2. "revise": Apply specific revision suggestions to improve the section text.
   - Input: A single suggestion from the review
   - Output: Modified section text with the suggestion applied

3. "get_full_survey": Include the complete current survey content in context
   - Useful for cross-section coherence checking

4. "finish": Complete the revision for this section when satisfactory
   - CRITICAL: You can only call "finish" when ALL THREE review scores are >= 8:
     * Readability >= 8
     * Depth >= 8
     * Framework >= 8
   - If any score is below 8, you must continue to revise or gather more context first

[WORKFLOW - CRITICAL - FOLLOW THIS ORDER]
You MUST follow this workflow strictly:
1. First, if you have NOT yet reviewed the section, call "review" to analyze and get scores + suggestions
2. If you have already reviewed, you MUST use the EXISTING suggestions from memory to call "revise" - DO NOT call "review" again!
3. When all current suggestions are met, you can call "review" again.
4. Call "finish" only when ALL THREE scores are >= 8, or when you have addressed all suggestions

[Priority Guidelines]
- Focus on the most impactful suggestions first
- Prioritize suggestions that address the LOWEST-scoring dimensions
- Use external knowledge (keynotes, pseudocode) when the section needs more depth
- Ensure logical flow between sections using previous/next section context

---

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "input": "...", "reason": "..."}},
        {{"operation": "...", "input": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "revise", "input" should contain the EXACT suggestion text to apply
- For "get_full_survey", no input needed
- For "finish", no input needed
- Include 1 to {max_plan_steps} operations per plan to complete the revision efficiently

Generate JSON directly without any other things.
"""

# Section Review skill prompt
REVIEW_SKILL_PROMPT = """
You are an expert reviewer for an academic survey paper with deep analysis and insights concerning topic: {topic}. You are reviewing the section: {section_title}.

The draft section may contain inline paper citations in angle brackets (e.g., "<Attention is All You Need>").

### Task:
1. Read the Draft Text for the given section and perform a short, careful review focused on clarity, logical flow, technical accuracy, and depth for an academic survey.
2. Produce a clear, specific, and actionable **list of revision suggestion list**. 
3. Each suggestion should concisely explain *what* to change, *why* and *how* to implement it. When appropriate, point to the exact sentence or short excerpt from the draft to anchor the suggestion.
4. You will be provided with the previous and next sections for context. Use them to ensure logical flow and coherence across sections.
5. Only give suggestions on the current section. 
6. There are some basic requirements as follow. If the section does not meet any, provide relevant modification suggestions.
7. If the section is satisfactory, provide an empty suggestion list.
8. Suggest sorting by importance, with important ones coming first.
9. You are suggested to give no more than 5 suggestions

### Review Dimensions (Score 0-10 for each):
1. **Readability**: Clarity, logical flow, formatting, avoiding verbose phrases or repetitions
2. **Depth**: Comprehensiveness, insightfulness, citation coverage, novel analysis
3. **Framework**: Alignment with section outline, structural coherence, academic rigor

### Scoring Criteria (Apply these guidelines consistently):

**Readability (0-10):**
- **0-3**: Difficult to understand; poor logical flow; repetitive or verbose; lacks proper formatting
- **4-5**: Some clarity issues; occasional logical jumps; minor repetitions or overly long sentences
- **6-7**: Generally clear; reasonable flow; minor formatting issues; some sentences could be tightened
- **8-9**: Clear and well-structured; smooth logical flow; proper formatting; concise prose
- **10**: Exceptional clarity; publication-ready prose; elegant transitions between paragraphs

**Depth (0-10):**
- **0-3**: Superficial; mainly lists methods without analysis; few or irrelevant citations; no insights
- **4-5**: Basic coverage; limited analysis; citations present but not well-integrated; some generic statements
- **6-7**: Good coverage; some insightful analysis; citations support key points; identifies trends or patterns
- **8-9**: Comprehensive; deep analysis with novel perspectives; well-chosen citations; clear connections to field
- **10**: Outstanding depth; original insights that advance understanding; comprehensive and well-cited analysis

**Framework (0-10):**
- **0-3**: Misaligned with outline; incoherent structure; lacks academic rigor; terminology imprecise
- **4-5**: Partially follows outline; some structural issues; minor inconsistencies in terminology
- **6-7**: Follows outline reasonably; acceptable structure; consistent terminology; generally rigorous
- **8-9**: Well-aligned with outline; coherent structure; precise terminology; strong academic rigor
- **10**: Perfect alignment with outline; exemplary structure; rigorous and precise throughout

### Requirements:
1. You should only provide suggestions on content. DO NOT give any suggestions that involve changing the titles of the section and any subsections.
2. The section text should have around {section_least_words} words. Current section length: {current_section_length} words.
3. All citations in the section must be in correct format: <paper_title> (like <Attention is All You Need>).
4. The goal is to ultimately complete a survey section with depth, insights and can boost further development, rather than simply listing methods.
5. The section content should have deep insights, novelty and analysis under the field of the section. 
6. The section content should be clear and coherent. Avoid over verbose phrase or any repetitions.
7. The section content should be elegantly formatted and have good readability, avoid extremely long sentences or paragraphs.
8. The content of the section should be strictly consistent with the outline of the section provided below.
9. The section content should be logically coherent and academically styled with academic rigor and precise description.
10. The paragraph should contain sufficient citations to support the viewpoints and provide specific and in-depth analysis
11. You should only change the content of the current section.

### Input:
Previous Section:
{previous_section_text}

Next Section:
{next_section_text}

Current Section:
{section_text}

Section Outline:
{section_outline}

### Output format (exact JSON):
{{
    "scores": {{
        "readability": <score_1>,
        "depth": <score_2>,
        "framework" <score_3>: 
    }},
    "suggestions": [
        "suggestion 1 with specific details(string)",
        "suggestion 2 with specific details(string)",
        ...
    ]
}}

- the content under "suggestions" should be exact List of String
"""

# Section Revise skill prompt
REVISE_SKILL_PROMPT = """
You are a revise assistant. The section content of a survey concerning {topic} is provided below. The section name is: {section_title}.

You must propose at most ONE exact textual substitution per response according to the suggestion of reviewer.
If you think the document requires changes, choose one that you think is most important to address next, and output ONE JSON object (and nothing else).

Section Outline (You should not change):
{section_outline}

Original Section Text:
{section_text}

Citation Text:
{citations}

Guidance:
- Give one minimal but precise modification.
- Revise the paragraph to enhance its readability, logicality and depth.
- Your modifications **MUST** be consistent with the overall structure, logic and the scope (title) of the section.
- If the suggestion requires additional context from papers, incorporate that context while making the revision.
- You can only revise section content. Do NOT make any revision plan including changing subsection title(#### .....).
- The revise should meets the standards of a top-tier academic literature review

Reviewer Suggestion:
{reviewer_suggestion}

External Context:
{external_context}

**Revision Goal:**
Your revision target: produce a survey section that meets the quality standards of a top-tier academic conference survey, with rigorous logic, excellent readability, and sufficient depth.

**Output format:**
{{
    "action":"replace", 
    "originalText":"<the exact substring to replace>", 
    "newText":"<the replacement text>"
}}

The originalText field must match EXACTLY ONE substring in the document.
If you believe no edits are required, output exactly: {{"action":"done"}}.
"""


# ============================================================================
# Prompts for Whole Survey Revision (AgenticSurveyRevisor)
# ============================================================================

AGENT_OPERATE_PROMPT_SURVEY = """
You are an intelligent agent that reviews and revises an entire academic survey paper.

[Survey Information]
- Topic: {topic}
- Survey Title: {survey_title}

[Current Survey Text (Preview)]
{current_survey_text}

[Survey Outline]
{survey_outline}

[Memory - Operation History and Results]
{memory}

[Current Context Summary]
{context_summary}

---

Your task is to plan a sequence of operations to review and revise the entire survey.

Available operations (skills):
1. "review": Analyze the entire survey and provide scores and suggestions for improvement in three dimensions:
   - Readability (clarity, logical flow, formatting)
   - Depth (comprehensiveness, insightfulness, citation coverage)
   - Coherence (cross-section consistency, smooth transitions)
   
2. "revise": Apply specific revision suggestions to improve the survey text.
   - Input: A single suggestion from the review
   - Output: Modified survey text with the suggestion applied

3. "get_pseudocode": Retrieve pseudocode for a specific repository
   - Input: repo_name
   - Context to include: what specific aspect to focus on

4. "get_code_report": Include code report content in context
   - Input: specific code report or reference to include

5. "finish": Complete the revision when the survey is satisfactory
   - CRITICAL: You can only call "finish" when ALL THREE review scores are >= 8:
     * Readability >= 8
     * Depth >= 8
     * Coherence >= 8
   - If any score is below 8, you must continue to revise or gather more context first

[WORKFLOW - CRITICAL - FOLLOW THIS ORDER]
You MUST follow this workflow strictly:
1. First, if you have NOT yet reviewed the survey, call "review" to analyze and get scores + suggestions
2. If you have already reviewed, you MUST use the EXISTING suggestions from memory to call "revise" - DO NOT call "review" again!
3. When all current suggestions are met, you can call "review" again.
4. Call "finish" only when ALL THREE scores are >= 8, or when you have addressed all suggestions

[Priority Guidelines]
- Focus on the most impactful suggestions first
- Prioritize suggestions that address the LOWEST-scoring dimensions
- Use external knowledge (keynotes, pseudocode) when the survey needs more depth

---

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "input": "...", "reason": "..."}},
        {{"operation": "...", "input": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "revise", "input" should contain the EXACT suggestion text to apply
- For "get_pseudocode", "input" should be the repo_name
- For "finish", no input needed
- Include 1 to {max_plan_steps} operations per plan to complete the revision efficiently

Generate JSON directly without any other things.
"""

AGENT_OPERATE_PROMPT_SURVEY_NO_CODE = """
You are an intelligent agent that reviews and revises an entire academic survey paper.

[Survey Information]
- Topic: {topic}
- Survey Title: {survey_title}

[Current Survey Text (Preview)]
{current_survey_text}

[Survey Outline]
{survey_outline}

[Memory - Operation History and Results]
{memory}

[Current Context Summary]
{context_summary}

---

Your task is to plan a sequence of operations to review and revise the entire survey.

Available operations (skills):
1. "review": Analyze the entire survey and provide scores and suggestions for improvement in three dimensions:
   - Readability (clarity, logical flow, formatting)
   - Depth (comprehensiveness, insightfulness, citation coverage)
   - Coherence (cross-section consistency, smooth transitions)
   
2. "revise": Apply specific revision suggestions to improve the survey text.
   - Input: A single suggestion from the review
   - Output: Modified survey text with the suggestion applied

3. "finish": Complete the revision when the survey is satisfactory
   - CRITICAL: You can only call "finish" when ALL THREE review scores are >= 8:
     * Readability >= 8
     * Depth >= 8
     * Coherence >= 8
   - If any score is below 8, you must continue to revise or gather more context first

[WORKFLOW - CRITICAL - FOLLOW THIS ORDER]
You MUST follow this workflow strictly:
1. First, if you have NOT yet reviewed the survey, call "review" to analyze and get scores + suggestions
2. If you have already reviewed, you MUST use the EXISTING suggestions from memory to call "revise" - DO NOT call "review" again!
3. When all current suggestions are met, you may call "review" again.
6. Call "finish" only when ALL THREE scores are >= 8, or when you have addressed all suggestions

[Priority Guidelines]
- Focus on the most impactful suggestions first
- Focus on the Memory content to learn the actions you have taken, and plan future actions with a reasonable workflow
- Prioritize suggestions that address the LOWEST-scoring dimensions
- Use external knowledge (keynotes, pseudocode) when the survey needs more depth
---

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "input": "...", "reason": "..."}},
        {{"operation": "...", "input": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "revise", "input" should contain the EXACT suggestion text to apply
- For "finish", no input needed
- Include 1 to {max_plan_steps} operations per plan to complete the revision efficiently

Generate JSON directly without any other things.
"""

# Whole Survey Review skill prompt
SURVEY_REVIEW_SKILL_PROMPT = """
You are an expert reviewer for an academic survey paper concerning topic: {topic}. You are reviewing the entire survey.

The draft survey may contain inline paper citations in angle brackets (e.g., "<Attention is All You Need>").

### Task:
1. Read the Draft Survey and perform a comprehensive review focused on clarity, logical flow, technical accuracy, depth, and cross-section coherence.
2. Produce a clear, specific, and actionable **list of revision suggestions**.
3. Each suggestion should concisely explain *what* to change, *why* and *how* to implement it. When appropriate, point to the exact section or paragraph to anchor the suggestion.
4. Focus on issues that span multiple sections or affect the overall survey quality.
5. Only give suggestions on content and structure, NOT on section titles or outline structure.
6. If the survey is satisfactory, provide an empty suggestion list.
7. Suggest sorting by importance, with important ones coming first.
8. You are suggested to give no more than 5 suggestions

### Review Dimensions (Score 0-10 for each):
1. **Readability**: Clarity, logical flow, formatting, avoiding verbose phrases or repetitions
2. **Depth**: Comprehensiveness, insightfulness, citation coverage, novel analysis
3. **Coherence**: Cross-section consistency, smooth transitions, unified narrative

### Scoring Criteria (Apply these guidelines consistently):

**Readability (0-10):**
- **0-3**: Difficult to understand; poor logical flow; repetitive or verbose; lacks proper formatting
- **4-5**: Some clarity issues; occasional logical jumps; minor repetitions or overly long sentences
- **6-7**: Generally clear; reasonable flow; minor formatting issues; some sentences could be tightened
- **8-9**: Clear and well-structured; smooth logical flow; proper formatting; concise prose
- **10**: Exceptional clarity; publication-ready prose; elegant transitions between paragraphs

**Depth (0-10):**
- **0-3**: Superficial; mainly lists methods without analysis; few or irrelevant citations; no insights
- **4-5**: Basic coverage; limited analysis; citations present but not well-integrated; some generic statements
- **6-7**: Good coverage; some insightful analysis; citations support key points; identifies trends or patterns
- **8-9**: Comprehensive; deep analysis with novel perspectives; well-chosen citations; clear connections to field
- **10**: Outstanding depth; original insights that advance understanding; comprehensive and well-cited analysis

**Coherence (0-10):**
- **0-3**: Sections disconnected; jarring transitions; inconsistent terminology across sections; fragmented narrative
- **4-5**: Some connections between sections; occasional abrupt transitions; minor terminology inconsistencies
- **6-7**: Reasonable section transitions; generally consistent terminology; coherent overall narrative
- **8-9**: Smooth transitions between sections; consistent terminology throughout; unified narrative flow
- **10**: Seamless integration of all sections; exemplary coherence; unified and compelling narrative

### Requirements:
1. You should only provide suggestions on content. DO NOT give any suggestions that involve changing section titles or the overall outline structure.
2. All citations in the survey must be in correct format: <paper_title> (like <Attention is All You Need>).
3. The goal is to ultimately complete a survey with depth, insights and can boost further development, rather than simply listing methods.
4. The survey content should have deep insights, novelty and analysis under the field.
5. The survey content should be clear and coherent. Avoid over verbose phrases or any repetitions.
6. The survey content should be elegantly formatted and have good readability.
7. The content of each section should be logically coherent and academically styled with academic rigor and precise description.
8. The paragraphs should contain sufficient citations to support the viewpoints and provide specific and in-depth analysis.

### Input:
Current Survey:
{survey_text}

Survey Outline:
{survey_outline}

### Output format (exact JSON):
{{
    "scores": {{
        "readability": <score_1>,
        "depth": <score_2>,
        "framework" <score_3>: 
    }},
    "suggestions": [
        "suggestion 1 with specific details (string)",
        "suggestion 2 with specific details (string)",
        ...
    ]
}}
- the content under "suggestions" should be exact List of String
"""

# Whole Survey Revise skill prompt
SURVEY_REVISE_SKILL_PROMPT = """
You are a revise assistant. The content of a survey concerning {topic} is provided below.

You must propose at most ONE exact textual substitution per response according to the suggestion of reviewer.
If you think the document requires changes, choose one that you think is most important to address next, and output ONE JSON object (and nothing else).

The originalText field must match EXACTLY ONE substring in the document.
If you believe no edits are required, output exactly: {{"action":"done"}}.

Survey Outline (Do not change structure):
{survey_outline}

Original Survey Text:
{survey_text}

Citation Text:
{citations}

Guidance:
- Give one minimal but precise modification.
- Revise the survey to enhance its readability, logicality, depth, and cross-section coherence.
- Your modifications **MUST** be consistent with the overall structure, logic and scope of the survey.
- If the suggestion requires additional context from papers, incorporate that context while making the revision.
- The revise should meets the standards of a top-tier academic literature review

Reviewer Suggestion:
{reviewer_suggestion}

External Context:
{external_context}

**Revision Goal:**
Your revision target: produce a survey that meets the quality standards of a top-tier academic conference survey, with rigorous logic, excellent readability, and sufficient depth.

**Output format: (STRICT)**
{{
    "action":"replace", 
    "originalText":"<the exact substring to replace>", 
    "newText":"<the replacement text>"
}}

"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AgentContext:
    """Structured input for the Section Agentic Revisor."""
    topic: str
    survey_title: str
    section_index: int
    total_sections: int
    section_title: str
    section_description: str
    current_section_text: str
    previous_section_text: str = ""
    next_section_text: str = ""
    section_outline: str = ""
    memory: List[Dict] = field(default_factory=list)
    context: str = ""
    context_summary: str = ""
    review_scores: Dict = field(default_factory=dict)
    current_suggestions: List[str] = field(default_factory=list)
    has_reviewed: bool = False
    revision_count: int = 0
    consecutive_review_count: int = 0  # Track consecutive review calls


@dataclass
class AgentContextSurvey:
    """Structured input for the Whole Survey Agentic Revisor."""
    topic: str
    survey_title: str
    current_survey_text: str
    survey_outline: str = ""
    memory: List[Dict] = field(default_factory=list)
    context: str = ""
    context_summary: str = ""
    review_scores: Dict = field(default_factory=dict)
    current_suggestions: List[str] = field(default_factory=list)
    has_reviewed: bool = False
    revision_count: int = 0
    consecutive_review_count: int = 0  # Track consecutive review calls

# ============================================================================
# Base Class
# ============================================================================

class BaseAgenticRevisor:
    """
    Base class for agent-based survey revisors.
    Contains common functionality shared between section and survey revisors.
    """
    
    def __init__(self, config, chat_agent: ChatAgent, work_analyzer, database, code_report: str = None, include_code = False):
        self.chat_agent = chat_agent
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.work_analyzer = work_analyzer
        self.database = database
        self.code_report = code_report
        self.include_code = include_code
        
        # Configuration
        self.default_max_steps = 15
        self.memory_token_threshold = 70000
        self.memory_result_max_length = 5000
        self.max_context_summary_length = 1000
        
        # Initialize tiktoken encoding for token counting
        try:
            self.encodings = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.logger.warning("Failed to load tiktoken encoding, using character-based estimation")
            self.encodings = None
    
    # ------------------------------------------------------------------------
    # Common utility methods
    # ------------------------------------------------------------------------
    
    def _get_pseudocode(self, repo_name: str, context_focus: str = "") -> str:
        """Retrieve pseudocode for a repository."""
        try:
            from modules.code_collector import CodeAnalyzer, CodeCollector
            code_collector = CodeCollector(self.config)
            repo_cache_path = code_collector.repo_cache_path
            repo_dir = os.path.join(repo_cache_path, repo_name)
            
            if not os.path.isdir(repo_dir):
                self.logger.warning(f"Repository {repo_name} not found in cache")
                return f"[Pseudocode for {repo_name} not available]"
            
            model_name = self.chat_agent.model_name
            
            for suffix in ["", "_concise"]:
                cache_path = os.path.join(repo_dir, f"pseudocode_{model_name}{suffix}.json")
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        pseudocode = f.read()
                    self.logger.info(f"Retrieved pseudocode for {repo_name} from cache (exact model match)")
                    if context_focus:
                        return f"[Pseudocode for {repo_name} - focusing on: {context_focus}]\n{pseudocode}"
                    return f"[Pseudocode for {repo_name}]\n{pseudocode}"
            
            for filename in os.listdir(repo_dir):
                if filename.startswith("pseudocode_") and filename.endswith(".json"):
                    cache_path = os.path.join(repo_dir, filename)
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            pseudocode = f.read()
                        self.logger.info(f"Retrieved pseudocode for {repo_name} from cache (fallback to {filename})")
                        if context_focus:
                            return f"[Pseudocode for {repo_name} - focusing on: {context_focus}]\n{pseudocode}"
                        return f"[Pseudocode for {repo_name}]\n{pseudocode}"
                    except Exception:
                        continue
            
            # for filename in os.listdir(repo_dir):
            #     if filename.startswith("mainfest_") and filename.endswith(".json"):
            #         mainfest_path = os.path.join(repo_dir, filename)
            #         with open(mainfest_path, "r", encoding="utf-8") as f:
            #             import json
            #             mainfest = json.load(f)
            #         self.logger.info(f"Found mainfest for {repo_name}, no cached pseudocode")
            #         return f"[Repository {repo_name} structure available but pseudocode not yet generated]"
            
            self.logger.warning(f"No pseudocode found for {repo_name}")
            return f"[Pseudocode for {repo_name} not available]"
        except Exception as e:
            self.logger.error(f"Error retrieving pseudocode for {repo_name}: {e}")
            return f"[Error retrieving pseudocode for {repo_name}: {e}]"
    
    def _get_keynote(self, paper_id_or_title: str, context_focus: str = "") -> str:
        """Retrieve keynote for a paper."""
        try:
            try:
                paper_info = self.work_analyzer.work_collector.get_paper_with_title(paper_id_or_title)
                if paper_info and "paper_id" in paper_info:
                    keynote = self.work_analyzer.get_paper_keynote(paper_info["paper_id"])
                    self.logger.info(f"Retrieved keynote for paper title {paper_id_or_title}")
                    if context_focus:
                        return f"[Keynote for '{paper_id_or_title}' - focusing on: {context_focus}]\n{keynote}"
                    return f"[Keynote for paper '{paper_id_or_title}']\n{keynote}"
            except Exception:
                pass

            try:
                keynote = self.work_analyzer.get_paper_keynote(paper_id_or_title)
                self.logger.info(f"Retrieved keynote for paper ID {paper_id_or_title}")
                if context_focus:
                    return f"[Keynote for {paper_id_or_title} - focusing on: {context_focus}]\n{keynote}"
                return f"[Keynote for paper {paper_id_or_title}]\n{keynote}"
            except Exception:
                pass
            
            self.logger.warning(f"No keynote found for {paper_id_or_title}")
            return f"[Keynote for {paper_id_or_title} not available]"
        except Exception as e:
            self.logger.error(f"Error retrieving keynote for {paper_id_or_title}: {e}")
            return f"[Error retrieving keynote for {paper_id_or_title}: {e}]"
    
    def _apply_revision_to_text(self, original_text: str, revision_json: dict) -> str:
        """Apply revision to original text based on revision json."""
        if revision_json.get("action") == "done":
            return original_text

        old_str = revision_json.get("originalText", "")
        new_str = revision_json.get("newText", "")

        if not old_str:
            self.logger.warning("'originalText' is empty or do not have the key.")
            return original_text

        idx = original_text.find(old_str)
        if idx == -1:
            self.logger.error(f"Could not find exact substring to replace.\nTarget: {old_str[:50]}...")
            return original_text
        
        new_text = original_text[:idx] + new_str + original_text[idx + len(old_str):]
        if self.config.BasicInfo.debug:
            self.logger.info(f"Applied revision: Replaced {len(old_str)} chars with {len(new_str)} chars.")
        return new_text
    
    def _check_memory_compression(self, agent_ctx) -> bool:
        """Check if memory needs compression based on token count."""
        memory_str = self._format_memory(agent_ctx.memory)
        token_count = self._count_tokens(memory_str)
        
        if token_count > self.memory_token_threshold:
            self.logger.info(f"Memory token count ({token_count}) exceeds threshold ({self.memory_token_threshold}), compressing...")
            agent_ctx.memory = self._compress_memory(agent_ctx.memory)
            return True
        return False
    
    def _validate_plan(self, response: str, info_dict: dict = None):
        """Validate plan response format."""
        result = extract_json(response)
        if not isinstance(result, dict):
            raise ValueError(f"Response is not a dict: {type(result)}")
        if "plan" not in result:
            raise ValueError("Missing 'plan' field in response")
        plan = result.get("plan")
        if not isinstance(plan, list):
            raise ValueError("'plan' field is not a list")
        for op in plan:
            if "operation" not in op:
                raise ValueError("Each operation must have an 'operation' field")
        return True, plan
    
    def _format_context_for_memory(self, operation: str, input_data: str, context_result: str = "", memory_entry: dict = None) -> dict:
        """Format a memory entry for context retrieval operations."""
        memory_entry = memory_entry or {}
        memory_entry["context_result"] = context_result
        return memory_entry
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken encoding."""
        if self.encodings is None:
            return len(text) // 4
        try:
            return len(self.encodings.encode(text))
        except Exception:
            return len(text) // 4
    
    def _compress_memory(self, memory: List[Dict]) -> List[Dict]:
        """
        Compress memory by:
        1. First, truncating context_result fields from query operations in the first half of entries
        2. Then, removing the oldest half of entries
        3. If still over threshold, recursively call itself
        """
        if len(memory) <= 1:
            return memory

        # Step 1: Truncate context_result from query operations in the first half
        mid_point = len(memory) // 2
        for entry in memory[:mid_point]:
            if entry.get("operation") in ("get_pseudocode", "get_keynote", "get_code_report", "get_full_survey"):
                if "context_result" in entry and isinstance(entry["context_result"], str):
                    entry["context_result"] = entry["context_result"][:self.memory_result_max_length] + "(compressed)..."

        # Step 2: Remove the oldest half
        compressed = memory[mid_point:]

        # Step 3: Check if still over threshold, recursively compress if needed
        memory_str = self._format_memory(compressed)
        token_count = self._count_tokens(memory_str)
        if token_count > self.memory_token_threshold:
            self.logger.info(f"Memory still over threshold ({token_count} tokens), recursively compressing...")
            return self._compress_memory(compressed)

        self.logger.info(f"Memory compressed: {len(memory)} -> {len(compressed)} entries")
        return compressed
    
    def _format_memory(self, memory: List[Dict]) -> str:
        """
        Format memory (operation history with results) as string.
        """
        if not memory:
            return "No operations performed yet."
        
        lines = []
        for i, op in enumerate(memory):
            op_type = op.get('operation', '')
            reason = op.get('reason', '')
            
            lines.append(f"{i+1}. {op_type}: {reason}")
            
            if op_type == "review":
                scores = op.get('scores', {})
                suggestions = op.get('suggestions', [])
                if self.config.BasicInfo.debug:
                    for idx, suggestion in enumerate(suggestions):
                        if not isinstance(suggestion, str):
                            self.logger.error(f"suggestion: {suggestion} not str: {type(suggestion)} in _format_memory")
                            suggestions[idx] = str(suggestion)

                if scores:
                    lines.append(f"   Scores: readability={scores.get('readability', 0)}, depth={scores.get('depth', 0)}, framework={scores.get('framework', 0)}")
                if suggestions:
                    suggestions_text = "; ".join(suggestions[:5])
                    if len(suggestions_text) > self.memory_result_max_length // 2:
                        suggestions_text = suggestions_text[:self.memory_result_max_length // 2] + "(truncated)..."
                    lines.append(f"   Suggestions: {suggestions_text}")
                    
            elif op_type == "revise":
                result = op.get('result', '')
                if result:
                    lines.append(f"   Result: {result}")
                    
            elif op_type in ["get_pseudocode", "get_keynote"]:
                repo_name = op.get('repo_name', '')
                paper_id = op.get('paper_id', '')
                target = repo_name or paper_id
                context_result = op.get('context_result', '')
                if context_result:
                    if len(context_result) > self.memory_result_max_length:
                        context_result = context_result[:self.memory_result_max_length] + "(truncated)..."
                    lines.append(f"   Retrieved for {target}: {context_result}")
                result = op.get('result', '')
                if result:
                    lines.append(f"   Result: {result}")
                    
            elif op_type == "get_code_report":
                lines.append(f"   Result: {op.get('result', '')}")
                
            elif op_type == "finish":
                final_scores = op.get('final_scores', {})
                revision_count = op.get('revision_count', 0)
                if final_scores:
                    lines.append(f"   Final scores: {final_scores}")
                lines.append(f"   Revision count: {revision_count}")
                result = op.get('result', '')
                if result:
                    lines.append(f"   Result: {result}")
            else:
                result = op.get('result', '')
                if result:
                    if len(result) > self.memory_result_max_length:
                        result = result[:self.memory_result_max_length] + "(truncated)..."
                    lines.append(f"   Result: {result}")
        for line in lines:
            if not isinstance(line, str):
                self.logger.warning(f"line not string in agentic reviser: {type(line)}")
                self.logger.warning(f"Trying to convert: {line}")
                line = str(line)
        return "\n".join(lines)


# ============================================================================
# Section Revision Class
# ============================================================================

class AgenticRevisor(BaseAgenticRevisor):
    """
    Agent-based survey section revisor that uses a planner with various skills
    to review and revise survey sections.
    """
    
    def __init__(self, config, chat_agent: ChatAgent, work_analyzer, database, code_report: str = None, include_code = False):
        super().__init__(config, chat_agent, work_analyzer, database, code_report, include_code)
        self.logger = get_logger("AgenticRevisor")
        
        # Section-specific configuration
        self.section_review_retry = config.ModuleInfo.SurveyGenerator.section_review_retry
        self.section_revise_retry = config.ModuleInfo.SurveyGenerator.section_revise_retry
        self.section_review_temperature = config.ModuleInfo.SurveyGenerator.section_review_temperature
        self.section_revise_temperature = config.ModuleInfo.SurveyGenerator.section_revise_temperature
        self.section_revision_RAG_topk = config.ModuleInfo.SurveyGenerator.section_revision_RAG_topk
        self.section_least_words = config.ModuleInfo.SurveyGenerator.section_least_words or "no limit"
    
    def _format_section_outline(self, section_outline, include_description=True):
        """Format section outline as string."""
        if isinstance(section_outline, str):
            return section_outline
        if not isinstance(section_outline, dict):
            return str(section_outline)
        
        text = f"- Section title: {section_outline.get('title', '')}\n"
        if include_description:
            text += f"  Section description: {section_outline.get('description', '')}\n"
        for subsection in section_outline.get("subsections", []):
            text += f"  - Subsection title: {subsection.get('title', '')}\n"
            if include_description:
                text += f"    Subsection description: {subsection.get('description', '')}\n"
        return text
    
    def _call_review(self, agent_ctx: AgentContext) -> Dict:
        """Call review skill to get scores and suggestions."""
        prompt = REVIEW_SKILL_PROMPT.format(
            topic=agent_ctx.topic,
            section_title=agent_ctx.section_title,
            section_least_words=self.section_least_words,
            current_section_length=len(agent_ctx.current_section_text.split()),
            previous_section_text=agent_ctx.previous_section_text or "(No previous section)",
            next_section_text=agent_ctx.next_section_text or "(No next section)",
            section_text=agent_ctx.current_section_text,
            section_outline=agent_ctx.section_outline
        )
        
        valid = False
        for retry_time in range(self.section_review_retry):
            try:
                response = self.chat_agent.remote_chat(prompt, temperature=self.section_review_temperature)
                result = extract_json(response)
                
                if isinstance(result, dict) and "scores" in result and "suggestions" in result:
                    scores = result.get("scores", {})
                    suggestions = result.get("suggestions", [])

                    if not isinstance(suggestions, list):
                        self.logger.warning(f"Type error in AgenticRevisor: suggestions not list: {type(suggestions)}")
                        raise TypeError(f"Reviewer suggestion not string: {suggestions}")
                    else:
                        # Convert each non-string suggestion to string in-place
                        for idx, suggestion in enumerate(suggestions):
                            if not isinstance(suggestion, str):
                                self.logger.warning(f"Type error in AgenticRevisor: suggestions {idx} not str: {type(suggestion)}")
                                self.logger.warning(f"Trying to convert: {suggestion}")
                                suggestions[idx] = str(suggestion)
                    
                    for key in ["readability", "depth", "framework"]:
                        if key not in scores:
                            scores[key] = 0
                        elif not (0 <= scores[key] <= 10):
                            scores[key] = max(0, min(10, scores[key]))
                    
                    return {"scores": scores, "suggestions": suggestions}
                
                raise ValueError(f"Invalid review response format: {result}")
                
            except Exception as e:
                self.logger.error(f"Review attempt {retry_time + 1} failed: {e}")
                continue
        
        raise ValueError("Failed to generate valid review after retries")
    
    def _call_revise(self, agent_ctx: AgentContext, suggestion: str) -> Tuple[str, Dict]:
        """Call revise skill to apply a suggestion."""
        citations = self.database.query_and_text(
            agent_ctx.section_title or agent_ctx.topic,
            self.section_revision_RAG_topk
        )
        
        prompt = REVISE_SKILL_PROMPT.format(
            topic=agent_ctx.topic,
            section_title=agent_ctx.section_title,
            section_outline=agent_ctx.section_outline,
            section_text=agent_ctx.current_section_text,
            citations=citations,
            reviewer_suggestion=suggestion,
            external_context=f"\n\n[Additional Context from Skills]\n{agent_ctx.context}" if agent_ctx.context else ""
        )
        
        valid = False
        parsed_json = None
        for retry_time in range(self.section_revise_retry):
            try:
                response = self.chat_agent.remote_chat(prompt, temperature=self.section_revise_temperature)
                parsed_json = extract_json(response)
                
                if not isinstance(parsed_json, dict):
                    raise ValueError("Parsed JSON is not a dict.")
                
                if "###" in parsed_json.get("newText", "") or "###" in parsed_json.get("originalText", ""):
                    raise ValueError("Revision contains '###' which may break markdown structure.")
                
                valid = True
                break
                
            except Exception as e:
                self.logger.error(f"Revise attempt {retry_time + 1} failed: {e}")
                self.logger.info(f"raw response: {response}")
                continue
        
        if not valid or parsed_json is None:
            raise ValueError("Failed to generate valid revision after retries")
        
        if parsed_json.get("action") == "done":
            return agent_ctx.current_section_text, parsed_json
        elif parsed_json.get("action") == "replace":
            new_text = self._apply_revision_to_text(agent_ctx.current_section_text, parsed_json)
            return new_text, parsed_json
        else:
            raise ValueError(f"Invalid revision action: {parsed_json.get('action')}")
    
    def _execute_operation(self, op: dict, agent_ctx: AgentContext, full_survey_text: str = "") -> Tuple[str, Dict]:
        """Execute a single operation for section revision."""
        operation = op.get("operation", "")
        input_data = op.get("input", "")
        reason = op.get("reason", "")
        
        memory_entry = {"operation": operation, "reason": reason}
        
        if operation == "review":
            self.logger.info(f"Calling review skill for section {agent_ctx.section_index}")
            try:
                result = self._call_review(agent_ctx)
                
                agent_ctx.review_scores = result.get("scores", {})
                agent_ctx.current_suggestions = result.get("suggestions", [])
                agent_ctx.has_reviewed = True
                
                memory_entry["scores"] = result.get("scores", {})
                memory_entry["suggestions"] = result.get("suggestions", [])
                memory_entry["result"] = f"Reviewed with scores: {result.get('scores', {})}"
            except Exception as e:
                self.logger.error(f"Error occurs in agentic refinement review: {e}")
                
                memory_entry["scores"] = {}
                memory_entry["suggestions"] = []
                memory_entry["result"] = f"Fail to use review skill: {e}"
            
        elif operation == "revise":
            if not agent_ctx.has_reviewed:
                self.logger.warning("revise called before review - calling review first")
                review_op = {"operation": "review", "reason": "Forced review before revise"}
                self._execute_operation(review_op, agent_ctx, full_survey_text)
            
            suggestion = input_data
            self.logger.info(f"Calling revise skill with suggestion: {truncate_text(suggestion, 100)}")
            error = ""
            try:
                new_text, revision_result = self._call_revise(agent_ctx, suggestion)
            except Exception as e:
                self.logger.error(f"Error occurs in calling revise skill: {e}")
                error = e

            if error:
                new_text = agent_ctx.current_section_text
                memory_entry["result"] = f"Error occurs: {error}, use original text"
            elif new_text != agent_ctx.current_section_text:
                agent_ctx.current_section_text = new_text
                agent_ctx.revision_count += 1
                memory_entry["result"] = f"Applied revision: {revision_result.get('action', 'unknown')}"
            else:
                memory_entry["result"] = "No change made"
            
            agent_ctx.context = ""
            agent_ctx.context_summary = ""
            
        elif operation == "get_pseudocode":
            repo_name = input_data
            self.logger.info(f"Retrieving pseudocode for repo: {repo_name}")
            
            context_result = self._get_pseudocode(repo_name)
            agent_ctx.context += f"\n\n=== Pseudocode: {repo_name} ===\n{context_result}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, self.max_context_summary_length)
            
            memory_entry["repo_name"] = repo_name
            memory_entry["context_result"] = context_result
            memory_entry["result"] = f"Retrieved pseudocode for {repo_name}"
            
        elif operation == "get_keynote":
            paper_id_or_title = input_data
            self.logger.info(f"Retrieving keynote for paper: {paper_id_or_title}")
            
            context_result = self._get_keynote(paper_id_or_title)
            agent_ctx.context += f"\n\n=== Paper Keynote: {paper_id_or_title} ===\n{context_result}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, self.max_context_summary_length)
            
            memory_entry["paper_id"] = paper_id_or_title
            memory_entry["context_result"] = context_result
            memory_entry["result"] = f"Retrieved keynote for {paper_id_or_title}"
            
        elif operation == "get_code_report":
            self.logger.info("Adding code report to context")
            
            if self.code_report:
                code_report_intro = CODE_REPORT_PROMPT.format(code_report="")
                agent_ctx.context += f"{code_report_intro}\n\n=== Code Report ===\n{self.code_report}\n"
                agent_ctx.context_summary = "Code report available (follow usage principles)"
                memory_entry["result"] = "Added code report with usage principles"
            else:
                memory_entry["result"] = "No code report available"
            
        elif operation == "get_full_survey":
            self.logger.info(f"Adding full survey to context ({len(full_survey_text)} chars)")
            
            agent_ctx.context += f"\n\n=== Full Survey ===\n{full_survey_text}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, self.max_context_summary_length)
            
            memory_entry["survey_content"] = full_survey_text
            memory_entry["result"] = f"Added full survey ({len(full_survey_text)} chars) to context"
            
        elif operation == "finish":
            self.logger.info("Agent decided to finish revision")
            memory_entry["final_scores"] = agent_ctx.review_scores
            memory_entry["revision_count"] = agent_ctx.revision_count
            memory_entry["result"] = f"Finished with {agent_ctx.revision_count} revisions"
            
        else:
            self.logger.error(f"Unknown operation: {operation}")
            memory_entry["result"] = f"Unknown operation: {operation}"
        
        return operation, memory_entry
    
    def agentic_revise_section(
        self,
        section_text: str,
        previous_section_text: str,
        next_section_text: str,
        section_outline: dict,
        section_index: int,
        total_sections: int,
        full_survey_text: str = "",
        max_steps: int = None,
    ) -> str:
        """Agentically review and revise a section using the planner agent."""
        agent_ctx = AgentContext(
            topic=self.config.BasicInfo.topic,
            survey_title=self.config.BasicInfo.topic + " Survey",
            section_index=section_index,
            total_sections=total_sections,
            section_title=section_outline.get('title', ''),
            section_description=section_outline.get('description', ''),
            current_section_text=section_text,
            previous_section_text=previous_section_text,
            next_section_text=next_section_text,
            section_outline=self._format_section_outline(section_outline),
            memory=[],
            context="",
            context_summary="",
            review_scores={},
            current_suggestions=[],
            has_reviewed=False,
            revision_count=0
        )
        
        if self.code_report:
            code_report_intro = CODE_REPORT_PROMPT.format(code_report="")
            agent_ctx.context = f"{code_report_intro}\n\n=== Code Report ===\n{self.code_report}\n"
            agent_ctx.context_summary = "Code report available (follow usage principles)"
        
        max_steps = max_steps or self.default_max_steps
        
        self.logger.info(f"Starting agentic revision for section {section_index}/{total_sections}: {agent_ctx.section_title}")
        
        for step in range(max_steps):
            self.logger.info(f"Agent step {step + 1}/{max_steps}")
            self._check_memory_compression(agent_ctx)

            if self.include_code:
                template = AGENT_OPERATE_PROMPT
            else:
                template = AGENT_OPERATE_PROMPT_NO_CODE
            prompt = template.format(
                topic=agent_ctx.topic,
                survey_title=agent_ctx.survey_title,
                section_index=agent_ctx.section_index,
                total_sections=agent_ctx.total_sections,
                section_title=agent_ctx.section_title,
                section_description=agent_ctx.section_description,
                current_section_text=truncate_text(agent_ctx.current_section_text, 3000),
                previous_section_text=truncate_text(agent_ctx.previous_section_text, 1000) if agent_ctx.previous_section_text else "(No previous section)",
                next_section_text=truncate_text(agent_ctx.next_section_text, 1000) if agent_ctx.next_section_text else "(No next section)",
                section_outline=agent_ctx.section_outline,
                memory=self._format_memory(agent_ctx.memory),
                # context=agent_ctx.context if agent_ctx.context else "No external context retrieved yet.",
                context_summary=agent_ctx.context_summary if agent_ctx.context_summary else "No context available",
                max_plan_steps = 5
            )
            
            print(f"[AGENTIC REFINE SECTION DEBUG:]\n{prompt}")

            try:
                plan_result = self.chat_agent.remote_chat_with_retry(
                    prompt=prompt,
                    validate_fn=self._validate_plan,
                    max_retry=3,
                    temperature=0.7
                )
            except Exception as e:
                self.logger.error(f"Failed to parse agent response: {e}, skipped this round")
                agent_ctx.memory.append({"operation": "error", "reason": str(e), "result": "Plan parsing failed"})
                continue
            
            for op in plan_result:
                op_type = op.get("operation", "")
                
                # Track consecutive reviews
                if op_type == "review":
                    agent_ctx.consecutive_review_count += 1
                else:
                    agent_ctx.consecutive_review_count = 0
                
                # Constraint 1: If 2+ consecutive reviews, replace with random revise from suggestions
                if op_type == "review" and agent_ctx.consecutive_review_count >= 2:
                    self.logger.warning(f"Consecutive review detected ({agent_ctx.consecutive_review_count}), replacing with random revise")
                    if agent_ctx.current_suggestions and len(agent_ctx.current_suggestions) > 0:
                        random_suggestion = random.choice(agent_ctx.current_suggestions)
                        self.logger.info(f"Selected random suggestion: {truncate_text(random_suggestion, 100)}")
                        op = {"operation": "revise", "input": random_suggestion, "reason": "Auto-revise: consecutive review replaced with suggestion"}
                    else:
                        self.logger.warning("No suggestions available for auto-revise, skipping")
                        continue
                
                # Constraint 2: If finish is called with scores < 8, replace with random revise
                if op_type == "finish":
                    scores = agent_ctx.review_scores
                    has_low_score = False
                    for key in ["readability", "depth", "framework"]:
                        score = scores.get(key, 0)
                        if score < 8:
                            has_low_score = True
                            self.logger.warning(f"Finish called but {key} score ({score}) < 8, replacing with random revise")
                            break
                    
                    if has_low_score:
                        if agent_ctx.current_suggestions and len(agent_ctx.current_suggestions) > 0:
                            random_suggestion = random.choice(agent_ctx.current_suggestions)
                            self.logger.info(f"Selected random suggestion: {truncate_text(random_suggestion, 100)}")
                            op = {"operation": "revise", "input": random_suggestion, "reason": "Auto-revise: finish blocked due to low scores"}
                            # Don't finish, continue with revise
                        else:
                            self.logger.warning("No suggestions available for auto-revise, continuing without finish")
                            continue
                    else:
                        # All scores >= 8, allow finish
                        op_type, memory_entry = self._execute_operation(op, agent_ctx, full_survey_text)
                        agent_ctx.memory.append(memory_entry)
                        self.logger.info(f"Agent decided to finish at step {step}")
                        break
                else:
                    op_type, memory_entry = self._execute_operation(op, agent_ctx, full_survey_text)
                    agent_ctx.memory.append(memory_entry)
                    
                    if op_type == "finish":
                        self.logger.info(f"Agent decided to finish at step {step}")
                        break
            
            if any(m.get("operation") == "finish" for m in agent_ctx.memory[-len(plan_result):] if plan_result):
                break
        
        self.logger.info(f"Agentic revision completed for section {section_index} after {len(agent_ctx.memory)} operations")
        self.logger.info(f"Final review scores: {agent_ctx.review_scores}")
        self.logger.info(f"Total revisions made: {agent_ctx.revision_count}")
        self.logger.info(f"[SECTION MEMORY DEBUG:] {self._format_memory(agent_ctx.memory)}")

        return agent_ctx.current_section_text


# ============================================================================
# Whole Survey Revision Class
# ============================================================================

class AgenticSurveyRevisor(BaseAgenticRevisor):
    """
    Agent-based whole survey revisor that uses a planner with various skills
    to review and revise the entire survey at once.
    """
    
    def __init__(self, config, chat_agent: ChatAgent, work_analyzer, database, code_report: str = None, include_code = False):
        super().__init__(config, chat_agent, work_analyzer, database, code_report, include_code)
        self.logger = get_logger("AgenticSurveyRevisor")
        
        # Survey-specific configuration
        self.survey_review_retry = config.ModuleInfo.SurveyGenerator.section_review_retry
        self.survey_revise_retry = config.ModuleInfo.SurveyGenerator.section_revise_retry
        self.survey_review_temperature = config.ModuleInfo.SurveyGenerator.section_review_temperature
        self.survey_revise_temperature = config.ModuleInfo.SurveyGenerator.section_revise_temperature
        self.survey_revision_RAG_topk = config.ModuleInfo.SurveyGenerator.section_revision_RAG_topk
    
    def _format_survey_outline(self, outline, include_description=True):
        """Format survey outline as string."""
        if isinstance(outline, str):
            return outline
        if not isinstance(outline, dict):
            return str(outline)
        
        text = ""
        for section_outline in outline.get("sections", []):
            text += f"- Section title: {section_outline.get('title', '')}\n"
            if include_description:
                text += f"  Section description: {section_outline.get('description', '')}\n"
            for subsection in section_outline.get("subsections", []):
                text += f"  - Subsection title: {subsection.get('title', '')}\n"
                if include_description:
                    text += f"    Subsection description: {subsection.get('description', '')}\n"
        return text
    
    def _call_review(self, agent_ctx: AgentContextSurvey) -> Dict:
        """Call review skill to get scores and suggestions for the whole survey."""
        prompt = SURVEY_REVIEW_SKILL_PROMPT.format(
            topic=agent_ctx.topic,
            survey_text=agent_ctx.current_survey_text,
            survey_outline=agent_ctx.survey_outline
        )
        
        valid = False
        for retry_time in range(self.survey_review_retry):
            try:
                response = self.chat_agent.remote_chat(prompt, temperature=self.survey_review_temperature)
                result = extract_json(response)
                
                if isinstance(result, dict) and "scores" in result and "suggestions" in result:
                    scores = result.get("scores", {})
                    suggestions = result.get("suggestions", [])

                    if not isinstance(suggestions, list):
                        self.logger.warning(f"Type error in AgenticRevisor: suggestions not list: {type(suggestions)}")
                        raise TypeError(f"Reviewer suggestion not string: {suggestions}")
                    else:
                        # Convert each non-string suggestion to string in-place
                        for idx, suggestion in enumerate(suggestions):
                            if not isinstance(suggestion, str):
                                self.logger.warning(f"Type error in AgenticRevisor: suggestions {idx} not str: {type(suggestion)}")
                                self.logger.warning(f"Trying to convert: {suggestion}")
                                suggestions[idx] = str(suggestion)
                    
                    for key in ["readability", "depth", "coherence"]:
                        if key not in scores:
                            scores[key] = 0
                        elif not (0 <= scores[key] <= 10):
                            scores[key] = max(0, min(10, scores[key]))
                    
                    return {"scores": scores, "suggestions": suggestions}
                
                raise ValueError(f"Invalid review response format: {result}")
                
            except Exception as e:
                self.logger.error(f"Survey review attempt {retry_time + 1} failed: {e}")
                continue
        
        raise ValueError("Failed to generate valid survey review after retries")
    
    def _call_revise(self, agent_ctx: AgentContextSurvey, suggestion: str) -> Tuple[str, Dict]:
        """Call revise skill to apply a suggestion to the whole survey."""
        citations = self.database.query_and_text(agent_ctx.topic, self.survey_revision_RAG_topk)
        
        prompt = SURVEY_REVISE_SKILL_PROMPT.format(
            topic=agent_ctx.topic,
            survey_outline=agent_ctx.survey_outline,
            survey_text=agent_ctx.current_survey_text,
            citations=citations,
            reviewer_suggestion=suggestion,
            external_context=f"\n\n[Additional Context from Skills]\n{agent_ctx.context}" if agent_ctx.context else ""
        )
        
        valid = False
        parsed_json = None
        for retry_time in range(self.survey_revise_retry):
            try:
                response = self.chat_agent.remote_chat(prompt, temperature=self.survey_revise_temperature)
                parsed_json = extract_json(response)
                
                if not isinstance(parsed_json, dict):
                    raise ValueError("Parsed JSON is not a dict.")
                
                if "###" in parsed_json.get("newText", "") or "###" in parsed_json.get("originalText", ""):
                    raise ValueError("Revision contains '###' which may break markdown structure.")
                
                valid = True
                break
                
            except Exception as e:
                self.logger.error(f"Survey revise attempt {retry_time + 1} failed: {e}")
                continue
        
        if not valid or parsed_json is None:
            raise ValueError("Failed to generate valid survey revision after retries")
        
        if parsed_json.get("action") == "done":
            return agent_ctx.current_survey_text, parsed_json
        elif parsed_json.get("action") == "replace":
            new_text = self._apply_revision_to_text(agent_ctx.current_survey_text, parsed_json)
            return new_text, parsed_json
        else:
            raise ValueError(f"Invalid revision action: {parsed_json.get('action')}")
    
    def _execute_operation(self, op: dict, agent_ctx: AgentContextSurvey) -> Tuple[str, Dict]:
        """Execute a single operation for whole survey revision."""
        operation = op.get("operation", "")
        input_data = op.get("input", "")
        reason = op.get("reason", "")
        
        memory_entry = {"operation": operation, "reason": reason}
        
        if operation == "review":
            self.logger.info("Calling review skill for entire survey")
            try:
                result = self._call_review(agent_ctx)
                
                agent_ctx.review_scores = result.get("scores", {})
                agent_ctx.current_suggestions = result.get("suggestions", [])
                agent_ctx.has_reviewed = True
                
                memory_entry["scores"] = result.get("scores", {})
                memory_entry["suggestions"] = result.get("suggestions", [])
                memory_entry["result"] = f"Reviewed with scores: {result.get('scores', {})}"
            except Exception as e:
                self.logger.error(f"Error occurs in agentic refinement review: {e}")
                
                memory_entry["scores"] = {}
                memory_entry["suggestions"] = []
                memory_entry["result"] = f"Fail to use review skill: {e}"
            
        elif operation == "revise":
            if not agent_ctx.has_reviewed:
                self.logger.warning("revise called before review - calling review first")
                review_op = {"operation": "review", "reason": "Forced review before revise"}
                self._execute_operation(review_op, agent_ctx)
            
            suggestion = input_data
            self.logger.info(f"Calling revise skill with suggestion: {truncate_text(suggestion, 100)}")
            error = ""
            try:
                new_text, revision_result = self._call_revise(agent_ctx, suggestion)
            except Exception as e:
                self.logger.error(f"Error occurs in calling revise skill: {e}")
                error = e

            if error:
                new_text = agent_ctx.current_survey_text
                memory_entry["result"] = f"Error occurs: {error}, use original text"
            elif new_text != agent_ctx.current_survey_text:
                agent_ctx.current_survey_text = new_text
                agent_ctx.revision_count += 1
                memory_entry["result"] = f"Applied revision: {revision_result.get('action', 'unknown')}"
            else:
                memory_entry["result"] = "No change made"
            
            agent_ctx.context = ""
            agent_ctx.context_summary = ""
            
        elif operation == "get_pseudocode":
            repo_name = input_data
            self.logger.info(f"Retrieving pseudocode for repo: {repo_name}")
            
            context_result = self._get_pseudocode(repo_name)
            agent_ctx.context += f"\n\n=== Pseudocode: {repo_name} ===\n{context_result}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, self.max_context_summary_length)
            
            memory_entry["repo_name"] = repo_name
            memory_entry["context_result"] = context_result
            memory_entry["result"] = f"Retrieved pseudocode for {repo_name}"
            
        elif operation == "get_keynote":
            paper_id_or_title = input_data
            self.logger.info(f"Retrieving keynote for paper: {paper_id_or_title}")
            
            context_result = self._get_keynote(paper_id_or_title)
            agent_ctx.context += f"\n\n=== Paper Keynote: {paper_id_or_title} ===\n{context_result}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, self.max_context_summary_length)
            
            memory_entry["paper_id"] = paper_id_or_title
            memory_entry["context_result"] = context_result
            memory_entry["result"] = f"Retrieved keynote for {paper_id_or_title}"
            
        elif operation == "get_code_report":
            self.logger.info("Adding code report to context")
            
            if self.code_report:
                code_report_intro = CODE_REPORT_PROMPT.format(code_report="")
                agent_ctx.context += f"{code_report_intro}\n\n=== Code Report ===\n{self.code_report}\n"
                agent_ctx.context_summary = "Code report available (follow usage principles)"
                memory_entry["result"] = "Added code report with usage principles"
            else:
                memory_entry["result"] = "No code report available"
            
        elif operation == "finish":
            self.logger.info("Agent decided to finish survey revision")
            memory_entry["final_scores"] = agent_ctx.review_scores
            memory_entry["revision_count"] = agent_ctx.revision_count
            memory_entry["result"] = f"Finished with {agent_ctx.revision_count} revisions"
            
        else:
            self.logger.error(f"Unknown operation: {operation}")
            memory_entry["result"] = f"Unknown operation: {operation}"
        
        return operation, memory_entry
    
    def agentic_revise_survey(self, survey_text: str, outline: dict, max_steps: int = None) -> str:
        """Agentically review and revise the entire survey using the planner agent."""
        agent_ctx = AgentContextSurvey(
            topic=self.config.BasicInfo.topic,
            survey_title=self.config.BasicInfo.topic + " Survey",
            current_survey_text=survey_text,
            survey_outline=self._format_survey_outline(outline),
            memory=[],
            context="",
            context_summary="",
            review_scores={},
            current_suggestions=[],
            has_reviewed=False,
            revision_count=0
        )
        
        if self.code_report:
            code_report_intro = CODE_REPORT_PROMPT.format(code_report="")
            agent_ctx.context = f"{code_report_intro}\n\n=== Code Report ===\n{self.code_report}\n"
            agent_ctx.context_summary = "Code report available (follow usage principles)"
        
        max_steps = max_steps or self.default_max_steps
        
        self.logger.info(f"Starting agentic revision for entire survey (max {max_steps} steps)")
        
        for step in range(max_steps):
            self.logger.info(f"Agent step {step + 1}/{max_steps}")
            self._check_memory_compression(agent_ctx)

            if self.include_code:
                template = AGENT_OPERATE_PROMPT_SURVEY
            else:
                template = AGENT_OPERATE_PROMPT_SURVEY_NO_CODE

            prompt = template.format(
                topic=agent_ctx.topic,
                survey_title=agent_ctx.survey_title,
                current_survey_text=truncate_text(agent_ctx.current_survey_text, 5000),
                survey_outline=agent_ctx.survey_outline,
                memory=self._format_memory(agent_ctx.memory),
                # context=agent_ctx.context if agent_ctx.context else "No external context retrieved yet.",
                context_summary=agent_ctx.context_summary if agent_ctx.context_summary else "No context available",
                max_plan_steps = 5 
            )

            print(f"[AGENTIC REFINE SURVEY DEBUG:]\n{prompt}")

            try:
                plan_result = self.chat_agent.remote_chat_with_retry(
                    prompt=prompt,
                    validate_fn=self._validate_plan,
                    max_retry=3,
                    temperature=0.7
                )
            except Exception as e:
                self.logger.error(f"Failed to parse agent response: {e}, skipped this round")
                agent_ctx.memory.append({"operation": "error", "reason": str(e), "result": "Plan parsing failed"})
                continue
            
            for op in plan_result:
                op_type = op.get("operation", "")
                
                # Track consecutive reviews
                if op_type == "review":
                    agent_ctx.consecutive_review_count += 1
                else:
                    agent_ctx.consecutive_review_count = 0
                
                # Constraint 1: If 2+ consecutive reviews, replace with random revise from suggestions
                if op_type == "review" and agent_ctx.consecutive_review_count >= 2:
                    self.logger.warning(f"Consecutive review detected ({agent_ctx.consecutive_review_count}), replacing with random revise")
                    if agent_ctx.current_suggestions and len(agent_ctx.current_suggestions) > 0:
                        random_suggestion = random.choice(agent_ctx.current_suggestions)
                        self.logger.info(f"Selected random suggestion: {truncate_text(random_suggestion, 100)}")
                        op = {"operation": "revise", "input": random_suggestion, "reason": "Auto-revise: consecutive review replaced with suggestion"}
                    else:
                        self.logger.warning("No suggestions available for auto-revise, skipping")
                        continue
                
                # Constraint 2: If finish is called with scores < 8, replace with random revise
                if op_type == "finish":
                    scores = agent_ctx.review_scores
                    has_low_score = False
                    for key in ["readability", "depth", "coherence"]:
                        score = scores.get(key, 0)
                        if score < 8:
                            has_low_score = True
                            self.logger.warning(f"Finish called but {key} score ({score}) < 8, replacing with random revise")
                            break
                    
                    if has_low_score:
                        if agent_ctx.current_suggestions and len(agent_ctx.current_suggestions) > 0:
                            random_suggestion = random.choice(agent_ctx.current_suggestions)
                            self.logger.info(f"Selected random suggestion: {truncate_text(random_suggestion, 100)}")
                            op = {"operation": "revise", "input": random_suggestion, "reason": "Auto-revise: finish blocked due to low scores"}
                        else:
                            self.logger.warning("No suggestions available for auto-revise, continuing without finish")
                            continue
                    else:
                        # All scores >= 8, allow finish
                        op_type, memory_entry = self._execute_operation(op, agent_ctx)
                        agent_ctx.memory.append(memory_entry)
                        self.logger.info(f"Agent decided to finish at step {step}")
                        break
                else:
                    op_type, memory_entry = self._execute_operation(op, agent_ctx)
                    agent_ctx.memory.append(memory_entry)
                    
                    if op_type == "finish":
                        self.logger.info(f"Agent decided to finish at step {step}")
                        break
            
            if any(m.get("operation") == "finish" for m in agent_ctx.memory[-len(plan_result):] if plan_result):
                break
        
        self.logger.info(f"Agentic survey revision completed after {len(agent_ctx.memory)} operations")
        self.logger.info(f"Final review scores: {agent_ctx.review_scores}")
        self.logger.info(f"Total revisions made: {agent_ctx.revision_count}")
        self.logger.info(f"[SURVEY MEMORY DEBUG:] {self._format_memory(agent_ctx.memory)}")

        return agent_ctx.current_survey_text


# ============================================================================
# Wrapper Functions
# ============================================================================

def agentic_revise_survey_in_parts(
    survey_generator,
    draft: dict,
    outline: dict,
    max_steps: int,
    code_report: str = None,
    use_agentic: bool = True
) -> dict:
    """
    Agentically review and revise survey sections using the agentic revisor.
    """
    sections = draft.get("section_drafts", []) or []
    if len(sections) == 0:
        survey_generator.logger.error("No sections found in draft for review and revise.")
        raise ValueError("No sections in draft.")
    
    include_code = code_report is not None

    agentic_revisor = AgenticRevisor(
        config=survey_generator.config,
        chat_agent=survey_generator.chat_agent,
        work_analyzer=survey_generator.work_analyzer,
        database=survey_generator.database,
        code_report=code_report,
        include_code = include_code
    )
    
    full_survey_text = draft.get("full_draft", "")
    outline_sections = outline.get('sections', [])
    max_parallel = getattr(survey_generator.config.ModuleInfo.SurveyGenerator, 'revise_section_in_parallel', 1)
    
    if max_parallel <= 1:
        revised_sections = []
        for idx, section_text in enumerate(sections):
            section_outline = outline_sections[idx] if idx < len(outline_sections) else {}
            section_title = section_outline.get('title', 'No Title')
            
            survey_generator.logger.info(
                f"\n\n***** Agentic Reviewing and Revising Section {idx + 1}/{len(sections)}: {section_title} *****"
            )
            
            previous_section_text = sections[idx - 1] if idx > 0 else ""
            next_section_text = sections[idx + 1] if idx + 1 < len(sections) else ""
            
            revised_text = agentic_revisor.agentic_revise_section(
                section_text=section_text,
                previous_section_text=previous_section_text,
                next_section_text=next_section_text,
                section_outline=section_outline,
                section_index=idx + 1,
                total_sections=len(sections),
                full_survey_text=full_survey_text,
                max_steps = max_steps
            )
            
            revised_sections.append(revised_text)
            full_survey_text = outline.get("title", survey_generator.config.BasicInfo.topic + " Survey") + "\n\n" + "\n\n".join(revised_sections)
    else:
        max_parallel = min(max_parallel, len(sections))
        survey_generator.logger.info(f"\n\n***** Agentic Processing {len(sections)} sections in parallel (max {max_parallel} workers) *****")
        
        def revise_section_with_context(idx):
            section_text = sections[idx]
            section_outline = outline_sections[idx] if idx < len(outline_sections) else {}
            section_title = section_outline.get('title', 'No Title')
            
            survey_generator.logger.info(f"Agentic Reviewing and Revising Section {idx + 1}/{len(sections)}: {section_title}")
            
            previous_section_text = sections[idx - 1] if idx > 0 else ""
            next_section_text = sections[idx + 1] if idx + 1 < len(sections) else ""
            
            return agentic_revisor.agentic_revise_section(
                section_text=section_text,
                previous_section_text=previous_section_text,
                next_section_text=next_section_text,
                section_outline=section_outline,
                section_index=idx + 1,
                total_sections=len(sections),
                full_survey_text=full_survey_text,
                max_steps=max_steps
            )
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_to_idx = {executor.submit(revise_section_with_context, idx): idx for idx in range(len(sections))}
            revised_sections = [None] * len(sections)
            
            for future in tqdm(as_completed(future_to_idx), total=len(sections), desc="Agentic revising sections in parallel", unit="section"):
                idx = future_to_idx[future]
                try:
                    revised_sections[idx] = future.result()
                except Exception as exc:
                    survey_generator.logger.error(f"Section {idx} generated an exception: {exc}")
                    revised_sections[idx] = sections[idx]
    
    draft['section_drafts'] = revised_sections
    draft["full_draft"] = outline.get("title", survey_generator.config.BasicInfo.topic + " Survey") + "\n\n" + "\n\n".join(revised_sections)

    return draft


def agentic_revise_survey_whole(
    survey_generator,
    survey: str,
    outline: dict,
    max_steps: int,
    code_report: str = None,
) -> str:
    """
    Agentically review and revise the entire survey.
    """
    
    survey_generator.logger.info("Using agentic approach for whole survey review and revise")
    
    include_code = code_report is not None

    agentic_survey_revisor = AgenticSurveyRevisor(
        config=survey_generator.config,
        chat_agent=survey_generator.chat_agent,
        work_analyzer=survey_generator.work_analyzer,
        database=survey_generator.database,
        code_report=code_report,
        include_code = include_code
    )
    
    revised_survey = agentic_survey_revisor.agentic_revise_survey(
        survey_text=survey,
        outline=outline,
        max_steps = max_steps
    )

    return revised_survey


# ============================================================================
# For backward compatibility - add methods to SurveyGenerator class
# ============================================================================

# def add_agentic_method_to_survey_generator():
#     """Add agentic_revise_survey_in_parts and agentic_revise_survey methods to SurveyGenerator class."""
#     from modules.survey_generator import SurveyGenerator
    
#     SurveyGenerator.agentic_revise_survey_in_parts = lambda self, draft, outline, code_report=None, use_agentic=True: agentic_revise_survey_in_parts(
#         survey_generator=self,
#         draft=draft,
#         outline=outline,
#         code_report=code_report,
#     )
    
#     SurveyGenerator.agentic_revise_survey = lambda self, survey, outline, code_report=None, use_agentic=True: agentic_revise_survey_whole(
#         survey_generator=self,
#         survey=survey,
#         outline=outline,
#         code_report=code_report,
#     )
