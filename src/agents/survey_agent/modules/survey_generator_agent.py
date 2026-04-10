"""
Agentic Survey Revisor: An agent-based approach to review and revise survey sections
using a central planner with multiple skills.
"""

import json
import os
import tiktoken
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
from utils.utils import extract_json
from modules.pe import SECTION_REVIEW, SECTION_REVISE


# Agent prompt for planning operations
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

[Context - Retrieved Content from External Sources]
{context}

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

4. "get_keynote": Retrieve keynote/summary for a specific paper
   - Input: paper_id or paper_title
   - Context to include: what specific aspect to focus on

5. "get_code_report": Include code report content in context
   - Input: specific code report or reference to include

6. "get_full_survey": Include the complete current survey content in context
   - Useful for cross-section coherence checking

7. "finish": Complete the revision for this section when satisfactory

[OPERATION REQUIREMENTS - CRITICAL]
1. You MUST call "review" first to analyze the section before making any revisions
2. After review, call "revise" with suggestions to improve the section
3. You can call "get_pseudocode", "get_keynote", or "get_code_report" to gather additional context
4. Call "get_full_survey" when you need to check cross-section coherence
5. Always call "finish" when the section is satisfactory or no further improvements can be made
6. You should iterate review-revise cycles to progressively improve the section

[Priority Guidelines]
- Focus on the most impactful suggestions first
- Use external knowledge (keynotes, pseudocode) when the section needs more depth or technical accuracy
- Ensure logical flow between sections using previous/next section context

---

Output format (JSON):
{{
    "plan": [
        {{"operation": "...", "input": "...", "reason": "..."}},
        ...
    ]
}}

- Each item in "plan" is one operation to execute in order
- For "revise", "input" should contain the suggestion to apply
- For "get_pseudocode", "input" should be the repo_name
- For "get_keynote", "input" should be the paper_id or paper_title
- For "finish", no input needed
- Include at least 1 operation, up to 3 operations per plan
- You MUST call "review" before "revise" in your first plan

Generate JSON directly without any other things.
"""

# Review skill prompt
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
        "readability": 7,
        "depth": 6,
        "framework": 8
    }},
    "suggestions": [
        "suggestion 1 with specific details",
        "suggestion 2 with specific details",
        ...
    ]
}}
"""

# Revise skill prompt
REVISE_SKILL_PROMPT = """
You are a revise assistant. The section content of a survey concerning {topic} is provided below. The section name is: {section_title}.

You must propose at most ONE exact textual substitution per response according to the suggestion of reviewer.
If you think the document requires changes, choose one that you think is most important to address next, and output ONE JSON object (and nothing else).

**Output format:**
{{
    "action":"replace", 
    "originalText":"<the exact substring to replace>", 
    "newText":"<the replacement text>"
}}

The originalText field must match EXACTLY ONE substring in the document.
If you believe no edits are required, output exactly: {{"action":"done"}}.

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

Reviewer Suggestion:
{reviewer_suggestion}

{external_context}
"""


@dataclass
class AgentContext:
    """Structured input for the Agentic Revisor."""
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
    context: str = ""  # External context from skills
    context_summary: str = ""  # Brief summary of current context
    review_scores: Dict = field(default_factory=dict)  # {"readability": 0-10, "depth": 0-10, "framework": 0-10}
    current_suggestions: List[str] = field(default_factory=list)
    has_reviewed: bool = False  # Track if review has been called
    revision_count: int = 0  # Track number of revisions made


def compress_memory(memory: List[Dict], encodings=None) -> List[Dict]:
    """
    Compress memory by removing the oldest half of entries.
    
    Args:
        memory: List of memory entries
        encodings: Optional tiktoken encodings for token counting
        
    Returns:
        Compressed memory list (oldest half removed)
    """
    if len(memory) <= 1:
        return memory
    
    mid_point = len(memory) // 2
    compressed = memory[mid_point:]
    
    logger = get_logger("AgenticRevisor")
    logger.info(f"Memory compressed: {len(memory)} -> {len(compressed)} entries (removed oldest {mid_point} entries)")
    
    return compressed


def count_tokens(text: str, encodings=None) -> int:
    """Count tokens in text using tiktoken encoding."""
    if encodings is None:
        return len(text) // 4
    try:
        return len(encodings.encode(text))
    except Exception:
        return len(text) // 4


def truncate_text(text: str, max_length: int, suffix: str = "(truncated)...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if not text:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_memory(memory: List[Dict], memory_result_max_length: int = 4000) -> str:
    """
    Format memory (operation history with results) as string.
    
    Different operation types have different formatting:
    - review: shows scores and suggestions
    - revise: shows applied revision
    - get_pseudocode/get_keynote: shows retrieved content
    - finish: shows final status
    
    Args:
        memory: List of memory entries
        memory_result_max_length: Max characters for memory results
        
    Returns:
        Formatted memory string
    """
    if not memory:
        return "No operations performed yet."
    
    lines = []
    for i, op in enumerate(memory):
        op_type = op.get('operation', '')
        reason = op.get('reason', '')
        
        lines.append(f"{i+1}. {op_type}: {reason}")
        
        # Format based on operation type
        if op_type == "review":
            scores = op.get('scores', {})
            suggestions = op.get('suggestions', [])
            if scores:
                lines.append(f"   Scores: readability={scores.get('readability', 0)}, depth={scores.get('depth', 0)}, framework={scores.get('framework', 0)}")
            if suggestions:
                suggestions_text = "; ".join(suggestions[:5])  # Limit to 5 suggestions
                if len(suggestions_text) > memory_result_max_length // 2:
                    suggestions_text = suggestions_text[:memory_result_max_length // 2] + "(truncated)..."
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
                if len(context_result) > memory_result_max_length:
                    context_result = context_result[:memory_result_max_length] + "(truncated)..."
                lines.append(f"   Retrieved for {target}: {context_result}")
            result = op.get('result', '')
            if result:
                lines.append(f"   Result: {result}")
                
        elif op_type == "get_code_report":
            lines.append(f"   Result: {op.get('result', '')}")
            
        elif op_type == "get_full_survey":
            survey_content = op.get('survey_content', '')
            if survey_content:
                if len(survey_content) > memory_result_max_length:
                    survey_content = survey_content[:memory_result_max_length] + "(truncated)..."
                lines.append(f"   Survey content ({len(survey_content)} chars): {survey_content}")
            result = op.get('result', '')
            if result:
                lines.append(f"   Result: {result}")
                
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
            # Generic fallback
            result = op.get('result', '')
            if result:
                if len(result) > memory_result_max_length:
                    result = result[:memory_result_max_length] + "(truncated)..."
                lines.append(f"   Result: {result}")
    
    return "\n".join(lines)


class AgenticRevisor:
    """
    Agent-based survey section revisor that uses a planner with various skills
    to review and revise survey sections.
    """
    
    def __init__(self, config, chat_agent: ChatAgent, work_analyzer, database, code_report: str = None):
        self.chat_agent = chat_agent
        self.logger = get_logger("AgenticRevisor")
        self.config = config
        self.work_analyzer = work_analyzer
        self.database = database
        self.code_report = code_report
        
        # Configuration
        self.default_max_steps = 15
        self.memory_token_threshold = 40000
        self.use_memory_result_limit = True
        self.memory_result_max_length = 4000
        self.section_review_retry = config.ModuleInfo.SurveyGenerator.section_review_retry
        self.section_revise_retry = config.ModuleInfo.SurveyGenerator.section_revise_retry
        self.section_review_temperature = config.ModuleInfo.SurveyGenerator.section_review_temperature
        self.section_revise_temperature = config.ModuleInfo.SurveyGenerator.section_revise_temperature
        self.section_revision_RAG_topk = config.ModuleInfo.SurveyGenerator.section_revision_RAG_topk
        self.section_least_words = config.ModuleInfo.SurveyGenerator.section_least_words or "no limit"
        
        # Initialize tiktoken encoding for token counting
        try:
            self.encodings = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.logger.warning("Failed to load tiktoken encoding, using character-based estimation")
            self.encodings = None
    
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
    
    def _get_pseudocode(self, repo_name: str, context_focus: str = "") -> str:
        """Retrieve pseudocode for a repository."""
        try:
            # Try to read from cache
            from modules.code_collector import CodeAnalyzer, CodeCollector
            code_collector = CodeCollector(self.config)
            repo_cache_path = code_collector.repo_cache_path
            repo_dir = os.path.join(repo_cache_path, repo_name)
            
            # If repo directory doesn't exist, return not available
            if not os.path.isdir(repo_dir):
                self.logger.warning(f"Repository {repo_name} not found in cache")
                return f"[Pseudocode for {repo_name} not available]"
            
            # Try exact model_name match first, then fallback to any available model
            model_name = self.chat_agent.model_name
            
            # Try both full and concise pseudocode with exact model_name first
            for suffix in ["", "_concise"]:
                cache_path = os.path.join(repo_dir, f"pseudocode_{model_name}{suffix}.json")
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        pseudocode = f.read()
                    self.logger.info(f"Retrieved pseudocode for {repo_name} from cache (exact model match)")
                    if context_focus:
                        return f"[Pseudocode for {repo_name} - focusing on: {context_focus}]\n{pseudocode}"
                    return f"[Pseudocode for {repo_name}]\n{pseudocode}"
            
            # Fallback: find any available pseudocode file in the repo directory
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
            
            # If no pseudocode found, try mainfest
            for filename in os.listdir(repo_dir):
                if filename.startswith("mainfest_") and filename.endswith(".json"):
                    mainfest_path = os.path.join(repo_dir, filename)
                    with open(mainfest_path, "r", encoding="utf-8") as f:
                        import json
                        mainfest = json.load(f)
                    self.logger.info(f"Found mainfest for {repo_name}, no cached pseudocode")
                    return f"[Repository {repo_name} structure available but pseudocode not yet generated]"
            
            self.logger.warning(f"No pseudocode found for {repo_name}")
            return f"[Pseudocode for {repo_name} not available]"
        except Exception as e:
            self.logger.error(f"Error retrieving pseudocode for {repo_name}: {e}")
            return f"[Error retrieving pseudocode for {repo_name}: {e}]"
    
    def _get_keynote(self, paper_id_or_title: str, context_focus: str = "") -> str:
        """Retrieve keynote for a paper."""
        try:
            # First try to get by paper_id
            try:
                keynote = self.work_analyzer.get_paper_keynote(paper_id_or_title)
                self.logger.info(f"Retrieved keynote for paper ID {paper_id_or_title}")
                if context_focus:
                    return f"[Keynote for {paper_id_or_title} - focusing on: {context_focus}]\n{keynote}"
                return f"[Keynote for paper {paper_id_or_title}]\n{keynote}"
            except Exception:
                pass
            
            # Try to get by title
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
                response = self.chat_agent.remote_chat(
                    prompt,
                    temperature=self.section_review_temperature,
                )
                result = extract_json(response)
                
                if isinstance(result, dict) and "scores" in result and "suggestions" in result:
                    scores = result.get("scores", {})
                    suggestions = result.get("suggestions", [])
                    
                    # Validate scores
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
                response = self.chat_agent.remote_chat(
                    prompt,
                    temperature=self.section_revise_temperature,
                )
                parsed_json = extract_json(response)
                
                if not isinstance(parsed_json, dict):
                    raise ValueError("Parsed JSON is not a dict.")
                
                if "###" in parsed_json.get("newText", "") or "###" in parsed_json.get("originalText", ""):
                    raise ValueError("Revision contains '###' which may break markdown structure.")
                
                valid = True
                break
                
            except Exception as e:
                self.logger.error(f"Revise attempt {retry_time + 1} failed: {e}")
                continue
        
        if not valid or parsed_json is None:
            raise ValueError("Failed to generate valid revision after retries")
        
        # Apply revision to text
        if parsed_json.get("action") == "done":
            return agent_ctx.current_section_text, parsed_json
        elif parsed_json.get("action") == "replace":
            new_text = self._apply_revision_to_text(agent_ctx.current_section_text, parsed_json)
            return new_text, parsed_json
        else:
            raise ValueError(f"Invalid revision action: {parsed_json.get('action')}")
    
    def _check_memory_compression(self, agent_ctx: AgentContext) -> bool:
        """Check if memory needs compression based on token count."""
        memory_str = format_memory(agent_ctx.memory)
        token_count = count_tokens(memory_str, self.encodings)
        
        if token_count > self.memory_token_threshold:
            self.logger.info(f"Memory token count ({token_count}) exceeds threshold ({self.memory_token_threshold}), compressing...")
            agent_ctx.memory = compress_memory(agent_ctx.memory, self.encodings)
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
    
    def _execute_operation(self, op: dict, agent_ctx: AgentContext, full_survey_text: str = "") -> Tuple[str, Dict]:
        """
        Execute a single operation and return the operation type and memory entry.
        
        Returns:
            Tuple of (op_type, memory_entry)
        """
        operation = op.get("operation", "")
        input_data = op.get("input", "")
        reason = op.get("reason", "")
        
        memory_entry = {"operation": operation, "reason": reason}
        
        if operation == "review":
            self.logger.info(f"Calling review skill for section {agent_ctx.section_index}")
            result = self._call_review(agent_ctx)
            
            agent_ctx.review_scores = result.get("scores", {})
            agent_ctx.current_suggestions = result.get("suggestions", [])
            agent_ctx.has_reviewed = True
            
            memory_entry["scores"] = result.get("scores", {})
            memory_entry["suggestions"] = result.get("suggestions", [])
            memory_entry["result"] = f"Reviewed with scores: {result.get('scores', {})}"
            
        elif operation == "revise":
            if not agent_ctx.has_reviewed:
                self.logger.warning("revise called before review - calling review first")
                review_op = {"operation": "review", "reason": "Forced review before revise"}
                self._execute_operation(review_op, agent_ctx, full_survey_text)
            
            suggestion = input_data
            self.logger.info(f"Calling revise skill with suggestion: {truncate_text(suggestion, 100)}")
            
            new_text, revision_result = self._call_revise(agent_ctx, suggestion)
            
            if new_text != agent_ctx.current_section_text:
                agent_ctx.current_section_text = new_text
                agent_ctx.revision_count += 1
                memory_entry["result"] = f"Applied revision: {revision_result.get('action', 'unknown')}"
            else:
                memory_entry["result"] = "No change made"
            
            # Clear context after revision
            agent_ctx.context = ""
            agent_ctx.context_summary = ""
            
        elif operation == "get_pseudocode":
            repo_name = input_data
            self.logger.info(f"Retrieving pseudocode for repo: {repo_name}")
            
            context_result = self._get_pseudocode(repo_name)
            agent_ctx.context += f"\n\n=== Pseudocode: {repo_name} ===\n{context_result}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, 500)
            
            memory_entry["repo_name"] = repo_name
            memory_entry["context_result"] = context_result
            memory_entry["result"] = f"Retrieved pseudocode for {repo_name}"
            
        elif operation == "get_keynote":
            paper_id_or_title = input_data
            self.logger.info(f"Retrieving keynote for paper: {paper_id_or_title}")
            
            context_result = self._get_keynote(paper_id_or_title)
            agent_ctx.context += f"\n\n=== Paper Keynote: {paper_id_or_title} ===\n{context_result}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, 500)
            
            memory_entry["paper_id"] = paper_id_or_title
            memory_entry["context_result"] = context_result
            memory_entry["result"] = f"Retrieved keynote for {paper_id_or_title}"
            
        elif operation == "get_code_report":
            self.logger.info("Adding code report to context")
            
            if self.code_report:
                agent_ctx.context += f"\n\n=== Code Report ===\n{self.code_report}\n"
                agent_ctx.context_summary = truncate_text(agent_ctx.context, 500)
                memory_entry["result"] = "Added code report to context"
            else:
                memory_entry["result"] = "No code report available"
            
        elif operation == "get_full_survey":
            self.logger.info(f"Adding full survey to context ({len(full_survey_text)} chars)")
            
            agent_ctx.context += f"\n\n=== Full Survey ===\n{full_survey_text}\n"
            agent_ctx.context_summary = truncate_text(agent_ctx.context, 500)
            
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
        """
        Agentically review and revise a section using the planner agent.
        
        Args:
            section_text: Current section text
            previous_section_text: Previous section text for context
            next_section_text: Next section text for context
            section_outline: Section outline dict
            section_index: Index of current section (1-based)
            total_sections: Total number of sections
            full_survey_text: Complete survey text for context
            max_steps: Maximum number of agent steps
            
        Returns:
            Revised section text
        """
        # Build AgentContext
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
        
        # Add code report to initial context if available
        if self.code_report:
            agent_ctx.context = f"\n\n=== Code Report ===\n{self.code_report}\n"
            agent_ctx.context_summary = "Code report available for technical context"
        
        max_steps = max_steps or self.default_max_steps
        
        self.logger.info(f"Starting agentic revision for section {section_index}/{total_sections}: {agent_ctx.section_title}")
        
        for step in range(max_steps):
            self.logger.info(f"Agent step {step + 1}/{max_steps}")

            # Check memory compression before building prompt
            self._check_memory_compression(agent_ctx)

            # Build the prompt
            prompt = AGENT_OPERATE_PROMPT.format(
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
                memory=format_memory(agent_ctx.memory),
                context=agent_ctx.context if agent_ctx.context else "No external context retrieved yet.",
                context_summary=agent_ctx.context_summary if agent_ctx.context_summary else "No context available"
            )

            # Get agent's plan
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
            
            # Execute each operation in the plan
            for op in plan_result:
                op_type, memory_entry = self._execute_operation(op, agent_ctx, full_survey_text)
                agent_ctx.memory.append(memory_entry)
                
                # Check for finish
                if op_type == "finish":
                    self.logger.info(f"Agent decided to finish at step {step}")
                    break
            
            # Check if finished
            if any(m.get("operation") == "finish" for m in agent_ctx.memory[-len(plan_result):] if plan_result):
                break
        
        self.logger.info(f"Agentic revision completed for section {section_index} after {len(agent_ctx.memory)} operations")
        self.logger.info(f"Final review scores: {agent_ctx.review_scores}")
        self.logger.info(f"Total revisions made: {agent_ctx.revision_count}")

        return agent_ctx.current_section_text


def agentic_revise_survey_in_parts(
    survey_generator,
    draft: dict,
    outline: dict,
    code_report: str = None,
    use_agentic: bool = True
) -> dict:
    """
    Agentically review and revise survey sections using the agentic revisor.
    
    This function has the EXACT SAME interface as review_and_revise_survey_in_parts
    in survey_generator.py, ensuring backward compatibility.
    
    Args:
        survey_generator: The SurveyGenerator instance (provides config, chat_agent, work_analyzer, database)
        draft: Draft dict with "section_drafts", "full_draft", etc.
        outline: Outline dict with sections
        code_report: Optional code report to include in context
        use_agentic: Whether to use the agentic approach (default: True)
        
    Returns:
        Revised draft dict with updated section_drafts and full_draft
    """
    if not survey_generator.config.ModuleInfo.SurveyGenerator.enable_review_and_revise:
        survey_generator.logger.info("Review and revise module is disabled. Skipping...")
        return draft
    
    sections = draft.get("section_drafts", []) or []
    if len(sections) == 0:
        survey_generator.logger.error("No sections found in draft for review and revise.")
        raise ValueError("No sections in draft.")
    
    # If not using agentic approach, fall back to the original method
    if not use_agentic:
        survey_generator.logger.info("Using non-agentic review and revise approach")
        return survey_generator.review_and_revise_survey_in_parts(draft, outline)
    
    survey_generator.logger.info("Using agentic review and revise approach")
    
    # Create agentic revisor
    agentic_revisor = AgenticRevisor(
        config=survey_generator.config,
        chat_agent=survey_generator.chat_agent,
        work_analyzer=survey_generator.work_analyzer,
        database=survey_generator.database,
        code_report=code_report
    )
    
    # Get full survey text for context
    full_survey_text = draft.get("full_draft", "")
    
    revised_sections = []
    outline_sections = outline.get('sections', [])
    
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
            full_survey_text=full_survey_text
        )
        
        revised_sections.append(revised_text)
        
        # Update full survey text for next section's context
        full_survey_text = outline.get("title", survey_generator.config.BasicInfo.topic + " Survey") + "\n\n" + "\n\n".join(revised_sections)
    
    draft['section_drafts'] = revised_sections
    draft["full_draft"] = outline.get("title", survey_generator.config.BasicInfo.topic + " Survey") + "\n\n" + "\n\n".join(revised_sections)

    return draft


# For backward compatibility - add the method to SurveyGenerator class
def add_agentic_method_to_survey_generator():
    """Add agentic_revise_survey_in_parts method to SurveyGenerator class."""
    from modules.survey_generator import SurveyGenerator
    
    SurveyGenerator.agentic_revise_survey_in_parts = lambda self, draft, outline, code_report=None, use_agentic=True: agentic_revise_survey_in_parts(
        survey_generator=self,
        draft=draft,
        outline=outline,
        code_report=code_report,
        use_agentic=use_agentic
    )
