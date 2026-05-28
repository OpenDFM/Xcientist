"""
Code Report Generator Module

This module provides functionality to generate comprehensive code analysis reports
from multiple papers' pseudocode, organized by different dimensions.

Input format: List[dict] where each dict contains:
    - "paper_id": str - the paper identifier
    - "pseudocode_report": str - the pseudocode report for that paper
"""

import os
import sys
import json
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.config_utils import merge_with_default_survey_config
from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
# from modules.work_collector import WorkCollector
# from modules.code_collector import CodeCollector, CodeAnalyzer


# Prompt template for batch code report generation
# All prompts are in English as requested
BATCH_CODE_REPORT_PROMPT = """
You are an expert at analyzing research paper code implementations and generating comprehensive reports with forward-looking guidance.

Your task is to analyze a batch of papers on the topic: "{topic}"

For each paper, you have basic paper information and code pseudocode reports:

{pseudocode_reports}

---

TASK: Generate a comprehensive code analysis report for this batch of papers, with emphasis on identifying gaps and future research directions.

IMPORTANT: Go beyond surface-level descriptions. Analyze the ACTUAL IMPLEMENTATION DETAILS from the pseudocode.

## Report Structure Requirements:

### 1. Executive Summary
- What are the dominant implementation patterns?
- What are the major gaps and underexplored opportunities?
- What are the most promising future research directions?

### 2. Data Structure Classification and Problem Modeling
- Analyze how different papers model the problem and what data structures they use
- Categorize them into meaningful groups
- For each data structure: WHY it was chosen, WHAT trade-offs it involves

### 3. Core Algorithm Classification
- Identify patterns in core algorithms across papers
- For each algorithm provide:
  * Specific function/class names from the pseudocode
  * Step-by-step logic flow (input → processing → output)
  * Key hyperparameters and their purposes
  * Concrete implementation details (e.g., HOW attention is computed, not just "uses attention")

### 4. Optimization and Acceleration Strategy Classification
- Computational optimizations (vectorization, batching, caching)
- Memory optimizations (gradient checkpointing, mixed precision)
- Training tricks and convergence improvements
- For each optimization: explain the MECHANISM and EXPECTED IMPACT

### 5. Custom Dimensions (Topic-Specific)
- For LLM topics: workflow management, context window handling, agent architectures
- For RL topics: reward shaping, exploration mechanisms, credit assignment
- For CV topics: data augmentation pipelines, architecture patterns
- Add other meaningful dimensions based on code patterns observed

### 6. Critical Gap Analysis
For each gap identified:
- Specific description of what's missing or underexplored
- Evidence from the code implementations supporting this gap
- Why this gap matters for the field

### 7. Future Research Directions
Based on gap analysis, provide concrete future directions:
- Specific research problems to tackle
- Potential approaches to address each gap
- Expected challenges and how to overcome them
- Connections to implementation patterns observed in the code

## General Requirements:
- Each claim should be traceable to specific parts of the pseudocode(Most important)
- Compare approaches with SPECIFIC technical differences
- Include actual code details: variable names, function names, class names

Output format:
Provide a structured report with clear sections for each dimension.
Emphasize the Future Research Directions section.
"""
INTEGRATED_REPORT_PROMPT = """
You are an expert at synthesizing multiple batch reports into a comprehensive integrated report.

You have already generated reports for several batches of papers on the topic: "{topic}"

Here are the batch reports:
{combined_reports}

---

TASK: Integrate all batch reports into a single comprehensive report

IMPORTANT: Maintain the depth and specificity from individual batch reports. Do NOT simplify or generalize the implementation details.

## Required Report Structure:

### 1. Executive Summary
- Dominant implementation patterns across all papers
- Major gaps and underexplored opportunities
- Most promising future research directions

### 2. Problem Modeling and Data Structure Classification
- Unified view of problem modeling approaches
- Data structure categories with specific examples
- Trade-offs analysis preserved from original reports

### 3. Core Algorithm Classification
- Algorithm patterns across all papers
- Specific function/class names, logic flows, hyperparameters
- Implementation details with code-level examples

### 4. Optimization and Acceleration Strategy Classification
- Optimization techniques identified
- Specific mechanisms and expected impacts
- Computational and memory optimizations

### 5. Custom Dimensions (Topic-Specific)
- Workflow patterns (for LLM topics)
- Agent architectures and context handling
- Other topic-relevant dimensions

### 6. Critical Gap Analysis (IMPORTANT)
For each gap identified:
- Specific description of what's missing
- Evidence from implementation patterns
- Why this gap matters

### 7. Future Research Directions (MOST IMPORTANT)
Based on gap analysis, provide:
- Specific research problems to tackle
- Potential approaches to address gaps
- Expected challenges and solutions
- Connections to observed implementation patterns

### 8. Conclusion
- Key takeaways from the synthesis
- Forward-looking insights

The final report should be:
- Comprehensive: Cover all important aspects from all batches
- Coherent: Flow logically from section to section
- Non-redundant: Avoid repeating same points when merging batches
- Specific: Provide meaningful synthesis with actual implementation details when necessary
- Actionable: Emphasize future research directions
- Concise: Avoid keeping trivial parts when intergrating batches
- Deep: Integrate with analysis rather than simple enumeration

Output format:
Provide a well-structured comprehensive report with clear sections.
The report should be at most 6000 words and 700 lines
Emphasize the Future Research Directions section.
"""


BATCH_FRAMEWORK_ENV_PROMPT = """
You are an expert at analyzing research paper code implementations and providing guidance on framework selection and environment configuration.

Your task is to analyze a batch of repositories to generate guidance on framework selection and environment setup.

For each repository, you have:
1. Paper general information: repository name, paper title, paper abstract, paper pseudocode report
2. Paper environment information: requirements content, README content: 
{combined_content}
---

TASK: Generate framework selection and environment configuration guidance for this batch

Required Content:

1. **Framework Analysis**:
   - Identify frameworks used (PyTorch, TensorFlow, JAX, etc.)
   - Analyze version requirements and compatibility
   - Assess framework suitability for different scenarios

2. **Environment Configuration**:
   - Key dependencies and their versions
   - Special requirements (GPU, CUDA, etc.)
   - Setup best practices

3. **Base Framework Assessment**:
   - Code quality and documentation quality
   - Extensibility and maintainability
   - Suitability for different research directions

Requirements:
- Comprehensive: Cover all important aspects mentioned above
- Coherent: Flow logically from section to section
- Specific: Provide concrete examples from the repositories
- Concise: Avoid keeping trivial parts when intergrating batches

Output format:
Provide a structured report with clear sections.
"""

INTEGRATED_FRAMEWORK_ENV_PROMPT = """
You are an expert at synthesizing multiple batch reports into a comprehensive framework and environment guidance report.

You have already generated reports for several batches of repositories on the topic: "{topic}"

Here are the batch reports:
{combined_reports}

---

TASK: Integrate all batch reports into a comprehensive framework selection and environment configuration guide

Requirements:
1. Synthesize information from all batches into a unified report
2. Remove duplicates and consolidate similar findings
3. Ensure all key insights from each batch are preserved
4. Structure the report with the following sections:
   - Framework Selection Analysis
   - Environment Configuration Guidance
   - Base Framework Recommendations
   - Sub-direction Analysis
5. Add a summary section at the beginning
6. Add a conclusion section with actionable recommendations

The final report should be:
- Practical: Provide actionable recommendations
- Comprehensive: Cover all important aspects
- Well-organized: Clear headings and structured sections
- Non-redundant: Avoid repeating the same points
- Coherent: Flow logically from section to section
- Actionable: Emphasize future research directions
- Concise: Avoid keeping trivial parts when intergrating batches
- Deep: Integrate with analysis rather than simple enumeration

Output format:
Provide a well-structured comprehensive report with clear sections.
The report should be at most 5000 words and 700 lines
"""

class CodeReportGenerator:
    """
    A class to generate comprehensive code analysis reports from multiple papers.
    
    This generator:
    1. Takes input as List[dict] with paper_id and pseudocode_report
    2. Processes them in batches
    3. Generates dimension-specific analysis for each batch directly from pseudocode reports
    4. Integrates all batch reports into a final comprehensive report
    """
    
    def __init__(self, config, 
                 work_collector = None, 
                 code_collector = None,
                 code_analyzer = None):
        """
        Initialize the CodeReportGenerator.
        
        Args:
            config: Configuration object containing API settings
            work_collector: Optional WorkCollector instance (will create if not provided)
            code_collector: Optional CodeCollector instance
            code_analyzer: Optional CodeAnalyzer instance
        """
        self.config = config
        self.logger = get_logger("CodeReportGenerator")
        self.chat_agent = ChatAgent(config)
        self.work_collector = work_collector
        self.max_context_chars_for_integrate = int(self.config.APIInfo.llm_max_context_length * 2.5)
        
        # Code collector and analyzer (optional, for backwards compatibility)
        self.code_collector = code_collector
        self.code_analyzer = code_analyzer
        
        # Cache path for pseudocode (if needed)
        if self.code_collector:
            self.repo_cache_path = self.code_collector.repo_cache_path
        else:
            raise ValueError("codereporter lack valid code_collector to initialize")
        
        # Batch size for processing papers
        self.batch_size = 5
        
        # Common file names for requirements and readme
        self.requirement_files = ['requirements.txt', 'requirements.txt', 'setup.py', 'pyproject.toml', 'setup.cfg']
        self.readme_files = ['README.md', 'README.rst', 'README.txt', 'README', "readme.txt", "readme.md", "readme.rst", "readme"]
    
    def read_repo_files(self, repo_name: str) -> Dict[str, Dict[str, str]]:
        """
        Read requirement and README files from a cloned repository.
        
        Args:
            repo_name: The name of the repository (folder name in cache)
            
        Returns:
            Dict containing 'requirements' and 'readme' content:
            {
                'requirements': {'file_name': 'content' or 'NOT_FOUND'},
                'readme': {'file_name': 'content' or 'NOT_FOUND'}
            }
        """
        repo_path = os.path.join(self.repo_cache_path, repo_name)
        
        if not os.path.exists(repo_path):
            self.logger.error(f"Repository not found in cache: {repo_path}")
            return {
                'requirements': {'NOT_FOUND': 'Repository not found'},
                'readme': {'NOT_FOUND': 'Repository not found'}
            }
        
        result = {
            'requirements': {},
            'readme': {}
        }
        
        # Search for requirements files
        for req_file in self.requirement_files:
            req_path = os.path.join(repo_path, req_file)
            if os.path.exists(req_path):
                try:
                    with open(req_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        result['requirements'][req_file] = content
                        self.logger.info(f"Found requirements file: {req_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to read {req_file}: {e}")
                    result['requirements'][req_file] = f"Error reading file: {str(e)}"
        
        # If no requirements file found, search for any file containing 'requirements'
        if not result['requirements']:
            try:
                for fname in os.listdir(repo_path):
                    if 'requirements' in fname.lower() and not fname.startswith('.'):
                        req_path = os.path.join(repo_path, fname)
                        if os.path.isfile(req_path):
                            try:
                                with open(req_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    result['requirements'][fname] = content
                                    self.logger.info(f"Found requirements file: {fname}")
                            except Exception as e:
                                self.logger.warning(f"Failed to read {fname}: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to list directory: {e}")
        
        # If still no requirements found, mark as NOT_FOUND
        if not result['requirements']:
            result['requirements']['NOT_FOUND'] = "No requirements file found"
        
        # Search for README files
        for readme_file in self.readme_files:
            readme_path = os.path.join(repo_path, readme_file)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Limit README content to first 15000 chars to avoid too long input
                        if len(content) > 15000:
                            content = content[:15000] + "\n\n... (truncated)"
                        result['readme'][readme_file] = content
                        self.logger.info(f"Found README file: {readme_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to read {readme_file}: {e}")
                    result['readme'][readme_file] = f"Error reading file: {str(e)}"
        
        # If no README found, search for any file starting with 'README'
        if not result['readme']:
            try:
                for fname in os.listdir(repo_path):
                    if fname.lower().startswith('readme') and not fname.startswith('.'):
                        readme_path = os.path.join(repo_path, fname)
                        if os.path.isfile(readme_path):
                            try:
                                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if len(content) > 15000:
                                        content = content[:15000] + "\n\n... (truncated)"
                                    result['readme'][fname] = content
                                    self.logger.info(f"Found README file: {fname}")
                            except Exception as e:
                                self.logger.warning(f"Failed to read {fname}: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to list directory: {e}")
        
        # If still no README found, mark as NOT_FOUND
        if not result['readme']:
            result['readme']['NOT_FOUND'] = "No README file found"
        
        return result
    
    def _generate_batch_framework_env_report(self, 
                                             topic: str, 
                                             repo_data_batches: List[List[Dict]],
                                             max_readme: int = 1,
                                             max_requirements: int = 2,
                                             max_batches: int = 20) -> List[str]:
        """
        Generate framework and environment guidance reports for multiple batches of repositories.
        
        Args:
            topic: The research topic
            repo_data_batches: List of batches, where each batch is a list of repo data dictionaries
            
        Returns:
            List of generated batch report strings
        """
        if not repo_data_batches:
            self.logger.warning("No repository data batches provided")
            return []
        
        self.logger.info(f"Processing {len(repo_data_batches)} batches of repositories")
        
        # Build prompts for each batch
        prompts = []
        batch_info = []  # Track info for logging
        
        for batch_idx, repo_data in enumerate(repo_data_batches):
            if not repo_data:
                self.logger.warning(f"Empty batch at index {batch_idx}, skipping")
                continue
            
            # Build the content for this batch
            repos_content = []
            
            for i, repo in enumerate(repo_data):
                repo_name = repo.get('repo_name', f'Repo_{i+1}')
                requirements = repo.get('requirements', {})
                readme = repo.get('readme', {})
                paper_abstract = repo.get('paper_abstract', "")
                paper_title = repo.get('paper_title', "")
                pseudocode = repo.get('pseudocode', 'pseudocode not found')
                
                # Format requirements content
                req_content = ""
                req_num = 0
                for fname, content in requirements.items():
                    if fname != 'NOT_FOUND':
                        req_content += f"\n--- {fname} ---\n{content}\n"
                        req_num += 1
                    if req_num >= max_requirements:
                        break
                if not req_content:
                    req_content = requirements.get('NOT_FOUND', 'No requirements found')
                
                # Format README content
                readme_content = ""
                readme_num = 0
                for fname, content in readme.items():
                    if fname != 'NOT_FOUND':
                        readme_content += f"\n--- {fname} ---\n{content}\n"
                        readme_num += 1
                    if readme_num >= max_readme:
                        break
                if not readme_content:
                    readme_content = readme.get('NOT_FOUND', 'No README found')
                
                content = f"=== Repository {i+1}: {repo_name} ===\n"
                content += f"[Repository Information]:(for reference and understanding the repo)\n"
                content += f"Paper_title: {paper_title}\nPaper abstract: {paper_abstract}\n"
                content += f"Paper_Pseudocode: {pseudocode}\n"
                content += f"[Readme and Requiremnets]:(report core contents source)\n"
                content += f"Requirements:\n{req_content}\n\n"
                content += f"README:\n{readme_content}"
                repos_content.append(content)
            
            combined_content = "\n\n".join(repos_content)
            
            prompt = BATCH_FRAMEWORK_ENV_PROMPT.format(
                combined_content = combined_content
            )
            prompts.append(prompt)
            batch_info.append({
                'batch_idx': batch_idx,
                'num_repos': len(repo_data),
                'repo_names': [r.get('repo_name', f'Repo_{i}') for i, r in enumerate(repo_data)]
            })
            
            self.logger.info(f"Prepared batch {batch_idx + 1}/{len(repo_data_batches)} with {len(repo_data)} repositories")
        
        if not prompts:
            self.logger.error("No valid prompts generated from batches")
            return []
        
        self.logger.info(f"Generating {len(prompts)} framework/env batch reports via API...")
        
        def _env_report_validate_fn(result, info_dict = None):
            if not isinstance(result, str):
                raise TypeError("batch env_report result not string")
            if len(result) < 100:
                raise ValueError("batch env_report result too short")
            return True, result
        
        try:
            results = self.chat_agent.batch_remote_chat_with_retry(
                prompts=prompts,
                validate_fn=_env_report_validate_fn,
                max_retry=6,
                desc="Generating batch env reports",
                temperature=0.3,
            )
            
            # Log success info
            if isinstance(results, list):
                self.logger.info(f"Successfully generated {len(results)} framework/env batch reports")
                for i, result in enumerate(results):
                    if result:
                        self.logger.info(f"Framework batch {i+1} report length: {len(result)} chars")
                    else:
                        self.logger.warning(f"Framework batch {i+1} returned empty result")
            else:
                self.logger.warning(f"Unexpected result type from batch_remote_chat_with_retry: {type(results)}")
            
            return results if isinstance(results, list) else []
            
        except Exception as e:
            self.logger.error(f"Failed to generate framework/env reports: {e}")
            return []
    
    def _integrate_framework_env_reports(self, topic: str, batch_reports: List[str], max_context_chars: int = None) -> str:
        """
        Integrate multiple batch reports into a final comprehensive framework/env report.
        Uses multi-round fallback when combined reports exceed context window.
        
        Args:
            topic: The research topic
            batch_reports: List of batch report strings
            max_context_chars: Maximum characters per integration call (default: 150000)
            
        Returns:
            The integrated report
        """
        if max_context_chars is None:
            max_context_chars = self.max_context_chars_for_integrate

        if not batch_reports:
            return "No reports to integrate."
        
        if len(batch_reports) == 1:
            # Only one batch, no integration needed
            return batch_reports[0]
        
        # First attempt: try integrating all reports at once
        combined_reports = "\n\n==== BATCH SEPARATOR ====\n\n".join(
            f"=== Batch {i+1} ===\n{report}" 
            for i, report in enumerate(batch_reports)
        )
        
        # Check if combined content fits in context
        prompt_length = len(INTEGRATED_FRAMEWORK_ENV_PROMPT.format(topic=topic, combined_reports=combined_reports))
        
        if prompt_length <= max_context_chars:
            self.logger.info("Integrating framework/env batch reports into final report...")
            try:
                result = self.chat_agent.remote_chat(
                    text_content=INTEGRATED_FRAMEWORK_ENV_PROMPT.format(topic=topic, combined_reports=combined_reports),
                    temperature=0.3
                )
                return result
            except Exception as e:
                self.logger.warning(f"Failed to integrate all reports at once: {e}")
                # Fall through to multi-round fallback
        else:
            self.logger.info(f"Combined reports exceed context ({prompt_length} chars), using multi-round fallback...")
        
        # Multi-round fallback: integrate in smaller groups
        return self._multi_round_integrate_framework_env(topic, batch_reports, max_context_chars)
    
    def _multi_round_integrate_framework_env(self, topic: str, batch_reports: List[str], max_context_chars: int = None) -> str:
        """
        Multi-round integration when all batch reports don't fit in context.
        Integrates reports in pairs/groups, then recursively integrates results.
        
        Args:
            topic: The research topic
            batch_reports: List of batch report strings
            max_context_chars: Maximum characters per integration call
            
        Returns:
            The integrated report
        """
        self.logger.info(f"Starting multi-round integration for {len(batch_reports)} batch reports...")
        
        # Base case: if only one or two reports, integrate directly
        if len(batch_reports) <= 2:
            combined = "\n\n==== BATCH SEPARATOR ====\n\n".join(
                f"=== Batch {i+1} ===\n{report}" 
                for i, report in enumerate(batch_reports)
            )
            try:
                result = self.chat_agent.remote_chat(
                    text_content=INTEGRATED_FRAMEWORK_ENV_PROMPT.format(topic=topic, combined_reports=combined),
                    temperature=0.3
                )
                return result
            except Exception as e:
                self.logger.error(f"Failed to integrate: {e}")
                return "\n\n".join(batch_reports)
        
        # Group reports into pairs that fit within context
        groups = []
        current_group = []
        current_group_size = 0
        
        for i, report in enumerate(batch_reports):
            report_size = len(report)
            separator_size = len(f"\n\n==== BATCH SEPARATOR ====\n\n=== Batch {len(current_group) + 1} ===\n")
            
            # Check if adding this report would exceed context
            # Reserve space for prompt template (~500 chars)
            if current_group_size + report_size + separator_size > int(max_context_chars*0.9):
                if current_group:
                    groups.append(current_group)
                current_group = [report]
                current_group_size = report_size
            else:
                current_group.append(report)
                current_group_size += report_size + separator_size
        
        if current_group:
            groups.append(current_group)
        
        self.logger.info(f"Grouped {len(batch_reports)} reports into {len(groups)} integration groups")
        
        # Integrate each group
        integrated_groups = []
        for i, group in enumerate(groups):
            if len(group) == 1:
                integrated_groups.append(group[0])
                self.logger.info(f"Group {i+1}: single report, no integration needed")
            else:
                combined = "\n\n==== BATCH SEPARATOR ====\n\n".join(
                    f"=== Batch {j+1} ===\n{report}" 
                    for j, report in enumerate(group)
                )
                try:
                    result = self.chat_agent.remote_chat(
                        text_content=INTEGRATED_FRAMEWORK_ENV_PROMPT.format(topic=topic, combined_reports=combined),
                        temperature=0.3
                    )
                    integrated_groups.append(result)
                    self.logger.info(f"Group {i+1}: integrated {len(group)} reports into {len(result)} chars")
                except Exception as e:
                    self.logger.error(f"Failed to integrate group {i+1}: {e}")
                    integrated_groups.append("\n\n".join(group))
        
        # If only one group, return it
        if len(integrated_groups) == 1:
            return integrated_groups[0]
        
        # Recursively integrate the integrated groups
        self.logger.info(f"Recursively integrating {len(integrated_groups)} integrated groups...")
        return self._multi_round_integrate_framework_env(topic, integrated_groups, max_context_chars)
    
    def generate_framework_env_report(
        self, 
        paper_mainfests: List[Dict], 
        topic: str,
        batch_size: int = None,
        verbose: bool = True,
        use_concise_pseudocode: bool = True
    ) -> str:
        """
        Generate a comprehensive framework selection and environment configuration guidance report.
        
        This function:
        1. Takes input as List[dict] with repo information (repo_name, paper_id, etc.)
        2. Reads requirement.txt and README files for each repository
        3. Processes repositories in batches
        4. Generates framework/env guidance for each batch
        5. Integrates all batch reports into a final comprehensive report
        
        Args:
            repos: List of dictionaries, each containing:
                - "repo_name": str - the repository name
                - "paper_id": str (optional) - the paper identifier
                - Other keys are ignored
            topic: The research topic
            batch_size: Number of repos per batch (default: self.batch_size)
            verbose: Whether to print progress information
            
        Returns:
            The final integrated report string
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        repo_data = []
        for paper_mainfest in paper_mainfests:
            # Validate input format
            if not paper_mainfest:
                return "Error: No paper_mainfest provided."
            
            paper_title = paper_mainfest.get("paper_title", "")
            paper_abstract = paper_mainfest.get("paper_abstract", "")
            paper_id = paper_mainfest.get("paper_id", "")
            repo_num = len(paper_mainfest.get("repo_names", []))

            if len(paper_mainfest.get("pseudocodes", [])) != repo_num or len(paper_mainfest.get("concise_pseudocodes", [])) != repo_num:
                raise ValueError(f"Mismatch in number of repos and (concise) pseudocodes for paper {paper_mainfest.get('paper_id', 'unknown')}")
            
            for idx in range(repo_num):
                repo_name = paper_mainfest.get("repo_names", [])[idx]
                if use_concise_pseudocode:
                    pseudocode = paper_mainfest.get("concise_pseudocodes", [])[idx]
                else:
                    pseudocode = paper_mainfest.get("pseudocode", [])[idx]
                files_content = self.read_repo_files(repo_name)
                repo_data.append({
                    'repo_name': repo_name,
                    'paper_id': paper_id,
                    'paper_abstract': paper_abstract,
                    'pseudocode': pseudocode,
                    'paper_title': paper_title,
                    'requirements': files_content.get('requirements', {}),
                    'readme': files_content.get('readme', {})
                })
        
        # Split repos into batches
        batches = []
        for i in range(0, len(repo_data), batch_size):
            batches.append(repo_data[i:i + batch_size])
        
        if verbose:
            self.logger.info(f"Processing {len(batches)} batches...")
        
        batch_reports = self._generate_batch_framework_env_report(topic, batches)
        
        
        if not batch_reports:
            self.logger.error("No reports generated")
            return "Error: No reports could be generated for the given repositories."
        
        # Integrate all batch reports
        final_report = self._integrate_framework_env_reports(topic, batch_reports)
        
        if verbose:
            self.logger.info(f"Final framework/env report generated ({len(final_report)} chars)")
        
        return final_report
    
    def _generate_batch_report(self, topic: str, paper_batches: List[List[Dict]]) -> List[str]:
        """
        Generate code reports for multiple batches of papers directly from pseudocode reports.
        
        Args:
            topic: The research topic
            paper_batches: List of batches, where each batch is a list of paper data dictionaries
            
        Returns:
            List of generated batch report strings
        """
        if not paper_batches:
            self.logger.warning("No paper batches provided to _generate_batch_report")
            return []
        
        self.logger.info(f"Processing {len(paper_batches)} batches of papers")
        
        # Validate and build prompts for each batch
        prompts = []
        batch_info = []  # Track info for logging
        
        for batch_idx, papers_data in enumerate(paper_batches):
            if not papers_data:
                self.logger.warning(f"Empty batch at index {batch_idx}, skipping")
                continue
            
            # Build the prompt with all papers in the batch
            papers_content = []
            
            for i, paper in enumerate(papers_data):
                paper_id = paper.get('paper_id', f'Paper_{i+1}')
                pseudocode_report = paper.get('pseudocode_report', '')
                paper_title = paper.get('paper_title', f'Paper {i+1}')
                paper_abstract = paper.get('paper_abstract', 'No abstract available')

                content = f"=== Paper {i+1} (ID: {paper_id})\n paper title: {paper_title}\n paper abstract: {paper_abstract}\n ===Pseudocode Report:\n{pseudocode_report}"
                papers_content.append(content)
            
            combined_content = "\n\n".join(papers_content)
            
            prompt = BATCH_CODE_REPORT_PROMPT.format(
                topic=topic,
                pseudocode_reports=combined_content
            )
            prompts.append(prompt)
            batch_info.append({
                'batch_idx': batch_idx,
                'num_papers': len(papers_data),
                'paper_ids': [p.get('paper_id', f'Paper_{i}') for i, p in enumerate(papers_data)]
            })
            
            self.logger.info(f"Prepared batch {batch_idx + 1}/{len(paper_batches)} with {len(papers_data)} papers")
        
        if not prompts:
            self.logger.error("No valid prompts generated from batches")
            return []
        
        # Validate function for batch report
        def _validate_batch_report(result, info_dict = None):
            if not isinstance(result, str):
                raise ValueError(f"Generated batch code report is not a string type: {type(result)}.")
            if not result or len(result) < 100:
                raise ValueError(f"Generated batch code report is too short: {len(result)}, likely an error occurred, raw: {result}")
            return True, result

        self.logger.info(f"Generating {len(prompts)} batch reports via API...")
        
        try:
            results = self.chat_agent.batch_remote_chat_with_retry(
                prompts=prompts,
                validate_fn=_validate_batch_report,
                max_retry=6,
                desc="Generating batch code reports",
                temperature=0.3,
            )
            
            # Log success info
            if isinstance(results, list):
                self.logger.info(f"Successfully generated {len(results)} batch reports")
                for i, result in enumerate(results):
                    if result:
                        self.logger.info(f"Batch {i+1} report length: {len(result)} chars")
                    else:
                        self.logger.warning(f"Batch {i+1} returned empty result")
            else:
                self.logger.warning(f"Unexpected result type: {type(results)}")
            
            return results if isinstance(results, list) else []
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch reports: {e}")
            return []
    
    def _integrate_reports(self, topic: str, batch_reports: List[str], max_context_chars: int = None) -> str:
        """
        Integrate multiple batch reports into a final comprehensive report.
        Uses multi-round fallback when combined reports exceed context window.
        
        Args:
            topic: The research topic
            batch_reports: List of batch report strings
            max_context_chars: Maximum characters per integration call 
            
        Returns:
            The integrated report
        """
        if not batch_reports:
            return "No reports to integrate."
    
        if max_context_chars is None:
            max_context_chars = self.max_context_chars_for_integrate
        
        if len(batch_reports) == 1:
            # Only one batch, no integration needed
            return batch_reports[0]
        
        # First attempt: try integrating all reports at once
        combined_reports = "\n\n==== BATCH SEPARATOR ====\n\n".join(
            f"=== Batch {i+1} ===\n{report}" 
            for i, report in enumerate(batch_reports)
        )
        
        # Check if combined content fits in context
        prompt_length = len(INTEGRATED_REPORT_PROMPT.format(topic=topic, combined_reports=combined_reports))
        
        if prompt_length <= max_context_chars:
            self.logger.info("Integrating batch reports into final report...")
            try:
                result = self.chat_agent.remote_chat_with_retry(
                    prompt=INTEGRATED_REPORT_PROMPT.format(topic=topic, combined_reports=combined_reports),
                    temperature=0.3,
                    max_retry=6,
                )
                return result
            except Exception as e:
                self.logger.warning(f"Failed to integrate all reports at once: {e}")
                # Fall through to multi-round fallback
        else:
            self.logger.info(f"Combined reports exceed context ({prompt_length} chars), using multi-round fallback...")
        
        # Multi-round fallback: integrate in smaller groups
        return self._multi_round_integrate_reports(topic, batch_reports, max_context_chars)
    
    def _multi_round_integrate_reports(self, topic: str, batch_reports: List[str], max_context_chars: int = None) -> str:
        """
        Multi-round integration when all batch reports don't fit in context.
        Integrates reports in pairs/groups, then recursively integrates results.
        
        Args:
            topic: The research topic
            batch_reports: List of batch report strings
            max_context_chars: Maximum characters per integration call
            
        Returns:
            The integrated report
        """
        self.logger.info(f"Starting multi-round integration for {len(batch_reports)} batch reports...")
        
        # Base case: if only one or two reports, integrate directly
        if len(batch_reports) <= 2:
            combined = "\n\n==== BATCH SEPARATOR ====\n\n".join(
                f"=== Batch {i+1} ===\n{report}" 
                for i, report in enumerate(batch_reports)
            )
            try:
                result = self.chat_agent.remote_chat_with_retry(
                    prompt=INTEGRATED_REPORT_PROMPT.format(topic=topic, combined_reports=combined),
                    temperature=0.3,
                    max_retry=6,
                )
                return result
            except Exception as e:
                self.logger.error(f"Failed to integrate: {e}")
                return "\n\n".join(batch_reports)
        
        # Group reports into pairs that fit within context
        groups = []
        current_group = []
        current_group_size = 0
        
        for i, report in enumerate(batch_reports):
            report_size = len(report)
            separator_size = len(f"\n\n==== BATCH SEPARATOR ====\n\n=== Batch {len(current_group) + 1} ===\n")
            
            # Check if adding this report would exceed context
            # Reserve space for prompt template (~1500 chars)
            if current_group_size + report_size + separator_size > int(max_context_chars*0.9):
                if current_group:
                    groups.append(current_group)
                current_group = [report]
                current_group_size = report_size
            else:
                current_group.append(report)
                current_group_size += report_size + separator_size
        
        if current_group:
            groups.append(current_group)
        
        self.logger.info(f"Grouped {len(batch_reports)} reports into {len(groups)} integration groups")
        
        # Integrate each group
        integrated_groups = []
        for i, group in enumerate(groups):
            if len(group) == 1:
                integrated_groups.append(group[0])
                self.logger.info(f"Group {i+1}: single report, no integration needed")
            else:
                combined = "\n\n==== BATCH SEPARATOR ====\n\n".join(
                    f"=== Batch {j+1} ===\n{report}" 
                    for j, report in enumerate(group)
                )
                try:
                    result = self.chat_agent.remote_chat_with_retry(
                        prompt=INTEGRATED_REPORT_PROMPT.format(topic=topic, combined_reports=combined),
                        temperature=0.3,
                        max_retry=6,
                    )
                    integrated_groups.append(result)
                    self.logger.info(f"Group {i+1}: integrated {len(group)} reports into {len(result)} chars")
                except Exception as e:
                    self.logger.error(f"Failed to integrate group {i+1}: {e}")
                    integrated_groups.append("\n\n".join(group))
        
        # If only one group, return it
        if len(integrated_groups) == 1:
            return integrated_groups[0]
        
        # Recursively integrate the integrated groups
        self.logger.info(f"Recursively integrating {len(integrated_groups)} integrated groups...")
        return self._multi_round_integrate_reports(topic, integrated_groups, max_context_chars)
    
    def generate_report(
        self, 
        papers: List[Dict], 
        topic: str = None,
        batch_size: int = None,
        verbose: bool = True
    ) -> str:
        """
        Generate a comprehensive code analysis report for a list of papers.
        
        This function:
        1. Takes input as List[dict] with paper_id and pseudocode_report
        2. Processes papers in batches
        3. Generates a dimension-specific report for each batch directly from pseudocode reports
        4. Integrates all batch reports into a final comprehensive report
        
        Args:
            papers: List of dictionaries, each containing:
                - "paper_id": str - the paper identifier
                - "pseudocode_report": str - the pseudocode report for that paper
            topic: The research topic
            batch_size: Number of papers per batch (default: self.batch_size)
            verbose: Whether to print progress information
            
        Returns:
            The final integrated report string
        """
        if topic is None:
            topic = self.config.BasicInfo.topic
        if batch_size is None:
            batch_size = self.batch_size
        
        # Validate input format
        if not papers:
            return "Error: No papers provided."
        
        for i, paper in enumerate(papers):
            if not isinstance(paper, dict):
                return f"Error: Expected dict at index {i}, got {type(paper)}"
            if 'pseudocodes' not in paper:
                return f"Error: Missing 'pseudocodes' key in paper at index {i}"
        
        if verbose:
            self.logger.info(f"Generating code report for topic: {topic}")
            self.logger.info(f"Total papers: {len(papers)}, batch size: {batch_size}")
        
        # Split papers into batches
        batches = []
        for i in range(0, len(papers), batch_size):
            batches.append(papers[i:i + batch_size])
        
        if verbose:
            self.logger.info(f"Processing {len(batches)} batches...")
    
        
        batch_reports = self._generate_batch_report(topic, batches)
        
        if not batch_reports:
            self.logger.error("No reports generated")
            return "Error: No reports could be generated for the given papers."
        
        # Integrate all batch reports
        final_report = self._integrate_reports(topic, batch_reports)
        
        # Build repository name index from input papers (code-generated, not LLM-generated)
        # This avoids LLM hallucination by using the actual input data
        repo_names_set = set()
        repo_paper_mapping = {}  # repo_name -> paper_id
        
        for paper in papers:
            paper_id = paper.get('paper_id', '')
            # Try to get repo_names from various possible keys
            repo_names = paper.get('repo_names', [])
            if not repo_names:
                # Try alternative keys that might contain repo info
                repo_names = paper.get('repos', [])
            if not repo_names and 'pseudocodes' in paper:
                # If no explicit repo_names, try to infer from paper structure
                # Some papers may store repo info differently
                pass
            for repo_name in repo_names:
                if repo_name and repo_name not in ['NOT_FOUND', None, '']:
                    repo_names_set.add(repo_name)
                    if repo_name not in repo_paper_mapping:
                        repo_paper_mapping[repo_name] = []
                    if paper_id not in repo_paper_mapping[repo_name]:
                        repo_paper_mapping[repo_name].append(paper_id)
        
        # Append repository index to the report if we found any repos
        if repo_names_set:
            repo_index_header = "\n\n---\n\n## Repository Index\n\n"
            repo_index_content = "The following repositories were analyzed in this report:\n\n"
            
            # Sort repo names for consistent output
            sorted_repo_names = sorted(repo_names_set)
            for idx, repo_name in enumerate(sorted_repo_names, 1):
                paper_ids = repo_paper_mapping.get(repo_name, [])
                if paper_ids:
                    repo_index_content += f"{idx}. **{repo_name}** (Paper: {', '.join(paper_ids)})\n"
                else:
                    repo_index_content += f"{idx}. **{repo_name}**\n"
            
            final_report += repo_index_header + repo_index_content
            if verbose:
                self.logger.info(f"Appended repository index with {len(repo_names_set)} repos")
        
        if verbose:
            self.logger.info(f"Final report generated ({len(final_report)} chars)")
        
        return final_report
    
    def save_report(self, code_report: str, env_report: str, output_path: str = None):
        """
        Save the report to a file.
        
        Args:
            code_report: The code report string
            env_report: The environment report string
            output_path: Path to save the report
        """
        topic = getattr(self.config.BasicInfo, 'topic', 'survey')
        
        # Determine save directory
        if output_path and output_path.strip():
            save_dir = os.path.dirname(output_path)
        else:
            # Fallback to output_base_dir if save_path is empty
            output_base_dir = getattr(self.config.BasicInfo, 'save_path', None)
            if output_base_dir and output_base_dir.strip():
                save_dir = output_base_dir
            else:
                # Last resort: use current directory
                save_dir = os.getcwd()
                self.logger.warning(f"Both output_path and output_base_dir are empty, using current directory: {save_dir}")
        
        save_dir = os.path.join(save_dir, "analysis")

        os.makedirs(save_dir, exist_ok=True)
        
        # Create md and json paths with topic name
        if self.config.BasicInfo.adapter_mode:
            save_env_path = os.path.join(save_dir, "survey_env_report.md")
            save_report_path = os.path.join(save_dir, "survey_code_report.md")
        else:
            save_env_path = os.path.join(save_dir, f"{topic}_env_report.md")
            save_report_path = os.path.join(save_dir, f"{topic}_code_report.md")

        with open(save_env_path, "w", encoding="utf-8") as f:
            f.write(env_report)
        with open(save_report_path, "w", encoding="utf-8") as f:
            f.write(code_report)
        
        self.logger.info(f"Code report saved to {save_report_path}\nEnv report saved to {save_env_path}")


def generate_code_report(
    papers: List[Dict], 
    topic: str, 
    config = None,
    output_path: str = None,
    batch_size: int = 5,
    verbose: bool = True
) -> str:
    """
    Convenience function to generate a code report for a list of papers.
    
    Args:
        papers: List of dictionaries, each containing:
            - "paper_id": str - the paper identifier
            - "pseudocode_report": str - the pseudocode report for that paper
        topic: The research topic
        config: Configuration object (will use default if not provided)
        output_path: Optional path to save the report
        batch_size: Number of papers per batch
        verbose: Whether to print progress
        
    Returns:
        The generated report string
    """
    # Create config if not provided (will use defaults from environment)
    if config is None:
        # Try to get config from environment or use a dummy one
        try:
            import hydra
            from omegaconf import OmegaConf
            
            # Try to load default config
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "config", 
                "deep_survey_batch.yaml"
            )
            if os.path.exists(config_path):
                config = merge_with_default_survey_config(OmegaConf.load(config_path))
            else:
                config = OmegaConf.create({})
        except:
            config = OmegaConf.create({})
    
    # Create generator and generate report
    generator = CodeReportGenerator(config)
    report = generator.generate_report(
        papers=papers,
        topic=topic,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Save if output path provided
    if output_path:
        generator.save_report(report, output_path)
    
    return report


def generate_framework_env_guidance(
    repos: List[Dict], 
    topic: str, 
    config = None,
    output_path: str = None,
    batch_size: int = 5,
    verbose: bool = True
) -> str:
    """
    Convenience function to generate framework selection and environment configuration guidance.
    
    This function analyzes requirements.txt and README files from multiple repositories
    to generate comprehensive guidance on framework selection and environment setup.
    
    Input format: List[dict] where each dict contains:
        - "repo_name": str - the repository name (folder name in cache)
        - "paper_id": str (optional) - the paper identifier
        - Other keys are ignored but can be included
    
    Args:
        repos: List of dictionaries with repository information
        topic: The research topic
        config: Configuration object (will use default if not provided)
        output_path: Optional path to save the report
        batch_size: Number of repos per batch (default: 5)
        verbose: Whether to print progress
        
    Returns:
        The generated framework/env guidance report string
        
    Example:
        repos = [
            {"paper_id": "2601.12345", "repo_name": "some-repo"},
            {"paper_id": "2601.23456", "repo_name": "another-repo"},
            {"paper_id": "2601.34567", "repo_name": "third-repo"}
        ]
        topic = "LLM-based Agents"
        report = generate_framework_env_guidance(repos, topic, output_path="./outputs/framework_env_guide.txt")
        print(report)
    """
    # Create config if not provided (will use defaults from environment)
    if config is None:
        # Try to get config from environment or use a dummy one
        try:
            import hydra
            from omegaconf import OmegaConf
            
            # Try to load default config
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "config", 
                "deep_survey_batch.yaml"
            )
            if os.path.exists(config_path):
                config = merge_with_default_survey_config(OmegaConf.load(config_path))
            else:
                config = OmegaConf.create({})
        except:
            config = OmegaConf.create({})
    
    # Create generator and generate framework/env report
    generator = CodeReportGenerator(config)
    report = generator.generate_framework_env_report(
        repos=repos,
        topic=topic,
        batch_size=batch_size,
        verbose=verbose
    )
    
    # Save if output path provided
    if output_path:
        generator.save_report(report, output_path)
    
    return report


# Example usage
if __name__ == "__main__":
    # Example: Generate a report for a list of papers with pseudocode reports
    # papers = [
    #     {"paper_id": "2601.12345", "pseudocode_report": "..."},
    #     {"paper_id": "2601.23456", "pseudocode_report": "..."},
    #     {"paper_id": "2601.34567", "pseudocode_report": "..."}
    # ]
    # topic = "LLM-based Agents"
    # report = generate_code_report(papers, topic, output_path="./outputs/code_report.txt")
    # print(report)
    pass
