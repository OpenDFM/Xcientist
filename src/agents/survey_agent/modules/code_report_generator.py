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
from utils.rich_logger import get_logger
from utils.api_call import ChatAgent
# from modules.work_collector import WorkCollector
# from modules.code_collector import CodeCollector, CodeAnalyzer


# Prompt template for batch code report generation
# All prompts are in English as requested
BATCH_CODE_REPORT_PROMPT = """
You are an expert at analyzing research paper code implementations and generating comprehensive reports.

Your task is to analyze a batch of papers on the topic: "{topic}"

For each paper, you have basic paper information and code pseudocode reports:

{pseudocode_reports}

---

TASK: Generate a comprehensive code analysis report for this batch of papers

Requirements:
1. The report should include the following mandatory dimensions:
   - **Problem Modeling and Data Structure Classification**:
     Analyze how different papers model the problem, what data structures they use,
     and categorize them into meaningful groups.
   
   - **Core Algorithm Classification**:
     Analyze the core algorithms implemented in the code, identify patterns,
     and categorize them by approach (e.g., attention mechanisms, neural network architectures, etc.)
   
   - **Optimization and Acceleration Strategy Classification**:
     Analyze code-level optimization strategies probably including:
     * Computational optimizations (vectorization, batching, caching)
     * Memory optimizations (gradient checkpointing, mixed precision)
     * Training tricks and convergence improvements

2. Additionally, you should identify and add **custom dimensions** relevant to this specific topic, for example:
   - For LLM-related topics: consider workflow management, context window handling, prompt engineering patterns, agent architectures
   - For RL-related topics: consider reward shaping strategies, exploration mechanisms, credit assignment methods, reward hacking mitigation
   - For CV-related topics: consider data augmentation pipelines, model architecture patterns, transfer learning approaches
   - For any topic: identify other meaningful dimensions based on the code patterns you observe

3. For each dimension:
   - Provide specific code-level details that cannot be obtained from paper abstracts alone
   - Give concrete examples from the pseudocode
   - Compare different approaches across papers

4. The report should be:
   - Deep: Go beyond surface-level descriptions, analyze the actual implementation logic
   - Specific: Referring to specific code details
   - Well-organized: Use clear headings and structured sections
   - Comparative: Compare approaches across different papers when relevant

Output format:
Provide a structured report with clear sections for each dimension.
Use bullet points and concrete examples to illustrate your points.
"""
INTEGRATED_REPORT_PROMPT = """
You are an expert at synthesizing multiple batch reports into a comprehensive integrated report.

You have already generated reports for several batches of papers on the topic: "{topic}"

Here are the batch reports:
{combined_reports}

---

TASK: Integrate all batch reports into a single comprehensive report

Requirements:
1. Synthesize the information from all batches into a unified report
2. Remove duplicates and consolidate similar findings
3. Ensure all key insights from each batch are preserved
4. Maintain the same dimensional structure:
   - Problem Modeling and Data Structure Classification
   - Core Algorithm Classification
   - Optimization and Acceleration Strategy Classification
   - Custom Dimensions (topic-specific)
5. Add a summary section at the beginning that gives an overview of the topic
6. Add a conclusion section that synthesizes the key findings

The final report should be:
- Comprehensive: Cover all important aspects from all batches
- Coherent: Flow logically from section to section
- Non-redundant: Avoid repeating the same points multiple times
- Deep: Provide meaningful synthesis and insights

Output format:
Provide a well-structured comprehensive report with clear sections.
"""

# Prompt template for framework and environment selection guidance
# All prompts are in English as requested
FRAMEWORK_ENV_GUIDANCE_PROMPT = """
You are an expert at analyzing research paper code implementations and providing guidance on framework selection and environment configuration.

Your task is to analyze the requirements and README files from multiple repositories to generate comprehensive guidance on framework selection and environment setup.

For each repository, you have:
1. Repository Name: {repo_names}
2. Requirements Content: {requirements_content}
3. README Content: {readme_content}

---

TASK: Generate a comprehensive framework selection and environment configuration guidance report

Requirements:

1. **Framework Selection Analysis**:
   - Analyze which frameworks (PyTorch, TensorFlow, JAX, etc.) are used across different repositories
   - Identify the main framework choices and their versions
   - Analyze what each framework/environment is suitable for:
     * Which scenarios or sub-directions are best suited for each framework
     * What are the strengths and weaknesses of each framework choice
     * Performance considerations and community support

2. **Environment Configuration Guidance**:
   - Analyze common dependency patterns across repositories
   - Identify key packages and their version requirements
   - Provide guidance on environment setup best practices
   - Note any special requirements (GPU support, CUDA versions, etc.)

3. **Base Framework Recommendations**:
   - For each repository, assess its suitability as a base framework for further research
   - Consider factors like: code quality, documentation, extensibility, community activity
   - Identify which repositories are good starting points for different sub-directions
   - Categorize repositories by their suitability for:
     * Starting new projects from scratch
     * Extending existing implementations
     * Benchmarking and comparison
     * Learning and understanding the domain

4. **Sub-direction Analysis**:
   - Analyze which repositories are best suited for different sub-directions of research
   - Identify patterns in framework choices across different research directions
   - Provide recommendations based on specific research needs

5. The report should be:
   - Practical: Provide actionable recommendations
   - Well-organized: Use clear headings and structured sections
   - Comparative: Compare frameworks and repositories when relevant
   - Comprehensive: Cover all important aspects

Output format:
Provide a structured report with clear sections for each dimension.
Use bullet points and concrete examples to illustrate your points.
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

Requirements:

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

4. Be specific and provide concrete examples from the repositories

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

Output format:
Provide a well-structured comprehensive report with clear sections.
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
                                             max_batches: int = 20) -> str:
        """
        Generate a framework and environment guidance report for a batch of repositories.
        
        Args:
            topic: The research topic
            repo_data: List of repo data dictionaries with repo_name, requirements, and readme
            
        Returns:
            The generated report string
        """
        if not repo_data_batches:
            return "No repository data available for this batch."
        
        # Build the prompt with all repos in the batch
        repos_content = []
        prompts = []

        for repo_data in repo_data_batches:
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
                content += f"[Repository Information]:(for reference and understanding the repo)"
                content += f"Paper_title: {paper_title}\nPaper abstract: {paper_abstract}\n"
                content += f"Paper_Pseudocode: {pseudocode}\n"
                content += f"[Readme and Requiremnets]:(report core contents source)"
                content += f"Requirements:\n{req_content}\n\n"
                content += f"README:\n{readme_content}"
                repos_content.append(content)
            
            combined_content = "\n\n".join(repos_content)
            
            # # Format requirements content for prompt
            # requirements_content = ""
            # for repo in repo_data:
            #     repo_name = repo.get('repo_name', 'Unknown')
            #     requirements = repo.get('requirements', {})
            #     requirements_content += f"\n--- {repo_name} ---\n"
            #     for fname, content in requirements.items():
            #         if fname != 'NOT_FOUND':
            #             requirements_content += f"[{fname}]:\n{content}\n"
            
            # # Format README content for prompt
            # readme_content = ""
            # for repo in repo_data:
            #     repo_name = repo.get('repo_name', 'Unknown')
            #     readme = repo.get('readme', {})
            #     readme_content += f"\n--- {repo_name} ---\n"
            #     for fname, content in readme.items():
            #         if fname != 'NOT_FOUND':
            #             readme_content += f"[{fname}]:\n{content}\n"
            
            prompt = BATCH_FRAMEWORK_ENV_PROMPT.format(
                combined_content = combined_content
            )
            prompts.append(prompt)
        
        self.logger.info(f"Generating framework/env report for {len(prompts)} repositories batch...")
        
        def _env_report_validate_fn(result, info_dict = None):
            if not isinstance(result, str):
                raise TypeError("batch env_report result not string")
            if len(result) < 100:
                raise ValueError("batch env_report result too short")
            return True, result
        try:
            results = self.chat_agent.batch_remote_chat_with_retry(
                prompts=prompts,
                validate_fn= _env_report_validate_fn,
                max_retry = 3,
                desc = "Generating batch env report...",
                temperature=0.3,
            )
            return results
        except Exception as e:
            self.logger.error(f"Failed to generate framework/env report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _integrate_framework_env_reports(self, topic: str, batch_reports: List[str]) -> str:
        """
        Integrate multiple batch reports into a final comprehensive framework/env report.
        
        Args:
            topic: The research topic
            batch_reports: List of batch report strings
            
        Returns:
            The integrated report
        """
        if not batch_reports:
            return "No reports to integrate."
        
        if len(batch_reports) == 1:
            # Only one batch, no integration needed
            return batch_reports[0]
        
        # Combine all batch reports
        combined_reports = "\n\n==== BATCH SEPARATOR ====\n\n".join(
            f"=== Batch {i+1} ===\n{report}" 
            for i, report in enumerate(batch_reports)
        )
        
        self.logger.info("Integrating framework/env batch reports into final report...")
        
        try:
            result = self.chat_agent.remote_chat(
                text_content=INTEGRATED_FRAMEWORK_ENV_PROMPT.format(topic=topic, combined_reports=combined_reports),
                temperature=0.3
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to integrate framework/env reports: {e}")
            # If integration fails, just concatenate the reports
            return "\n\n".join(batch_reports)
    
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
    
    def _generate_batch_report(self, topic: str, paper_batchs: List[List[Dict]]) -> str:
        """
        Generate a code report for a batch of papers directly from pseudocode reports.
        
        Args:
            topic: The research topic
            papers_data: List of paper data dictionaries with paper_id and pseudocode_report
            
        Returns:
            The generated report string
        """
        prompts = []
        for papers_data in paper_batchs:
            if not papers_data:
                return "No papers data available for this batch."
            
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

            def _validate_batch_report(result, info_dict = None):
                if not result or len(result) < 100:
                    raise ValueError("Generated batch code report is too short, likely an error occurred.")
                if not isinstance(result, str):
                    raise ValueError("Generated batch code report is not a string.")
                return True, result

        self.logger.info(f"Generating batch report for {len(papers_data)} papers...")
        
        try:
            results = self.chat_agent.batch_remote_chat_with_retry(
                prompts = prompts,
                validate_fn=_validate_batch_report,
                max_retry=3,
                desc = "Generating batch code reports",
                temperature= 0.3,
            )
            return results
        except Exception as e:
            self.logger.error(f"Failed to generate batch report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _integrate_reports(self, topic: str, batch_reports: List[str]) -> str:
        """
        Integrate multiple batch reports into a final comprehensive report.
        
        Args:
            topic: The research topic
            batch_reports: List of batch report strings
            
        Returns:
            The integrated report
        """
        if not batch_reports:
            return "No reports to integrate."
        
        if len(batch_reports) == 1:
            # Only one batch, no integration needed
            return batch_reports[0]
        
        # Combine all batch reports
        combined_reports = "\n\n==== BATCH SEPARATOR ====\n\n".join(
            f"=== Batch {i+1} ===\n{report}" 
            for i, report in enumerate(batch_reports)
        )
        
        # Integration prompt in English
        
        self.logger.info("Integrating batch reports into final report...")
        
        try:
            result = self.chat_agent.remote_chat_with_retry(
                prompt=INTEGRATED_REPORT_PROMPT.format(topic=topic, combined_reports=combined_reports),
                temperature=0.3,
                max_retry = 3,
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to integrate reports: {e}")
            # If integration fails, just concatenate the reports
            return "\n\n".join(batch_reports)
    
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
        
        if verbose:
            self.logger.info(f"Final report generated ({len(final_report)} chars)")
        
        return final_report
    
    def save_report(self, report: str, output_path: str):
        """
        Save the report to a file.
        
        Args:
            report: The report string
            output_path: Path to save the report
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {output_path}")


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
                config = OmegaConf.load(config_path)
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
                config = OmegaConf.load(config_path)
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