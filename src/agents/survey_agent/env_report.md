# Comprehensive Framework Selection and Environment Configuration Guide for LLM-Based Agents

## Executive Summary

This guide synthesizes analysis across 13 repositories spanning AI agent research, software engineering automation, vulnerability detection, and biological protocol understanding. The research reveals three distinct architectural paradigms: **PyTorch-based agents** requiring local GPU infrastructure, **API-centric agents** emphasizing operational simplicity, and **multi-agent systems** enabling collaborative task decomposition. Framework selection depends primarily on research objectives: production deployment favors minimalist API-centric designs, research benchmarking requires evaluation-integrated frameworks, and domain-specific applications demand specialized tooling. Key findings indicate a significant industry shift toward simplified agent architectures, with minimal scaffolding achieving competitive performance against complex multi-tool systems. Environment configuration complexity scales with capability requirements, from simple pip installations to full GPU cluster setups with CUDA 11.x/12.x compatibility.

---

## 1. Framework Selection Analysis

### 1.1 Architectural Paradigms

The analyzed repositories reveal three fundamental approaches to LLM-based agent construction, each with distinct tradeoffs between flexibility, operational burden, and research applicability.

**API-Centric Architecture** represents the emerging industry standard. Mini-swe-agent exemplifies this paradigm with a core stack requiring only LiteLLM, Pydantic 2.0, and standard CLI utilities. This approach treats LLMs as black-box services, eliminating local ML framework dependencies. The design philosophy, articulated as "one year later, as LMs have become more capable, a lot of this is not needed at all to build a useful agent," reflects broader industry recognition that sophisticated tool systems may be unnecessary with sufficiently capable foundation models. API-centric agents achieve >74% on SWE-bench verified with approximately 100 lines of Python, demonstrating that scaffold complexity does not correlate with capability.

**PyTorch-Based Architecture** remains essential for research requiring fine-grained model control. AutoCodeRover employs PyTorch 2.2.1 with CUDA 12.x support, enabling custom fault localization algorithms and potential model fine-tuning. This approach provides programmatic access to intermediate representations through Tree-sitter AST analysis, supporting research into spectrum-based fault localization and code search strategies. The trade-off includes substantially higher operational complexity: 16GB+ VRAM requirements, 30GB+ disk space for Docker images, and CUDA driver compatibility constraints.

**Multi-Agent Orchestration** enables collaborative task decomposition through role-based architectures. AgileCoder implements phase-based workflows with configurable role definitions (Product Manager, Developer, Tester) managed through JSON configurations. Agent-as-a-judge extends this concept with tool-augmented judge agents capable of evaluating other agents' trajectories. These architectures excel in complex software engineering tasks requiring diverse expertise but introduce coordination overhead and debugging complexity.

### 1.2 Framework Ecosystem Distribution

| Domain | Primary Frameworks | Python Version | GPU Requirement |
|--------|-------------------|-----------------|------------------|
| Code Agents | PyTorch, LiteLLM, Tree-sitter | 3.10-3.11 | Varies |
| Evaluation | Transformers, LiteLLM, Poetry | 3.11 | Optional |
| Vulnerability Detection | PyTorch 1.12-2.0, Transformers 4.24-4.36 | 3.9-3.10 | Recommended |
| Domain Benchmarks | Transformers, sentence-transformers | 3.10 | Varies |

### 1.3 Critical Version Constraints

Version management presents significant challenges across repositories. **Agent-as-a-judge** enforces `numpy<2.0` due to breaking changes in array operations—a constraint that may conflict with other packages requiring modern numpy versions. **USENIX_2024** (vulnerability detection) uses PyTorch 1.12.0 with transformers 4.24.0, representing a stable but dated configuration incompatible with newer model architectures requiring transformers 4.36+. **JeeWMS** (vulnerability detection target) mandates JDK 1.8 and MySQL 5.7, excluding newer versions entirely.

LiteLLM version management requires particular attention. Mini-swe-agent specifies `>=1.75.5` while blocking versions 1.82.7 and 1.82.8 for security vulnerabilities. Agent-as-a-judge uses litellm==1.50.0, suggesting compatibility constraints with the evaluation toolchain.

---

## 2. Environment Configuration Guidance

### 2.1 Dependency Management Strategies

Repository dependency management approaches vary significantly in complexity and reproducibility guarantees.

**Poetry-based Management** (agent-as-a-judge) provides the highest reproducibility through lockfile generation and dependency group separation. The pyproject.toml structure separates runtime from development dependencies, enabling lean production deployments:

```toml
[tool.poetry.dependencies]
python = "^3.11"
litellm = "^1.50.0"
networkx = "^3.3"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.3"
```

**Requirements.txt** (USENIX_2024, VulInstruct-temp) relies on explicit version pinning. This approach offers straightforward installation but may encounter dependency resolution conflicts when transitive dependencies require incompatible versions.

**Setuptools** (AgileCoder) balances package installation convenience with development flexibility through setup.py configurations.

### 2.2 Python Environment Strategy

| Repository Category | Recommended Python | Environment Tool | Setup Time |
|--------------------|--------------------|-------------------|------------|
| API-centric agents | 3.10+ | venv/pip | 2-5 minutes |
| Evaluation frameworks | 3.11 | conda/poetry | 5-10 minutes |
| PyTorch research | 3.9-3.10 | conda | 15-30 minutes |
| Java targets | N/A | System JDK | Varies |

For multi-repository research environments, a layered approach proves effective:

```bash
# Base conda environment for core ML dependencies
conda create -n llm-research python=3.10
conda activate llm-research

# PyTorch with CUDA support (verify CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core frameworks
pip install transformers>=4.36.0 sentence-transformers>=2.2.0
pip install datasets>=2.16.0 accelerate>=0.25.0

# Unified LLM interface
pip install litellm

# Evaluation utilities
pip install numpy pandas scipy tqdm
```

### 2.3 GPU Configuration Requirements

GPU requirements divide repositories into two categories. **Mandatory GPU** repositories include autoCodeRover (PyTorch 2.2.1, CUDA 12.x) and vulnerability detection research (PyTorch 1.12+). These require NVIDIA GPUs with CUDA compute capability 3.5+, 16GB+ VRAM for full functionality, and driver 525.60.13+ for CUDA 12.x support. **CPU-sufficient** repositories include mini-swe-agent and most API-centric designs, where GPU acceleration occurs server-side through model providers.

```bash
# Verify GPU availability and CUDA compatibility
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# For PyTorch 2.0+ with CUDA 11.8
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# For PyTorch 1.12 (older research)
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

### 2.4 API Key Configuration Patterns

All LLM-dependent repositories share environment variable-based API configuration:

```bash
# Standard configuration file structure (.env)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AZURE_OPENAI_KEY=...
OPENROUTER_API_KEY=...

# Provider-specific endpoints
AZURE_OPENAI_ENDPOINT=https://resource.openai.azure.com
```

LiteLLM enables unified access across providers, supporting OpenAI, Anthropic, Azure OpenAI, OpenRouter, and local Ollama deployments. This abstraction layer facilitates provider switching without code modifications—a critical feature for research comparing model performance across vendors.

---

## 3. Base Framework Recommendations

### 3.1 Production Deployment

**Primary Recommendation: Mini-SWE-Agent**

Mini-swe-agent provides optimal operational characteristics for production deployment. The minimal dependency footprint reduces maintenance burden, while comprehensive environment support (Docker, Podman, Singularity, Modal) accommodates diverse deployment targets. The architecture prioritizes hackability—core agent logic spans approximately 100 lines, enabling rapid customization without navigating complex abstractions.

```bash
# Installation
pip install uv && uvx mini-swe-agent

# Production configuration
mini --model gpt-4o --model-temperature 0.2
```

**Alternative for Multi-Provider Needs: Agent-as-a-Judge**

For deployments requiring unified multi-model evaluation, agent-as-a-judge's LiteLLM integration and comprehensive tool system provide necessary flexibility. The Poetry-based dependency management ensures reproducible environments.

### 3.2 Research Benchmarking

**Software Engineering Focus: AutoCodeRover**

AutoCodeRover integrates with SWE-bench for standardized evaluation while providing deeper program analysis capabilities through Tree-sitter AST parsing and spectrum-based fault localization. The PyTorch dependency enables custom model integration for research extending fault localization algorithms.

```bash
# SWE-bench evaluation setup
conda env create -f environment.yml
conda activate auto-code-rover
export OPENAI_KEY=sk-YOUR-KEY

python app/main.py swe-bench --model gpt-4o
```

**General Agent Evaluation: Mini-SWE-Agent**

Mini-swe-agent's "bash-only" philosophy creates ideal conditions for measuring model capability rather than scaffold performance. Listed on SWE-bench leaderboard for model comparison, it provides cleaner ablation study foundations than complex multi-tool systems.

### 3.3 Program Analysis Research

**Code Search and Fault Localization: AutoCodeRover**

Tree-sitter integration provides programmatic AST access for research extending code search strategies. The architecture separates concerns across app/, services/, and lib/ directories, enabling targeted modification of localization algorithms without restructuring entire systems.

**Static Analysis Enhancement: ZeroFalse**

For SARIF-based static analysis augmentation, ZeroFalse provides OpenRouter integration and parallel analysis capabilities. The streamlit dashboard enables interactive result exploration.

### 3.4 Domain-Specific Applications

**Vulnerability Detection: USENIX_2024 Stack**

PyTorch 1.12.0 with transformers 4.24.0 provides tested stability for security-specific models (CodeBERT, GraphCodeBERT). The research prototype architecture prioritizes reproducibility over production readiness.

**Biological Protocol Understanding: BioProtocolBench**

Dual inference support (API-based via OpenAI-compatible endpoints, local via HuggingFace models) enables flexible evaluation across model types. The benchmark provides standardized metrics across five task categories: Protocol QA, Step Ordering, Error Correction, Protocol Generation, and Experimental Reasoning.

**Enterprise Application Analysis: JeeWMS**

As a vulnerability detection target, JeeWMS requires specific infrastructure: JDK 1.8, MySQL 5.7, and Tomcat 7. The Spring/Hibernate architecture provides authentic enterprise complexity for security research.

---

## 4. Sub-Direction Analysis

### 4.1 Agent Architecture Trends

The repositories reveal significant architectural evolution. **From Complex Scaffolds to Minimalist Designs**: Mini-swe-agent demonstrates that sophisticated tool systems may be unnecessary with capable LLMs. The agent achieves competitive results without explicit tool definitions, suggesting that foundation model capability increasingly determines agent performance independent of scaffold complexity.

**Multi-Agent Collaboration Patterns**: AgileCoder's role-based architecture and agent-as-a-judge's evaluation paradigm represent complementary approaches to agent collaboration. Role-based systems distribute expertise across specialized agents, while judge-based systems enable agent self-evaluation and refinement.

**Evaluation-Driven Development**: The emergence of agent-as-a-judge reflects growing emphasis on rigorous agent evaluation. Integration with standardized benchmarks (SWE-bench, DevAI) enables reproducible capability assessment and systematic improvement tracking.

### 4.2 Framework Ecosystem Patterns

**HuggingFace Ecosystem Dominance**: Transformers and datasets libraries represent the de facto standard for LLM research. Both VulInstruct-temp and bioprotocolbench use HuggingFace dataset formats, enabling interoperability across repositories.

**Unified LLM Interfaces**: LiteLLM's abstraction layer appears across multiple repositories (mini-swe-agent, agent-as-a-judge, ZeroFalse), indicating industry recognition of provider diversity challenges. This pattern facilitates research across model providers without code modification.

**Code Analysis Tooling**: Tree-sitter integration appears across agent-as-a-judge, FFmpeg, and auto-code-rover, demonstrating its utility for language-agnostic code structure analysis. The 0.21.3 + 1.8.0 version combination requires careful management to avoid parser initialization failures.

### 4.3 Hardware Scaling Patterns

Resource requirements correlate with capability scope:

| Capability Level | VRAM | RAM | Storage | Typical Setup |
|-----------------|------|-----|---------|---------------|
| API-only | N/A | 8GB | 2GB | Standard laptop |
| Local inference (7B) | 8GB | 16GB | 10GB | Single GPU |
| Local inference (70B) | 40GB | 64GB | 50GB | A100 40GB |
| Training/fine-tuning | 80GB+ | 128GB+ | 100GB+ | Multi-GPU cluster |

---

## 5. Troubleshooting and Mitigation

### 5.1 Common Environment Issues

**Numpy 2.0 Incompatibility**

Agent-as-a-judge explicitly constrains numpy<2.0. Explicit version pinning prevents inadvertent upgrades:

```bash
pip install "numpy<2.0"
python -c "import numpy; print(numpy.__version__)"
```

**PyTorch CUDA Mismatches**

Version-specific CUDA requirements frequently cause runtime failures. For PyTorch 1.12.0, install matching cudatoolkit:

```bash
conda install cudatoolkit=11.6
python -c "import torch; print(torch.version.cuda)"
```

**Deprecated OpenAI API**

AgileCoder uses openai==0.28.1, which will eventually become unsupported. Migration to openai>=1.0.0 requires API pattern updates but enables access to newer model capabilities.

**Missing libclang**

C/C++ analysis requires system-level LLVM installation:

```bash
# Ubuntu/Debian
sudo apt-get install libclang-dev
# macOS
brew install llvm
# Conda
conda install -c conda-forge libclang
```

### 5.2 Dependency Conflict Mitigation

**Complex PyTorch Environments**

For repositories with extensive dependency trees, conda environment creation handles resolution more effectively than pip:

```bash
conda env create -f environment.yml
conda env create -n llm-research python=3.10
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Poetry Resolution Failures**

Agent-as-a-judge's complex dependency graph may cause resolution failures. Update poetry and clear cache:

```bash
pip install --upgrade poetry
poetry cache clear pypi --all
poetry lock --no-update
poetry install
```

---

## 6. Actionable Recommendations

### 6.1 For Practitioners

1. **Start with API-centric architectures** for rapid prototyping and production deployment. Mini-swe-agent provides the fastest path from concept to execution with minimal operational overhead.

2. **Use Docker for environment isolation** across all deployments. Containerization ensures consistent execution environments and simplifies dependency management.

3. **Configure via YAML** for reproducible experiments. Both mini-swe-agent and agent-as-a-judge support YAML configuration, enabling version-controlled experiment definitions.

4. **Adopt LiteLLM for multi-provider support** to avoid vendor lock-in and enable comparative evaluation across model families.

### 6.2 For Researchers

1. **Use autoCodeRover** for program analysis research requiring AST access, fault localization algorithms, or custom model integration. The PyTorch foundation enables programmatic model behavior modification.

2. **Use mini-swe-agent** for foundation model evaluation, where scaffold transparency ensures measurement of model capability rather than tool performance.

3. **Integrate with SWE-bench** for standardized software engineering task evaluation. Both primary frameworks support SWE-bench evaluation for reproducible benchmarking.

4. **Preserve working environments** for dated dependencies (PyTorch 1.12.0, transformers 4.24.0) to ensure experimental reproducibility.

### 6.3 For Educators

1. **Use mini-swe-agent** for teaching agent fundamentals due to its minimal codebase (~100 lines core logic) enabling rapid comprehension.

2. **Use autoCodeRover** for advanced courses demonstrating production-scale agent engineering, including Docker deployment, GPU configuration, and evaluation infrastructure.

3. **Supplement with survey repositories** (awesome-lifelong-llm-agent, LLM-Agent-Paper-List) for literature review assignments and taxonomy understanding.

### 6.4 For Repository Maintainers

1. **Add explicit requirements.txt** or environment.yml to enable environment reproduction. This represents the highest-impact improvement for research reproducibility.

2. **Include Docker support** for complex dependencies, particularly those requiring CUDA, tree-sitter language bindings, or system-level libraries.

3. **Document hardware requirements** explicitly, including GPU memory, RAM, and disk space for model downloads and testbed environments.

4. **Provide minimal working examples** demonstrating core functionality with standardized benchmark tasks.

---

## 7. Conclusion

The analysis of 13 repositories reveals a maturing LLM-based agent ecosystem transitioning from complex multi-tool scaffolds toward minimalist, API-centric designs. Framework selection should align with research objectives: production deployments benefit from mini-swe-agent's operational simplicity, research benchmarking requires evaluation-integrated frameworks like autoCodeRover, and domain-specific applications demand specialized tooling configurations.

The architectural trend toward simplified agent designs reflects foundation model capability improvements, suggesting that scaffold engineering will increasingly focus on task decomposition and evaluation rather than tool implementation. Multi-agent systems and evaluation frameworks represent growth areas where human expertise in orchestration and assessment remains essential.

Environment configuration complexity correlates directly with capability requirements. API-centric agents require minimal setup (minutes), PyTorch-based research systems demand significant configuration investment (15-30+ minutes), and training workloads require dedicated GPU infrastructure. Researchers should match environment complexity to research objectives, avoiding unnecessary infrastructure investment for tasks achievable through API-based inference.

Future research directions include evaluation framework standardization, cross-domain benchmark development, and systematic comparison of agent architectures across capability dimensions. The repositories analyzed provide foundational infrastructure for this research, with unified LLM interfaces (LiteLLM) and standardized benchmarks (SWE-bench) enabling reproducible comparison studies.

---

**Appendix: Quick Reference**

| Repository | Setup Command | Min Python | GPU | Primary Use |
|------------|--------------|------------|-----|-------------|
| mini-swe-agent | `uvx mini-swe-agent` | 3.10 | No | Production, evaluation |
| auto-code-rover | `conda env create -f environment.yml` | 3.x | Yes (16GB) | Research, program analysis |
| agent-as-a-judge | `poetry install` | 3.11 | Optional | Evaluation framework |
| AgileCoder | `pip install agilecoder` | 3.8 | No | Multi-agent development |
| USENIX_2024 | `bash install_requirements.sh` | 3.9 | Yes (40GB) | Vulnerability detection |