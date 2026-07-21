---
name: workspace-navigator
description: Navigate the workspace, find and read project-related files, then write blog analysis to output directory
triggers:
  - project
  - workspace
  - explore
  - structure
  - 查看
  - 项目
---

# Workspace Navigator Skill

You are a project navigation expert. When the user asks to explore a project, follow these steps:

## Project Paths

- **Source workspace**: `/home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
- **Output directory**: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}`

## Workflow

### 1. Identify Project Name
- Extract the project folder name from user input
- Example: user says "explore dcr_v1", then project_name = "dcr_v1"

### 2. List Project Directory
- Use bash tool to execute: `ls -la /home/zhang/projects/ResearchAgent-main/src/agents/experiment_agent/workspaces/{project_name}`
- Explore subdirectories recursively

### 3. Identify Important Files
Priority:
- README.md, setup.py, pyproject.toml
- Core source code files
- docs/, examples/, scripts/

### 4. Read Key Files
- Use FileEditorTool to read important files
- Keep track of which files you read (these will be referenced)

### 5. Write Blog Analysis

Output directory: `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/`

Write file: `blog_idea.md`

```markdown
# [Project Name] Blog Analysis

## Project Overview
[Brief description - what this project does and its value]

## Architecture
[Main components and how they work together]

## Tech Stack
[Languages, frameworks, key libraries]

## Blog Outline

**Title Guidelines**:
- For [Title to be decided], find a proper title for it:
- Section titles should be related to the project and clearly indicate the content
- Titles should be engaging and attract readers (use action verbs, interesting phrases)
- **DO NOT use questions** (e.g., avoid "How does X work?" or "Why is Y important?")
- Instead, use declarative statements (e.g., "How X Works" → "X's Core Mechanism")

### 1. TL;DR
**Important**: This section should cover one-sentence summary of the research paper's core contribution and key finding

### 2. Introduction
**Important**: This section should cover background context, existing problems, and why this research is needed

### 3. [Title to be decided]
**Important**: This section should cover the overall architecture and briefly describe the innovative method - give readers a high-level overview of the system components and how they connect.

### 4. [Title to be decided]
**Important**: This section should cover in-depth explanation of each component/approach (2-4 key points)

### 5. [Title to be decided] (if applicable)
**Important**: This section should cover experimental setup, results, and analysis

### 6. [Title to be decided]
**Important**: This section should cover key contributions, real-world impact, and future directions

### 7. References
**Important**: This section should cover related work, citations, and resources

## Key Files Analyzed
- [file1]: [brief description]
- [file2]: [brief description]
- [file3]: [brief description]

## Candidate Papers for Citation

After writing the outline, find relevant academic papers to cite in the blog:

- **Target**: Aim to find and list approximately 10 papers (if possible. quality first)
- Papers should cover: background/motivation, related methods, and comparison points

### 1. Search for Related Papers
- Use SearchCoreNodesTool to search for papers related to the project keywords
- Also check the references section of Markdown files that may be related to the project in: /home/zhang/projects/ResearchAgent-main/src/agents/survey_agent/outputs/test/

### 2. Get Paper Details
- For promising paper titles, use SearchPaperAbstractTool to get full abstract and details to check if they are truly relevant

### 3. Download PDFs
- Use DownloadPaperPdfTool to download the promising papers to:
  `/home/zhang/projects/ResearchAgent-main/src/agents/blog_agent/workspaces/{project_name}/`

### 4. Verify PDF Content
- Read each downloaded PDF to check if it contains expected useful information for the blog
- If the PDF has no useful information, delete it using bash rm command

### 5. List Useful Papers
Add to the end of blog_idea.md:

## Papers for Citation

| Paper | Relevance | Suggested Citation Location |
|-------|-----------|----------------------------|
| [Title] | [Why it's relevant] | [e.g., Introduction/Related Work] |
```

## Verify Outline Against Template

After writing the outline, verify that it covers all required topics from the blog template:

**Required Topics**:
1. **Summary**: A brief one-sentence summary (TL;DR)
2. **Introduction**: Background context and why this research is needed
3. **Method/Architecture**: the overall architecture and briefly describe the innovative method
4. **Detailed Analysis**: Component-level explanation with sub-sections
5. **Experiments**: Experimental setup and results (if applicable)
6. **Contributions**: Key contributions and impact
7. **References**: Citations

**Process**:
1. Read the current outline in blog_idea.md
2. Check if the outline covers each required topic (names can differ)
3. If any topic is missing, add a corresponding section to the outline
4. Repeat until all required topics are covered
