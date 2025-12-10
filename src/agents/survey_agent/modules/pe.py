# All prompts used in the project are stored here.

PAPER_RELATEDNESS_BASED_ON_TITLE_AND_ABSTRACT = """You are assisting in filtering papers for a literature review.

Below is the SEED PAPER that defines the target research direction.

--- SEED PAPER ---
TITLE: {seed_title}
ABSTRACT: {seed_abstract}
-------------------

Evaluate the following CANDIDATE PAPER.

TITLE: {candidate_title}
ABSTRACT: {candidate_abstract}

Task:
Compare the candidate paper to the seed paper and judge how strongly it fits the same research direction.
Identify both key similarities and key differences.

Return a JSON object with:
- relevance_score: integer from 0 to 5
- category: "core", "related", or "irrelevant"
- reason: one concise sentence describing the key similarity/difference

Scoring guideline:
5 = Direct continuation / same problem / same method line  
4 = Strongly related  
3 = Some peripheral similarity  
1-2 = Weak overlap  
0 = Unrelated
"""

PAPER_DEEP_READING = """%%
{paper_markdown_text}
%%

Act as a senior researcher reading this paper. 
Your task is to produce a **deep, structured understanding** of the paper. 

Guidelines:

[
- Read the paper carefully and summarize its key contributions, methods, results, and significance.
- Include your own critical reflections, insights, and possible future directions.
- Generate a TL;DR paragraph capturing the core idea.
- Organize the output in JSON format, but do not fix the keys; include whatever fields make sense for this paper.
- Include relevant examples, observations, or specific results when appropriate.
- Focus on depth, clarity, and coherence, rather than adhering to a rigid template.
- Only use information contained in the paper; do not add external knowledge.
]
"""

PAPER_CLUSTERING = """You are a research assistant specializing in analyzing scientific papers.

Existing clusters of papers:

{existing_clusters_json}

New batch of papers (each with a keynote):

{new_batch_json}

Task:
- Update the clusters by adding new papers, reorganizing existing papers, merging or splitting clusters if needed.
- Assign a name and a brief summary to each cluster.
- Aim to create clusters that are as detailed as possible, capturing subtle differences, while still making sense.
- A single paper may belong to multiple clusters if its research area reasonably intersects with multiple themes (multi-assignment is allowed).
- Output the full updated cluster list strictly in JSON format.

Output format requirements:

The JSON should be a list of cluster objects.  
Each cluster object should include the following fields:
- cluster_name: string, the name of the cluster
- summary: string, a brief description of the cluster
- papers: list of paper objects, each with:
    - id: string, unique identifier of the paper
    - title: string, title of the paper
    - tldr: string, concise summary or TL;DR of the paper
"""

PROPOSE_QUESTIONS_FOR_CLUSTER = """You are an expert research assistant. I will give you a set of closely related papers, each with a keynote.

Your task is to read all the keynotes and propose a list of questions that would help someone deeply understand the ideas in these papers.

Guidelines:
- You may decide what questions are meaningful, surprising, or worth investigating.
- Questions should preferably involve relationships among multiple papers, such as comparisons, differences, shared assumptions, or conflicting claims.
- Do NOT mention paper IDs inside the question text.
- Each question must list the related paper IDs.
- You are encouraged to think critically, speculate, compare ideas, question assumptions, identify gaps, or propose future directions.
- Feel free to ask questions that are subtle, unconventional, or creative—your role is to guide deep thinking rather than summarize.

Input:
{cluster_content}

Output (strictly in JSON format):
[
  {{
    "question": "...",
    "related_papers": ["paper_id_1", "paper_id_2", ...]
  }}
]
"""

ANSWER_QUESTION_FOR_PAPERS = """You are an expert research assistant. I will give you a question and a set of related papers.

Your task is to provide a comprehensive answer to the question based on the content of the related papers.

Guidelines:
- Read the question carefully and understand what is being asked.
- Synthesize information from all the related papers to construct a well-rounded answer.
- Provide citations to the papers by their IDs when referencing specific information.
- Aim for clarity, depth, and coherence in your answer.

Input:
Question: {question}
Related Papers:
{related_papers_content}

Output:
Provide a detailed answer to the question, citing relevant papers by their IDs.
"""

INTRA_CLUSTER_ANALYSIS = """You are an expert research analyst. I will provide several groups of research papers, each with a list of questions and discussion notes.

Your task:
- Read all content and produce a summary analysis that extracts key cross-group insights.
- You may analyze freely: identify patterns, differences, potential connections, unresolved issues, or research gaps.
- The goal is to generate meaningful insights and analytical synthesis, not just list the original content.

Optional approaches (not required):
- Connect concepts or themes that appear across multiple groups
- Highlight commonalities and differences in methods or conclusions
- Identify key unresolved challenges or gaps
- Suggest possible future research directions

Requirements:
- Do not repeat the original text verbatim.
- Output a concise analytical summary in clear prose.

Input:
{cluster_analysis_content}

Output:
"""

SURVEY_OUTLINE_GENERATION = """You are an expert research survey generator. Your task is to iteratively update an existing survey outline using a batch of new paper keynotes, the current outline, and the analysis results of relevant papers for the topic/subtopic being written.

**Guidance:**
- The analysis results contain summarized insights, comparisons, trends, and key points extracted from prior work. They should be the **important source of guidance** when updating the outline.
- Paper keynotes are also useful: use them to **supplement, validate, or provide additional details** for the sections/subsections.
- Balance both sources to create a comprehensive, accurate, and logically organized survey outline.

**Requirements:**
1. Use the current outline as the base structure. Keep existing sections/subsections unless updated or merged.
2. Update the outline by:
   - Adding new sections/subsections for emerging topics or trends.
   - Merging similar topics/subsections to avoid redundancy.
   - Revising descriptions to reflect key insights, comparisons, trends, and challenges from the analysis results.
   - Supplementing descriptions with relevant points from paper keynotes.
3. Update "papers_to_use" for each section/subsection, including papers (paper ids) from the current batch and existing outline, ensuring that only paper IDs from the current batch, existing outline or relevant papers analysis are included—no unrelated IDs should appear.
4. Maintain clarity, logical structure, and a survey-style narrative.
5. Output strictly in JSON format, as shown below.

**Input:**
- current outline: {current_outline}
- relevant papers analysis: {papers_analysis}
- new paper keynotes: {paper_keynotes}

**Output JSON format:**
{{
    "title" : "Survey Title",
    "sections": [
        {{
            "title": "Section title",
            "description": "Summary of content to include, emphasizing insights, comparisons, and trends from analysis, supplemented by keynotes",
            "papers_to_use": ["Paper 1", "Paper 2"],
            "subsections": [
                {{
                    "title": "Subsection title",
                    "description": "Content description reflecting key insights, trends, comparisons from analysis, supplemented by keynotes",
                    "papers_to_use": ["Paper 1"]
                }}
            ]
        }}
]
}}

**Instruction to LLM:**
- Use the insights, trends, and comparisons from relevant paper analysis together with points from new paper keynotes to update the outline. Both sources should inform the content of each section/subsection.
- Add, merge, or revise sections/subsections as appropriate, keeping the outline structured and coherent.
- Ensure that each section/subsection reflects the combined contributions of the analysis results and the keynotes, capturing important insights, trends, and examples.
"""

SUBSECTION_DRAFT = """You are writing a subsection of an academic survey paper.

### Subsection Title:
{title}

### Guidance:
Write a coherent subsection that explains, analyzes, or discusses the topic indicated in the title and description.  
You should synthesize relevant information from the papers, analysis results.  
The content should be academically structured and readable, with emphasis on insights, trends, and comparisons where appropriate.
You may include examples from papers to support the discussion, but do not simply list papers.

### Information to use:
- Subsection description:
{description}

- Relevant papers:
{papers}

- Insights from analysis:
{relevant_analysis}

### Output:
- A well-written subsection in several coherent paragraphs.
- Academic style; focused on synthesis and analysis, not just reporting.
- Only cite papers that appear in Relevant Papers, using a format such as (Paper ID: 2023.10567) wherever appropriate.
- Do not generate a bibliography or reference list here.
"""

SECTION_DRAFT = """You are writing a section of an academic survey paper.

### Section Title:
{title}

### Guidance:
Write a coherent section that provides a high-level synthesis of its subsections.  
Use the subsection drafts to highlight insights, trends, comparisons, and relationships between ideas.  
The section should be academically structured and readable, and it should integrate the content of the subsections rather than simply repeating them.  
You may also include examples or references from papers to support the discussion, but do not just list papers or subsection texts.

### Information to use:
- Section description:
{description}

- Subsection drafts:
{subsection_drafts}

- Relevant papers:
{papers}

### Output:
- A well-written section in several coherent paragraphs.
- Academic style; focused on synthesis, comparison, and trends.
- Only cite papers that appear in Relevant Papers, using a format such as (Paper ID: 2023.10567) wherever appropriate.
- Do not generate a bibliography or reference list here.
"""

DRAFT_REFINEMENT = """You are refining an academic survey paper draft. The draft currently contains paper IDs as citations (e.g., "2406.10252").

### Task:
1. Refine the draft to improve clarity, coherence, and academic style.
2. Replace all paper ID citations in the draft with numbered references in square brackets, in the order of first appearance.
3. Each citation number corresponds to a single paper ID.
4. Keep all original content and ideas.
5. Integrate citations smoothly and naturally within the text.

### Input:
Draft Text:
{draft_text}

### Output:
Provide a **JSON object** with the following structure:

{{
  "refined_survey": "<refined survey text with numbered citations>",
  "references": [<paper_id for citation 1>, <paper_id for citation 2>, ...]
}}

- `refined_survey` is a string containing the survey text with citations replaced by `[number]`.
- `references` is a list of objects ordered by citation number, each mapping the index to its corresponding paper IDs.
- Ensure JSON is valid and parsable.
- Do not generate a bibliography or reference list here.
"""
