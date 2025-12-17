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
You are encouraged to cite more papers from the relevant papers to strengthen your points.

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
- Only cite papers that appear in Relevant Papers, using format: (Paper ID: 2023.10567) to cite wherever appropriate.
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
You are encouraged to cite more papers from the relevant papers to strengthen your points.

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
- Only cite papers that appear in Relevant Papers, using format: (Paper ID: 2023.10567) to cite wherever appropriate.
- Do not generate a bibliography or reference list here.
"""

DRAFT_REFINEMENT = """You are refining an academic survey paper draft. The draft currently contains paper IDs as citations (e.g., "2406.10252").

### Task:
1. Refine the draft to improve clarity, coherence, and academic style.
2. Replace all paper ID citations with format like (Paper ID: 2023.10567) in the draft with numbered references in square brackets, in the order of first appearance.
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

ERROR_FEEDBACK_PROMPT = """The previous draft attempt encountered the following issues:
{info}
Please use this feedback to improve the next draft attempt. 
For example, if previous attempt cited non-existent or error papers, avoid citing the same paper_id and ensure all cited papers are from the provided relevant papers list.
"""

###----EVALUATOR PROMPTS----###
EVAL_CRITERIA = {
    # ---------- Core Quality ----------
    "Synthesis Quality": {
        "aspect": "Core Quality",
        "description": "Synthesis quality: distinguishes true integration and conceptual grouping of literature from mere enumeration of papers.",
        "score 1":  "No synthesis — the text is a disconnected list of papers with no comparison, grouping, or integrative statements.",
        "score 2":  "Very weak synthesis — occasional grouping but mostly descriptive lists; links between works are absent or mistaken.",
        "score 3":  "Minimal synthesis — some attempts to group papers or note similarities, but links are shallow and inconsistent.",
        "score 4":  "Basic synthesis — groups and comparisons exist but are superficial and miss deeper patterns or tensions.",
        "score 5":  "Moderate synthesis — reasonable grouping and some comparative statements; important cross-paper themes are noted but not developed.",
        "score 6":  "Good synthesis — clear groups and comparisons, with emerging themes and some cross-cutting analysis.",
        "score 7":  "Strong synthesis — integrates findings across multiple works, highlights causes of differences, and builds partial conceptual links.",
        "score 8":  "Very strong synthesis — conveys nuanced relationships among studies, articulates patterns and contradictions, and forms clear conceptual maps.",
        "score 9":  "Excellent synthesis — deeply integrates diverse sources, proposes coherent frameworks, and resolves or explains major discrepancies.",
        "score 10": "Outstanding synthesis — creates novel unifying frameworks or taxonomies, transforms raw literature into new conceptual insight."
    },

    "Organization": {
        "aspect": "Core Quality",
        "description": "Organization: evaluates logical flow, section/subsection ordering, and ease of following the argument across the survey.",
        "score 1":  "No logical order — sections are chaotic and the reader cannot form a coherent view of the topic.",
        "score 2":  "Very poor ordering — frequent abrupt jumps and misplaced content; reader must constantly backtrack.",
        "score 3":  "Weak organization — some logical chunks but many misplaced paragraphs or unclear section boundaries.",
        "score 4":  "Below-average organization — parts make sense but transitions are rough and structure sometimes redundant.",
        "score 5":  "Adequate organization — overall structure is serviceable though some sections could be rearranged for clarity.",
        "score 6":  "Good organization — clear sections and logical progression with occasional weak transitions or minor redundancies.",
        "score 7":  "Strong organization — well-ordered sections, clear flow, and purposeful subsectioning.",
        "score 8":  "Very strong organization — sections build on each other smoothly; transitions are purposeful and guide the reader.",
        "score 9":  "Excellent organization — highly coherent structure, logical sequencing, and efficient, non-redundant layout.",
        "score 10": "Exemplary organization — impeccable flow, perfect chapter/subsection design, and the structure itself aids understanding."
    },

    # ---------- Writing Quality ----------
    "Readability": {
        "aspect": "Writing Quality",
        "description": "Readability: clarity and accessibility for the intended academic audience (sentence-level clarity, pacing, and jargon handling).",
        "score 1":  "Unreadable — sentences are confusing, grammar errors abound, and jargon is opaque.",
        "score 2":  "Very poor readability — frequent grammar issues and unclear sentences; reader struggles to extract meaning.",
        "score 3":  "Low readability — many awkward sentences and unclear phrasing; frequent re-reading required.",
        "score 4":  "Below-average readability — understandable in places but style is verbose or clumsy; jargon often unexplained.",
        "score 5":  "Fair readability — generally clear though some sentences or paragraphs are convoluted.",
        "score 6":  "Good readability — clear prose overall with occasional dense passages or uneven pacing.",
        "score 7":  "Very good readability — fluent writing, appropriate use of terminology, and easy to follow.",
        "score 8":  "Excellent readability — crisp sentences, well-balanced pacing, and accessible to the target academic audience.",
        "score 9":  "Near-perfect readability — highly engaging academic prose with precise phrasing and smooth flow.",
        "score 10": "Outstanding readability — exemplary clarity and elegance; highly accessible without loss of technical precision."
    },

    "Academic Rigor": {
        "aspect": "Writing Quality",
        "description": "Academic rigor: fidelity to scholarly standards (proper citations, fair representation of prior work, and methodological transparency).",
        "score 1":  "No rigor — citations missing or blatantly incorrect; claims unsupported and methods misrepresented.",
        "score 2":  "Very low rigor — many unsupported claims, poor citation practice, and factual errors.",
        "score 3":  "Low rigor — partial or inconsistent citation and occasional misrepresentation of methods/results.",
        "score 4":  "Below-average rigor — citations present but incomplete and some claims lack supporting evidence.",
        "score 5":  "Moderate rigor — generally correct citations and basic methodological descriptions but missing nuance.",
        "score 6":  "Good rigor — solid citation practice and fair representation of methods with minor gaps.",
        "score 7":  "Strong rigor — careful referencing, transparent presentation of methods and limitations.",
        "score 8":  "Very strong rigor — thorough citations, balanced critique of methods, and clear evidence-backed claims.",
        "score 9":  "Excellent rigor — meticulous referencing, rigorous methodological critique, and consistently supported claims.",
        "score 10": "Exceptional rigor — exemplary scholarship: exhaustive, accurate citations and deep, transparent methodological analysis."
    },

    "Clarity": {
        "aspect": "Writing Quality",
        "description": "Clarity: precision in technical descriptions, definitions, and explanations so readers can unambiguously understand methods and results.",
        "score 1":  "Opaque — technical terms undefined, descriptions vague or misleading.",
        "score 2":  "Very unclear — frequent ambiguity in definitions and technical statements.",
        "score 3":  "Unclear in places — several technical passages lack precision or necessary definitional context.",
        "score 4":  "Somewhat clear — important terms sometimes defined, but many explanations remain imprecise.",
        "score 5":  "Moderately clear — most terms and methods are defined, but some technical ambiguity persists.",
        "score 6":  "Generally clear — technical descriptions are understandable though occasionally terse or under-specified.",
        "score 7":  "Clear — good precision in explanations and consistent definitions of key concepts.",
        "score 8":  "Very clear — technical passages are well-explained and precise; readers can reproduce reasoning.",
        "score 9":  "Extremely clear — excellent definitional consistency and precise, unambiguous technical exposition.",
        "score 10": "Perfect clarity — crystal-clear technical writing enabling immediate reproducibility and understanding."
    },

    "Coherence": {
        "aspect": "Writing Quality",
        "description": "Coherence: internal consistency of claims, definitions, and conclusions across sections (no contradictions or drifting terms).",
        "score 1":  "Contradictory — multiple direct contradictions and inconsistent use of terms across the text.",
        "score 2":  "Very incoherent — frequent inconsistencies and shifting definitions or claims.",
        "score 3":  "Inconsistent — notable contradictions or drifting terminology that confuse the narrative.",
        "score 4":  "Some inconsistencies — occasional contradictions or uneven use of terms across sections.",
        "score 5":  "Partly coherent — most sections consistent but a few conflicting statements remain.",
        "score 6":  "Generally coherent — consistent claims and terminology with minor lapses.",
        "score 7":  "Coherent — internally consistent throughout, with good alignment between sections and conclusions.",
        "score 8":  "Very coherent — strong internal consistency; different parts reinforce a unified message.",
        "score 9":  "Highly coherent — near-perfect consistency and thematic harmony across the survey.",
        "score 10": "Exceptionally coherent — flawless internal consistency; every section supports the central narrative."
    },

    # ---------- Content Depth ----------
    "Comprehensiveness": {
        "aspect": "Content Depth",
        "description": "Comprehensiveness: breadth of coverage across subtopics, methods, datasets, and perspectives relevant to the research area.",
        "score 1":  "Extremely narrow — only a tiny slice of the topic covered; many core areas absent.",
        "score 2":  "Very limited — major subtopics missing; coverage skewed to a few papers or approaches.",
        "score 3":  "Sparse — several important areas omitted and coverage lacks depth.",
        "score 4":  "Partial — some important areas covered but many relevant subtopics absent or lightly treated.",
        "score 5":  "Moderate — key areas present but notable gaps in methods, datasets, or perspectives.",
        "score 6":  "Reasonably comprehensive — most core areas covered though some peripheral topics missing.",
        "score 7":  "Comprehensive — broad coverage including major methods and datasets with minor omissions.",
        "score 8":  "Very comprehensive — covers almost all relevant facets with useful detail across areas.",
        "score 9":  "Extensive — deep and wide coverage across subfields and perspectives; only small gaps remain.",
        "score 10": "Exhaustive — near-complete coverage across the domain, methods, datasets, and viewpoints."
    },

    "Critical Analysis": {
        "aspect": "Content Depth",
        "description": "Critical analysis: depth of evaluation, fairness of critique, and the comparative assessment of strengths/weaknesses across works.",
        "score 1":  "No critique — purely descriptive with no critical evaluation of methods or findings.",
        "score 2":  "Very weak critique — token criticisms that are superficial or inaccurate.",
        "score 3":  "Shallow critique — limited or one-sided criticisms lacking evidence or depth.",
        "score 4":  "Some critique — occasional meaningful evaluation but inconsistent and incomplete.",
        "score 5":  "Moderate critique — reasonable evaluations but lacking systematic comparative rigor.",
        "score 6":  "Good critique — fair assessments, identifies key weaknesses and trade-offs in several areas.",
        "score 7":  "Strong critique — systematic comparisons, balanced judgments, and evidence-backed critiques.",
        "score 8":  "Very strong critique — deep analysis of methodological limitations, assumptions, and empirical gaps.",
        "score 9":  "Excellent critique — rigorous, balanced, and insightful comparative evaluation across major works.",
        "score 10": "Masterful critique — transformative critique that clarifies fundamental limits, suggests remedies, and reshapes thinking."
    },

    "Novelty and Insights": {
        "aspect": "Content Depth",
        "description": "Novelty and insights: the degree to which the survey produces original synthesis, useful conceptual reframing, or fresh research questions.",
        "score 1":  "No novelty — restates existing content without any new perspective or insight.",
        "score 2":  "Very little novelty — trivial observations only; no original framing or notable insight.",
        "score 3":  "Low novelty — occasional small observations, but no meaningful new contribution.",
        "score 4":  "Limited novelty — a few modest insights but mostly derivative.",
        "score 5":  "Some novelty — useful observations that add modest clarity or re-organization.",
        "score 6":  "Notable novelty — several original observations or useful reframings emerge.",
        "score 7":  "Strong novelty — clear new insights or a helpful conceptual framing that aids understanding.",
        "score 8":  "Very strong novelty — original synthesis that changes how parts of the field are seen.",
        "score 9":  "Excellent novelty — significant conceptual contribution or new taxonomy with wide applicability.",
        "score 10": "Exceptional novelty — groundbreaking synthesis or insight that opens new directions or major re-interpretation."
    },

    "Future Directions": {
        "aspect": "Content Depth",
        "description": "Future directions: quality and specificity of identified open problems, research trajectories, and actionable next steps for the field.",
        "score 1":  "No future directions — none suggested or suggestions are irrelevant/obvious to the point of uselessness.",
        "score 2":  "Very weak — vague or trivial future work statements without justification.",
        "score 3":  "Weak — a few general suggestions that lack specificity or connection to gaps.",
        "score 4":  "Limited — some potential directions mentioned but they are broad and under-motivated.",
        "score 5":  "Moderate — plausible future directions linked to gaps but lacking concrete research paths.",
        "score 6":  "Good — relevant and justified directions with some sense of how to pursue them.",
        "score 7":  "Strong — clear, actionable research directions that address concrete gaps and trade-offs.",
        "score 8":  "Very strong — well-motivated, specific proposals and near-term experiments or benchmarks suggested.",
        "score 9":  "Excellent — insightful and prioritized research agenda with clear milestones and evaluation ideas.",
        "score 10": "Outstanding — visionary yet practical roadmap for the field, with detailed, high-impact research programs and measurable goals."
    }
}

JUDGE_WITH_CRITERIA_PROMPT = """Here is an academic survey about the topic "{TOPIC}":
{SURVEY}
Please evaluate this survey about the topic "{TOPIC}" based on the criterion above provided below, and give a score from 1 to 10 according to the score description: 
--- 
Criterion Description: {Criterion_Description} 
--- 
Score 1 Description: {Score_1_Description} 
Score 2 Description: {Score_2_Description} 
Score 3 Description: {Score_3_Description} 
Score 4 Description: {Score_4_Description} 
Score 5 Description: {Score_5_Description} 
Score 6 Description: {Score_6_Description} 
Score 7 Description: {Score_7_Description} 
Score 8 Description: {Score_8_Description} 
Score 9 Description: {Score_9_Description} 
Score 10 Description: {Score_10_Description} 
--- 
Return the score without any other information:
"""

NLI_PROMPT = """---
Claim:
{CLAIM}
---
Source: {SOURCE}
Claim: {CLAIM}
---
Is the Claim faithful to the Source? 
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.
Only reply with 'Yes' or 'No':
"""
