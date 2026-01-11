# Paper Constitution (ICML Guidelines)

This file defines the high-level standards, philosophy, and aesthetic requirements for producing a top-tier ICML conference paper. It serves as the "superego" of the writing process, ensuring quality, rigor, and narrative coherence.

## 1. High-Level Philosophy: What Makes a "Good" Paper?

A top-tier machine learning paper is not just a report of experiments; it is a **persuasive narrative** centered on a scientific contribution.

-   **Clarity is King**: If the reader cannot understand the method in 5 minutes, the paper fails.
-   **Novelty**: The contribution (method, insight, or analysis) must be non-trivial and distinct from prior work.
-   **Rigor**: Claims must be strictly supported by evidence. Overclaiming is a fatal flaw.
-   **Significance**: The problem solved must be worth solving.

## 2. General Constraints

-   **Page Limit**: Strictly **8 pages** for the main body (Abstract through Conclusion).
    -   References and Appendix are unlimited.
    -   **Target**: Aim for a full 8 pages. A 7-page paper often implies missing depth.
-   **Anonymity**: Double-blind review. No authors, affiliations, or github links.
-   **Format**: Use the official `icml2024.sty` (or equivalent current year) style.

## 3. Section Guidelines (The Narrative Arc)

### 3.1 Abstract (200-300 words)
-   **Purpose**: The "elevator pitch". Must stand alone.
-   **Structure**:
    1.  **Context**: 1 sentence on the general problem.
    2.  **Gap**: 1 sentence on why existing solutions fail.
    3.  **Insight/Method**: 1-2 sentences on your proposed solution (The "Key Idea").
    4.  **Results**: 1-2 sentences on the quantitative improvement (e.g., "We achieve SOTA on X, improving accuracy by Y%").
    5.  **Takeaway**: 1 sentence on the broader impact.

### 3.2 Introduction (1 - 1.5 pages)
-   **Purpose**: Hook the reader and set up the problem.
-   **Flow**:
    1.  **Broad Context**: Why should anyone care?
    2.  **Specific Problem**: What is hard/unsolved?
    3.  **Current Approaches & Failures**: Why hasn't this been solved yet? (The "Villain").
    4.  **Our Approach**: The "Hero". Key insight.
    5.  **Contributions**: A bulleted list of 3 concrete contributions (Method, Theory, Empirics).
-   **Tip**: Include a "Teaser Figure" on Page 1 (top right or top center) that visually summarizes the method/problem.

### 3.3 Related Work (0.5 - 0.75 pages)
-   **Purpose**: Position the work, not just list citations.
-   **Style**: Group by topic (e.g., "Generative Models for X", "Efficient Transformers").
-   **Critical nuance**: Explicitly state how your work differs (e.g., "While [A] addresses X, it fails at Y, whereas we...").

### 3.4 Method (2 - 2.5 pages)
-   **Purpose**: Enable reproducibility and demonstrate technical correctness.
-   **Components**:
    -   **Preliminaries**: Define notation and problem statement formally.
    -   **Core Method**: The algorithm/model. Use a diagram!
    -   **Theoretical Justification** (optional but recommended): Why does it work?
-   **Tone**: Precise, mathematical, yet accessible. Avoid wall-of-text; use subsections.

### 3.5 Experiments (2 - 2.5 pages)
-   **Purpose**: Prove the claims made in the Introduction.
-   **Structure**:
    1.  **Setup**: Datasets, baselines, metrics (keep brief, move details to Appendix).
    2.  **Main Results**: Comparison against SOTA. Tables/Plots.
    3.  **Ablation Studies**: Dissect the method. What components matter?
    4.  **Analysis/Visualization**: Qualitative understanding (Why does it work?).
-   **Requirement**: Every claim must have a corresponding experiment.

### 3.6 Limitations (0.25 - 0.5 pages)
-   **Purpose**: Show intellectual honesty and maturity.
-   **Content**: When does the method fail? What are the computational costs?
-   **Tone**: Constructive, not self-deprecating.

### 3.7 Conclusion (0.25 pages)
-   **Purpose**: Summarize and look forward.
-   **Content**: Recap the main contribution and suggest 1-2 future directions.

## 4. Aesthetics & Design Recommendations

-   **Visual Hierarchy**:
    -   Use **bolding** for key terms when first introduced.
    -   Ensure section headers are clear.
-   **Figures**:
    -   **Vector Graphics**: Always use PDF/EPS, never PNG/JPG for plots/diagrams (unless showing raw images).
    -   **Captions**: Captions should be self-contained. A reader should understand the figure without reading the text.
    -   **Font Sizes**: Axis labels and legends must be legible (approx. same size as caption text).
-   **Tables**:
    -   Use `booktabs` (toprule, midrule, bottomrule). Avoid vertical lines.
    -   Highlight best results in **bold**.

## 5. Narrative Logic & Storytelling

A strong narrative is what distinguishes a "clear accept" from a "borderline". The logic must be robust, linear, and inevitable.

### 5.1 The Core Tension
Every good paper resolves a tension. Identify yours early:
-   **Accuracy vs. Efficiency**: "Existing methods are accurate but slow; we are just as accurate but 10x faster."
-   **Generality vs. Specificity**: "Methods work for X but fail for Y; we unify them."
-   **Theory vs. Practice**: "Theory predicts X, but practice shows Y; we bridge this gap."
**Rule**: If there is no tension, there is no story.

### 5.2 Robust Narrative Patterns
Choose one of these standard patterns to structure your logic. Do not reinvent the wheel.

**Pattern A: The "Gap" (Most Common)**
1.  **Status Quo**: Method A is the standard for Task X.
2.  **The Villain**: However, Method A fails when condition Z applies (The Gap).
3.  **The Insight**: This failure happens because of reason R.
4.  **The Hero**: We propose Method B, which fixes R.
5.  **The Proof**: We show B works on Z without breaking X.

**Pattern B: The "Mystery" (Insight-Driven)**
1.  **Observation**: We observe a weird phenomenon P in current models.
2.  **Investigation**: Why does P happen? We hypothesize H.
3.  **Confirmation**: We design experiments to verify H.
4.  **Solution**: Based on H, we design a simple fix F.
5.  **Result**: F eliminates P and improves performance.

**Pattern C: The "Democratization" (Efficiency/Systems)**
1.  **Problem**: Task X is solved, but requires massive compute/data (Elite only).
2.  **Goal**: We want to make X accessible to everyone.
3.  **Bottleneck**: The cost comes from component C.
4.  **Innovation**: We replace C with efficient approximation C'.
5.  **Trade-off**: We lose negligible accuracy for 10x speedup.

### 5.3 The "Hourglass" Flow
-   **Start Wide** (Intro): "Deep Learning transforms the world."
-   **Narrow Down** (Problem): "But RNNs forget long sequences."
-   **Deep Dive** (Method): "Here is the LSTM gate equation."
-   **Expand Out** (Results): "LSTM beats RNN on translation."
-   **End Wide** (Conclusion): "This enables infinite-context reasoning."

### 5.4 Writing Mechanics for Logic

-   **The "No Surprise" Rule**: The Introduction is a contract. The Method fulfills it. The Experiments prove it. Never introduce a major new motivation in the middle of the paper.
-   **Chekhov's Gun**: If you introduce a complex equation or concept in Section 3, you **must** use it or analyze it in Section 5. If it doesn't affect the results, cut it.
-   **Topic Sentences**: The first sentence of every paragraph should summarize that paragraph. A reader skimming **only the first sentences** should still understand the full argument.
-   **Signposting**: Tell the reader where you are going. "In this section, we first..., then..."
-   **Active Voice**: "We propose..." is better than "It is proposed...".
-   **The "So What?" Test**: Every paragraph should have a purpose. Ask "So what?" after writing it. If the answer is unclear, delete or rewrite.
