# StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization

Ziliang Wang1∗‡ , Xuhui Zheng1,2∗ †, Kang ${ \bf A } { \bf n } ^ { 1 , 3 * \dagger }$ ,   
Cijun Ouyang1, Jialu Cai1, Yuhang Wang1‡, Yichao Wu1§ 1SenseTime 2Nanjing University 3Shenzhen University {wangziliang1, ouyangcijun,   
caijialu, wangyuhang, wuyichao}@sensetime.com   
zhengxuhui@smail.nju.edu.cn, ankang@gml.ac.cn

# Abstract

Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained questionanswering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving $1 1 . 2 \%$ and $4 . 2 \%$ absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our code will be released on https://github.com/Zillwang/StepSearch

# 1 Introduction

Recent breakthroughs in Large Language Models (LLMs) have demonstrated unprecedented capabilities in sophisticated linguistic comprehension and generative tasks.

Reinforcement learning enhanced architectures(e.g., OpenAI-o3 [1], DeepSeek-R1 [2], and Kimi1.5 [3]) employ policy-gradient methods (PPO [4], GRPO [5]) to advance multi-hop logical reasoning [6]. However, complex multi-hop QA still suffers from intrinsic knowledge gaps [7] and static, inefficient knowledge-assimilation mechanisms [8, 9]. To address limited modeling of internal dependencies, recent work has adopted prompting strategies, RAG architectures, and tailored learning paradigms [10, 11].

Chain-of-Thought (CoT) prompting [12] decomposes complex inference into sequential subtasks but remains highly sensitive to prompt formulation and does not eliminate hallucinations. Retrieval-Augmented Generation (RAG) [11] dynamically incorporates external corpora to bridge knowledge gaps and suppress spurious content [13, 14, 15]; embedding structured knowledge graphs further enforces semantic coherence through explicit entity relations [16, 17]. Agentic frameworks—combining self-reflection, strategic planning, and multi-agent collaboration—facilitate adaptive task decomposition and iterative refinement [18, 19]. Advanced retrieval tactics (query reformulation, reranking, hybrid vector–keyword indexing) bolster multi-hop reasoning while filtering noise [20, 21]. Nonetheless, reliance on proprietary knowledge bases demands frequent updates to avert data obsolescence.

![](images/4d057cbc13e5de1191e760818ecfeae4a03103de049f9fd2e63184bfae4f1a5b.jpg)  
Figure 1: Performance comparisons between StepSearch and other methods on four multi-hop Q&A benchmarks with Qwen2.5-3B-Base as the base LLM. Our method achieves significant improvement.

Training-based paradigms endow LLMs with adaptive tool use by integrating external information sources (e.g., search engines) directly into the training loop. Supervised fine-tuning (SFT) frameworks—such as ToolFormer [9], ToolKengPT [22], and related efforts [23, 24]—substantially boost performance in specialized, knowledge-intensive tasks but suffer from poor out-of-domain generalization [25].

Recent advances have adopted reinforcement learning to learn dynamic retrieval policies, enabling models to iteratively query and integrate external knowledge based on the static RAG paradigm [26, 27]. Methods such as R1-Searcher [28], Search-R1 [29], ReSearch [30] and ZeroSearch [31] rely on answer and format-level rewards, empower agents to autonomously invoke search tools and achieve QA performance surpassing conventional RAG. DeepResearcher [32] further extends this paradigm to unconstrained online search environments, highlighting the scalability and potential of search-RL approaches. However, existing RL-based search agents depend on coarse global rewards, lacking fine-grained supervision of intermediate queries and multi-step retrievals—an approach inadequate for the dependencies inherent in complex multi-hop reasoning.

Process-level supervision enables the design of fine-grained reward functions that steer strategic query planning and enhance retrieval quality in complex search environments [33, 34, 35, 36]. However, existing step-reward methods—such as R1-VL [37] for pure logical reasoning and RPO [38] lack true token-level supervision for interactive retrieval tasks. Moreover, most multi-hop QA frameworks omit explicit guidance on query trajectories (e.g., intermediate search keywords or document usage), leaving a critical gap in search-path modeling. To address these shortcomings, We propose StepSearch, a reinforcement learning framework that integrates iterative retrieval with explicit step-wise supervision for search llms (Figure 2). Built on a pipeline that generates subquestion-aligned search-keyword trajectories, it also introduces a regenerated public multi-hop dataset for sequential retrieval benchmarking. By augmenting PPO with token-level rewards that combine information gain and redundancy penalties, StepSearch boosts policy convergence and improves retrieval fidelity and QA accuracy.

In general, our core contribution lies in:

![](images/d210f38e3f91f5131e76c00ab7ee1d4cc3089bb0a59e30291f4ba67e509d35b2.jpg)  
Figure 2: Step-wise search involves interactive rounds, with information gain being rewarded and redundancy penalised. Each interaction evaluates thinking and searching behaviour based on the retrieved results, with the final answer being used as the basis for global rewards.

• Universal multi-hop search data. We develop a novel MuSiQue-based pipeline, contributing 60k filtered sub-question search keywords that generalize across retrieval datasets.

• StepSearch: Step-wise RL with dual rewards. We augment PPO with token-level rewards—information gain and redundancy penalties—for both query formulation and document retrieval.

• State-of-the-art performance. StepSearch outperforms standard RL baselines by $5 . 7 \%$ , $9 . 1 \%$ $1 0 . 0 \%$ , and $1 5 . 2 \%$ absolutely on diverse multi-hop QA benchmarks.

# 2 Methodology

# 2.1 Data Augmentation Pipeline

In this pilot study, we construct a multi-turn Q&A dataset with subquestion-level search trajectories. Starting from the MusiQue [39] dataset, our pipeline show as (Figure. 3):

(a) Leverage GPT-4o. to enrich decomposed MuSiQue questions with coherent subquestion–answer pairs, then derive $N$ search queries per step for retrieval.   
(b) Each enhanced step question is then reformulated into a set of $N$ search queries to facilitate information retrieval.   
(c) Queries are issued to $M$ sources (e.g., Google, Bing, Wiki-18), and only those returning valid results in at least $\lceil M / 2 \rceil$ sources are retained.

# 2.2 Train LLM with Search Actions

To rapidly instill tool-augmented reasoning, we design a minimalist prompt template comprising three chain-of-thought demonstration pairs plus a dedicated label for retrieved results. Prompt templates for training can be found in the Appendix A, this schema enforces a consistent structure across reasoning, retrieval relying only on zero-shot guidance.

Rollout and Mask for Retrieve: (1) We run the loop of <think>...</think>, <search>. . . </search>, <information>. . . </information> iteratively, appending external docs until LLM returns <answer>. . . </answer> or the action budget is reached. (2) During RL training, we optimize a composite loss but mask out all <information>. . . </information> segments from gradient computation, thereby decoupling parameter updates from retrieval artifacts and focusing learning on the model’s internal reasoning and search-policy parameters, as established in prior search-RL work [28, 29, 30, 31, 32].

![](images/b0805929c748cde0aa20a618eaaf0f724ba88889fe2e148b25ed1eb94f504e7d.jpg)  
Figure 3: Data pipeline for generating the corresponding search query for the Q&A intermediate process.

# 2.3 StepSearch

In retrieval-augmented RL, carefully crafted rewards are pivotal to convergence and reasoning efficacy. In addition to the standard format and final-answer reward $r _ { \mathrm { a n s w e r } }$ , we introduce a search-key reward $r _ { \mathrm { k e y } }$ to promote informative query issuance directly. Our method further diverges from vanilla PPO by segmenting each turn into think search answer phases and assigning token-level rewards: each token earns an information-gain signal $\mathcal { G } ^ { t }$ and incurs a redundancy penalty $\mathcal { P } ^ { t }$ . This precise, process-aware supervision compels the model to decompose multi-hop queries into focused search subtasks, adapt its retrieval strategy dynamically, and integrate external evidence more effectively, yielding faster convergence and higher accuracy on complex reasoning benchmarks.

Our optimization algorithm combines the above-mentioned Search Steps supervision reward based on the currently widely used actor-critic approach PPO [4], denote as StePPO. For each sample input $x \backsim \mathcal { D }$ , obtain output $o$ from the old policy $\pi _ { \theta _ { o l d } }$ , let $I ( y _ { t } )$ be the token-loss masking indicator, it equals 1 when $o _ { t }$ is generated by actor LLM else 0 for retrieved tokens. Then optimize the policy $\pi _ { \theta }$ with the reference policy $\pi _ { \theta _ { r e f } }$ by maximizing the following objective:

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { S t e P P O } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , o \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x ) } } \\ & { \Bigg \{ \frac { 1 } { \sum _ { t = 1 } ^ { | \sigma | } I ( o _ { t } ) } \sum _ { t = 1 : I ( o _ { t } ) = 1 } ^ { | \omega | } \left[ \frac { \pi _ { \theta } \big ( o _ { t } | x , o _ { < t } \big ) } { \pi _ { \theta _ { \mathrm { o l d } } } \big ( o _ { t } | x , o _ { < t } \big ) } A _ { t } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } \big ( o _ { t } | x , o _ { < t } \big ) } { \pi _ { \theta _ { \mathrm { o l d } } } \big ( o _ { t } | x , o _ { < t } \big ) } , 1 - \epsilon , 1 + \epsilon \right) A _ { t } \right] \Bigg \} , } \end{array}
$$

here, $\epsilon$ is a hyper-parameter for clipping to stalblilize training, and $A _ { t }$ represents the estimated advantage computed with GAE algorithm [40], based on future rewards $r _ { \geq t }$ , which is composed of the gloabal and step-wise search round rewards, and a learned value function $V _ { \phi }$ . The global reward is set at the last position of the output, while the step-wise reward is set at the last token of each round of search behavior.

# 2.3.1 Type 1 Reward: Global Signal

Format Requirement: To ensure the model adopts the prescribed multi-step "search $^ +$ reason" workflow and correctly initiates search actions across iterative reasoning rounds, we enforce strict format validation as a hard constraint rather than implement it as an explicit reward. The required output format is defined as follows:

![](images/a61c71db2db05411afe3591110c527a7a9f822e84a8daddb22067ff7a160e3b0.jpg)  
Figure 4: Overview of StepSearch. At each step, the model issues queries to an external engine and receives snippets. Search Step Reward score, combining information gain and redundancy penalty, are applied to tokens within each round, while the global reward, based on final answer accuracy and keyword hit rate, is applied at the last token. Retrieved content is masked during training to isolate the model’s generative parameters.

• Only the search queries in the proper <search>. . . </search> pairs will be extracted and used to call search tools, and the answer must be in the <answer>. . . </answer> pair. • At least one round of "think" and "search" behaviour • Only one <answer>. . . </answer> tag pair to answer the question, and it must be at the end.

Answer Reward: We follow the classic reinforcement learning method and calculate the degree of the match by using the word-level f1 method between the answer and ground truth. Let $P N$ represent the word count of the predicted answer, $R N$ for word count of the golden answer and $I N$ stands for the word count of the intersection between them, then the answer reward $r _ { \mathrm { a n s w e r } }$ can be defined as:

$$
\mathrm { F 1 } ( x , y ) = \frac { 2 * I N } { P N + R N }
$$

$$
r _ { \mathrm { a n s w e r } } = { \left\{ \begin{array} { l l } { \mathrm { F 1 } ( a _ { \mathrm { p r e d } } , a _ { \mathrm { g t } } ) , } & { { \mathrm { f o r m a t ~ i s ~ c o r r e c t } } , } \\ { 0 , } & { { \mathrm { f o r m a t ~ i s ~ i n c o r r e c t } } . } \end{array} \right. }
$$

Search Keys Reward: We quantify the search-key reward by measuring the alignment between each emitted query and the reference keywords assigned to its corresponding subtask. Concretely, we compute a word-level F1 score—capturing token overlap to assess query quality. To guarantee adherence to the prescribed interaction protocol, this reward is granted only when the model’s search emission conforms to the required format, ensuring that policy updates reinforce both correct structure and effective retrieval behaviour. Suppose there are $T$ rounds of queries $Q = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { T } \}$ , $K _ { i } = \{ k _ { i 1 } , k _ { i 2 } , \ldots , k _ { i N _ { i } } \} , i = 1 , \ldots , \bar { M }$ corresponding $M$ subquestions which each contains $N _ { i }$ related golden queries, thus, the search keyword reward can be calculated as:

$$
f _ { i j t } = \mathrm { F } 1 ( q _ { t } , k _ { i j } ) ,
$$

$$
r _ { \mathrm { k e y } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left( \operatorname* { m a x } _ { 1 \leq j \leq N _ { i } } \left( \operatorname* { m a x } _ { 1 \leq t \leq T } f _ { i j t } \right) \right) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \operatorname* { m a x } _ { 1 \leq j \leq N _ { i } } \operatorname* { m a x } _ { 1 \leq t \leq T } f _ { i j t } .
$$

Type 1 Reward: Set $\gamma _ { \mathrm { k e y } }$ as the scale factor and the final reward rule can be expressed by the following formula:

$$
r _ { \mathrm { o v e r a l l } } = r _ { \mathrm { a n s w e r } } + \gamma _ { \mathrm { k e y } } \cdot r _ { \mathrm { k e y } } .
$$

# 2.3.2 Type 2 Reward: Search Step

The step-wise reward $r _ { \mathrm { s t e p } } ^ { t }$ of each round of search behaviour can be expressed as information gain $\mathcal { G } ^ { t }$ minus redundancy penalty $\mathcal { P } ^ { t }$ as Type 2 Reward:

$$
\begin{array} { r } { r _ { \mathrm { s t e p } } ^ { t } = \mathcal { G } ^ { t } - \mathcal { P } ^ { t } . } \end{array}
$$

Information Gains: To quantify the utility of each search action, we measure the marginal information contribution of its retrieved documents in reducing uncertainty about the target answer at the current reasoning stage.

Let $D ^ { g } = \{ d _ { 1 } ^ { g } , \ldots , d _ { n } ^ { g } \}$ denote the $n$ ground-truth documents required to resolve problem $p$ at search turn $t$ , where each $d _ { i } ^ { g }$ contains the gold-standard information for a specific subtask. We maintain a memory vector $\mathbf { \bar { \boldsymbol { M } } } ^ { t } = [ m _ { 1 } ^ { t } , \dots , m _ { n } ^ { t } ] ,$ , in which $m _ { i } ^ { t }$ records the maximum similarity observed to date between any retrieved document and $d _ { i } ^ { g }$ . At turn $t$ , the agent retrieves a set $D ^ { r ( t ) } = \{ d _ { 1 } ^ { r ( t ) } , \ldots , d _ { k } ^ { r ( t ) } \}$ of $k$ documents; we denote by $c _ { j } ^ { t }$ the similarity between $\underset { . } { d _ { j } ^ { r ( t ) } }$ and its corresponding golden document(s). To evaluate this alignment, we adopt a submodular coverage function instantiated with cosine similarity over $T F { - } I D F$ [41] representations, which naturally enforces diminishing returns and penalizes redundant retrievals.

First initialize $m _ { i } ^ { t }$ to 0, the current matching degree of the round $t$ search results can be calculated based on each golden info document, and the highest similarity among the search documents in this round is taken as $c _ { i } ^ { t }$ :

$$
c _ { i } ^ { t } = \operatorname* { m a x } _ { 1 \leq j \leq k } \frac { \overrightarrow { d _ { i } ^ { \acute { \beta } } } \cdot \overrightarrow { d _ { j } ^ { r ( t ) } } } { \vert \vert \overrightarrow { d _ { i } ^ { \acute { \beta } } } \vert \vert \cdot \vert \vert \overrightarrow { d _ { j } ^ { r ( t ) } } \vert \vert } , \quad i = 1 , \cdot \cdot , n .
$$

The valuable information gain $\triangle _ { i } ^ { t }$ on golden document $d _ { i } ^ { g }$ of this round $t$ is calculated based on the current matching degree $c _ { i } ^ { t }$ of this round and the global maximum matching degree $m _ { i } ^ { t }$ of the previous round:

$$
\triangle _ { i } ^ { t } = \operatorname* { m a x } ( c _ { i } ^ { t } - m _ { i } ^ { t } , 0 ) , \quad i = 1 , \cdot \cdot \cdot , n ,
$$

then, the overall information gain value of tth round is the average gain of $\mathbf { n }$ golden info documents in the current round:

$$
\mathcal { G } ^ { t } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \triangle _ { i } ^ { t } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m a x } ( c _ { i } ^ { t } - m _ { i } ^ { t } , 0 ) .
$$

Finally, the accumulated global maximum matching record value is updated for evaluation in the next round of search behavior:

$$
\begin{array} { r } { m _ { i } ^ { t } = m a x ( m _ { i } ^ { t - 1 } , c _ { i } ^ { t } ) , \quad i = 1 , \cdots , n . } \end{array}
$$

Then update the current maximum information matching degree record for subsequent iterative calculations.

# Redundancy Penalty:

During search-stage supervision, we observed that repetitive confirmatory queries both waste budget and amplify hallucinations without effective feedback. To counter this, we track a cumulative retrieval history $H ^ { t }$ (with $H ^ { 0 } = \varnothing$ ) and let each round’s retrieved set be $I ^ { t }$ . Any query whose results overlap with $\dot { H } ^ { t - 1 }$ incurs a redundancy penalty, discouraging low-value repetition and promoting novel, informative retrievals. At the end of round $t$ , we update

$$
H ^ { t } = H ^ { t - 1 } \cup I ^ { t } ,
$$

the redundancy penalty value $\mathcal { P } ^ { t }$ of the tth round can be expressed by counting the proportion of the documents retrieved $\dot { D } ^ { r ( t ) }$ in this round that are repeated in any previous round:

$$
\mathcal { P } ^ { t } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } 1 ( d _ { j } ^ { r ( t ) } \in H ^ { t - 1 } ) ,
$$

where $\mathbf { 1 } ( \cdot )$ is the indicator function.

# 3 Experiment

# 3.1 Dataset and Evaluation Metrics

During training with process supervision, Our empirical benchmarks span four established multihop Q&A datasets: (1) HotpotQA [42], (2) MuSiQue [39], (3) 2WikiMultiHopQA [43], and (4) Bamboogle [44].

To maintain alignment with prior work [45, 28] and guarantee fair evaluation, we report the canonical word-level F1 and Exact Match (EM) scores. We eschew third-party LLM judges due to their reproducibility and stability limitations.

# 3.2 Baselines

We evaluate StepSearch against a diverse set of representative baselines, chosen to cover both prompting and reinforcement-learning paradigms as well as both static and dynamic retrieval strategies:

• Naive Generation: Direct generation and Chain-of-Thought (CoT) [12] reasoning;   
• RAG: naive Retrieval-Augmented Generation (RAG) and IRCoT [46] which use retrieval and the chain-of-thought (CoT) guide each other;   
• SFT: Supervised fine-tuning (SFT) [47] without a search engine;   
• Large Reasoning Model: RL-based fine-tuning (R1) [2] without a search engine and reasoning with in-process search (Search-o1) [19];   
• Search with RL: Existing outstanding reinforcement learning methods combined with external search engines including Search-R1 [28], ZeroSearch [31] and ReSearch [30]. To ensure a fair comparison, we adopt the original open-source model checkpoints and their published prompt configurations, and standardize all retrieval and hyperparameter settings across experiments.

# 3.3 Training Details

We conduct experiments on 4 models from Qwen family [48]: Qwen-2.5-7B (Base/Instruct) and Qwen-2.5-3B (Base/Instruct). In order to support the training of StepSearch, we generated our dataset with process supervision reference information based on the MuSiQue [39] dataset using the search key synthesis pipeline in Section 2.1.

During training, we employ E5 [49] as the retriever over our synthesized dataset. For evaluation, we augment the corpus with the 2018 Wikipedia dump [50], as in Search-R1 [28], and uniformly retrieve $k = 3$ documents. Prompt-based baselines use Instruct models, whereas RL methods are evaluated on both Base and Instruct variants to gauge cross-model robustness. A more detailed experimental setup can be found in the Appendix B.

# 3.4 Main Results

The main results comparing StepSearch with baseline methods across the four datasets (containing different retrieval bases) are presented in Table 1. The results in these tables summarize the following key findings: (1) StepSearch consistently outperforms strong baseline Search-RL methods. Our method performance advantage holds for both in-domain multi-hop (i.e., MuSiQue) and out-ofdomain (i.e., HotpotQA, 2WikiMultiHopQA, and Bamboogle) datasets, demonstrating the robustness of our method. (2) StepSearch exhibits robust generalization, particularly in smaller-scale models. Under models of different sizes and types (base and instruction), our method generally shows better performance than the strong baseline model. The process supervision method can be plug-and-play combined with the PPO algorithm to improve the performance of Search-RL tasks smaller models are greatly motivated to improve their search capabilities. (3) StepSearch shows higher adaptability to out-of-domain knowledge bases. Using only a knowledge base with a smaller amount of retrieval database (about $0 . 3 5 \%$ ) and training data (about $1 1 \%$ ) can show even better adaptability compared to models that are trained on larger datasets, and our methodology guarantees top results for searches in out-of-domain retrieval databases than others.

<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA†</td><td colspan="2">2Wikit</td><td colspan="2">MuSiQue†</td><td colspan="2">Bamboogle†</td><td colspan="2">MuSiQue*</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="9">Qwen2.5-3b-Base/Instruct</td><td></td><td></td></tr><tr><td>Naive Geneartion</td><td>0.145</td><td>0.237</td><td>0.249</td><td>0.356</td><td>0.018</td><td>0.079</td><td>0.030</td><td>0.086</td><td></td><td></td></tr><tr><td>RAG</td><td>0.251</td><td>0.359</td><td>0.221</td><td>0.316</td><td>0.051</td><td>0.135</td><td>0.076</td><td>0.161</td><td></td><td></td></tr><tr><td>SFT</td><td>0.191</td><td>0.299</td><td>0.248</td><td>0.356</td><td>0.039</td><td>0.110</td><td>0.112</td><td>0.181</td><td></td><td></td></tr><tr><td>IRCoT</td><td>0.171</td><td>0.241</td><td>0.196</td><td>0.265</td><td>0.059</td><td>0.123</td><td>0.245</td><td>0.356</td><td>-</td><td></td></tr><tr><td>R1-base</td><td>0.191</td><td>0.306</td><td>0.277</td><td>0.334</td><td>0.051</td><td>0.106</td><td>0.232</td><td>0.305</td><td></td><td></td></tr><tr><td>R1-instruct</td><td>0.210</td><td>0.299</td><td>0.288</td><td>0.389</td><td>0.065</td><td>0.136</td><td>0.213</td><td>0.317</td><td></td><td></td></tr><tr><td>Search-o1</td><td>0.240</td><td>0.326</td><td>0.207</td><td>0.309</td><td>0.045</td><td>0.117</td><td>0.316</td><td>0.436</td><td></td><td></td></tr><tr><td>ZeroSearch-base</td><td>0.260</td><td>0.354</td><td>0.234</td><td>0.281</td><td>0.056</td><td>0.116</td><td>0.096</td><td>0.193</td><td>0.072</td><td>0.151</td></tr><tr><td>ZeroSearch-instruct</td><td>0.265</td><td>0355</td><td>0.233</td><td>0.278</td><td>0.059</td><td>0.121</td><td>0.144</td><td>0.243</td><td>0.073</td><td>0.147</td></tr><tr><td>Search-R1-baset</td><td>0.272</td><td>0.361</td><td>0.248</td><td>0.296</td><td>0.081</td><td>0.146</td><td>0.176</td><td>0.270</td><td>0.118</td><td>0.193</td></tr><tr><td>Search-R1-instruct†</td><td>0.304</td><td>0.401</td><td>0.293</td><td>0.352</td><td>0.120</td><td>0.188</td><td>0.240</td><td>0.344</td><td>0.162</td><td>0.240</td></tr><tr><td>StepSearch-base</td><td>0.329</td><td>0.434</td><td>0.339</td><td>0.395</td><td>0.181</td><td>0.273</td><td>0.328</td><td>0.419</td><td>0.274</td><td>0.375</td></tr><tr><td>StepSearch-instruct*</td><td>0.345</td><td>0.452</td><td>0.320</td><td>0.385</td><td>0.174</td><td>0.261</td><td>0.344</td><td>0.452</td><td>0.258</td><td>0.357</td></tr><tr><td colspan="9">Qwen2.5-7b-Base/Instruct</td><td></td><td></td></tr><tr><td>Naive Geneartion</td><td>0.187</td><td>0.291</td><td>0.246</td><td>0.352</td><td>0.027</td><td>0.083</td><td>0.123</td><td>0.242</td><td></td><td></td></tr><tr><td>SFT</td><td>0.196</td><td>0.175</td><td>0.269</td><td>0.374</td><td>0.054</td><td>0.131</td><td>0.110</td><td>0.203</td><td></td><td></td></tr><tr><td>IRCoT</td><td>0.141</td><td>0.232</td><td>0.142</td><td>0.241</td><td>0.072</td><td>0.159</td><td>0.216</td><td>0.319</td><td></td><td></td></tr><tr><td>RAG</td><td>0.287</td><td>0.391</td><td>0.231</td><td>0.226</td><td>0.061</td><td>0.142</td><td>0.214</td><td>0.316</td><td></td><td></td></tr><tr><td>R1-base</td><td>0.234</td><td>0.326</td><td>0.270</td><td>0.368</td><td>0.076</td><td>0.151</td><td>0.287</td><td>0.395</td><td></td><td></td></tr><tr><td>R1-instruct</td><td>0.241</td><td>0.345</td><td>0.287</td><td>0.392</td><td>0.079</td><td>0.154</td><td>0.284</td><td>0.397</td><td></td><td></td></tr><tr><td>Search-o1</td><td>0.193</td><td>0.288</td><td>0.181</td><td>0.289</td><td>0.053</td><td>0.127</td><td>0.302</td><td>0.427</td><td></td><td></td></tr><tr><td>ZeroSearch-base</td><td>0.294</td><td>0.394</td><td>0.275</td><td>0.324</td><td>0.102</td><td>0.175</td><td>0.258</td><td>0.373</td><td>0.134</td><td>0.218</td></tr><tr><td>ZeroSearch-instruct</td><td>0.325</td><td>0.432</td><td>0.309</td><td>0.370</td><td>0.120</td><td>0.204</td><td>0.267</td><td>0.409</td><td>0.184</td><td>0.280</td></tr><tr><td>Research-base*</td><td>0.294</td><td>0.388</td><td>0.264</td><td>0.313</td><td>0.143</td><td>0.230</td><td>0.373</td><td>0.449</td><td>0.206</td><td>0.309</td></tr><tr><td>Research-instruct*</td><td>0.362</td><td>0.471</td><td>0.354</td><td>0.416</td><td>0.184</td><td>0.271</td><td>0.424</td><td>0.544</td><td>0.250</td><td>0.348</td></tr><tr><td>Search-R1-baset</td><td>0.432</td><td>0.547</td><td>0.350</td><td>0.411</td><td>0.206</td><td>0.290</td><td>0.430</td><td>0.545</td><td>0.305</td><td>0.401</td></tr><tr><td>Search-R1-instruct†</td><td>0.394</td><td>0.502</td><td>0.312</td><td>0.376</td><td>0.181</td><td>0.262</td><td>0.384</td><td>0.501</td><td>0.268</td><td>0.352</td></tr><tr><td>StepSearch-base*</td><td>0.380</td><td>0.493</td><td>0.385</td><td>0.450</td><td>0.216</td><td>0.324</td><td>0.467</td><td>0.573</td><td>0.346</td><td>0.461</td></tr><tr><td>StepSearch-instruct*</td><td>0.386</td><td>0.502</td><td>0.366</td><td>0.431</td><td>0.226</td><td>0.312</td><td>0.400</td><td>0.534</td><td>0.339</td><td>0.443</td></tr></table>

Table 1: The main results of StepSearch on 4 multi-hop Q&A datasets using different retrieval databases. Search-R1 is trained based on $\mathrm { N Q + 1 }$ HotpotQA dataset (170k) while ours and ReSearch are on MuSiQue (19k), "†" refers to train or test on wiki-18 knowledge base and " $" * "$ for our customized base build on MuSiQue.

# 4 Further Analysis

# 4.1 Different RL Comparison

![](images/ef42feb6a48e9dd4d45b60338634608d9e1afa0f90ce5906cca952a7f1408b7b.jpg)  
Figure 5: Comparison of StePPO vs. PPO.

Leveraging Qwen2.5-Base (3B/7B), we compare GRPO and PPO against StePPO. Results in Table 2 and training curves in Figure 5 show that (1) StePPO drives high-quality, low-cost generation (Figure 5b), achieving higher accuracy in fewer rounds with shorter outputs due to enriched search supervision; (2) PPO-based algorithm delivers the most stable training, while GRPO (both Base and Instruct) is prone to reward collapse at higher learning rates (Figure 5a); and (3) StePPO attains the highest convergence speed and peak effectiveness, outperforming both PPO and GRPO in final F1 (Figure 5a, Table 2).

<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">2Wiki</td><td colspan="2">MuSiQue</td><td colspan="2">Bamboogle</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="9">Qwen2.5-7b-Base</td></tr><tr><td>StePPO</td><td>0.380</td><td>0.493</td><td>0.385</td><td>0.450</td><td>0.216</td><td>0.324</td><td>0.467</td><td>0.573</td></tr><tr><td>PPO</td><td>0.374</td><td>0.479</td><td>0.282</td><td>0.329</td><td>0.198</td><td>0.280</td><td>0.432</td><td>0.549</td></tr><tr><td>GRPO</td><td>0.351</td><td>0.462</td><td>0.266</td><td>0.345</td><td>0.202</td><td>0.291</td><td>0.400</td><td>0.512</td></tr><tr><td colspan="9">Qwen2.5-3b-Base</td></tr><tr><td>StePPO</td><td>0.329</td><td>0.434</td><td>0.339</td><td>0.395</td><td>0.181</td><td>0.273</td><td>0.328</td><td>0.419</td></tr><tr><td>PPO</td><td>0.223</td><td>0.315</td><td>0.225</td><td>0.273</td><td>0.090</td><td>0.163</td><td>0.176</td><td>0.266</td></tr><tr><td>GRPO</td><td>0.256</td><td>0.366</td><td>0.256</td><td>0.325</td><td>0.114</td><td>0.190</td><td>0.224</td><td>0.314</td></tr></table>

Table 2: Performance of models trained by different RL algorithms on multi-hop Q&A datasets. PPO and GRPO are trained on the reward of final answer F1. The retrieval is based on Wikipedia knowledge from 2018, as is the main experiment.

# 4.2 Ablation Study

<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">2Wiki</td><td colspan="2">MuSiQue</td><td colspan="2">Bamboogle</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="9">Qwen2.5-7b-Base</td></tr><tr><td>StePPO</td><td>0.380</td><td>0.493</td><td>0.385</td><td>0.450</td><td>0.216</td><td>0.324</td><td>0.467</td><td>0.573</td></tr><tr><td>w/o key reward</td><td>0.404</td><td>0.528</td><td>0.388</td><td>0.468</td><td>0.204</td><td>0.315</td><td>0.432</td><td>0.542</td></tr><tr><td>w/o penalty</td><td>0.377</td><td>0.494</td><td>0.300</td><td>0.367</td><td>0.190</td><td>0.286</td><td>0.392</td><td>0.502</td></tr><tr><td>w/o infon gain</td><td>0.365</td><td>0.468</td><td>0.3651</td><td>0.422</td><td>0.208</td><td>0.303</td><td>0.421</td><td>0.540</td></tr><tr><td>w/o steps reward</td><td>0.361</td><td>0.475</td><td>0.360</td><td>0.433</td><td>0.192</td><td>0.283</td><td>0.384</td><td>0.485</td></tr><tr><td colspan="9">Qwen2.5-3b-Base</td></tr><tr><td>PPO</td><td>0.228</td><td>0.315</td><td>0.225</td><td>0.273</td><td>0.090</td><td>0.163</td><td>0.176</td><td>0.266</td></tr><tr><td>ow key reward</td><td>0.259</td><td>0.375</td><td>0.178</td><td>0.282</td><td>0.127</td><td>0.218</td><td>0.232</td><td>0.334</td></tr><tr><td>ow penalty</td><td>0.258</td><td>0.364</td><td>0.227</td><td>0.279</td><td>0.083</td><td>0.177</td><td>0.192</td><td>0.312</td></tr><tr><td>ow info gain</td><td>0.323</td><td>0.432</td><td>0.355</td><td>0.425</td><td>0.169</td><td>0.249</td><td>0.344</td><td>0.439</td></tr><tr><td>ow steps reward</td><td>0.328</td><td>0.437</td><td>0.326</td><td>0.391</td><td>0.185</td><td>0.282</td><td>0.360</td><td>0.487</td></tr><tr><td>ow sub-answer</td><td>0.339</td><td>0.448</td><td>0.293</td><td>0.354</td><td>0.176</td><td>0.258</td><td>0.312</td><td>0.432</td></tr></table>

Table 3: Results on models 7b and 3b respectively. $\overline { { \bf { \Omega } _ { w } / o ^ { \prime } } }$ represent ‘with out’ while $\overline { { \cdot _ { o w } } } ^ { , }$ for ’only with’, ’sub-answer’ represents a process supervision rewards based on intermediate sub-answers.

To further validate the effectiveness of StePPO’s search process reward mechanism, we conducted extensive ablation experiments based on the Qwen2.5-7B-Base model. The Table 3 below shows the evaluation of each configuration model on different datasets, and Figure 6. shows the experimental process record. The experimental results revealed these phenomena: (1) StePPO has more prominent advantages in small parameter models (Figure 6a, Table 3). Compared with the classic RL algorithm, our method achieves more obvious search answer quality on the 3B parameter model than the 7B model and has the advantage of convergence speed.; (2) Redundancy penalty alone does not optimize the search ability (ow-rp in Figure 6a), but it can force the model to perform high-quality, low-repetition effective search when applying information gain calculation (ow-ss in Figure 6a), thereby achieving a higher capacity ceiling; (3) Searching keyword reward values can significantly improve the model convergence speed (ow-skr in Figure 6a and 6c), but without process supervision, it may lead to hallucination and reward collapse problems more quickly, causing the model to respond incoherently and fail to converge; (4) The effectiveness of fine-grained process rewards, the step-wise token-level reward mechanism (ow-ss) has more obvious advantages and stability compared to the global reward of search process (ow-skr).

In addition, we have tried to let the model answer the sub-task answers in the intermediate process and provide feedback in the global reward. Experiments have shown that the keywords in the supervised search behavior process are similar to the sub-question answers, and can bring obvious improvements in results on various data sets. The success of various methods has further proved the effectiveness of process supervision. The prompt template can be found in the Appendix A.

![](images/f6c3fe3915b3bba244807dd0ae24b75154a0c9b378a61de5fc64d223c38f20c5.jpg)  
Figure 6: Training dynamics of correctness, response length, and search-key scores in ablation experiments. $\ ' _ { w / o } '$ represent ‘with out’ while ‘ow’ for ’only with’, $\ ' _ { r p }$ ’ stands for ’redundancy penalty’,’ig’ for ’information gain’, $\ ' _ { s s } \ '$ is ’step score’ $( ` i g ^ { , } + ` r p ^ { , } )$ and ’skr’ means the global reward ’search key reward’

# 4.3 Case Study

More cases about the effect and content of model responses with different parameter configurations can be found in the Appendix C.

# 5 Limitations

Despite the advances demonstrated by our retrieval-augmented reasoning framework, it remains subject to several important limitations. Evaluation has been restricted to text-only question answering, leaving open the question of how well the approach generalizes to multimodal inputs (e.g., images, audio) and to tasks that cross paradigmatic boundaries. And we have tested only at relatively modest parameter scales; scaling to larger models (e.g., 14 B, 32 B) may exacerbate issues such as reward collapse and unstable training dynamics, necessitating novel stabilization and regularization strategies. StepSearch currently only uses 19k training data from MuSiQue for training. We don’t know how StepSearch will perform when using the same scale of training data as Search-R1. Future work will need to address these gaps to realize a truly generalizable, robust, and scalable retrieval-augmented agent.

# 6 Future Work

Looking ahead, we plan to enhance the agent’s post-retrieval comprehension and metacognitive reflection, enabling more nuanced integration of external evidence; enrich the diversity and complexity of search behaviors to better mirror real-world information-seeking dynamics; and extend our framework beyond text-only QA to a truly multimodal, cross-paradigm setting. Concurrently, we will explore scaling StePPO to larger parameter regimes (e.g., $1 4 \mathrm { ~ B ~ }$ and $3 2 \mathrm { ~ B ~ }$ ) and develop robust countermeasures against reward collapse to ensure stable, efficient training at scale.

References   
[1] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024.   
[2] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025.   
[3] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025.   
[4] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal

policy optimization algorithms, 2017.

[5] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.   
[6] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning, 2025.   
[7] Joohyun Lee and Minji Roh. Multi-reranker: Maximizing performance of retrieval-augmented generation in the financerag challenge, 2024.   
[8] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O. Arik. Long-context llms meet rag: Overcoming challenges for long inputs in rag, 2024.   
[9] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539– 68551, 2023.   
[10] Avinash Patil. Advancing reasoning in large language models: Promising methods and approaches, 2025.   
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.   
[12] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
[13] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A survey, 2024.   
[14] Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. A comprehensive survey of retrievalaugmented generation (rag): Evolution, current landscape and future directions, 2024.   
[15] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, page 6491–6501, New York, NY, USA, 2024. Association for Computing Machinery.   
[16] Karthik Soman, Peter W Rose, John H Morris, Rabia E Akbas, Brett Smith, Braian Peetoom, Catalina Villouta-Reyes, Gabriel Cerono, Yongmei Shi, Angela Rizk-Jackson, Sharat Israni, Charlotte A Nelson, Sui Huang, and Sergio E Baranzini. Biomedical knowledge graph-optimized prompt generation for large language models, 2024.   
[17] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization, 2025.   
[18] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented generation: A survey on agentic rag, 2025.   
[19] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models, 2025.   
[20] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Rajaram Naik, Pengshan Cai, and Alfio Gliozzo. Re2g: Retrieve, rerank, generate, 2022.   
[21] Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki. Blended rag: Improving rag (retriever-augmented generation) accuracy with semantic search and hybrid query-based retrievers. In 2024 IEEE 7th International Conference on Multimedia Information Processing and Retrieval (MIPR), volume 24, page 155–161. IEEE, August 2024.   
[22] Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. Advances in neural information processing systems, 36:45870–45894, 2023.   
[23] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. Frontiers of Computer Science, 19(8):198343, 2025.   
[24] Zhengliang Shi, Shen Gao, Lingyong Yan, Yue Feng, Xiuyi Chen, Zhumin Chen, Dawei Yin, Suzan Verberne, and Zhaochun Ren. Tool learning in the wild: Empowering language models as automatic tool agents, 2025.   
[25] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training, 2025.   
[26] Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Julia Hockenmaier, and Tong Zhang. Rag-rl: Advancing retrieval-augmented generation via rl and curriculum learning, 2025.   
[27] Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, and Jiawei Han. Deepretrieval: Hacking real search engines and retrievers with large language models via reinforcement learning, 2025.   
[28] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025.   
[29] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning, 2025.   
[30] Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen. Research: Learning to reason with search for llms via reinforcement learning, 2025.   
[31] Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, and Yan Zhang. Zerosearch: Incentivize the search capability of llms without searching, 2025.   
[32] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments, 2025.   
[33] Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, and Sujian Li. Chain-of-thought matters: Improving long-context language models with reasoning path supervision, 2025.   
[34] Zihuiwen Ye, Luckeciano Carvalho Melo, Younesse Kaddar, Phil Blunsom, Sam Staton, and Yarin Gal. Uncertainty-aware step-wise verification with generative reward models, 2025.   
[35] Yufan Ye, Ting Zhang, Wenbin Jiang, and Hua Huang. Process-supervised reinforcement learning for code generation, 2025.   
[36] Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, and Furu Wei. Chainof-retrieval augmented generation. CoRR, abs/2501.14342, 2025.   
[37] Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, and Dacheng Tao. R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization, 2025.   
[38] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Rithesh Murthy, Liangwei Yang, Zuxin Liu, Tian Lan, Ming Zhu, Juntao Tan, Shirley Kokane, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, and Caiming Xiong. Pract: Optimizing principled reasoning and acting of llm agent, 2024.   
[39] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.   
[40] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.   
[41] Juan Ramos et al. Using tf-idf to determine word relevance in document queries. Proceedings of the First Instructional Conference on Machine Learning, 242(1):29–48, 2003.   
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.   
[43] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.   
[44] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350, 2022.   
[45] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation in llms. Advances in Neural Information Processing Systems, 37:121156–121184, 2024.   
[46] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10014–10037, 2023.   
[47] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.   
[48] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025.   
[49] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533, 2022.   
[50] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP (1), pages 6769–6781, 2020.   
[51] Author Sheng and Others. Hybridflow: A hybrid approach to large language model training. arXiv preprint arXiv:2401.12345, 2024.

# A Prompt for Research Plan on Question Answering

To rapidly instill tool-augmented reasoning, we design a minimalist prompt template comprising three chain-of-thought demonstration pairs plus a dedicated label for retrieved results. Prompt templates for training can be found in the Table 4, this schema enforces a consistent structure across reasoning, retrieval relying only on zero-shot guidance.

In addition, we have tried to let the model answer the sub-task answers in the intermediate process and provide feedback in the global reward. Experiments have shown that the keywords in the supervised search behavior process are similar to the sub-question answers, and can bring obvious improvements in results on various data sets. The success of various methods has further proved the effectiveness of process supervision. The prompt template for this response can be found in the Table 5.

# B Experiment Setups

Our implementation is based on Search-R1 [28], and our training is conducted using Verl [51]. Our experiments are carried out on two series of models: Qwen-2.5-3B and Qwen-2.5-7B [48]. The MuSiQue [39] training set processed through our pipeline is used for training, while the full 2WikiMultiHopQA [43], Bamboogle [44], HotpotQA [42], and MuSiQue test or validation sets are used for evaluation. EM and F1 score are employed as evaluation metrics.

We train for a total of 500 steps. The learning rates for the policy and value large models are set to $7 e - 7$ and $7 e - 6$ , respectively, with warm-up ratios of 0.285 and 0.015 learning rate warm-up steps ratio. Training is conducted across two nodes with 16 H800 GPUs. The total batch size, mini-batch size, and micro-batch size are set to 256, 64, and 32, respectively. To optimize GPU memory usage, we employ Fully Sharded Data Parallel (FSDP) with CPU offloading, and the GPU memory utilization ratio is set to 0.7.

For rollout sampling, both the temperature and top_ $p$ are set to 1.0. The KL-divergence regularization coefficient $\beta$ and the clipping ratio are set to $1 e - 3$ and 0.2, respectively.

# C Case Study

We observe the sampling cases during the training process, and some of the sampling results are shown in the Table 6, 7. Here, we label the parts related to the Step Score Reward in different ways. By comparing the results generated by the model at different stages of training, we can see that in the early stages, the model repeatedly initiates similar search keywords, which wastes computational resources and makes it more likely to fail to answer correctly before reaching the search count. On the one hand, the low quality of the designed query means that the required document cannot be found directly, and on the other hand, the model cannot trust the correct document and tends to reconfirm.

In addition, we observed in our experiments that when the GLOBAL level of search key score was applied as the only process reward, the model was more likely to discover the trick of the format reward. This led to anomalies in generation and interaction behaviours and ultimately caused a training crash. As shown in the Table 8, the model issued a query request without using the correct search format and even made up information for phantom answers.

Question: What educational institution publishes the Quarterly, named after the state where Eastwood Park Historic District is located?

# Ground Truth: [’University of North Dakota’, ’UND’]

StepSearch:   
<think> My goal is to find the educational institution that publishes the Quarterly, which is named after the state where Eastwood Park Historic District is located. </think>   
<search> Eastwood Park Historic District </search>   
<information> Doc 1<## Title: Eastwood Park Historic District ##> The Eastwood Park Historic District is a historic district... Doc 2<## Title: Eastwood, Houston ##> Eastwood is a historic neighborhood in the East End ... Doc 3<## Title: Ravenswood Manor Historic District ##> The Ravenswood Manor Historic District is a ... </information>   
<think> I found that Eastwood Park Historic District is located in Minot, North Dakota, but I need to identify the state’s name to find the Quarterly’s publisher. I suggest searching for the location of Eastwood Park Historic District and the state it is named after. </think>   
<search> location of Eastwood Park Historic District </search>   
<information> Doc 1<## Title: Eastwood Park Historic District ##> The Eastwood Park Historic District is a historic district... Doc 2<## Title: Eastwood, Houston ##> Eastwood is a historic neighborhood ... Doc 3<## Title: Eastwood Village ##> Eastwood Village, formerly Eastwood Mall, was a shopping mall...</information>   
<think> The information now confirms that the Eastwood Park Historic District is in Minot, North Dakota. Now, I will search for educational institutions in Minot, North Dakota and their associated Quarters. </think>   
<search> educational institutions in Minot, North Dakota </search>   
<information> Doc 1<## Title: Eastwood Park Historic District ##> The Eastwood Park Historic District is a historic district ... Doc 2<## Title: North Dakota School for the Deaf ##> The North Dakota School for the Deaf (NDSD) is a state-funded residential school... Doc 3<## Title: Oklahoma City ##> Private career and technology education schools in Oklahoma City include Oklahoma Technology Institute... </information>   
<think>The search results provide information about the North Dakota Quarterly, which is a quarterly literary journal published by the University of North Dakota. This Quarterly is located in Minot, North Dakota, which is consistent with the Eastwood Park Historic District location. I have gathered enough information and identified the educational institution that publishes the Quarterly, named after the state where Eastwood Park Historic District is located. </think>   
<answer> University of North Dakota </answer>

Table 6: A case at the early train steps of redundancy penalty and effective information gain role in StePPO training. Red-font and underlined text indicates a golden document containing a staged answer, gray text is a question-independent document, and strikethrough text indicates a redundant retrieval document.

Question: The city where KBQI is found, is located in which county, and state?

Ground Truth: [’Bernalillo County, New Mexico’, ’Bernalillo County’]

StepSearch:   
<think> First, I need to identify the city where KBQI is located. Then, I will find the county and state in which that city is situated. </think>   
<search> where is KBQI located </search>   
<information> Doc 1<## Title: KBQI ##> KBQI (107.9 FM, "Big I 107.9") is a radio station in Albuquerque... Doc 2<## Title: KBIK ##> KBIK (102.9 FM) is a radio station licensed in Independence, Kansas... Doc 3<## Title: KBIF ##> KBIF (900 AM) is a radio station broadcasting a World Ethnic format to the Fresno... </information>   
<think> From the information, I found out that KBQI is located in Albuquerque. Now I need to find the county and state in which Albuquerque is situated. </think>   
<search> Albuquerque county and state </search>   
<information> Doc 1<## Title: Bernalillo County Metropolitan Court ##> The Bernalillo County Metropolitan Court is the Judicial system of the metropolitan areas of Albuquerque, New Mexico and Bernalillo County, New Mexico. Doc 2<## Title: Prieta Mesa ##> Prieta Mesa is a large mesa located northwest of Albuquerque in Sandoval County, New Mexico... Doc 3<## Title: Embudo, New Mexico ##> Embudo (also Embudo Station) is an unincorporated community in Rio Arriba County, New Mexico, United States... </information>   
<think> Albuquerque is located in Bernalillo County, New Mexico. </think>   
<answer> Bernalillo County, New Mexico </answer>

Question: When was the leader who wanted to unify Germany (Prussia) born?

# Ground Truth: [’1862’]

StepSearch:   
<think> First, I need to know who the leader was that wanted to unify Germany (Prussia). Then, I will search for the birth date of that leader. </think>   
<search> Who is the leader that wanted to unify Germany (Prussia)? </search>   
<information> Doc 1<## Title: Unification of Germany ##> Historians debate whether Otto von Bismarck... Doc 2<## Title: William I, German Emperor ##> William I, or in German Wilhelm I. (full name: William Frederick Louis... Doc 3<## Title: William I, German Emperor ##> William I, ( ; 22 March 1797 – 9 March 1888) of the House... </information>   
<think> Based on the information, the leader who wanted to unify Germany (Prussia) was Otto von Bismarck. Now, I will search for the birth date of Otto von Bismarck. </think>   
<search> When was Otto von Bismarck born? </search” (!!! wrong format search action !!!) <information> (Fake) Doc 1<## Title: Otto von Bismarck ##> Otto Eduard Leopold, Prince of Bismarck, Duke of Lauenburg (German: Ottheinrich Leo pratz von Bismarck... </information> (Fake)   
<think> I found that Otto von Bismarck was born on 1 April 1815. So, the answer is 1 April 1815. </think>   
<search> When was Otto von Bismarck born? </search) (!!! wrong format search action !!!) <information> (Fake) Doc 1<## Title: Otto von Bismarck ##> Otto Eduard Leopold, Prince of Bismarck, Duke of Lauenburg (German: Ottheinrich Leo pratz von Bismarck... </information> (Fake)   
<search> When was Otto von Bismarck born? </search» (!!! wrong format search action !!!) <search> When was Otto von Bismarck born? </search] (!!! wrong format search action !!!) <think> Based on the information provided, the leader who wanted to unify Germany (Prussia) was Otto von Bismarck. Therefore, the answer is 1 April 1815. </think>   
<answer> 1 April 1815 </answer>

# Ground Truth: [’Haider al - Abadi’]

<think>To answer the question of who the current prime minister of the nation that Yussef AlSuwayed was from is, let’s break down the information step by step. </think>   
<sub_question1>What country is Yussef Al-Suwayed from?</sub_question1>   
<think>To determine Yussef Al-Suwayed’s country of origin, I need to search for this information. </think>   
<search>Who is Yussef Al-Suwayed and what country is he from?</search>   
<information> Doc 1<## Title: Yussef Al-Suwayed ##>Yussef Al-Suwayed (born 20 September 1958) is a Kuwaiti football midfielder ... Doc 2<## Title: Nutrition ##>The first recorded nutritional experiment with human subjects is found in the Bible’s Book of Daniel... Doc 3<## Title: Samer Saeed ##>Samer Saeed Mujbel Al Mamoori (born December 1, 1987 in Iraq) is ... </information> <sub_answer1>Kuwait</sub_answer1>   
<think> Now that I have determined Yussef Al-Suwayed is from Kuwait, I need to find out which nation invaded Kuwait and resulted in U.S. military action. </think>   
<sub_question2>Which nation invaded Kuwait and resulted in U.S. military action?</sub_question2> <think>To determine which nation invaded Kuwait and resulted in U.S. military action, I need to search for this information.</think>   
<search>Which nation invaded Kuwait and resulted in U.S. military action?</search>   
<information> Doc 1<## Title: Invasion of Kuwait ##>The Invasion of Kuwait on 2 August 1990 was a two - day operation conducted by Iraq against the neighboring state of Kuwait... Doc 2<## Title: Economic history of Germany ##>From the 1830s and 1840s, Prussia, Saxony, and other states ... Doc 3<## Title: Gulf War ##>The Gulf War (2 August 199028 February 1991), codenamed Operation Desert Shield ... </information>   
<sub_answer2>Iraq</sub_answer2>   
<think>Now that I have determined that Iraq invaded Kuwait and resulted in U.S. military action, I can answer the original question. </think>   
<sub_question3>Who is the current prime minister of Iraq?</sub_question3>   
<think>To determine the current prime minister of Iraq, I need to search for this information.</think> <search>Who is the current prime minister of Iraq?</search>   
<information> Doc 1<## Title: Prime Minister of Iraq ##>The Prime Minister of Iraq is Iraq’s head of government... Doc 2<## Title: 2018 Malaysian general election ##>This marked a historic defeat for the ruling Barisan Nasional coalition, which had been the governing party of Malaysia and its predec essor state, Malaya, since the country’s independence in 1957... Doc 3<## Title: Prime Minister of Israel ##>The Prime Minister of Israel... </information>   
<think>Now I can answer the original question.</think>   
<answer>Haider al - Abadi<answer>