# OTC: Optimal Tool Calls via Reinforcement Learning

Hongru Wangα, Cheng $\mathbf { Q i a n } ^ { \beta }$ , Wanjun Zhongδ, Xiusi $\mathbf { C h e n } ^ { \beta }$ , Jiahao ${ \bf Q } { \bf i } { \bf u } ^ { \sigma }$ , Shijue Huangµ, Bowen $\mathbf { J i n } ^ { \beta }$ , Mengdi Wangσ, Kam-Fai $\mathbf { W o n g } ^ { \alpha }$ , Heng $\mathbf { J i } ^ { \beta }$ αThe Chinese University of Hong Kong, βUniversity of Illinois Urbana-Champaign σPrinceton University, $^ \delta \mathsf { S u n }$ Yat-sen University, $\mu$ Hong Kong University of Science and Technology hrwang, kfwong@se.cuhk.edu.hk, hengji@illinois.edu

# Abstract

Tool-integrated reasoning (TIR) augments large language models (LLMs) with the ability to invoke external tools, such as search engines and code interpreters, to solve tasks beyond the capabilities of language-only reasoning. While reinforcement learning (RL) has shown promise in improving TIR by optimizing final answer correctness, existing approaches often overlook the efficiency and cost associated with tool usage. This can lead to suboptimal behavior, including excessive tool calls that increase computational and financial overhead, or insufficient tool use that compromises answer quality. In this work, we propose Optimal Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective RL-based framework that encourages models to produce accurate answers with minimal tool calls. Our method introduces a tool-integrated reward that jointly considers correctness and tool efficiency, promoting high tool productivity. We instantiate this framework within both Proximal Policy Optimization (PPO) and Group Relative Preference Optimization (GRPO), resulting in OTC-PPO and OTC-GRPO. Experiments with Qwen-2.5 and Qwen-Math across multiple QA benchmarks show that our approach reduces tool calls by up to $7 3 . 1 \%$ and improves tool productivity by up to $2 2 9 . 4 \%$ , while maintaining comparable answer accuracy. To the best of our knowledge, this is the first RL-based framework that explicitly optimizes tool-use efficiency in TIR 1.

# 1 Introduction

![](images/6cf7e9c0da30199e978e4f7d833b2e2bad10a1572dd8e9eaf6f18538b799d97f.jpg)  
Figure 1: The performance of OTC-PO v.s. Search-R1 on Qwen2.5-7B-Base model. Our OTC achieves similar final performance but much higher tool calling efficiency comparing with the original method.

Recent advances in large language models (LLMs) have demonstrated impressive reasoning capabilities when fine-tuned via reinforcement learning (RL) with simple rule-based rewards. This has led to the emergence of large reasoning models (LRMs) - such as OpenAI’s o1 [1], DeepSeek-R1 [2], and QwQ [3] - which exhibit creative and diverse behaviors in extended Chain-of-Thought (CoT) reasoning [4]. These models develop meta-reasoning capabilities reminiscent of human cognition, such as reflection and decomposition, which have been shown to enhance performance on complex tasks [5–7]. However, the internal reasoning capacity of LRMs alone is often insufficient or even infeasible for tasks that require interaction with external environments, such as accessing up-to-date domain knowledge or performing precise computations. To overcome these limitations, tool-integrated reasoning (TIR) has emerged as a promising paradigm [8–12], enabling LLMs to interface with external tools like search engines, calculators, or code interpreters. By bridging language understanding with external capabilities, TIR significantly expands the problem-solving scope of LLMs beyond language-only reasoning.

While recent efforts have sought to extend reinforcement learning (RL) to tool-integrated reasoning (TIR) by leveraging rule-based rewards tied to final answer correctness [13–15], these approaches fall short in guiding models to use tools judiciously. Despite achieving strong performance and improved generalization, current methods often encourage indiscriminate tool usage, introducing critical challenges during both training and inference. On the one hand, frequent and unnecessary tool calls incur substantial computational and temporal overhead, as executing tools—especially at scale—demands significant infrastructure and increases operational costs, limiting deployment feasibility. We also observe a pattern of cognitive offloading [16], in which the model extensively relies on external tools, thereby limiting the development and utilization of its internal reasoning capabilities. This issue becomes more pronounced as model size increases. On the other hand, optimizing solely for answer accuracy neglects the inherent costs (e.g., time, financial) and effectiveness of individual tool usage, which may lead to inefficient reasoning behaviors. In particular, models may excessively rely on external tools even when fewer or no tools are actually needed (i.e., Tool Overuse) [11], or conversely, avoid using tools altogether even when external assistance is crucial (i.e., Tool Underuse) [17]. Both patterns hinder optimal reasoning and highlight the need for more refined training objectives - ones that balance correctness with efficient and contextually appropriate tool usage.

We aim to develop such a method based on a foundational assumption: for each question and each model, there exists an optimal number of tool calls, defined as the minimal number required for the model to arrive at a correct answer. Crucially, this optimal number is not fixed across models or questions. For the same question, models with different reasoning or representation capabilities may require different levels of tool assistance. Similarly, even for a single model, the number of necessary tool calls can vary significantly depending on the complexity of different question. For example, some questions may be solvable through internal reasoning alone - requiring zero tool calls2 - while others may necessitate one or more tool interactions to yield a correct answer. This observation highlights the limitations of supervised fine-tuning (SFT), which typically enforces a uniform policy across diverse model-question pairs. In contrast, reinforcement learning (RL) provides a more flexible framework, enabling reward signals to be dynamically adjusted based on each model’s behavior and tool usage. Ideally, among all tool-integrated reasoning trajectories that lead to the correct answer for the same question, the reward function should assign the highest reward to the trajectory that uses the optimal (i.e., minimal) number of tool calls. Other correct trajectories that involve excessive tool calls - by definition, suboptimal - should be rewarded less3. This formulation allows the model to learn policies that jointly optimize for both answer correctness and efficient, context-aware tool use on a per-question, per-model basis.

We propose Optimal Tool Calls controlled Policy Optimization (OTC-PO), a simple yet effective RL-based method that enables large language models (LLMs) to learn precise and adaptive control over external tool usage. OTC-PO trains models to optimize the number of tool calls required to reach a correct solution, thereby reducing both training and inference costs without sacrificing accuracy. To achieve this, we introduce a tool-integrated reward that modulates traditional reward signals, such as correctness, with a scaling coefficient reflecting tool efficiency. This encourages the model to prioritize correct answers that require fewer tool invocations. Our formulation shifts the optimization objective from correctness alone to tool productivity, defined as the ratio between task benefit (e.g., answer accuracy) and tool usage cost (e.g., number of tool calls). To the best of our knowledge, this is the first RL-based framework that explicitly optimizes for both the efficiency and effectiveness of toolintegrated reasoning. OTC-PO is lightweight and broadly applicable, requiring only minimal changes to standard RL pipelines - just a few lines of code - making it easy to adopt in existing systems. We validate our approach across two common tool modalities: web search and code execution. Using multiple LLMs—including Qwen-2.5-3B/7B-Base and Qwen2.5-Math-1.5B/7B-Base, we show that OTC-PO significantly reduces tool calls during reasoning trajectories while maintaining comparable accuracy to conventional approaches. In summary, the key contributions are as follows:

• We are the first to i) systematically address the tool efficiency problem via RL, which has been largely overlooked in prior research; ii) identify the cognitive offloading phenomenon   
in TIR of LLMs; iii) introduce the concept of tool productivity to measure the effectiveness and efficiency of TIR.   
• We propose a simple, scalable, and generalizable OTC-PO algorithm to encourage the model   
to use optimal tool calls to solve the problem and maximize the tool productivity, building on top of a fundamental observation that there is an optimal number of tool calls for each question and model pair. We note that it is compatible with various RL algorithms and can be easily implemented with only a few line changes.   
• We implement OTC-PPO and OTC-GRPO as two typical methods without losing the adaptability and generalization, and the experimental results on several benchmarks and baselines demonstrate significant reductions in tool call cost while preserving most of the accuracy in both in-domain and out-of-domain evaluation.

# 2 Related Work

Tool Utilization for LLMs. Teaching LLMs to use tools enables them to interact with external environments while overcoming several inherent limitations such as restricted access to up-to-date or domain-specific knowledge and poor mathmatical operation capabilities. There are three major methods which can achieve this goal: 1) prompting engineering [18–23], which guides the model’s behavior through carefully designed input templates or few-shot examples without modifying model weights; 2) supervised finetuning on tool-integrated reasoning datasets [24, 8, 25, 26, 11], where the model learns from annotated trajectories that demonstrate when and how to use tools in context; and 3) reinforcement learning [13, 14, 27, 28], which allows the model to directly learn tool-use strategies by interacting with an environment and optimizing long-term rewards, enabling more adaptive and goal-directed behaviors. While prompting and supervised fine-tuning have shown promising results, they rely heavily on expert-designed prompts and tool-integrated annotations, which limits their scalability and generalization. As a result, recent efforts have shifted toward reinforcement learning, which demonstrates improved performance and generalization through simple rule-based rewards.

Tool-integrated Reasoning Efficiency. Only few of studies from prompting engineering and supervised fine-tuning attention on tool-integrated reasoning efficiency issues in terms of the cost of tool usages during the reasoning [23, 17, 11]. In detail, Wang et al. [23] first propose a promptingbased framework: Self Divide-and-Conquer (SelfDC) to leverage the self-aware confidence score of LLMs to decide whether or not need to call tools during reasoning, achieving better trade-off between effectiveness and efficiency in the context of RAG. Furthermore, several works follow this direction and explore more boarder applications and larger tool spaces [11, 17, 29]. For example, SMART [11] collect the well-designed dataset to finetune the model to only call tools when the knowledge is outside the inherent parametric knowledge of LLMs. Despite these advancements, most existing approaches still rely on complex prompt engineering or annotated datasets, which hinders their adaptability and scalability to new scenarios. In contrast, the efficiency of tool-integrated reasoning within reinforcement learning frameworks remains largely underexplored.

Reward Shaping in RL. Reward shaping plays a critical role in reinforcement learning for toolusing language models, as it directly shapes the behavior the model learns to optimize [30, 31]. Recent advances have introduced several reward signals for LLMs to consider the correctness and the language-only reasoning efficiency together such as the length of responses and difficulty of the questions [32–34]. For instances, Length Controlled Policy Optimization (LCPO) [32] is proposed to satisfy the length constraints while optimizing reasoning performance and some efforts try to dynamically allocate inference time compute based on task complexity [33, 35].

# 3 Methodology

In this section, we first provide a definition of task considering the both effectiveness and efficiency of tool-integrated reasoning, followed by general RL framework and then our proposed Optimal Tool Call-controlled Policy Optimization (OTC-PO).

# 3.1 Task Definition

Given a question $q$ , and an environment $\mathcal { E }$ that provides access to a set of tools $\mathcal { T } = \{ t _ { 0 } , t _ { 1 } , . . . t _ { n } \}$ , the language model $\mathcal { M }$ can optionally interact with the environment by calling specific tools in $\tau$ , obtaining the corresponding tool results from $\mathcal { E }$ , and iteratively repeating this processing until the final answer is driven. Without losing generalization, the tool-integrated reasoning trajectory $\tau _ { k }$ at step $k$ is defined as follows:

$$
\tau _ { k } = ( r _ { 0 } , t c _ { 0 } , o _ { 0 } ) , ( r _ { 1 } , t c _ { 1 } , o _ { 1 } ) , . . . ( r _ { k } , t c _ { k } , o _ { k } ) ,
$$

where $r _ { i } , t c _ { i } , o _ { i }$ denotes the reasoning, tool call and returned observation respectively. Importantly, we also account for reasoning steps that do not involve tool usage. Suppose the step $p$ does not need to call tools, then the $t c _ { p }$ and $o _ { p }$ become empty string, the reasoning content $r _ { p }$ can either be merged with the subsequent reasoning step $r _ { p + 1 }$ to form the new $r _ { p + 1 } ^ { * }$ , or, if $p$ is the last step 4, be used directly to derive the final answer. The objective of the task is to generate the correct answer $\hat { a }$ with the minimal cost of tools in the full trajectory $\tau$ as follows:

$$
\operatorname * { a r g m i n } _ { \tau } \mathrm { C o s t } ( \tau ) \quad \mathrm { s u b j e c t t o } \quad \mathcal { M } ( q , \tau ) = \hat { a } ,
$$

Here the cost is measured as the number of tool calls within the trajectory $\tau$ considering its simplicity and generalization. Thus the model is encouraged to not only generate correct answer but also minimize the tool calls.

# 3.2 Tool-integrated Reinforcement Learning

Inspired by recent success to use RL for better reasoning in LLMs, several efforts try to extend RL to tool-integrated reasoning with the objective functions can be defined as follows:

$$
\displaystyle { \operatorname* { m a x } _ { \pi _ { \theta } } \mathbb { E } _ { q \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot \vert q ; \mathcal { E } ) } [ r _ { \phi } ( q , y ) ] - \beta \mathbb { D } _ { k l } [ \pi _ { \theta } ( y \mid q ; \mathcal { E } )  \pi _ { \mathrm { r e f } } ( y \mid q ; \mathcal { E } ) ] , }
$$

where $\pi _ { \theta } , \pi _ { r e f }$ stand for the policy model and reference model respectively, $r _ { \phi }$ is the reward function and $\mathbb { D } _ { k l }$ is the KL-divergence measure. $q$ is the question drawn from the dataset $\mathcal { D }$ and $y$ is the generate outputs consists of the tool-integrated reasoning trajectory $\tau$ and the final answer $a$ . To optimize this goal, there are two well-established policy-gradient RL methods: Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO).

PPO in TRL. Proximal Policy Optimization (PPO) is a widely used policy-gradient method in lots of tasks [36]. Given our formulation, PPO updates the policy, using trajectories sampled from the previous policy, and maximizes the following objective:

$$
\mathcal { T } _ { \mathrm { P r O } } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , y \sim \pi _ { \mathrm { o d d } } } \left[ \frac { 1 } { \sum _ { t = 1 } ^ { | y | } \mathbb { I } ( y _ { t } ) } \sum _ { t = 1 } ^ { | y | } \mathbb { I } ( y _ { t } ) \cdot \operatorname* { m i n } \left( \rho _ { t } A _ { t } , \exp ( \rho _ { t } , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) \right] ,
$$

where $\pi _ { \theta }$ and $\pi _ { o l d }$ are current and previous policy models, and $\scriptstyle p _ { t } = { \frac { \pi _ { \theta } ( y _ { t } | q , y _ { < t } ; { \mathcal { E } } ) } { \pi _ { o l d } ( y _ { t } | q , y _ { < t } ; { \mathcal { E } } ) } }$ . Here, $\mathbb { I } ( y _ { t } )$ is an indicator function marking whether token $y _ { t }$ is generated by the model (i.e., $r _ { i }$ and $t c _ { i }$ ) or returned from the environment $\mathcal { E }$ (i.e., $o _ { i }$ ). The advantage estimate $A _ { t }$ is computed via Generalized Advantage Estimation (GAE) and $\epsilon$ a PPO clipping threshold to constrain the policy update.

![](images/c479c79c7e8bf4156ff7f424ec215d0e9b421d9bfec8b837949a43029bce8ab6.jpg)  
Figure 2: An overview of OTC-GRPO Algorithm.

GRPO in TRL. To improve the stability of policy optimization in language models and avoid reliance on an additional value function approximation, Group Relative Policy Optimization (GRPO) is introduced which uses the relative ranking of multiple sampled outputs as a baseline for computing advantages, rather than fitting a value function. For each input question $q$ , GRPO samples a group of $G$ response $\left\{ y _ { 1 } , y _ { 2 } , . . . , y _ { G } \right\}$ from the reference policy $\pi _ { r e f }$ . The trained policy $\pi _ { \theta }$ is then updated by maximizing the following objective:

$$
\begin{array} { r l r } & { } & { \mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = { \mathbb E } _ { q \sim \mathcal { D } , \left\{ y _ { i } \right\} _ { i = 1 } ^ { G } \sim \tau _ { \mathrm { o l d } } ( \cdot | q ; \mathcal { E } ) } \Bigg [ \cfrac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { \sum _ { t = 1 } ^ { | y _ { i } | } \mathbb { I } ( y _ { i , t } ) } \sum _ { t = 1 } ^ { | y _ { i } | } \mathbb { I } ( y _ { i , t } ) } \\ & { } & { \cdot \operatorname* { m i n } \left( p _ { t } \hat { A } _ { i , t } , \ \mathrm { c l i p } \left( p _ { t } , 1 - \epsilon , \ 1 + \epsilon \right) \hat { A } _ { i , t } \right) \Bigg ] - \beta { \mathbb D } _ { \mathrm { K L } } [ \pi _ { \theta } \ \| \ \pi _ { \mathrm { r e f } } ] , } \end{array}
$$

Here, = πθ(yi,t|x,yi,<t;R)πold(yi,t|x,yi,<t;R) , Aˆi,t denotes the advantage at token t in response yi, computed based on the relative ranking of rewards within the group, $\beta$ controlling the KL-regularization strength. The clipping threshold $\epsilon$ ensures stable updates.

Reward Design. Reward signals serve as the optimization objective and directly guide the behavior of the policy model during training. Most prior predominantly focuses on rule-based correctness rewards [13, 14], as illustrated below:

$$
r _ { \phi } ( q , y ) = r _ { c o r r e c t } = 1 { \mathrm { i f } } a = { \hat { a } } \operatorname { e l s e } 0 ,
$$

where $a$ is the extracted predicted answer from the response $y$ and $\hat { a }$ is the ground truth answer. Some of them further consider the format reward by assigning additional reward besides the correctness such as $r _ { f o r m a t }$ if the model generate the response in the required format. We discuss this in Appendix.

# 3.3 OTC-PO: Optimal Tool Call-controlled Policy Optimization

Instead of only considering the correctness (and optionally format) of the answer, we argue it is important to consider the effectiveness and efficiency of tool-integrated reasoning trajectory. Specifically, different questions may require different numbers of tool interactions depending on the internal capabilities of model, and for each question–model pair, there exists an optimal (i.e., minimal) number of tool calls needed to reach the correct answer. Given the real-world cost and latency associated with tool calls, it is crucial to move beyond coarse, answer-only rewards and instead provide finer-grained signals that capture the overall quality of TIR trajectory.

To this end, we propose Optimal Tool Call-controlled Policy Optimization (OTC-PO), can be easily integrated with any RL algorithms, and is broadly applicable across different models and settings, aiming to preserve overall accuracy while significantly reducing the cost of tool interactions. At the heart of OTC-PO is a reward mechanism that ranks the current number of tool calls $m$ relative to the optimal number of tool calls $n$ given the question and model. In most realistic settings, the true optimal number $n$ is unknown in advance. To address this, OTC-PO approximates $n$ by tracking the minimal number of tool calls observed across different trajectories to arrive the correct answer for the same question. This approach enables the model to learn efficient tool use without requiring prior knowledge of the optimal tool budget. Nonetheless, the framework can be naturally extended to scenarios where the optimal tool call number is known a priori (e.g., $n = 0$ for language-only reasoning). We instantiate OTC-PO in two variants: OTC-PPO and OTC-GRPO, tailored to different underlying RL algorithms.

OTC-PPO. Since there is no multiple trajectories for same question $q$ in single optimization step, we design the tool reward according to the number of tool calls $m$ in the current trajectory as follows:

$$
r _ { t o o l } = \cos ( \frac { m * \pi } { 2 m + c } )
$$

where $c$ is the smooth constant that controls the reward decay rate. The smaller the value, the faster the punishment and the more it encourages less use of tools; the larger the value, the more tolerant it is. In this way, among trajectories, those requiring more tool calls will receive lower rewards than those achieved with fewer tool calls. Although we can track the $m$ here for approximation of $n$ in later steps, we empirically find it requires much longer training steps due to poor sampling efficiency in PPO.

OTC-GRPO. We first identify the trajectories $\{ \tau ^ { 1 } , \tau ^ { 2 } , . . . , \tau ^ { p } \}$ leading to correct answer from the group for the question $q$ , and get the corresponding number of tool calls for each trajectory: $\mathcal { C } = \mathbf { \bar { \{ } }  c ^ { 1 } , \mathbf { \bar { { } } } c ^ { 2 } , . . . , c ^ { p } \}$ , and we can calculate the minimal tool calls $k = \operatorname* { m i n } ( \mathcal { C } )$ , serving as the local approximation of optimal tool calls for $q$ . Furthermore, we can update $k$ during multiple epochs to approximate the global optimal tool calls if the policy model finds the better solution with less than $k$ calls in later iterations. We use $n$ to indicate the approximation of optimal tool calls, and $m$ the number of tool calls in the current trajectory for $q$ . Therefore we design the reward as follows:

$$
r _ { t o o l } = \{ { \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } f ( m , n ) = n = 0 } \\ { \cos ( { \frac { m * \pi } { 2 m + c } } ) } & { { \mathrm { i f ~ } } n = 0 } \\ { \sin ( { \frac { f ( m , n ) * \pi } { 2 n } } ) } & { { \mathrm { o t h e r w i s e } } } \end{array} } &  f ( m , n ) = \{ \begin{array} { l l } { 0 , } & { { \mathrm { i f ~ } } m = 0 { \mathrm { ~ a n d ~ } } n = 0 } \\ { m , } & { { \mathrm { i f ~ } } n = 0 } \\ { { \displaystyle { \frac { 2 n m } { m + n } } } , } & { { \mathrm { o t h e r w i s e } } } \end{array} 
$$

where $f ( m , n )$ is the mapping function to re-map the $m$ to the range from 0 to $2 n$ . Briefly, the key here is to assign the highest reward 1 (a.k.a, $\sin ( \pi / 2 ) ,$ ) when the policy model achieves optimal tool calls $n$ , and when actual number of tool calls $m$ deviates from $n$ , either exceeding or falling short, the model receives a comparatively lower reward 5. Thus, we can assign different rewards dynamically according to the different $n$ and $m$ for the question $q$ .

Tool-integrated Reward Design. Inspired by recent study [33], we regard the tool-integrated reward $r _ { t o o l }$ as a coefficient of conventional reward function $r _ { \phi } ( q , y )$ , leading to the final toolintegrated reward function:

$$
r _ { \phi } ^ { t o o l } ( q , y ) = \alpha * r _ { t o o l } * r _ { \phi } ( q , y )
$$

where $\alpha$ is a hyperparameter that controls the scaling of the tool reward as $r _ { \phi }$ is usually same for correct or wrong answer. Notably, this multiplicative structure ensures that tool efficiency is mainly rewarded when the primary task objective (e.g., answer correctness) is satisfied. For example, when the final answer is incorrect, $r _ { \phi }$ is 0, effectively disabling the influence of $r _ { t o o l }$ . This design helps mitigate reward hacking by preventing the policy model from being incentivized to use tools without ultimately producing a correct answer. If the answer is correct, $r _ { \phi }$ is 1 and then $r _ { t o o l }$ begins to dominate the scale of the reward, thus encouraging less tool usage. We highlight this design is nontrivial since there are several advantages of this design: i) It maximally preserves overall accuracy, with a theoretical justification [33]; ii) It mitigates the risk of reward hacking compared to additive forms (i.e., $r _ { t o o l } + r _ { \phi } )$ , which we found to be unstable and sub-optimal 6; iii) It generalizes well to different formulations of $r _ { \phi }$ , such as $r _ { \phi } = r _ { c o r r e c t }$ or $r _ { \phi } = r _ { c o r r e c t } + r _ { f o r m a t }$ . We provide the theoretical justification, more explanations and examples in the Appendix.

Table 1: The results of OTC-PO with different baselines in search. The results except Search-R1 are directly copied from original paper [13]. We highlight the relative improvements compared with the Search-R1 in terms of TC and TP. It is clear that our method achieves comparable performance with much lower TC and higher TP.   

<table><tr><td rowspan="2">Models</td><td colspan="3">NQ</td><td colspan="3">HotpotQA</td></tr><tr><td>EM (↑)</td><td>TC(↓)</td><td>TP (↑)</td><td>EM (↑)</td><td>TC(↓)</td><td>TP(↑)</td></tr><tr><td colspan="7">Qwen2.5-3B(-Base)</td></tr><tr><td>R1-Base</td><td>0.226</td><td></td><td></td><td>0.201</td><td></td><td></td></tr><tr><td>SFT</td><td>0.249</td><td></td><td></td><td>0.186</td><td></td><td></td></tr><tr><td>RAG</td><td>0.348</td><td>1.0</td><td>0.348</td><td>0.255</td><td>1.0</td><td>0.255</td></tr><tr><td>IRCoT</td><td>0.111</td><td>10.0</td><td>0.011</td><td>0.164</td><td>10.0</td><td>0.016</td></tr><tr><td>Search-R1-PPO</td><td>0.403</td><td>1.738</td><td>0.232</td><td>0.279</td><td>1.716</td><td>0.163</td></tr><tr><td>OTC-PPO</td><td>0.355</td><td></td><td>1.010(41.9%) 0.351( 51.3%)</td><td>0.260</td><td></td><td>1.026(740.2%)0.253( 55.2%)</td></tr><tr><td>OTC-GRPO</td><td>0.444</td><td>1.008 ( 42.0%)</td><td>0.440 ( 89.7%)</td><td>0.365</td><td>1.387 (19.2%)</td><td>0.263 ( 61.3%)</td></tr><tr><td colspan="7">Qwen2.5-7B(-Base)</td></tr><tr><td>R1-Base</td><td>0.270</td><td></td><td></td><td>0.242</td><td></td><td></td></tr><tr><td>SFT</td><td>0.318</td><td></td><td></td><td>0.217</td><td></td><td></td></tr><tr><td>RAG</td><td>0.349</td><td>1.0</td><td>0.349</td><td>0.299</td><td>1.0</td><td>0.299</td></tr><tr><td>IRCoT</td><td>0.224</td><td>9.999</td><td>0.022</td><td>0.133</td><td>9.982</td><td>0.013</td></tr><tr><td>Search-R1-PPO</td><td>0.449</td><td>3.282</td><td>0.136</td><td>0.380</td><td>3.741</td><td>0.102</td></tr><tr><td>OTC-PPO</td><td>0.446</td><td></td><td>1.040(68.3%)  0.429( 215.4%)</td><td></td><td></td><td>0.3831.464(60.9%) 0.262( 156.9%)</td></tr><tr><td>OTC-GRPO</td><td>0.444</td><td>0.990 (?69.8%)</td><td>0.448 ( 229.4%)</td><td>0.366</td><td>1.005 ( 73.1%)</td><td>0.364( 256.9%)</td></tr></table>

# 4 Experiments

# 4.1 Set up

Datasets and Baselines. We mainly follow the Search-R1 [13] and ToRL [14] setting and use same baselines and datasets for the fair and comprehensive evaluation. Specifically, we use NQ and HotpotQA as training dataset for search, and we use the dataset provided in ToRL for code. We also directly compare our methods with several baselines such as SFT, Base-RL, retrieval-augmented generation baselines (i.e., RAG, IRCoT) and more importantly the Search-R1 and ToRL themselves.

Evaluation Metrics. Besides the exact match (EM) and the average tool calls (TC), we additionally define a new metric, tool productivity (TP), which measures the effectiveness and efficiency of tool calls during inference. Specifically, TP is defined as the number of correctly answered questions per unit of tool call: = PNi=1 I{yi=ˆyi}PN tc 7, where I is the indicator function that equals 1 if the predicted answer $\hat { y } _ { i }$ matches the ground truth $y _ { i }$ , and $t c _ { i }$ denotes the number of tool calls used in the $i _ { t h }$ instance. This metric reflects how efficiently the model converts tool usage into correct answers, capturing both utility and cost in a single measure. While EM provides a standard measure of accuracy, it does not reflect the underlying reasoning cost. Therefore, we consider TP as more informative indicators of agentic reasoning efficiency – highlighting not only whether the model can provide correct answer, but how economically it arrives at that correctness.

Implementation Details. We re-use the same parameter in Search-R1 [13] and ToRL [14] respectively. There are only minor modification we make to suit our method: i) We slightly change the template in Search-R1 to tell the model that it only need to call tolls when necessary (Appendix), and we do not change the template in ToRL; ii) We set the max turn (maximal number of tool calls) in Search-R1 (ToRL) to 4 (3) to better study the effects of our methods with multiple tool calls are allowed. We set $c$ as corresponding max turns or maximal tool limits, $\alpha$ as 1. We conduct our experiments on 8 A100-80G GPU and re-produce the results of Search-R1 and ToRL independently due to the different value of maximal tool calls in their original setting. We implement OTC-GRPO using the global approximation of optimal tool calls since it leads to more stable and better optimization.

# 4.2 Main Results

Search as Tool. Table 1 shows the results for search. There are several key insights can be drawn: i) OTC-PPO achieves significant improvement in terms of TC and TP compared with Search-R1, and OTC-GRPO can further boost the performance due to more accurate approximation, resulting in up to a $7 3 . 1 \%$ reduction in TC and a $2 2 9 . 4 \%$ increase in TP; ii) It is found that our method will not sacrifice the accuracy too much especially for larger LLMs, as evidenced by our OTC-PPO achieves comparable EM score with Search-R1 on Qwen2.5-7B model; iii) Different models have different tool use behaviors on different datasets. If we look at the Search-R1, we can find that 3B model tends to use less tool calls compared with 7B model. This reveals a critical issue: as language models scale up, they tend to over-rely on external tool calls unless explicitly penalized for such behavior. This not only exacerbates the problem with larger models but also leads to an underutilization of their inherent reasoning capabilities. The number of TC is also various across the datasets, which relates to many factors such as inherent capabilities of different models (i.e., self-awareness) and complexity of different datasets, leading to varying optimal number of tool calls for each question and model; iv) As the model size increases, the TC and TP get bigger boost no matter in OTC-PPO or OTC-GRPO. We highlight the ii) and iv) are very important since they compose the great potential scalability for our methods as model scales.

<table><tr><td rowspan="2">Models</td><td>AIME24</td><td></td><td>AIME25</td></tr><tr><td>EM (↑) TC(↓)</td><td>TP (↑) EM (↑)</td><td>TC (↓) TP(↑)</td></tr><tr><td>Qwen2.5-Math-1.5B(-Base) Qwen2.5-Math-1.5B-Ins Qwen2.5-Math-1.5B-Ins-TIR</td><td>10.0 13.3 =</td><td>10.0 13.3 =</td><td>=</td></tr><tr><td>ToRL OTC-GRPO</td><td>26.7 2.0 20.0 0.6</td><td>13.4 33.3</td><td>23.3 2.3 10.1 17.5 1.3 13.5</td></tr></table>

Table 2: The results of OTC-PO with different baselines in code. We note this is ONLY preliminary results without hyper-parameter probing.

Table 3: Tool behavioral advantage analysis of OTC-PO against Search-R1.   

<table><tr><td rowspan="2">Method</td><td colspan="3">NQ</td><td colspan="3">HotpotQA</td></tr><tr><td>A</td><td>B</td><td>C</td><td>A</td><td>B</td><td>C</td></tr><tr><td colspan="7">Qwen2.5-3B(-Base)</td></tr><tr><td>OTC-PPO</td><td>87.6</td><td>0.17</td><td>4.60</td><td>88.8</td><td>0.53</td><td>4.30</td></tr><tr><td>OTC-GRPO</td><td>89.8</td><td>0.25</td><td>6.12</td><td>78.1</td><td>5.96</td><td>10.43</td></tr><tr><td colspan="7">Qwen2.5-7B(-Base)</td></tr><tr><td>OTC-PPO</td><td>86.6</td><td>0.03</td><td>6.45</td><td>82.6</td><td>0.08</td><td>8.60</td></tr><tr><td>OTC-GRPO</td><td>82.0</td><td>0.0</td><td>7.26</td><td>79.9</td><td>0.0</td><td>8.15</td></tr></table>

Code as Tool. Table 2 shows the results for code. We already observe similar findings as mentioned above and will release more results soon.

# 5 Analysis

We mainly conduct our analysis using search as a tool in this section and leave code in the Appendix.

# 5.1 Tool Use Behavior Analysis

Training. Figure 3 shows the training behaviors of different methods. Our method not only achieves comparable results with fewer tool calls and shorter responses, but also enables faster and more efficient training optimization. This is particularly important, as it significantly reduces the time and cost associated with real-time tool interactions during training, both in terms of financial expenses and computational overhead.

Inference. We identify several representative tool-use behavior types of our proposed method: i) the answer is same with less tool calls compared with baseline (A); ii) the answer is same with more tool calls compared with baseline (B); and iii) our method is able to produce the correct answer using fewer tool calls, whereas the baseline fails to answer correctly (C). Table 3 shows the results of our methods compared with Search-R1. Notably, our method achieves same answers with baseline using fewer tool calls in mostly $80 \%$ of test cases (i.e., A). When also considering cases where our method answers correctly but uses less tools (i.e., C), it demonstrates a favorable balance between effectiveness and efficiency on approximately $90 \%$ of test cases across different models and datasets. These results highlight the strength of our reward design in promoting both accuracy and tool-use efficiency.

![](images/584a2268a41b6846ab422196cce2ae5f3e9127eb9d22e13b83dd7df84830f291.jpg)  
Figure 3: (a): Changes of number of search calls during the training; (b): Runing time analysis between OTC-PPO and Search-R1-PPO; and (c): Change of response length during the training. As the training proceeds, it is clear that our method uses less tool calls with shorter responses and less training time.

![](images/9aa708cd8395ed50ad320f4aa94d4f213d5cb38730771924225be0de142829ff.jpg)  
Figure 4: The Out-of-domain performance of OTC-PO and Search-R1 in TP.

<table><tr><td rowspan="2">Models</td><td colspan="2">TriviaQA</td><td colspan="2">PopQA</td><td colspan="2">2Wiki</td><td colspan="2">Musique</td><td colspan="2">Bamboogle</td></tr><tr><td>EM(↑)</td><td>TC(↓）</td><td>EM (↑）</td><td>TC(↓）</td><td>EM(↑)</td><td>TC(↓)</td><td>EM (↑)</td><td>TC(↓）</td><td>EM(↑)</td><td>TC(↓)</td></tr><tr><td colspan="9">Qwen2.5-3B(-Base)</td><td></td></tr><tr><td>Search-R1-PPO</td><td>0.566</td><td>1.580</td><td>0.425</td><td>1.631</td><td>0.258</td><td>1.675</td><td>0.051</td><td>1.922</td><td>0.063</td><td>1.766</td></tr><tr><td>OTC-PPO&quot;</td><td>0.551</td><td>1.008</td><td>0.409</td><td>1.009</td><td>0.235</td><td>1.050</td><td>0.045</td><td>1.051</td><td>-0.063</td><td>-1.016</td></tr><tr><td>OTC-GRPO</td><td>0.608</td><td>1.046</td><td>0.441</td><td>1.030</td><td>0.341</td><td>1.561</td><td>0.124</td><td>1.734</td><td>0.266</td><td>1.547</td></tr><tr><td colspan="9">Qwen2.5-7B(-Base)</td><td></td></tr><tr><td>Search-R1-PPO</td><td>0.596</td><td>3.353</td><td>0.420</td><td>3.315</td><td>0.326</td><td>4.116</td><td>0.135</td><td>4.294</td><td>0.375</td><td>3.641</td></tr><tr><td>OTC-PPO</td><td>0.623</td><td>1.066</td><td>0.425</td><td>1.083</td><td>0.363</td><td>1.868</td><td>0.152</td><td>1.942</td><td>0.391</td><td>1.828</td></tr><tr><td>OTC-GRPO</td><td>0.597</td><td>0.430</td><td>0.431</td><td>0.739</td><td>0.311</td><td>0.938</td><td>0.130</td><td>1.224</td><td>0.250</td><td>0.781</td></tr></table>

Table 4: The results of Out-of-Domain (OOD) evaluation of OTC against Search-R1 in EM and TC.

# 5.2 Out-of-domain Evaluation

We run out-of-domain evaluation on more QA benchmarks. Figure 4 shows the results of TP and Table 4 shows the results of EM and TC. Generally, we find that our method achieves highest TP across all the model and benchmark. More importantly, we observe that OTC-GRPO is better than OTC-PPO on TP and as the model size increases, our method showcases bigger improvement, serving as another evidence of great scalability of our method. Notably, we find that OTC-PPO outperforms the Search-R1-PPO in terms of EM on all benchmarks with Qwen2.5-7B-Base model with much less tool calls, demonstrating the better generalization capabilities of our methods.

# 5.3 Case Study

Figure 5 presents a case study to better illustrate the advantages of our proposed OTC-PO. There are several important findings can be drawn from the case study: i) it is evident that Search-R1 issues multiple redundant and unnecessary search calls. This behavior stems from a lack of incentive to minimize tool usage, as the method is not penalized for excessive tool calls. Consequently, it fails to optimize the efficiency and effectiveness of tool calls, resulting in multiple information-poor queries; 2) both OTC-PPO and OTC-GRPO can reduce the number of unnecessary tool calls significantly but they achieve this in different ways. OTC-PPO utilizes more accurate search by generating

# Case Study: Trade-off between Reasoning and Tool Use

Question: Are Giuseppe Verdi and Ambroise Thomas both Opera composers ? Answer: Yes

Search-R1: <think> I need to determine if Giuseppe Verdi and Ambroise Thomas are both   
opera composers. </think>   
<search> Giuseppe Verdi </search>   
<observation> ... </observation>   
<search> Ambroise Thomas </search>   
<observation> ... </observation>   
<search> Are Giuseppe Verdi and Ambroise Thomas both Opera composers? </search>   
<observation> ... </observation>   
<think> Based on the observations, both Giuseppe Verdi and Ambroise Thomas are described   
as composers, specifically opera composers. </think>   
<answer> yes </answer> (Tool Calls ${ \bf \mu } = 3$ )

![](images/0b5a77daf5c26d5b22991544cd168d278a269fa249bed1dce63d86b3d1d6aeca.jpg)  
Figure 5: An example that all three methods can provide correct answer but with different cost in terms of reasoning and external tool calls.

information-rich query and OTC-GRPO choose to leverage the internal reasoning capabilities as much as possible since it is encouraged to solve the problem without external tool calls. In summary, our proposed method not only minimize external tool calls, but also maxmize internal reasoning capabilities of LLMs at the same time, resulting in more smart and accurate TIR. This is very important and valuable for almost all agentic reasoning tasks 8.

# 6 Conclusion

In this work, we introduce Optimal Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective reinforcement learning framework that explicitly encourages language models to generate correct answers with optimal tool calls. Unlike prior work that primarily focuses on final answer correctness, our approach incorporates a tool-integrated reward that accounts for both effectiveness and efficiency of tool usage, thereby promoting tool-use behaviors that are both smart and costeffective. To the best of our knowledge, this is the first work that explicitly addresses tool-use efficiency in TIR from a RL perspective, and our method offers a scalable and principled solution to make LLMs more capable and economical agents in multiple situations and benchmarks. We emphasize that this is an ongoing project, and we plan to release additional results, analyses, and extensions in future versions. We hope our work can serve as a foundation for further exploration into efficient and scalable tool-integrated reasoning with language models.

# Acknowledgement

We thank the open-source community for their invaluable contributions and the spirit of open collaboration that makes research like this possible. In particular, we would like to acknowledge the developers and maintainers of Search-R1, ToRL, VeRL, SandboxFusion, and many others.

# References

[1] OpenAI Team. Openai o1 system card, 2024.

[2] DeepSeek-AI Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.   
[3] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.   
[4] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.   
[5] Hongru WANG, Deng Cai, Wanjun Zhong, Shijue Huang, Jeff Z. Pan, Zeming Liu, and Kam-Fai Wong. Self-reasoning language models: Unfold hidden reasoning chains with few reasoning catalyst. In Workshop on Reasoning and Planning for Large Language Models, 2025.   
[6] Yiwei Qin, Xuefeng Li, Haoyang Zou, Yixiu Liu, Shijie Xia, Zhen Huang, Yixin Ye, Weizhe Yuan, Hector Liu, Yuanzhi Li, and Pengfei Liu. O1 replication journey: A strategic progress report – part 1, 2024.   
[7] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025.   
[8] Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. ToRA: A tool-integrated reasoning agent for mathematical problem solving. In The Twelfth International Conference on Learning Representations, 2024.   
[9] Chengpeng Li, Mingfeng Xue, Zhenru Zhang, Jiaxi Yang, Beichen Zhang, Xiang Wang, Bowen Yu, Binyuan Hui, Junyang Lin, and Dayiheng Liu. Start: Self-taught reasoner with tools, 2025.   
[10] Hongru Wang, Minda Hu, Yang Deng, Rui Wang, Fei Mi, Weichao Wang, Yasheng Wang, Wai-Chung Kwan, Irwin King, and Kam-Fai Wong. Large language models as source planner for personalized knowledge-grounded dialogues. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9556–9569, Singapore, December 2023. Association for Computational Linguistics.   
[11] Cheng Qian, Emre Can Acikgoz, Hongru Wang, Xiusi Chen, Avirup Sil, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Smart: Self-aware agent for tool overuse mitigation. arXiv preprint arXiv:2502.11435, 2025.   
[12] Hongru Wang, Yujia Qin, Yankai Lin, Jeff Z. Pan, and Kam-Fai Wong. Empowering large language models: Tool learning for real-world interaction. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 2983–2986, New York, NY, USA, 2024. Association for Computing Machinery.   
[13] Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025.

[14] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl, 2025.

[15] Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan, Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou, and Weipeng Chen. Research: Learning to reason with search for llms via reinforcement learning, 2025.

[16] Evan F Risko and Sam J Gilbert. Cognitive offloading. Trends in cognitive sciences, 20(9):676– 688, 2016.

[17] Yuanhao Shen, Xiaodan Zhu, and Lei Chen. SMARTCAL: An approach to self-aware tool-use evaluation and calibration. In Franck Dernoncourt, Daniel Preo¸tiuc-Pietro, and Anastasia Shimorina, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 774–789, Miami, Florida, US, November 2024. Association for Computational Linguistics.

[18] Cheng Qian, Chi Han, Yi Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. CREATOR: Tool creation for disentangling abstract and concrete reasoning of large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6922–6939, Singapore, December 2023. Association for Computational Linguistics.

[19] Hongru Wang, Rui Wang, Boyang Xue, Heming Xia, Jingtao Cao, Zeming Liu, Jeff Z. Pan, and Kam-Fai Wong. AppBench: Planning of multiple APIs from various APPs for complex user instruction. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15322–15336, Miami, Florida, USA, November 2024. Association for Computational Linguistics.

[20] Cheng Qian, Shihao Liang, Yujia Qin, Yining Ye, Xin Cong, Yankai Lin, Yesai Wu, Zhiyuan Liu, and Maosong Sun. Investigate-consolidate-exploit: A general strategy for inter-task agent self-evolution. arXiv preprint arXiv:2401.13996, 2024.

[21] Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi R. Fung, Hao Peng, and Heng Ji. Craft: Customizing llms by creating and retrieving from specialized toolsets, 2024.

[22] Cheng Qian, Peixuan Han, Qinyu Luo, Bingxiang He, Xiusi Chen, Yuji Zhang, Hongyi Du, Jiarui Yao, Xiaocheng Yang, Denghui Zhang, et al. Escapebench: Pushing language models to think outside the box. arXiv preprint arXiv:2412.13549, 2024.

[23] Hongru Wang, Boyang Xue, Baohang Zhou, Tianhua Zhang, Cunxiang Wang, Huimin Wang, Guanhua Chen, and Kam fai Wong. Self-dc: When to reason and when to act? self divide-andconquer for compositional unknown questions, 2025.

[24] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master $1 6 0 0 0 +$ real-world apis, 2023.

[25] Cheng Qian, Chenyan Xiong, Zhenghao Liu, and Zhiyuan Liu. Toolink: Linking toolkit creation and using through chain-of-solving on open-source model. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 831–854, 2024.

[26] Sijia Chen, Yibo Wang, Yi-Feng Wu, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Lijun Zhang. Advancing tool-augmented large language models: Integrating insights from errors in inference trees, 2025.

[27] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms, 2025.

[28] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. arXiv preprint, 2025.

[29] Wenjun Li, Dexun Li, Kuicai Dong, Cong Zhang, Hao Zhang, Weiwen Liu, Yasheng Wang, Ruiming Tang, and Yong Liu. Adaptive tool use in large language models with meta-cognition trigger, 2025.

[30] Adam Daniel Laud. Theory and application of reward shaping in reinforcement learning. University of Illinois at Urbana-Champaign, 2004.   
[31] Hongru Wang, Huimin Wang, Zezhong Wang, and Kam-Fai Wong. Integrating pretrained language model for dialogue policy evaluation. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6692–6696, 2022.   
[32] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning, 2025.   
[33] Daman Arora and Andrea Zanette. Training language models to reason efficiently, 2025.   
[34] Rui Wang, Hongru Wang, Boyang Xue, Jianhui Pang, Shudong Liu, Yi Chen, Jiahao Qiu, Derek Fai Wong, Heng Ji, and Kam-Fai Wong. Harnessing the reasoning economy: A survey of efficient reasoning for large language models, 2025.   
[35] Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning, 2025.   
[36] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.

# A Search Template

Answer the given question. You must conduct reasoning inside <think> and ${ < } /$ think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine tool by <search> query </search>, and it will return the top searched results between <observation> and ${ < } I$ observation>. You need to make every search call count and gain helpful results. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> xxx </answer>. Question: question.

# B Reward Function

# B.1 Figure Illustration

We draw the two types of $r _ { t o o l }$ defined in the main content for better understanding. Figure 6 shows the illustration of these reward functions. It is very clear that: 1) Left: as the number of tool calls increases, the $r _ { t o o l }$ decreases accordingly. Thus when multiple trajectories leads to correct answer, the one that use less tools will get higher reward and the one that do not use tool will get the highest reward; 2) Right: It is obvious that the diagonal achieves the highest reward as the number of tool calls is the optimal one, and as the $m$ increases, the color brightness becomes larger since multiple tool calls are involved.

# B.2 Theoretical Justification

We mainly follow the justification (Section 4.2) in the paper [33] to showcase our proposed reward design leads to reducing the tool calls without compromising accuracy. We strongly encourage readers to refer to the previous paper for complete details. All assumptions made in that work still hold in our setting, with the only difference being that our response is defined as $y = ( \tau , a )$ , which includes both the tool-integrated reasoning trajectory and the final answer, rather than language-only reasoning.

Let $\theta _ { e f f } ^ { * }$ denote the population-level parameters of the policy models obtained by maximizing Equation 9, i.e.,

$$
\theta _ { \mathrm { e f f } } ^ { \star } = \arg \operatorname* { m a x } _ { \theta } \left\{ \mathbb { E } _ { x \sim p } \mathbb { E } _ { y \sim p _ { \theta } ( x ) } \left[ \left( \alpha \ast r _ { t o o l } \right) \right] \right\}
$$

as $r _ { \phi } ( q , y )$ is mainly the indicator function about the correctness and format. Therefore, the population-level maximizer ${ p \theta _ { e f f } ^ { \ast } }$ is as accurate as the population-level maximizer $p _ { \theta } ^ { * }$ and Acc $( p _ { \theta _ { e f f } } ^ { * } ) = 1$ .

# B.3 Generalization of Reward

We use two major forms of $r _ { \phi }$ : i) $r _ { \phi } ^ { 1 } = r _ { c o r r e c t }$ ; and ii) $r _ { \phi } ^ { 2 } = r _ { c o r r e c t } + r _ { f o r m a t }$ to illustrate the generalization of our proposed tool-integrated reward design $r _ { \phi } ^ { t o o l } = \alpha * r _ { t o o l } * r _ { \phi }$ .

If the answer is correct, the $r _ { \phi } ^ { 1 }$ and $r _ { \phi } ^ { 2 }$ are both positive and fixed for all questions, therefore the $r _ { \phi } ^ { t o o l }$ is only determined by the number of $\alpha$ and $r _ { t o o l }$ . Thus it can assign higher score for less tools and lower score for more tools by definition of $r _ { t o o l }$ .

If the answer is wrong, the $r _ { \phi } ^ { 1 }$ will be 0, and therefore disables the influence of $r _ { t o o l }$ , reducing the reward hacking issue. The acceptable. Considering t $r _ { \phi } ^ { 2 }$ will be  cases of $r _ { f o r m a t }$ an, if $r _ { \phi } ^ { t o o l }$ becomes  0, then th $\alpha * r _ { t o o l } * r _ { f o r m a t }$ which isso 0; and $r _ { f o r m a t }$ $r _ { f o r m a t }$   
if $r _ { f o r m a t }$ is a positive, the $r _ { \phi } ^ { t o o l }$ still holds as less tools will be assigned more rewards. This is reasonable since less tool calls means less cost especially when the answer is wrong. We note that the reward gap here between wrong answer (i.e., $r _ { f o r m a t , }$ ) and correct answer $( i . e . , r _ { c o r r e c t } + r _ { f o r m a t } )$ is significant for the policy model to learn the desirable behavior.

![](images/c9a48f4e5c09e00735e8c1119b05e5ec1dd08557f6860d58dbce72a3a2815a42.jpg)

Figure 6: The illustration of different tool-use reward functions for OTC-PPO and OTC-GRPO.   

<table><tr><td rowspan="2">Method</td><td colspan="2">NQ</td><td colspan="2">HotpotQA</td><td colspan="2">TriviaQA</td><td colspan="2">PopQA</td><td colspan="2">2Wiki</td><td colspan="2">Musique</td><td colspan="2">Bamboogle</td></tr><tr><td>LT</td><td>MT</td><td>LT</td><td>MT</td><td>LT</td><td>MT</td><td>LT</td><td>MT</td><td>LT</td><td>MT</td><td>LT</td><td>MT</td><td>LT</td><td>MT</td></tr><tr><td colspan="9">Qwen2.5-3B(-Base)</td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>OTC-PPO</td><td>2,294</td><td>0</td><td>4,412</td><td>0</td><td>6,101</td><td>0</td><td>7,951</td><td>0</td><td>7,312</td><td>15</td><td>1,926</td><td>0</td><td>45</td><td>0</td></tr><tr><td>OTC-GRPO</td><td>2.362</td><td>1</td><td>2.538</td><td>104</td><td>5,850</td><td>2</td><td>7,662</td><td>2</td><td>2.802</td><td>225</td><td>789</td><td>42</td><td>18</td><td>1</td></tr><tr><td colspan="9">Qwen2.5-7B(-Base)</td><td colspan="7"></td></tr><tr><td>OTC-PPO</td><td>3,110</td><td>0</td><td>5,820</td><td>0</td><td>10,852</td><td>0</td><td>12,347</td><td>0</td><td>9,566</td><td>0</td><td>1990</td><td>0</td><td>46</td><td>0</td></tr><tr><td>OTC-GRPO</td><td>2,962</td><td>0</td><td>5,693</td><td>0</td><td>10,678</td><td>0</td><td>11,949</td><td>0</td><td>9,681</td><td>0</td><td>2.034</td><td>0</td><td>48</td><td>0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: Analysis of tool overuse and underuse of OTC-PO against Search-R1. The size of evaluation datasets is NQ (3,610), HotpotQA (7,142), TriviaQA (10,852), PopQA (12,347), 2Wiki (12,576), Musique (2,292), Bamboogle (64).

# C Analysis

# C.1 Tool Overuse v.s. Tool Underuse

We define two metrics: i) the number of test cases that our method can provide same answer with baseline but using less tools (i.e., LT); and ii) the number of test cases that our method can provide correct answer by using more tools while the baseline can not answer it correctly (i.e., MT). Table 5 shows the results. It is observed that smaller LLMs tend to underuse tools, while larger LLMs tend to overuse them when reinforcement learning optimization focuses solely on answer correctness. Specifically, we find our method not only can use less tools to arrive the same answer with the baseline but also can use more tools to provide correct answers, especially on small LLMs. When the model size get larger, it is found that MT always is 0, serving as another evidence that large LLMs over-rely on external tools even when it can not provide correct answer.

# C.2 More Examples

Question: The Bellagio resort owned by MGM Resorts international is located in which US city?

Answer: Las Vegas

Search-R1: <think> I need to find the US city where the Bellagio resort owned by MGM   
Resorts International is located. </think>   
<search> Bellagio resort owned by MGM Resorts International </search>   
<observation> . . . </observation>   
<search> Bellagio resort city </search>   
<observation> . . . </observation>   
<think> Based on the observations, the Bellagio resort owned by MGM Resorts International   
is located in Las Vegas, Nevada. </think>   
<answer> Las Vegas </answer> (Tool Calls ${ \ o } = 2$ ) OTC-GRPO: To answer this question, let’s first reason through the information provided: 1. We need to find the US city where the Bellagio resort, owned by MGM Resorts International, is located.   
2. The Bellagio resort is a well-known hotel and casino located in Las Vegas, Nevada. 3. Therefore, the US city where the Bellagio resort is located is Las Vegas.   
<answer> Las Vegas </answer> (Tool Calls $\mathbf { \mu = 0 }$ )   
Question: What science fantasy young adult series, told in first person, has a set of companion   
books narrating the stories of enslaved worlds and alien species?   
Answer: Animorphs   
Search-R1: <think> I need to find a science fantasy young adult series that is told in first   
person and has a set of companion books narrating the stories of enslaved worlds and alien   
species. </think>   
<search> science fantasy young adult series told in first person </search>   
<observation> ... <observation>   
<search> companion books animorphs enslaved worlds alien species </search>   
<observation> ... </observation>   
<search> science fantasy young adult series told in first person enslaved worlds alien species   
companion books </search>   
<observation> ... </observation>   
<think> Based on the observations, the science fantasy young adult series told in first person   
and has a set of companion books narrating the stories of enslaved worlds and alien species is   
“Animorphs”. </think>   
<answer> Animorphs </answer> (Tool Calls ${ \bf \Omega } = 3 { \bf \Omega }$ )

OTC-GRPO: To answer this question, let’s first reason through the information provided:

1. We need to find a science fantasy young adult series.   
2. The series is told in first person.   
3. It has a set of companion books narrating the stories of enslaved worlds and alien species. Let’s search for this information:   
<search> What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species? </search>   
<observation> ... </observation>   
Based on the search results, the science fantasy young adult series that fits the description is "Animorphs".   
<answer> Animorphs </answer> (Tool Calls $\mathbf { \lambda } = \mathbf { 1 }$ )