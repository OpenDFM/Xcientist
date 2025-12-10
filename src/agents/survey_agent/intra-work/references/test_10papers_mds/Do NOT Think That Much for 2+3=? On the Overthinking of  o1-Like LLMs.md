# Do NOT Think That Much for $\mathbf { 2 + 3 = ? }$ On the Overthinking of o1-Like LLMs

Xingyu Chen∗,1,2 , Jiahao $\mathbf { \boldsymbol { x } } _ { \mathbf { \lambda } \mathbf { \boldsymbol { u } } } { \ast } , 1$ , Tian Liang∗,1 , Zhiwei $\mathbf { H e } ^ { * , 1 , 2 }$ , Jianhui Pang1 , Dian ${ \bf { Y } } { \bf { u } } ^ { 1 }$ , Linfeng Song1 , Qiuzhi $\mathbf { L i u ^ { 1 } }$ , Mengfei Zhou2 , Zhuosheng Zhang2 , Rui Wang† 2 , Zhaopeng $\mathbf { T } \mathbf { u } ^ { \mathrm { { f 1 } } }$ , Haitao ${ \bf { M } } \mathbf { i } ^ { 1 }$ , and Dong ${ \bf { Y } } { \bf { u } } ^ { 1 }$

1Tencent AI Lab 2Shanghai Jiao Tong University

(a) Generated tokens on question “what is the answer of 2 plus 3?”

![](images/1c7ad9bc92b91935aedca079205af341ad0358531f8f7b22f646f1905699d525.jpg)  
Figure 1: Illustration of overthinking issue in Figure (a): o1-like models (right panel) spend much more tokens than conventional LLMs (left and middle panels). Our method reduces the overthinking issue when applied to QwQ-32B-Preview (Figure (b)).

![](images/3018045efff547e856f4840d1d7f9b7407f277f2fd3aa11d282869a7ea3b1fea.jpg)

# Abstract

The remarkable performance of models like the OpenAI o1 can be attributed to their ability to emulate human-like long-time thinking during inference. These models employ extended chain-of-thought (CoT) processes, exploring multiple strategies to enhance problem-solving capabilities. However, a critical question remains: How to intelligently and efficiently scale computational resources during testing. This paper presents the first comprehensive study on the prevalent issue of overthinking in these models, where excessive computational resources are allocated for simple problems with minimal benefit. We introduce novel efficiency metrics from both outcome and process perspectives to evaluate the rational use of computational resources by o1-like models. Using a self-training paradigm, we propose strategies to mitigate overthinking, streamlining reasoning processes without compromising accuracy. Experimental results show that our approach successfully reduces computational overhead while preserving model performance across a range of testsets with varying difficulty levels, such as GSM8K, MATH500, GPQA, and AIME.

# 1 Introduction

The OpenAI o1 model (OpenAI, 2024) and its replicas (Qwen, 2024; Guo et al., 2025; Kimi et al., 2025) exemplify the state-of-the-art in AI reasoning. Their success is largely attributed to mimicking human-like long-time thinking before responding to a question. Specifically, o1-like models cultivate a long chain-of-thoughts (CoT), explore multiple strategies, break down complex steps, and perform double-checking, which ultimately enhance their ability to tackle intricate reasoning tasks. This approach, known as “scaling test-time compute”, involves allocating more computational resources during the model’s inference phase to generally yield more accurate responses.

While effective, a critical yet underexplored question remains: Are we scaling test-time compute efficiently and intelligently? This study provides an initial exploration of this problem. We first observe that o1-like models exhibit significant overthinking issues. Specifically, they tend to expend excessive compute (in terms of tokens or thinking rounds) on questions that are exceptionally simple or for which the answer is already evident. For example, Figure 1(a) compares the token usage of o1-like models with conventional models when answering the question, “what is the answer of 2 plus $3 ? { } ^ { \prime \prime }$ On average, o1-like models consumed $^ { 1 , 9 5 3 \% }$ more tokens than conventional models to reach the same answer. Figure 2 illustrates a concrete example where o1-style thinking results in generating 13 solutions for this trivially simple question. Across extensive analyses of mathematical benchmarks, we found these overthinking patterns: (1) contribute minimally to improving accuracy, (2) lack diversity in reasoning strategies, and (3) occur more frequently with simple problems.

The overthinking observed in o1-like models reveals inefficiency in inference and highlights fundamental limitations in their reasoning and decision-making processes. We assert that reasoning involves not only accuracy but also the application of the appropriate level of complexity based on the problem’s requirements. This insight motivates our exploration of studying and mitigating overthinking. To address this, we propose two metrics from both outcome and process perspectives to evaluate o1-like models’ efficiency. These metrics help provide a comprehensive assessment of the efficiency of o1-like models, augmenting the commonly-used effectiveness metrics.

To mitigate overthinking without introducing external information, we adopt a self-training paradigm. With our proposed efficiency metrics, we streamline the generated responses by removing redundant solutions while maintaining basic reflexivity. Experimental results across testsets of varying difficulty levels (e.g., GSM8K, MATH500, GPQA, and AIME) demonstrate our approach’s effectiveness and robustness in mitigating overthinking issues. For instance, as shown in Figure 1(b), our approach can reduce token output by $4 8 . 6 \%$ while maintaining accuracy on the widely-used MATH500 testset as applied to QwQ-32B-Preview.

In summary, our contributions are three-fold:

1. We present the first study offering both a definitive explanation and comprehensive analysis of the overthinking issue, showing that o1-like LLMs often expend unnecessary computational resources on redundant solutions that contribute minimally to final outcomes.   
2. We introduce metrics considering both outcome and process aspects to assess the efficiency of o1-like models.   
3. We explore several strategies to mitigate overthinking, significantly reducing token generation while maintaining model performance across testsets of varying difficulty.

# 2 Observing Overthinking Issues

In this section, we present a comprehensive analysis of outputs generated by o1-like models. First, we provide a basic illustration of the solution distribution in responses from these models (§ 2.1). We then identify two inefficiencies in long CoT responses: their limited contribution to accuracy $( \ S 2 . 2 )$ and diversity $( \ S 2 . 3 )$ . To evaluate these inefficiencies empirically, we propose two efficiency metrics based on our observations. Finally, we present empirical results in $\ S 2 . 4$ and conclude that o1-like models often overthink, particularly with easier math problems.

![](images/4ca7df955728080ae4c448cf45287f58c5f6a57c527b453b484d2c6c8060e6e2.jpg)  
Figure 2: An example of overthinking issue for QwQ-32B-Preview model’s output response that consists of 13 solutions. We also list the outputs of other conventional LLMs for reference.

# 2.1 Solution Distribution of o1-Like Models

Experimental Setup We conduct experiments on three testsets:

• ASDIV (Miao et al., 2020): an English math word problem corpus with 2,305 instances, each annotated with its problem type and grade level (1 to 6, indicating difficulty). The test set covers three main problem types (i.e., basic arithmetic operations, aggregative operations, and additional domain knowledge required), typically found in elementary schools.

• GSM8K (Cobbe et al., 2021): a dataset of high-quality, linguistically diverse grade school math word problems created by human problem writers. The test set includes 1,319 problems, with solutions often involving a sequence of elementary calculations using basic arithmetic. A middle school student should be able to solve every problem.

• MATH500 (Hendrycks et al., 2021): a challenging dataset consisting of problems from high school math competitions across seven subjects (e.g., Prealgebra, Algebra, Number Theory) and difficulty levels based on AoPS (ranging from 1 to 5). Problems in these competitions range from level 1, the easiest, often found in AMC 8 exams, to level 5, like those in AIME.

The overall difficulty levels of the test sets are $\mathrm { A S D I V } < \mathrm { G S M 8 K } < \mathrm { M A T H 5 0 0 } .$

We mainly investigate two widely recognized o1-like models featuring a visible thinking process: Qwen-QwQ-32B-Preview (Qwen, 2024) and DeepSeek-R1 (DeepSeek, 2025).

Solution Distribution In this paper, we define solution as part of the full model generation that contains an answer explicitly. For example, in Figure 2, each solution in the QwQ generation contains the answer 5. We use the Llama-3.3-70B model to separate solutions from generated responses. Figure 3 shows the distribution of solutions in generated responses. Generally, o1-like models produce 2 to 4 solution rounds for most instances, covering ${ \hat { 7 } } 6 \%$ to $8 0 \%$ of cases for QwQ-32BPreview across the test sets and $5 9 \%$ to $6 3 \%$ for DeepSeek-R1. Regarding different test sets, o1-like models tend to generate more solutions for easier test sets. For instance, the average number of solutions of QwQ-32B-Preview on the easiest ASDIV test set is 3.5, whereas on the most difficult MATH500 test set, it is 3.2. The numbers for DeepSeek-R1 are respectively 4.5 and 4.3.

![](images/92045c3d4117641652e8b6d91ab21fecfb6c2bc36d71f00c2b4eeec2f689b7ba.jpg)  
Figure 3: Distribution of solution counts in generated responses for different test sets and models (QwQ-32B-Preview (“QwQ”) and DeepSeek-R1 $( ^ { \prime \prime } \mathrm { R 1 ^ { \prime \prime } } ) )$ .

![](images/8607831aa43555c7e75e205fd17f0d2524512482d1603e7edea1c8377551be72.jpg)  
Figure 4: Average rounds of solutions (“Solutions”) and number of tokens (“Tokens”) in generated QwQ-32B-Preview DeepSeek-R1QwQ-32B-Preview Deepresponses across different difficulty levels of the MATH500 test set.

4.0 4 4.0 4 ASDIVTo empirically validate this finding, we conducted an analysis across various difficulty levels in the   
4 3.8 4.03.84 3.8 4.03.8MATH500 test set, as illustrated in Figure 4. Both QwQ-32B-Preview and DeepSeek-R1 generate   
3.63.6 32.8 3.63.6 32.8more solution rounds for problems at easier levels 1-2 (e.g., averaging 3.7 rounds and 4.6 rounds,   
2.33.1 2.12.33.1 2.1respectively) compared to levels 4-5 (e.g., averaging 3.0 rounds and 3.9 rounds, respectively), despite   
3 1.5 2.9 21.4 1.412% 3 1.5 2.9 21.4 1.412%12%the number of tokens consistently increasing with the difficulty level. These results support our   
1.1 11.1 1% 4%claim that o1-like models tend to generate more solution rounds for easier math problems.

# 1 2 3 4 1 22.2 Efficiency on Accuracy Improvements

Intuition In the example in Figure 2, we observe that the initial round of solutions already yields the correct answer. Subsequent solutions, which account for the majority of generated tokens, do not enhance accuracy. Based on this observation, we empirically investigate whether later solutions contribute to accuracy improvements. Specifically, for all cases where o1-like models produce the correct answer in the response, we calculate the distribution of occurrences for the first correct answer, termed the “first correctness distribution”. If more correct answers appear in earlier solutions, then the subsequent solutions contribute minimally to accuracy improvement, indicating reduced efficiency.

Observation Figure 5 illustrates the first correctness distribution across the test sets and models. In more than $9 2 \%$ of cases, the initial round of solutions produces the correct answer. Notably, the first round generally comprises less than $6 0 \%$ of the total tokens generated, suggesting that the extended CoT might not significantly enhance accuracy. For instance, the average length of the first round of solutions for QwQ-32B-Preview on the ASDIV test set is 287 tokens, constituting only $3 8 . 7 \%$ of the entire response. These results suggest that later solutions marginally contribute to improvements in accuracy.

Outcome Efficiency Metric Based on the above observation, we propose an outcome efficiency metric to empirically evaluate how effectively later solutions contribute to accuracy improvements. The outcome efficiency metric, denoted $\xi _ { O } ,$ is defined by the following formula:

![](images/f6731892b4af33f131840ea50dc832402d5d000adaf17e6c85dbb97a9bb498dc.jpg)  
Figure 5: Distribution of occurrences for the first correct answer.

$$
\xi _ { O } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma _ { i } \frac { \hat { T } _ { i } } { T _ { i } }
$$

where $N$ is the number of instances in a given test set, $T _ { i }$ is the total number of tokens produced for the $i .$ -th instance, and $\hat { T } _ { i }$ denotes the efficient tokens that contribute to reaching the correct answer:

$$
\hat { T } _ { i } = \left\{ \begin{array} { l l } { \# \mathrm { t o k e n s ~ t o ~ f i r s t ~ a r r i v e ~ a t ~ c o r r e c t ~ a n s w e r } , } & { \sigma _ { i } = 1 } \\ { T _ { i } , } & { \sigma _ { i } = 0 } \end{array} \right.
$$

$\sigma _ { i }$ denotes whether the evaluated model can produce a correct answer in the response:

$$
\sigma _ { i } = \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { i f ~ a t ~ l e a s t ~ o n e ~ s o l u t i o n ~ i n ~ r e s p o n s e ~ i s ~ c o r r e c t } } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right.
$$

Intuitively, if a model correctly answers at an early stage, the tokens generated thereafter do not contribute to improving accuracy and are considered inefficient. Consider Figure 2 as an example: The first solution correctly addresses the problem with $\hat { T } = 3 9$ . Consequently, $\begin{array} { r } { \xi _ { O } = \frac { 3 9 } { 9 0 1 } = 4 . { \overset { \cdot } { 3 } } \% , } \end{array}$ which can be considered extremely inefficient.

# 2.3 Efficiency on Diverse Thinking

Intuition Some researchers might argue that while solving an easy math problem may appear straightforward, approaching it from different perspectives can deepen understanding and build flexibility in mathematical thinking, which is also valuable. Consider the example output of QwQ32B-Preview in Figure 2: Solution 1 states the basic fact that 2 plus 3 equals 5; Solution 2 breaks the addition into smaller steps; Solution 3 uses a counting objects analogy. These three solutions provide different reasoning strategies. However, Solution 4 repeats Solution 3, and Solution 5 repeats Solution 2 using similar reasoning strategies. In this section, we empirically examine the diversity among solutions within a response.

Observation To empirically evaluate whether later solutions provide new reasoning strategies, we introduce the “distinctness ratio” as the measure for the ratio of distinct solutions for each data index. Consider $R _ { i } = \{ s _ { i } ^ { 1 } , \ldots , s _ { i } ^ { m } , \ldots , s _ { i } ^ { M _ { i } } \}$ as the set of $M _ { i }$ solutions in the $i .$ -th instance response.

Let $S ^ { m } = \bigl \{ s _ { 1 } ^ { m } , \ldots , s _ { k } ^ { m } , \ldots , s _ { K } ^ { m } \bigr \}$ be the set of $m$ -th solutions in the responses of all instances in the test subset.1 The distinctness ratio is defined as:

$$
\mathrm { D i s } ^ { m } = \frac { \sum _ { k = 1 } ^ { K } \tau _ { k } ^ { m } } { K }
$$

where

$$
\tau _ { k } ^ { m } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f } } \Phi ( s _ { k } ^ { m } ) \not \subseteq \{ \Phi ( s _ { k } ^ { 1 } ) , \dots , \Phi ( s _ { k } ^ { m - 1 } ) \} } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.
$$

In this context, $\Phi ( s _ { k } ^ { m } )$ is the reasoning strategy of $s _ { k } ^ { m }$ . We use GPT-4o to cluster the solutions for each instance into groups via a prompt like (Ye et al., 2024).2 The clustering results for the QwQ-32B-Preview response in Figure 2 are:

cluster1 [Solution 1, Solution 6, Solution 11] stating or affirming the basic arithmetic fact that 2 plus 3 equals 5.   
cluster2 [Solution 2, Solution5] breaking down the addition into smaller, simpler steps to reach the result.   
cluster3 [Solution 3, Solution 4] using a practical analogy of counting objects to explain the addition.   
cluster4 [Solution 7] using subtraction as a reverse check to verify the addition result.   
cluster5 [Solution 8] using algebraic manipulation and solving simple equations to confirm the result.   
cluster6 [Solution 9, Solution 10] converting numbers into different systems (binary and Roman numerals) to verify the result.   
cluster7 [Solution 12, Solution 13] considering specific contexts or frameworks like modular arithmetic or programming which could change traditional addition results.

![](images/dce9aa625a1ff61f72b2c7fb7f0d17f56a20d08622598d456c182e60e6a31516.jpg)  
Figure 6 displays the distinctness ratio for each % QwQ: MATH500solution index. Intuitively, the ratio for Solution#1 is always $1 0 0 \%$ 1: ASDIV, as it has no preceding solutions, thus $\tau \equiv 1$ R1: MATH500 for all instances. Generally, the ratio decreases with higher indices, indicating that later solutions often repeat earlier ones. For example, the average distinctness ratio for Solution# $\geq 4$ across test sets decreases by $1 1 . 5 \%$ compared to Solution#3. The ratio of Solution#2 significantly de3% 2% 2%5% 2% 3%6% 3% 2%1% 1% 1%creases, underperforming Solution#3. By review2 3 ≥4ing outputs, we find that Solution#2 often doublechecks answers from Solution#1 using the same reasoning strategy. Subsequently, Solution#3 attempts to solve the problem using a new reasoning strategy.   
Figure 6: Ratio of whether a solution provides a new reasoning strategy for each index.

Process Efficiency Metric Based on the above observation, we propose a process efficiency metric to empirically evaluate the contribution of later solutions to solution diversity. The process efficiency metric, denoted $\xi _ { P } ,$ is calculated using the formula:

$$
\xi _ { P } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { D _ { i } } { T _ { i } }
$$

Table 1: Model efficiency results of strong LLMs.   

<table><tr><td rowspan="2">Models</td><td rowspan="2">Accuracy</td><td colspan="2">Response</td><td rowspan="2"></td><td colspan="2">Efficiency</td></tr><tr><td>#Solution</td><td>#Token</td><td>Outcome</td><td>Process</td></tr><tr><td colspan="7"></td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>95.6</td><td>ASDIV 1.0</td><td></td><td>166.4</td><td>95.6%</td><td>100.0%</td></tr><tr><td>Qwen2.5-Math-72B-Instruct</td><td>96.3</td><td></td><td>1.0</td><td>213.0</td><td>96.3%</td><td>100.0%</td></tr><tr><td>QwQ-32B-Preview</td><td>96.9</td><td></td><td>3.5</td><td>741.8</td><td>41.9%</td><td>66.5%</td></tr><tr><td>DeepSeek-R1</td><td>97.1</td><td></td><td>4.5</td><td>845.0</td><td>45.9 %</td><td>64.3%</td></tr><tr><td colspan="7"></td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>92.6</td><td>GSM8K</td><td>1.0</td><td>220.3</td><td>92.6%</td><td>100.0%</td></tr><tr><td>Qwen2.5-Math-72B-Instruct</td><td>95.8</td><td></td><td>1.0</td><td>317.4</td><td>95.8%</td><td>100.0%</td></tr><tr><td>QwQ-32B-Preview</td><td>94.8</td><td></td><td>3.1</td><td>772.8</td><td>50.7%</td><td>67.6%</td></tr><tr><td>DeepSeek-R1</td><td>96.4</td><td></td><td>4.3</td><td>1056.3</td><td>48.9%</td><td>62.0%</td></tr><tr><td colspan="7"></td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>75.4</td><td>MATH500</td><td></td><td>553.4</td><td></td><td></td></tr><tr><td>Qwen2.5-Math-72B-Instruct</td><td>86.8</td><td></td><td>1.0 1.0</td><td>593.1</td><td>75.4% 86.8%</td><td>100.0% 100.0%</td></tr><tr><td>QwQ-32B-Preview</td><td>93.0</td><td></td><td>3.2</td><td>2407.9</td><td>52.3%</td><td>71.2%</td></tr><tr><td>DeepSeek-R1</td><td>96.4</td><td></td><td>4.3</td><td>2704.3</td><td>51.0%</td><td>66.2%</td></tr></table>

where $D _ { i }$ represents the number of efficient tokens that contribute to the solutions’ diversity. Here, we intentionally exclude the factor $\sigma _ { i }$ to concentrate on diversity, independent of correctness. Let $T _ { i } ^ { m }$ denote the number of tokens in solution $s _ { i } ^ { m }$ . We define:

$$
D _ { i } = \sum _ { m = 1 } ^ { M } \tau _ { i } ^ { m } T _ { i } ^ { m }
$$

Intuitively, the tokens in a distinct solution are regarded as process efficient tokens. In the example shown in Figure 2, the 13 solutions are categorized into 7 distinct reasoning strategies. Consequently, tokens in Solutions 1, 2, 3, 7, 8, 9, and 12 are efficient, resulting in $\begin{array} { r } { \xi _ { P } = \frac { \widetilde { ( 3 9 + 1 0 9 + 3 9 + 2 9 + 2 9 + 1 9 + 5 9 ) } } { 9 0 1 } = } \end{array}$ $3 5 . 8 \%$ .

# 2.4 Empirical Efficiency Results

Table 1 presents the results on model efficiency. For comparison, we include two representative conventional LLMs: Llama-3.3-70B-Instruct and Qwen2.5-Math-72B-Instruct. These conventional LLMs produce only a single solution, meaning that $\begin{array} { r } { \frac { D _ { i } } { T _ { i } } = \frac { \hat { T } _ { i } } { T _ { i } } = 1 } \end{array}$ . Therefore, in these cases, the outcome efficiency metric $\begin{array} { r } { \xi _ { O } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma _ { i } } \end{array}$ i iequals accuracy, and the process efficiency metric $\xi _ { P } = 1 . 0$ In comparison, o1-like models generate significantly longer responses, which are less efficient in improving accuracy and solution diversity. We refer to the inefficient use of generated tokens as the “overthinking issue”. The experimental results demonstrate that while o1-like models have the capacity to generate multiple solutions, their efficiency is hindered by the overthinking issue.

Figure 7 presents the detailed efficiency results across various difficulty levels of the MATH500 test set. Notably, both models perform poorly on the simplest Level 1 problems, achieving less than $5 0 \%$ outcome efficiency, a pattern that corresponds with results observed on the easy ASDIV test set. These findings underscore that the overthinking issues faced by o1-like models are particularly pronounced with simpler math problems.

![](images/a896c9a550c05f15b3d430fc12d3877e4709983d5fd1e2a822a8c3a0f8e26d88.jpg)  
50% 46.4% 46.8% 47.2%DeepSeek-R1Figure 7: Efficiency results of (a) QwQ-32B-Preview and (b) DeepSeek-R1 across different difficulty 80%levels of the MATH500 testset.

1 2 3 4 5 1 2Table 2: Statistics on different types of generated responses based on the training data. “Greedy” 67.1% 65.4% 65.5% 66.3% denotes responses generated via greedy search; “Shortest” and “Longest” indicate the shortest and longest responses among 10 samples, respectively.

<table><tr><td rowspan="2">Response</td><td rowspan="2">#Solution</td><td rowspan="2">#Token</td><td colspan="2">Efficiency</td></tr><tr><td>Outcome</td><td>Process</td></tr><tr><td>Greedy</td><td>3.1</td><td>1434.8</td><td>55.6%</td><td>72.6%</td></tr><tr><td>Shortest</td><td>2.5</td><td>1051.3</td><td>69.8%</td><td>80.3%</td></tr><tr><td>Longest</td><td>4.1</td><td>2258.7</td><td>46.0%</td><td>66.4%</td></tr></table>

# 3 Mitigating Overthinking Issues

In this section, we explore several strategies aimed at enhancing the efficiency of o1-like models. We adopt the settings for LLM reasoning tasks and primarily utilize the self-training strategy (Zelikman et al., 2022; Ho et al., 2023), where the model itself generates the training data. Consistent with previous studies, we employ the PRM12K dataset (Lightman et al., 2024) as our training dataset to generate self-training data. The QwQ-32B-Preview model serves as our testing platform because it is available for post-training.

# 3.1 Length Preference Optimization

We began by assessing whether the model could produce more efficient responses. We generated 10 samples for each instance in the training dataset with a temperature of 1.0. We discard samples that failed to generate a correct answer. Table 2 presents the statistics of different types of generated responses. Our analysis of these sampled responses reveals that the shortest response performs better in terms of both outcome and process efficiency, using fewer rounds and tokens. These findings support our initiative to enhance model efficiency through self-improvement.

We explore several effective methods for self-improvement:

• Supervised Fine-Tuning (SFT; Wei et al. 2022a): This method involves fine-tuning a model using positive synthetic data. The model learns to map inputs to preferred outputs by minimizing the cross-entropy loss between predicted and actual outputs. SFT enables the model to mimic the behavior demonstrated in training examples.

Table 3: Statistics on different types of positive examples. “#S” denotes the number of solutions.   

<table><tr><td rowspan="2">Positive Example</td><td rowspan="2">#S</td><td rowspan="2">#Token</td><td colspan="2">Efficiency</td></tr><tr><td>Outcome</td><td>Process</td></tr><tr><td>Shortest Response</td><td>2.5</td><td>1051.3</td><td>69.8%</td><td>80.3%</td></tr><tr><td>FCS</td><td>1.1</td><td>681.0</td><td>99.5%</td><td>99.1%</td></tr><tr><td>FCS + Ref.</td><td>1.9</td><td>878.7</td><td>78.4%</td><td>82.4%</td></tr><tr><td>GDS</td><td>1.6</td><td>856.8</td><td>86.8%</td><td>94.2%</td></tr></table>

• Direct Preference Optimization (DPO; Rafailov et al. 2024): This method trains a model directly on human-preferred responses to increase the likelihood of preferred responses over unpreferred ones.

• Reasoning Preference Optimization (RPO; Pang et al. 2024; Liu et al. 2024): This approach modifies the DPO loss by adding a negative log-likelihood term on the preferred response. RPO enhances DPO training stability by preventing a decreased probability of selected responses.

• Simple Preference Optimization (SimPO; Meng et al. 2024): This method addresses the discrepancy between the reward function and the generation metric during inference found in other preference optimization methods. SimPO incorporates techniques like adaptive margin and length regularization into DPO training.

Apart from the SFT method, which uses only the shortest sampled response as training data, the other three preference optimization methods require contrastive instance pairs (positive, negative). It is straightforward to use the response generated by greedy search as the negative example, aligning with the real-time inference scenario. However, in our preliminary experiments, we found it less effective than using the longest sampled response as the negative example. One possible reason is that the longest sampled response provides a clearer contrastive signal.

# 3.2 Streamlining Responses to Enhance Efficiency

Although shorter sampled responses improve the efficiency of o1-like models, they still suffer from overthinking issues. Based on the observations in Section 2, where earlier solutions in the response are more efficient, we further streamline the responses to enhance efficiency. We propose three types of simplification strategies that differ in how they streamline the responses from the beginning:

• First-Correct Solutions (FCS): This strategy retains the earliest solutions that first arrive at the correct answer. $\mathbf { F C S + }$ Reflection: Since the majority of responses achieve the correct answer on the first solution (see Figure 5), maintaining only the First-Correct Solutions might cause o1-like models to revert to conventional LLM behavior. To counter this, we extend the approach to include the second solution that reaches the correct answer in positive examples, recalling the model’s long-reflective capability while maintaining efficiency.   
• Greedily Diverse Solutions (GDS): Figure 6 demonstrates that the distinctiveness of Solution#2 significantly decreases because the second solution often double-checks answers from the first using the same reasoning strategy. Consequently, $\mathrm { F C S + }$ Reflection may reduce efficiency. To address this issue, we propose a simple heuristic that greedily expands solutions providing new reasoning strategies. Additionally, this strategy includes more solutions when the second solution does not repeat the first, thereby increasing diversity.

For each instance, we select the shortest result of each type from 10 samples. Consequently, the three types of simplified responses may originate from different original responses. Table 3 presents the statistics for these simplified responses. Notably, all simplified responses enhance efficiency compared to the shortest response. “FCS” is the most efficient, both in terms of outcome and process, using the fewest number of solution rounds and tokens. ${ } ^ { \prime \prime } \mathrm { F C S + } ]$ Reflection” incorporates reflection, requiring approximately one additional solution round, which reduces both outcome and process efficiencies. “Greedily Diverse Solutions” serves as a compromise, balancing the number of solutions and tokens, and achieving moderate to high efficiency.

Table 4: Experimental results of the proposed efficiency enhancing methods.   

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Accuracy</td><td colspan="2">Response</td><td colspan="2">Efficiency</td></tr><tr><td>#Solution</td><td>#Token</td><td>Outcome</td><td>Process</td></tr><tr><td colspan="7">ASDIV</td></tr><tr><td>QwQ-32B-Preview</td><td>96.9</td><td>3.5</td><td>741.8</td><td>41.9%</td><td>66.5%</td></tr><tr><td>+SimPOFCS+Reflection</td><td>96.8</td><td>2.0</td><td>381.6</td><td>77.6%</td><td>86.0%</td></tr><tr><td colspan="6"></td></tr><tr><td>QwQ-32B-Preview</td><td>GSM8K 94.8</td><td>3.1</td><td>772.8</td><td>50.7%</td><td>67.6%</td></tr><tr><td>+SimPOFCS+Reflection</td><td>96.0</td><td>2.0</td><td>416.6</td><td>80.2%</td><td>87.2%</td></tr><tr><td colspan="6"></td></tr><tr><td>QwQ-32B-Preview</td><td>MATH500 93.0</td><td>3.2</td><td>2407.9</td><td>52.3%</td><td>71.2%</td></tr><tr><td>+SFTShortest Response</td><td>93.2</td><td>3.0</td><td>2359.5</td><td>60.4%</td><td>75.6%</td></tr><tr><td>+DPOshortest Response</td><td>94.0</td><td>2.7</td><td>1929.5</td><td>65.8%</td><td>79.1%</td></tr><tr><td>+RPOShortest Response</td><td>91.6</td><td>2.7</td><td>2015.7</td><td>64.8%</td><td>79.2%</td></tr><tr><td>+SimPOshortest Response</td><td>92.4</td><td>2.5</td><td>1871.8</td><td>67.6%</td><td>80.9%</td></tr><tr><td colspan="6">+SimPOFirst-Correct Solution</td></tr><tr><td>+SimPOFCS+Reflection (Ours)</td><td>91.0 92.8</td><td>1.4 1.9</td><td>1016.0 1330.7</td><td>88.7% 80.0%</td><td>98.1% 89.5%</td></tr><tr><td>+SimPOGreedily Diverse Solutions</td><td>91.8</td><td>1.7</td><td>1286.1</td><td>84.3%</td><td>93.6%</td></tr><tr><td colspan="6"></td></tr><tr><td>Qwen2.5-Math-72B-Instruct</td><td>GPQA 46.5</td><td>1.0</td><td>811.7</td><td>46.5%</td><td>100%</td></tr><tr><td>QwQ-32B-Preview</td><td>59.6</td><td>2.2</td><td>3228.4</td><td>51.4%</td><td>84.3%</td></tr><tr><td>+SimPOFCS+Reflection</td><td>59.1</td><td>1.7</td><td>2085.7</td><td>55.7%</td><td>90.4%</td></tr><tr><td colspan="6"></td></tr><tr><td>Qwen2.5-Math-72B-Instruct</td><td>AIME24 23.3</td><td>1.0</td><td>1204.5</td><td>23.3%</td><td>100.0%</td></tr><tr><td>QwQ-32B-Preview</td><td>46.7</td><td>2.6</td><td>9480.9</td><td>38.4%</td><td>84.4%</td></tr><tr><td>+SimPOFCS+Reflection</td><td>43.3</td><td>1.7</td><td>5154.5</td><td>39.8%</td><td>92.0%</td></tr></table>

# 3.3 Experimental Results

Table 4 presents the results of the proposed methods. We perform a detailed comparison on MATH500 and validate the most effective approach using the other test sets.

Performance of Length Preference Optimization Methods SFT only slightly reduces the number of solution rounds and tokens compared to the vanilla QwQ-32B-Preview model, underperforming the preference optimization methods. Among these methods, SimPO achieves the best results, reducing the number of generated tokens by $2 2 . 3 \%$ on MATH500. Consequently, SimPO is used as the default post-training method in the subsequent experiments.

Performance of Response Simplification Methods As anticipated, the First-Correction Solutions strategy achieves the greatest reduction in length. However, this method decreases performance on the difficult MATH500 test set, which may require more rounds of reflection. The ${ } ^ { \prime \prime } \bar { \mathrm { F C S + } }$ Reflection” approach alleviates this issue and surpasses the FCS method by $1 . 4 \%$ with an additional round of reflection. The “Greedily Diverse Solutions” strategy balances performance with the number of generated tokens. However, it significantly underperforms compared to “FCS+Reflection”, reinforcing our claim that the difficult MATH500 test set requires the deep inference provided by o1-like models. Hence, we adopt “FCS $+$ Reflection” as the default response simplification method.

Results on Challenging Test Sets Our approach enhances performance on easier testsets such as ASDIV and GSM8K with fewer tokens, demonstrating the effectiveness and versatility of our method in addressing overthinking issues. To address the concerns of some researchers that our approach might weaken the ability of o1-like models to tackle complex problems requiring long-term reasoning, we validate our method using more challenging GPQA and AIME test sets. As seen, our approach maintains model performance while using fewer tokens, demonstrating the robustness and generalization capability of our approach.

# 4 Related Work

# 4.1 Scaling Test-Time Compute

Enhancing model performance on complex tasks can be achieved by scaling test-time compute, which involves:

Expanding Search Space LLMs have strong reasoning abilities, but their auto-regressive decoding often misses optimal solutions. Self-consistency generates multiple responses and use majority voting to select the best answer (Wang et al., 2023b). Other approaches include best-of-n decoding, minimum Bayes risk decoding (Lightman et al., 2024; Li et al., 2023; Khanov et al., 2024; Heineman et al., 2024; Wu et al., 2024), and structured search methods such as Tree-of-Thought, Graph-ofThought, and Monte Carlo Tree Search (Yao et al., 2024; Besta et al., 2024; Luo et al., 2024; Tian et al., 2024; Wan et al., 2024).

Human-Like Thinking Patterns LLMs often use natural language reasoning. Techniques like chain-of-thought encourage step-by-step reasoning instead of direct answers (Wei et al., 2022b; Kojima et al., 2022). This has been expanded with methods like debating, self-correction, selfcritique, and plan-and-solve (Liang et al., 2024; Du et al., 2024; Xiong et al., 2023; Kumar et al., 2024; Kamoi et al., 2024; Ke et al., 2023; Lin et al., 2024; Yu et al., 2024; Wang et al., 2023a). Recent studies also explore latent space reasoning to mimic human cognition (Hao et al., 2024; Goyal et al., 2024). Advanced models combine these patterns into extensive chains-of-thought, improving accuracy with more reasoning time (OpenAI, 2024).

# 4.2 Efficient Thinking

Scaling the search space and scaling human-like thinking involves two distinct aspects of efficiency: efficient search and efficient thinking. However, few works specifically focus on efficient thinking in LLMs. Kimi et al. (2025) leveraged the long to short strategy to compress generation context. Zhao et al. (2024) encourages the model to terminate reasoning by saying $\bar { \mathbf { \Omega } } _ { \mathbf { I } } \mathbf { \Omega } _ { \mathrm { I } } ^ { - }$ don’t know” when the problem is hard to solve. Han et al. (2024) introduces token-budget-aware reasoning, where the model is prompted with a specified token budget to guide its reasoning process. There are also several contributions made to predict the distribution of the computation budget and allocate the computation power based on the prompt’s difficulty (Damani et al., 2024; Wang et al., 2024; Xu et al., 2024). Another line of work emphasizes the early stopping strategy to save computation budget while reasoning (Manvi et al., 2024; Li et al., 2024). Moreover, multi-agent framework utilizes large LLMs for difficult tasks while small LLMs for simple tasks (Kirchner et al., 2024; Damani et al., 2024)

In summary, all the aforementioned works consider conventional models rather than o1-like models with longer chains-of-thought. In contrast, our work first identifies the overthinking problem in o1-like model. Additionally, instead of limiting the reasoning space or leaving the token budget to be specified by the user, we aim to train the model to learn how to think efficiently.

# 5 Conclusion

This study identifies a key challenge in o1-like LLMs —- efficient and intelligent scaling of test-time computational resources. We have presented a comprehensive analysis of the overthinking issue in o1-like LLMs. By highlighting the overthinking phenomenon and proposing efficiency metrics, we enhance our understanding of resource utilization in o1-like models. Our self-training based approach effectively mitigates overthinking, reducing unnecessary computation while maintaining performance across reasoning benchmarks of varying difficulty levels.

This work not only improves model efficiency but also sets the groundwork for future research on optimizing computational resource allocation in AI reasoning tasks. Future directions include exploring adaptive compute strategies that dynamically adjust to problem complexity and refining efficiency metrics for broader model generalization.

# References

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17682–17690, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv:2110.14168, 2021.

Mehul Damani, Idan Shenfeld, Andi Peng, Andreea Bobu, and Jacob Andreas. Learning how hard to think: Input-adaptive allocation of lm computation, 2024. URL https://arxiv.org/abs/2410. 04707.

DeepSeek. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. 2025. URL https://api.semanticscholar.org/CorpusID:275789950.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, 2024.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=ph04CRkPdC.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tingxu Han, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen, and Zhenting Wang. Tokenbudget-aware llm reasoning. arXiv preprint arXiv:2412.18547, 2024.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2024. URL https: //arxiv.org/abs/2412.06769.

David Heineman, Yao Dou, and Wei Xu. Improving minimum bayes risk decoding with multiprompt. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 22525–22545, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS, 2021.

Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14852–14882, 2023.

Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. When can llms actually correct their own mistakes? a critical survey of self-correction of llms. Transactions of the Association for Computational Linguistics, 12:1417–1440, 2024.

Pei Ke, Bosi Wen, Zhuoer Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, et al. Critiquellm: Scaling llm-as-critic for effective and explainable evaluation of large language model generation. corr, abs/2311.18702. detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9004–9017, 2023.

Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. Args: Alignment as reward-guided search. In The Twelfth International Conference on Learning Representations, 2024.

Team Kimi, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, and Yuri Burda. Proververifier games improve legibility of llm outputs, 2024. URL https://arxiv.org/abs/2407.13692.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35: 22199–22213, 2022.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5315–5333, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.291. URL https://aclanthology. org/2023.acl-long.291.

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Xinglin Wang, Bin Sun, Heda Wang, and Kan Li. Escape sky-high cost: Early-stopping self-consistency for multi-step reasoning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id $=$ ndR8Ytrzhh.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multiagent debate. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 17889–17904, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.992. URL https://aclanthology.org/2024.emnlp-main.992.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id $=$ v8L0pN6EOi.

Zicheng Lin, Zhibin Gou, Tian Liang, Ruilin Luo, Haowei Liu, and Yujiu Yang. CriticBench: Benchmarking LLMs for critique-correct reasoning. In Lun-Wei ${ \mathrm { K u } } ,$ Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 1552–1587,

Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.findings-acl.91. URL https://aclanthology.org/2024.findings-acl.91.

Zhihan Liu, Miao Lu, Shenao Zhang, Boyi Liu, Hongyi Guo, Yingxiang Yang, Jose Blanchet, and Zhaoran Wang. Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer. arXiv preprint arXiv:2405.16436, 2024.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.

Rohin Manvi, Anikait Singh, and Stefano Ermon. Adaptive inference-time compute: Llms can predict if they can do better, even mid-generation, 2024. URL https://arxiv.org/abs/2410.02725.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a referencefree reward. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing english math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.

OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms, 2024.

Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason E Weston. Iterative reasoning preference optimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id $=$ 4XIKfvNYvx.

Qwen. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https: //qwenlm.github.io/blog/qwq-32b-preview/.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu. Toward selfimprovement of llms via imagination, searching, and criticizing. arXiv preprint arXiv:2404.12253, 2024.

Ziyu Wan, Xidong Feng, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, and Jun Wang. Alphazero-like tree-search can guide large language model decoding and training. In Forty-first International Conference on Machine Learning, 2024.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2609–2634, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/ v1/2023.acl-long.147. URL https://aclanthology.org/2023.acl-long.147.

Xinglin Wang, Shaoxiong Feng, Yiwei Li, Peiwen Yuan, Yueqi Zhang, Boyuan Pan, Heda Wang, Yao Hu, and Kan Li. Make every penny count: Difficulty-adaptive self-consistency for cost-efficient reasoning, 2024. URL https://arxiv.org/abs/2408.13457.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023b. URL https: //openreview.net/forum?id=1PL1NIMMrw.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022a.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022b.

Ian Wu, Patrick Fernandes, Amanda Bertsch, Seungone Kim, Sina Pakazad, and Graham Neubig. Better instruction-following through minimum bayes risk. arXiv preprint arXiv:2410.02902, 2024.

Kai Xiong, Xiao Ding, Yixin Cao, Ting Liu, and Bing Qin. Examining inter-consistency of large language models collaboration: An in-depth analysis via debate. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 7572–7590, 2023.

Mayi Xu, Yongqi Li, Ke Sun, and Tieyun Qian. Adaption-of-thought: Learning question difficulty improves large language models for reasoning. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5468–5495, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.313. URL https://aclanthology.org/2024.emnlp-main. 313/.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.

Junyi Ye, Jingyi Gu, Xinyun Zhao, Wenpeng Yin, and Guiling Wang. Assessing the creativity of llms in proposing novel solutions to mathematical problems. arXiv preprint arXiv:2410.18336, 2024.

Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, et al. Self-generated critiques boost reward modeling for language models. arXiv preprint arXiv:2411.16646, 2024.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Zirui Zhao, Hanze Dong, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Automatic curriculum expert iteration for reliable llm reasoning, 2024. URL https://arxiv.org/abs/2410.07627.

# A Appendix

# A.1 Case Overview for Deepseek-R1-Preview

![](images/a72537bf0c624e4161e3d2ecdf3046f4f749458db630f1394c72888e934dc859.jpg)

Figure 8: Deepseek-R1-Preview response for the query “What is the answer of 2 plus $3 ? { } ^ { \prime \prime }$

# A.2 Prompts for Clustering Solutions

Inspired by (Ye et al., 2024), we leverage GPT-4o to cluster the solutions for each instance into groups with the following prompt:

Criteria for clustering the mathematical solutions:   
1. If the solutions used to arrive at the solutions are fundamentally different from each other, such as algebraic manipulation versus geometric reasoning, they can be considered novel;   
2. Even if the results are the same, if the intermediate steps or processes involved in reaching those solutions vary significantly, the solutions can be considered different;   
3. If the solutions relies on different assumptions or conditions, they should be considered different from each other;   
4. A solution might generalize to a broader class of problems, while another solution might be specific to certain conditions. In such cases, they are considered distinct;   
5. If one solution is significantly simpler or more complex than the others, it can be regarded as essentially novel, even if they lead to the same result.

Given the following mathematical problem: \*\*\*problem\*\*\*

Solutions: Solution 1: ... Solution 2: ...

Please output the clusters strictly following the following format, each row containing a cluster, names, and reasons. Do not include any additional text or explanations outside of this format:

cluster1 [solution names] reason for cluster cluster2 [solution names] reason for cluster cluster3 [solution names] reason for cluster

For example:

cluster1 [Solution 1, Solution 3, Solution 5] similar algebraic approach using the volume formula and canceling terms or directly solving for the height.   
cluster2 [Solution 2, Solution 4] verifying the correctness and consistency of the formula and solution and considering unit checks or logical reasoning on how volume relates to base area and height.