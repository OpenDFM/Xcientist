import json
import time
import sys
import re
from typing import List, Dict, Any, Union
from openai import OpenAI

# ================= CONFIGURATION =================
API_KEY = "sk-c1n2jxt2llyucdaac9g09zflyijszu803gr7hhzv083oj2gq"  # 请填入你的Key
BASE_URL = "https://api.xiaomimimo.com/v1/"
MODEL_NAME = "mimo-v2-flash"
MAX_OUTER_ITERATIONS = 100  # 最大 Review 轮数

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= PROMPTS =================
original_text = """
    The Modality Interface: From Linear Projections to Deep Fusion and Generative Unification

The architectural design of the modality interface—the bridge that connects visual encoders with Large Language Models (LLMs)—has evolved into a central battleground for defining the capabilities of Multimodal Large Language Models (MLLMs). While early efforts, such as <Visual Instruction Tuning>, demonstrated the viability of connecting a visual encoder to an LLM using a simple linear projection layer, the field has rapidly shifted toward more sophisticated architectures. These newer designs aim to handle the "modality gap" more effectively, resulting in a spectrum of interfaces ranging from specialized compression modules to deep, interleaved fusion strategies.

A prominent approach involves the use of specialized modules to map visual tokens into the semantic space of the LLM. <MiniGPT-4> utilizes Q-Former (inspired by BLIP-2) to learn a set of learnable queries that aggregate visual features before they are fed into the LLM. This method proved that treating visual input as a sequence of compressed, highly relevant features allows an otherwise frozen LLM to understand complex modalities, unlocking emergent capabilities like website generation from sketches. Similarly, <KOSMOS-1> adopts a Perceiver-like architecture to handle interleaved multimodal inputs within a single Transformer. By employing a mapping network and special boundary tokens, it effectively unifies modalities. However, models relying on complex, pre-trained feature compressors like Q-Former (as seen in <BLIVA>, which integrates Q-Former-style embeddings with patch embeddings to better handle text-rich images) often introduce significant parameter overhead and latency.

In reaction to this complexity, recent research has sought to simplify the interface without sacrificing performance. <mPLUG-Owl2> introduces a "modality-adaptive module" that preserves modality-specific features, arguing that strict fusion can degrade text-only capabilities. It decouples the visual pathway while still sharing a universal language decoder. Conversely, <FIBER> argues for "Fusion in the Backbone." Instead of a separate connector, it interleaves cross-attention layers directly into the LLM’s encoder and decoder context. This deep fusion allows the model to reason about visual dependencies dynamically throughout the generation process, rather than relying on a static pre-fusion step. This trade-off—between specialized, compressed encoders (<MiniGPT-4>) and deep, integrated fusion (<FIBER>)—defines the current tension in interface design.

Simultaneously, a radical shift in philosophy has emerged, epitomized by <VisionLLM>. This work proposes treating images as a "foreign language." Rather than designing complex connectors, it aligns vision-centric tasks directly with the LLM’s language processing capabilities. It uses a language-guided image tokenizer and treats outputs (like bounding boxes) as text tokens, effectively unifying the architecture into a pure generative text decoder. This approach suggests that the modality interface might eventually disappear entirely, with the LLM handling all modalities natively. <GIT> further supports this generative unification, achieving state-of-the-art performance in image-to-text generation with a simple encoder-decoder architecture that requires no complex modality-specific modules. This evolution from simple linear projections (<Visual Instruction Tuning>) to specialized compressors (<MiniGPT-4>, <KOSMOS-1>), deep fusion (<FIBER>, <mPLUG-Owl2>), and finally generative unification (<VisionLLM>, <GIT>) represents the field’s search for the most efficient way to bridge the visual and linguistic worlds.

However, architectural innovation cannot compensate for insufficient training data, a point emphasized by <Static Augmentation>. This paper argues that standard supervised fine-tuning (SFT) data is often too narrow, limiting the model's ability to generalize across different instruction styles. This highlights that the "interface" is not just architecture; it is also the instructional data that teaches the model how to utilize that architecture. Even the most elegant connector, such as the "Spatial Vision Aggregator (SVA)" proposed in <Cambrian-1>, relies on a massive, high-quality dataset to fine-tune the interaction between visual backbones and the LLM.

Furthermore, the industry is trending toward architectures that prioritize "visual grounding"—explicitly linking text to visual regions. <KOSMOS-2> introduced grounding capabilities by training on grounded image-text pairs, allowing the model to output <region> tags corresponding to bounding boxes. While not strictly an architectural connector change, it alters the output interface to enforce alignment between the visual and textual streams. Similarly, <UniTAB> Unifies Text and Box outputs, proposing a shared token sequence that allows for grounded captioning. The inclusion of an <obj> token represents a structural change to the interface, enabling the model to explicitly align generated words with visual locations.

The trajectory of research suggests that while early interfaces focused on "how to connect" (via linear layers or projectors), the current focus is on "how to integrate." <mPLUG-Owl2> demonstrates that preserving modality-specific features is crucial for general intelligence, while <FIBER> suggests that dynamic fusion is key for fine-grained tasks. As we look at the evolution from <Visual Instruction Tuning>'s linear layer to <VisionLLM>'s generative decoder, the interface is transforming from a static bridge into a dynamic, intelligent router that determines which modality to prioritize and how to fuse context effectively.

The limitations of current interfaces are also becoming apparent in specific domains. For example, <BLIVA> addresses the failure of generalist connectors in text-rich images by explicitly injecting patch embeddings alongside learned query embeddings. This suggests that "one size fits all" interfaces may not be optimal. As the line between the visual encoder and the language model blurs, exemplified by <Downscaling Intelligence>'s proposal for visual extraction tuning, the modality interface is effectively becoming part of the reasoning engine itself, rather than a mere data conduit.

In conclusion, the modality interface has moved from a trivial linear layer to a core architectural component defining the model's reasoning capabilities. The choice between heavy, pre-trained connectors and deep, integrated fusion reflects a fundamental debate on whether visual processing should be heuristically compressed before reaching the LLM or treated as an integral part of the reasoning process. Works like <VisionLLM> and <GIT> suggest the ultimate interface may be the absence of one—a unified generative space—while current practical architectures like <Cambrian-1> and <FIBER> are pushing the boundaries of sophisticated, multi-stage fusion to maximize performance on complex reasoning tasks.

"""

SECTION_REVIEW = """You are an expert reviewer for an academic survey paper. The draft section may contain inline paper citations in angle brackets (e.g., "<Attention is All You Need>").

### Task:
1. Read the Draft Text for the given section and perform a short, careful review focused on clarity, logical flow, technical accuracy, and depth for an academic survey.
2. Produce a clear, specific, and actionable **list of revision suggestion list**. 
3. Each suggestion should concisely explain *what* to change, *why* and  *how* to implement it. When appropriate, point to the exact sentence or short excerpt from the draft to anchor the suggestion.
4. You will be provided with the previous and next sections for context. Use them to ensure logical flow and coherence across sections.
5. Only give suggestions on the current section. 
6. There are some basic requirements as follow. If the section does not meet any, provide relevant modification suggestions.
7. If the section is satisfactory, provide an empty suggestion list.

### Requirements:
1. The section text should have at least 3000 words.  Current section length: {current_section_length} words.
2. The section title should be in correct format with ### and index before the section title(like ### 1. Introduction).
3. The subsection titles should be in correct format with #### and index before the subsection title(like #### 1.1 Background).
4. All citations in the section must be in correct format: <paper_title> (like <Attention is All You Need>).
5. The section should be elegantly formatted and have good readability.
6. The content of the section should be consistent with its title and scope.
7. The section should be logically coherent and academically styled.
8. The paragraph should contain sufficient citations to support the viewpoints and provide specific and in-depth analysis

### Input:
Previous Section:
{previous_section_text}

Next Section:
{next_section_text}

Current Section:
{draft_text}

### Output format (exact):
[
  "suggestion 1",
  "suggestion 2",
  ...
]
"""

SECTION_REVISE  = """
You are a revise assistant. The section content of a survey concerning {topic} is provided below. You must propose at most ONE exact textual substitution per response according to the suggestion of reviewer.
If you think the document requires changes, choose one that you think is most important to address next, and output ONE JSON object (and nothing else).

**Output format:**
{{
    "action":"replace", 
    "originalText":"<the exact substring to replace>", 
    "newText":"<the replacement text>"
}}

The originalText field must match EXACTLY ONE substring in the document.
If you believe no edits are required, output exactly: {{"action":"done"}}.

Original Section Text:
{text}

Citation Text:
{citations}

Guidance:
- Give one minimal but precise modification.
- Revise the paragraph to enhance its logicality and depth.
- Your modifications **MUST** be consistent with the overall structure, logic and the scope(title) of the section.

Revision instructions:
- You can cite new literature from Citation Text if it helps improve the content's insight or completeness.
- You can only cite papers from the Citation Text. Make sure the title matches EXACTLY.
- The modification has multiple iterations, so make minimal changes: prefer a single precise substitution rather than rewriting whole paragraphs.

Reviewer Suggestion:
{reviewer_suggestion}

Strict rules:
- The field originalText must be an exact substring of the document (it may include newlines).
- Output only the JSON object described above. Do NOT output any prose.
- Do not return arrays. Only one JSON object.
- Keep the substitution minimal and precise.
- Any additions that are repetitive, spoil the format, inconsistent with the paragraph logic, or beyond the scope of the content are strictly prohibited

"""
citations = """
- key papers: Paper ID: 2510.09361                                                                                 
                             Title: BLINK-Twice: You see, but do you observe? A Reasoning Benchmark on Visual Perception                        
                             Keynote: {'TLDR': "The paper introduces BLINK-Twice, a vision-centric reasoning benchmark designed to evaluate     
                             multimodal large language models (MLLMs) on complex visual perception tasks, emphasizing the need for true visual  
                             observation over surface-level perception. The benchmark includes seven types of visual challenges, natural        
                             adversarial images to test models' reliance on visual content, and annotated reasoning chains for a fine-grained   
                             evaluation of the reasoning process, finding significant performance gaps in existing MLLMs in visual reasoning    
                             abilities.", 'KeyContributions': {'BenchmarkIntroduction': 'BLINK-Twice is presented as a new benchmark aimed      
                             specifically at enhancing and evaluating visual reasoning capabilities in MLLMs, contrasting with existing         
                             benchmarks that focus predominantly on textual reasoning.', 'VisualChallenges': "The benchmark encompasses seven   
                             carefully curated types of visual challenges, enabling a broad assessment of models' perception and reasoning      
                             capabilities, moving beyond mere recognition to deeper understanding.", 'AdversarialSamples': 'Utilizes natural    
                             adversarial samples generated from visually similar images but with different semantic meanings to compel models to
                             rely entirely on visual input, addressing the shortcuts often employed due to common sense.',                      
                             'ReasoningChainAnnotations': 'Includes detailed reasoning step annotations that provide a structured way to analyze
                             the reasoning quality and efficiency, focusing not only on the final answers but also on the reasoning process     
                             itself.'}, 'Methods': {'DatasetCreation': {'ImageCollection': '345 challenging base images were selected from a    
                             larger pool to ensure they offered high visual ambiguity and reasoning complexity.',                               
                             'AdversarialExampleGeneration': '103 adversarial samples were generated using an image-editing model (GPT-4o) to   
                             create visually indistinguishable yet semantically different images.'}, 'EvaluationProtocol': {'ModelsEvaluated':  
                             '20 leading MLLMs including 12 foundational and 8 reasoning-enhanced models.', 'PerformanceMetrics': 'Five         
                             different metrics were established: No-Acc, Yes-Acc, Q-Acc, I-Acc, and G-Acc, along with a CoT score based on the  
                             quality of reasoning chains.'}}, 'Results': {'PerformanceFindings': {'GeneralModelFlaws': "Models often exhibit    
                             flaws where they can 'see' but do not effectively 'observe,' highlighting significant gaps in their visual         
                             reasoning abilities.", 'ModelComparisons': 'The QVQ model demonstrated superior performance over its base versions,
                             affirming the benefits of incorporating reasoning enhancements. Closed-source models like Gemini-2.5 Pro performed 
                             notably well but still fell short in complex reasoning tasks.', 'ReasoningEfforts': 'While some models increased   
                             performance with reasoning enhancements, they struggled with redundancy and efficiency in reasoning chains.'},     
                             'StatisticalOutcomes': 'Evaluation results showed many models achieving below 0.5 on crucial I-Acc and G-Acc       
                             metrics, emphasizing the need for improved architectures in multimodal reasoning.'}, 'Significance':               
                             {'InnovativeBenchmarking': 'BLINK-Twice sets a high standard for future benchmarking of MLLMs, asserting that      
                             multimodal systems must enhance visual reasoning, rather than simplify their tasks by treating visual inputs as    
                             mere context.'}, 'CriticalReflections': {'ModelGeneralization': 'Current models may require further adaptation to  
                             embrace multimodal reasoning paradigms, specifically by integrating stronger collaborative processes between visual
                             and language modalities.', 'FutureInvestigations': 'Further research could explore systems for adaptive reasoning  
                             processes that dynamically gauge when to engage deeper visual observations, reducing unnecessary reasoning         
                             redundancies.'}, 'FutureDirections': ['Exploration of hybrid models that leverage reinforcement learning for       
                             improving visual reasoning.', 'Investigating adaptive reasoning strategies that dynamically adjust based on image  
                             complexity and content.', 'Developing new techniques to enhance model efficiency during reasoning processes without
                             sacrificing accuracy.']}                                                                                           
                                                                                                                                                
                             Paper ID: 2506.10022                                                                                               
                             Title: LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges                                     
                             Keynote: {'title': 'LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges', 'authors': ['Haoyang 
                             Li', 'Huan Gao', 'Zhiyuan Zhao', 'Zhiyu Lin', 'Junyu Gao', 'Xuelong Li'], 'abstract': {'summary': 'The paper       
                             introduces MalwareBench, a benchmark dataset comprising 3,520 jailbreaking prompts designed to evaluate the        
                             vulnerability of large language models (LLMs) in code generation contexts, particularly regarding malware          
                             generation. Experiments reveal that LLMs have a low rejection rate for malicious code requests, particularly when  
                             exposed to multiple jailbreak strategies. The findings underscore a significant deficiency in the security         
                             capabilities of current LLMs.'}, 'key_contributions': {'dataset_development': 'MalwareBench consists of 3,520      
                             jailbreaking prompts derived from 320 malicious code generation requirements covering 11 jailbreak methods and 29  
                             code functionalities.', 'evaluation_framework': 'The authors conducted thorough evaluations on 29 LLMs, including  
                             both closed-source and open-source models, assessing their ability to reject malicious prompts and the             
                             effectiveness of various jailbreak techniques.', 'insights': 'The results indicate that mainstream LLMs struggle   
                             significantly with rejecting malicious requests, especially when multiple jailbreak techniques are employed.'},    
                             'methods': {'dataset_creation': {'approach': "Questions were categorized into 6 primary classifications reflecting 
                             users' malicious intents and included a supplemental taxonomy for detailed assessments.",                          
                             'jailbreak_attack_methods': "The study utilized 11 distinct black-box jailbreak techniques to generate prompts     
                             aimed at bypassing the LLMs' defenses."}, 'experimental_design': {'models': '29 mainstream LLMs were tested,       
                             comprising various models like GPT-4o, Claude, and open-source models such as CodeLlama.', 'metrics': 'Reject rates
                             and quality scores of LLM responses were quantitatively analyzed, with scores rated from 1 to 4 based on the       
                             relevance and malicious intent of generated responses.'}}, 'results': {'rejection_rates': {'overall':              
                             {'average_rejection_rate': '60.93%', 'dropped_to': '39.92% with multiple jailbreak methods'}, 'model_performance': 
                             {'best_performer': {'model': 'OpenAI-o1', 'rejection_rate': '77.29%', 'average_score': '0.82'}, 'notable_finds':   
                             {'CodeLlama': {'rejection_rate': '76.86%', 'average_score': '0.46'}}}}, 'observations': {'passive_defense':        
                             'Smaller models often exhibit passive defenses, while larger models demonstrate a lack of proportional             
                             robustness.', 'hidden_logic': 'Some models provide responses that appear benign but harbor hidden malicious        
                             instructions.'}}, 'significance': {'contributions_to_security_research': 'The paper fills a critical gap in        
                             understanding LLM vulnerabilities in malware generation scenarios and provides a basis for future enhancements in  
                             model security.', 'implications_for_technology': 'The findings stress the urgent need for improved safety measures 
                             and configurations in LLMs to mitigate potential exploitation risks.'}, 'critical_reflections':                    
                             {'dataset_limitations': 'The reliance on a single model (Qwen-Turbo) for generating jailbreaking prompts could     
                             limit the generalizability of results.', 'scope_for_improvement': 'The current set of requirements does not        
                             encompass all possible real-world malicious scenarios, suggesting a need for expansion.',                          
                             'future_research_directions': "Further investigations should include evaluations using more complex black-box      
                             methods and a variety of models to strengthen the dataset's robustness."}, 'key_observations':                     
                             {'attack_method_effectiveness': 'Certain jailbreak methods, like the Benign Expression and DRA, proved notably     
                             effective at circumventing model defenses.', 'requirement_type_analysis': 'Models performed better against rough   
                             requirements compared to detailed ones, suggesting they respond adequately to simpler malicious prompts.'},        
                             'TL;DR': 'This paper presents MalwareBench, a dataset aimed at assessing the security of large language models     
                             against jailbreak attacks for malicious code generation, revealing significant vulnerability across various models 
                             and underscoring the need for enhanced safety mechanisms.'}                                                        
                                                                                                                                                
                             Paper ID: 2503.04833                                                                                               
                             Title: Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks                         
                             Keynote: {'title': 'Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks',          
                             'authors': ['Liming Lu', 'Shuchao Pang', 'Siyuan Liang', 'Haotian Zhu', 'Xiyu Zeng', 'Aishan Liu', 'Yunhuai Liu',  
                             'Yongbin Zhou'], 'affiliations': ['Nanjing University of Science and Technology', 'National University of          
                             Singapore', 'Beihang University', 'Peking University'], 'abstract': 'This paper presents a novel adversarial       
                             training paradigm, ProEAT, specifically designed to enhance the robustness of Multimodal Large Language Models     
                             (MLLMs) against jailbreak attacks. It employs an efficient projector layer for adversarial training while          
                             optimizing the loss function dynamically across visual and textual modalities. ProEAT achieves significant         
                             improvements in defense performance against various attack methods without sacrificing clean accuracy.',           
                             'key_contributions': ['Introduction of ProEAT, a projector-based adversarial training framework for MLLMs.',       
                             'Development of a dynamic weight adjustment mechanism that optimizes loss function allocation during training.',   
                             'Establishment of a joint optimization strategy across visual and textual modalities to tackle vulnerabilities     
                             related to jailbreak attacks.', 'Empirical results indicate that ProEAT achieves an average defense performance    
                             improvement of +34% over existing methods with only a 1% reduction in clean accuracy.'], 'methods': {'ProEAT       
                             Framework': {'description': 'An end-to-end adversarial training framework focused on a lightweight projector layer 
                             to manage large model parameters efficiently.', 'specifics': {'Adversarial Training': 'Utilizes a MSE loss         
                             objective for projecting features from clean and adversarial samples.', 'Dynamic Weight Adjustment': 'Incorporates 
                             an exponential moving average for adaptive balancing of loss weights between adversarial and normal objectives.',  
                             'Joint Multimodal Optimization': 'Trains both the projector and the language model in a unified manner to ensure   
                             robustness against cross-modal jailbreak attacks.'}}, 'Experimental Setup': {'models': ['LLaVA', 'Bunny',          
                             'mPLUG-Owl2'], 'datasets': ['JailBreakV-28K', 'MM-SafetyBench'], 'attack_methods': ['Logic', 'Persuade',           
                             'Template', 'FigStep', 'Query-Relevant']}}, 'results': {'defense_performance': {'LLaVA': {'ASR': {'No Defense':    
                             '0.64', 'ProEAT': '0.00'}, 'Clean Accuracy': '57.7%'}, 'Bunny': {'ASR': {'No Defense': '0.23', 'ProEAT': '0.00'},  
                             'Clean Accuracy': '50.5%'}, 'mPLUG-Owl2': {'ASR': {'No Defense': '0.59', 'ProEAT': '0.01'}, 'Clean Accuracy':      
                             '52.6%'}}, 'safety_assessment': {'MM-SafetyBench': {'ASR Examples': {'Illegal Activities': {'LLaVA': '0.65',       
                             'ProEAT': '0.00'}, 'Hate Speech': {'LLaVA': '0.43', 'ProEAT': '0.00'}}}}}, 'significance': "ProEAT significantly   
                             improves MLLMs' defenses against jailbreak attacks, ensuring safer deployment of these models in sensitive         
                             applications. The integration of dynamic optimization techniques minimizes the computational cost typically        
                             associated with adversarial training while maintaining model efficacy.", 'critical_reflections': ['The choice to   
                             focus on the projector layer as opposed to the entire model for adversarial training is a strong methodological    
                             contribution, enabling efficiency without compromising defense strength.', 'Future work could explore further      
                             optimization strategies or alternative architectures that could yield even stronger defenses without sacrificing   
                             model capabilities.', 'Real-world testing using embodied AI showcases practical applicability, highlighting the    
                             importance of security in operational contexts.'], 'future_directions': ['Exploration of additional adversarial    
                             attack vectors and their implications for model robustness.', 'Investigation of expansion into more complex        
                             multimodal scenarios beyond text and image, including audio or video integrations.', 'Potential adaptation of the  
                             framework to work with different architectures or to enhance existing defense methods for various MLLMs.'], 'TLDR':
                             'ProEAT introduces an innovative adversarial training framework tailored for MLLMs, utilizing a projector layer and
                             joint multimodal optimization, resulting in significant improvements in defense against jailbreak attacks while    
                             maintaining high model accuracy.'}                                                                                 
                                                                                                                                                
                             Paper ID: 2507.11932                                                                                               
                             Title: Hyperphantasia: A Benchmark for Evaluating the Mental Visualization Capabilities of Multimodal LLMs         
                             Keynote: {'title': 'Hyperphantasia: A Benchmark for Mental Visualization in Multimodal Large Language Models',     
                             'authors': ['Mohammad Shahab Sepehri', 'Berk Tinaz', 'Zalan Fabian', 'Mahdi Soltanolkotabi'], 'abstract':          
                             {'summary': 'The paper introduces Hyperphantasia, a benchmark designed to evaluate the mental visualization        
                             capabilities of Multimodal Large Language Models (MLLMs). It consists of four puzzle types assessed at three       
                             difficulty levels, highlighting significant gaps between model and human performance in tasks that require active  
                             visual reasoning rather than mere perception.'}, 'key_contributions': ['Introduction of Hyperphantasia, a synthetic
                             benchmark for assessing mental visualization skills in MLLMs.', 'Creation of four distinct puzzles divided into    
                             interpolation and extrapolation tasks, with each task designed to systematically evaluate different aspects of     
                             mental visualization.', 'Public availability of a dataset containing 1200 samples (400 per task type), facilitating
                             further research in cognitive capabilities of MLLMs.'], 'methods': {'benchmark_design': {'task_categories':        
                             {'Interpolation': ['Seven Segments: Completed from partial connections.', 'Connect the Dots: Identifying objects   
                             from outer contours.'], 'Extrapolation': ['Linear Trajectory: Predicting missing points in a triangle.', 'Parabolic
                             Trajectory: Predicting impact points from incomplete trajectories.']}, 'difficulty_levels': ['Easy: Minimal        
                             inference required.', 'Medium: Moderate complexity.', 'Hard: Extensive reasoning needed.']}, 'evaluation_protocol':
                             ['Comparison of MLLM performance across tasks and difficulty levels.', 'Involvement of human participants for      
                             baseline accuracy assessment.', 'Use of reinforcement learning to enhance model performance on extrapolation       
                             tasks.']}, 'results': {'model_performance': {'summary': 'Current state-of-the-art MLLMs show inadequate mental     
                             visualization abilities, particularly in extrapolation tasks.', 'accuracy': {'Seven Segments': {'GPT-4o': '3%      
                             accuracy on Easy.', 'Gemini': '30% accuracy on Easy.'}, 'Connect the Dots': {'GPT-4o': '96% accuracy on Easy.',    
                             'Gemini': '67% accuracy on Hard.'}, 'Linear Trajectory': {'Claude': '60% accuracy on Easy.', 'Gemini': 'Overall    
                             <50% across tasks.'}, 'Parabolic Trajectory': {'Claude': 'Best performance but rapidly declines on harder          
                             tasks.'}}}, 'reinforcement_learning_findings': {'success': 'Models trained with diverse difficulty levels achieved 
                             better generalization to harder tasks.', 'failure': 'Models trained only on easier puzzles exhibited shallow       
                             heuristic reliance.'}}, 'significance': {'implications': 'The findings indicate a need for models to develop robust
                             visual reasoning beyond memorization and task-specific shortcuts. Reinforcement learning emerges as a viable method
                             to enhance these capabilities, but models still struggle with basic visual recognitions, highlighting gaps in MLLM 
                             training approaches.'}, 'critical_reflections': {'evaluation_gap': 'The evaluation shows that while LLMs exhibit   
                             surface-level competence in visual tasks, their reasoning abilities are limited and inconsistent.',                
                             'future_directions': ['Explore additional cognitive tasks to expand the range of mental visualization capabilities 
                             assessed.', 'Investigate how to improve model robustness to visual perturbations that MLLMs currently fail to      
                             interpret.', 'Enhance reinforcement learning strategies to ensure models develop flexible reasoning capabilities   
                             rather than shallow heuristics.']}, 'TL;DR': 'This paper presents Hyperphantasia, a benchmark assessing the mental 
                             visualization skills of MLLMs through novel tasks requiring internal visualization and reasoning. Results reveal   
                             significant performance gaps between these models and human capabilities, particularly highlighting limitations in 
                             extrapolation tasks and the potential for reinforcement learning to improve model performance.'}                   
                                                                                                                                                
                             Paper ID: 2410.06468                                                                                               
                             Title: Does Spatial Cognition Emerge in Frontier Models?                                                           
                             Keynote: {'title': 'Does Spatial Cognition Emerge in Frontier Models?', 'authors': ['Santhosh Kumar Ramakrishnan', 
                             'Erik Wijmans', 'Philipp Krahenbuhl', 'Vladlen Koltun'], 'key_contributions': ['Introduction of SPACE, a benchmark 
                             to evaluate spatial cognition in frontier models.', 'The benchmark incorporates classic animal cognition tasks     
                             adapted for large language models (LLMs) and vision-language models (VLMs).', 'Demonstration through evaluations   
                             that contemporary frontier models, including multimodal models, show significant deficiencies in spatial cognition 
                             compared to biological intelligence.'], 'methods': {'benchmark_structure': {'categories': [{'name': 'Large-scale   
                             Spatial Cognition', 'tasks': ['Direction estimation', 'Distance estimation', 'Map sketching', 'Route retracing',   
                             'Shortcut discovery']}, {'name': 'Small-scale Spatial Cognition', 'tasks': ['Mental rotation test (MRT)',          
                             'Perspective taking test (PTT)', 'Water level test (WLT)', 'Minnesota Paper Form Board test (MPFB)', 'Judgment of  
                             Line Orientation test (JLO)', 'Selective attention task (SAtt)', 'Maze completion task (MCT)', 'Corsi block-tapping
                             task (CBTT)', 'Spatial addition task (SAdd)', 'Cambridge spatial working memory test (CSWM)']}]},                  
                             'evaluation_framework': {'models': ['GPT-4', 'Claude 3.5 Sonnet', 'Llama3 family', 'Mistral models', 'Yi 1.5       
                             models'], 'modalities': ['Text-only presentations', 'Multimodal presentations (images + text)'],                   
                             'performance_metrics': ['Accuracy on tasks', 'Success weighted by path length (SPL) for interactive tasks']}},     
                             'results': {'large_scale_cognition': {'average_performance': {'GPT-4': {'direction_estimation': '32.0%',           
                             'distance_estimation': '36.5%', 'map_sketching': '33.3%', 'route_retracing': '6.6%', 'shortcut_discovery': '6.4%'},
                             'chance_baseline': '25%', 'human_performance': {'average': '79-100%'}}, 'observations': 'On several large-scale    
                             cognition tasks, models performed near chance levels even with allocentric (map) views.'}, 'small_scale_cognition':
                             {'average_performance': {'GPT-4': {'MRT': '41.9%', 'PTT': '55.5%', 'SAtt': '98.8%', 'MCT': '2.2%'},                
                             'human_performance': {'average': '65-100%'}}, 'observations': 'Models showed significantly lower performance on    
                             tasks requiring spatial transformation capabilities.'}}, 'significance': {'implications_for_ai_research': 'The     
                             findings suggest contemporary frontier models have a fundamentally different form of intelligence than biological  
                             entities, particularly falling short in basic spatial reasoning, which is critical for higher cognitive            
                             processes.', 'cautionary_note': 'These results warrant caution when drawing analogies between AI capabilities and  
                             human-like cognition.'}, 'critical_reflections': ['The stark contrast between animal cognition and AI in spatial   
                             reasoning raises questions about the future direction of model training—particularly regarding the role of embodied
                             cognition.', 'Future research could explore whether integrating embodied AI principles could enhance spatial       
                             reasoning abilities in these models.', 'The performance disparities across different modalities hint at the        
                             necessity for specific model architectures or training methods to bridge this gap.'], 'future_directions':         
                             {'entity_embeddings': 'Investigate how models can better embody spatial cognition through simulated physical       
                             interactions.', 'evaluation_expansion': 'Develop additional tasks that reflect ecological validity to ensure AI    
                             models can generalize spatial reasoning.', 'multimodal_learning': 'Explore how combining sensory modalities during 
                             training could improve spatial understanding.'}, 'TL;DR': 'This paper presents SPACE, a benchmark that reveals     
                             frontier models exhibit significant deficiencies in spatial cognition, drawing contrast with animal intelligence;  
                             suggesting a need for further research on fostering spatial reasoning in AI.'}                                     
                                                                                                                                                
                             Paper ID: 2406.09321                                                                                               
                             Title: JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models        
                             Keynote: {'title': 'JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language  
                             Models', 'authors': ['Delong Ran', 'Jinyuan Liu', 'Yichen Gong', 'Jingyi Zheng', 'Xinlei He', 'Tianshuo Cong',     
                             'Anyu Wang'], 'affiliations': ['Tsinghua University', 'Hong Kong University of Science and Technology              
                             (Guangzhou)'], 'key_contributions': {'1': 'Introduced a systematic taxonomy of jailbreak evaluators based on an    
                             analysis of nearly 90 jailbreak research papers published between May 2023 and April 2024.', '2': 'Identified the  
                             strengths and weaknesses of various evaluation methods used for evaluating jailbreak attempts, namely human        
                             annotation, string matching, prompting chat completion models, and consulting text classifiers.', '3': 'Developed  
                             JailbreakEval, an integrated toolkit that consolidates multiple safety evaluation methods, allowing flexible       
                             configurations and enabling both out-of-the-box assessments and customizable workflows.'}, 'methods':              
                             {'taxonomy_analysis': {'description': 'The authors conducted a comprehensive review of research focusing on        
                             jailbreaking large language models and categorized the evaluation methodologies into four main approaches.',       
                             'approaches': ['Human Annotation', 'String Matching', 'Chat Completion (both closed-source and open-source         
                             models)', 'Text Classification (both closed-source and open-source models)']}, 'JailbreakEval_toolkit':            
                             {'description': 'An integrated framework that encompasses well-established automated evaluators with configurable  
                             settings for tailored strategies.', 'functionalities': ['Out-of-the-box evaluations', 'Customizable evaluator      
                             configurations', 'Ensemble judgment capability to enhance evaluation reliability']}}, 'results':                   
                             {'performance_evaluation': {'datasets_used': ['JAILJUDGE', 'Safe-RLHF'], 'evaluators_performance':                 
                             {'accuracy_range': [0.47, 0.9], 'notable_evaluators': {'highest_accuracy': {'name': 'qi2023fine', 'accuracy': 0.9},
                             'string_match_evaluators': 'High recall rates indicate their potential as effective preliminary filters.'},        
                             'ensemble_method_results': {'description': 'Voting method combining top evaluators showed strong performance but   
                             slightly underperformed compared to the top individual evaluator.', 'importance': 'Highlights the need for optimal 
                             evaluator selection to maximize ensemble benefits.'}}}}, 'significance': {'impact_on_research': 'The comprehensive 
                             analysis and the introduction of JailbreakEval contribute to fostering consistent methodologies for assessing      
                             jailbreak attempts against LLMs, promoting transparency and reproducibility in safety evaluations.',               
                             'future_directions': {'1': 'Expand the JailbreakEval toolkit by integrating more advanced evaluators and           
                             methodologies.', '2': 'Investigate and develop new safety evaluators specifically designed for varied contexts in  
                             jailbreak evaluation.', '3': 'Explore the harmonization of safety standards amongst different evaluation           
                             methodologies to build consensus in the field.'}}, 'critical_reflection': {'observations': ['The reliance solely on
                             automated evaluators may miss nuanced judgments human annotators can provide. Therefore, a hybrid approach could be
                             valuable.', 'As the landscape of jailbreak attempts is continuously evolving, the evaluation metrics should be     
                             dynamically adapted to remain relevant.'], 'personal_insight': 'Enhancing collaboration between LLM safety         
                             researchers and engineers can yield shared insights leading to improved methodologies in combatting jailbreak      
                             vulnerabilities.'}, 'TL;DR': 'This paper analyzes nearly 90 research works on jailbreak evaluations of large       
                             language models, categorizes key evaluator methodologies, and introduces the JailbreakEval toolkit, which          
                             facilitates systematic and customizable assessments of jailbreak effectiveness, advancing consistency and          
                             reproducibility in this burgeoning field.'}                                                                        
                                                                                                                                                
                             Paper ID: 2411.14794                                                                                               
                             Title: VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame       
                             Selection                                                                                                          
                             Keynote: {'title': 'VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core
                             Frame Selection', 'authors': ['Songhao Han', 'Wei Huang', 'Hairong Shi', 'Le Zhuo', 'Xiu Su', 'Shifeng Zhang', 'Xu 
                             Zhou', 'Xiaojuan Qi', 'Yue Liao', 'Si Liu'], 'institutions': ['Beihang University', 'The University of Hong Kong', 
                             'Shanghai AI Lab', 'Central South University', 'Sangfor Technologies Inc.', 'CUHK'], 'key_contributions':          
                             [{'contribution': 'Introduction of VideoEspresso', 'description': 'A novel large-scale dataset for video           
                             question-answering (VideoQA) that includes fine-grained reasoning capabilities and multimodal annotations.',       
                             'significance': 'Addresses the limitations of existing VideoQA datasets that rely heavily on costly manual         
                             annotations or simplistic automatic processes.'}, {'contribution': 'Semantic-aware Key Frame Extraction',          
                             'description': 'Employs a method to extract key information while removing redundant frames, thereby maintaining   
                             essential spatiotemporal relationships.', 'significance': 'This improves efficiency and enhances the overall       
                             quality of the generated QA pairs.'}, {'contribution': 'Chain-of-Thought Annotations', 'description': 'Development 
                             of multimodal Chain-of-Thought (CoT) annotations that embed logical relationships and reasoning steps into the     
                             dataset.', 'significance': 'These annotations facilitate deeper reasoning processes, which can significantly       
                             enhance model performance in VideoQA tasks.'}, {'contribution': 'Hybrid LVLM Collaboration Framework',             
                             'description': 'Introduces a framework consisting of a Frame Selector and a two-stage instruction fine-tuned       
                             reasoning LVLM.', 'significance': 'Allows models to dynamically select relevant frames and leverage multimodal     
                             evidence for reasoning, greatly enhancing performance.'}], 'methods': {'data_collection': 'Raw video data is       
                             curated from various internet sources with diverse genres and manual review to ensure reasoning potential.',       
                             'redundancy_removal': 'Uses InternVL2-8B for frame-level captioning and BGE-M3 for semantic filtering to select    
                             frames with minimal redundancy.', 'QA_pair_construction': 'Multi-frame captions are grouped, and GPT-4o is prompted
                             to formulate reasoning questions and answers.', 'CoT_annotation': 'Utilizes GPT-4o to extract key information and  
                             evidence supporting answers, alongside temporal and spatial annotations.'}, 'results': {'benchmark_performance':   
                             {'tasks_evaluated': 14, 'compared_models': 9, 'improvements': 'Demonstrates superior performance over existing     
                             LVLMs on most tasks, achieving an average accuracy of 34.1%.'}, 'subjective_eval': 'Significant enhancements in    
                             logic, accuracy, and conciseness compared to baseline models, with notable performance in logical reasoning        
                             tasks.', 'efficiency': 'Uses 1.8% of the input frames compared to traditional methods, leading to reduced          
                             computational overhead.'}, 'critical_reflections': {'strengths': ['The dataset effectively combines broad semantic 
                             understanding with fine-grained temporal coherence.', 'The integration of multimodal reasoning via CoT can set new 
                             standards for future VideoQA datasets.'], 'weaknesses': ['The reliance on GPT-4o for generating pairs raises       
                             concerns about potential biases ingrained in its language model, impacting output quality.', 'The computational    
                             efficiency might have trade-offs as more complex reasoning tasks are introduced.']}, 'future_directions':          
                             ['Exploring the use of even more advanced LLMs to iteratively enhance the dataset.', 'Investigating more efficient 
                             methods for video data processing to further minimize computational costs.', 'Conducting user studies to analyze   
                             how well the models perform in real-world scenarios outside structured benchmarks.'], 'TL;DR': 'VideoEspresso      
                             introduces a comprehensive dataset for video question answering, utilizing an innovative frame-selection approach  
                             that preserves key information and incorporates multimodal Chain-of-Thought reasoning, significantly surpassing    
                             existing models in performance and efficiency.'}                                                                   
                                                                                                                                                
                             Paper ID: 2307.16125                                                                                               
                             Title: SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension                                      
                             Keynote: {'TL;DR': 'The paper introduces SEED-Bench, a large-scale benchmark with 19K multiple-choice questions    
                             designed to evaluate generative comprehension capabilities in Multimodal Large Language Models (MLLMs), spanning 12
                             dimensions across image and video modalities. The study highlights significant performance gaps among existing     
                             models, especially in fine-grained temporal understanding and text recognition, while revealing strengths in global
                             spatial comprehension and reasoning.', 'key_contributions': {'1': {'title': 'Introduction of SEED-Bench',          
                             'description': 'SEED-Bench is a comprehensive evaluation benchmark containing 19K multiple-choice questions derived
                             from human annotations, covering 12 evaluation dimensions for both image and video modalities.'}, '2': {'title':   
                             'Evaluation Dimensions', 'description': 'The benchmark assesses generative comprehension across numerous           
                             dimensions, including spatial understanding (e.g., scene understanding, instance location) and temporal            
                             understanding (e.g., action recognition, procedure understanding).'}, '3': {'title': 'Innovative Question          
                             Generation Pipeline', 'description': 'The paper details a robust question generation pipeline incorporating        
                             automatic filtering and human validation to ensure high-quality data with challenging questions.'}, '4': {'title': 
                             'Model Performance Evaluation', 'description': '18 models, consisting of LLMs, ImageLLMs, and VideoLLMs, were      
                             rigorously evaluated across all dimensions, providing critical insights into their capabilities and                
                             deficiencies.'}}, 'methods': {'question_generation': {'description': 'Automated question generation involves using 
                             pretrained models for extracting visual information, followed by using ChatGPT/GPT-4 for crafting multiple-choice  
                             questions based on that information.', 'process_steps': ['Extraction of visual information (image captions,        
                             instance descriptions, textual elements).', 'Generation of questions and answer choices using ChatGPT/GPT-4.',     
                             'Automatic filtering to eliminate easily answerable questions without visual input.', 'Human verification for      
                             accuracy and categorization of questions.']}, 'evaluation_strategy': {'description': 'Models are evaluated based on
                             answer ranking, calculating the likelihood of each answer choice being generated given a question and selecting the
                             highest-ranked option.'}}, 'results': {'summary': 'Significant performance disparities were revealed, with most    
                             MLLMs underperforming across numerous dimensions, particularly in fine-grained temporal reasoning and text         
                             recognition. InstructBLIP emerged as a leader, demonstrating superior performance in multiple evaluation           
                             dimensions.', 'specific_model_performance': [{'model': 'InstructBLIP Vicuna', 'overall_accuracy': 53.37,           
                             'spatial_accuracy': 58.76, 'temporal_accuracy': 38.05, 'rank': 1}, {'model': 'Blurip-2', 'overall_accuracy': 46.35,
                             'spatial_accuracy': 49.74, 'temporal_accuracy': 36.71, 'rank': 3}, {'model': 'LLaMA', 'overall_accuracy': 26.75,   
                             'spatial_accuracy': 26.56, 'temporal_accuracy': 27.27, 'rank': 18}], 'notable_findings': ['MLLMs exhibit strengths 
                             in global image comprehension but struggle with fine-grained tasks.', "InstructBLIP led in multiple dimensions but 
                             didn't excel in action recognition.", 'Most models performed poorly on text recognition tasks, indicating a gap in 
                             training for recognizing textual elements in images.']}, 'significance': {'description': 'The introduction of a    
                             large-scale, objective benchmark like SEED-Bench allows for a more comprehensive understanding of MLLM             
                             capabilities, aiding future developments in the field.', 'implications': 'By identifying specific weaknesses in    
                             existing models, the benchmark can guide further research efforts and encourage the development of models that can 
                             better handle fine-grained understanding tasks, especially in temporal reasoning and text recognition.'},          
                             'critical_reflections': {'strengths': ['The extensive scale of SEED-Bench (19K questions) allows for robust        
                             evaluations and comparisons.', 'A well-structured methodology for question generation and evaluation ensures       
                             high-quality data.'], 'weaknesses': ['The performance of VideoLLMs on temporal understanding being lower than      
                             expected raises questions about their current training strategies.', 'The reliance on human annotation, while      
                             improving data quality, could introduce biases if not managed carefully.']}, 'future_directions': {'1': 'Expand the
                             range of evaluation dimensions in SEED-Bench to include additional aspects of multimodal comprehension.', '2':     
                             'Enhance the performance of VideoLLMs through improved training techniques and datasets specifically targeting     
                             temporal reasoning.', '3': 'Develop models with stronger capabilities for text recognition in multimodal           
                             contexts.', '4': 'Investigate the potential for implementing real-time evaluation mechanisms or feedback loops that
                             can update and refine the benchmarks dynamically.'}}                                                               
                                                                                                                                                
                             Paper ID: 2305.14281                                                                                               
                             Title: Weakly-Supervised Learning of Visual Relations in Multimodal Pretraining                                    
                             Keynote: {'TLDR': 'The paper proposes two innovative pretraining methods—Verbalised Scene Graphs (VSG) and Masked  
                             Relation Classification (MRC)—to enhance vision-and-language models (VLMs) by leveraging structured visual relation
                             data. The proposed methods improve fine-grained understanding, particularly in spatial reasoning tasks while       
                             maintaining competitive performance on coarse-grained tasks.', 'key_contributions': [{'title': 'Introduction of    
                             Novel Pretraining Approaches', 'description': 'The paper presents VSG and MRC to integrate scene graph relations   
                             into VLMs, improving their understanding of fine-grained visual and textual relationships.'}, {'title': 'Empirical 
                             Validation on Diverse Benchmarks', 'description': 'The authors conduct extensive evaluations showing that the      
                             relation-enhanced models outperform baselines in fine-grained tasks, particularly spatial reasoning, and achieve   
                             competitive results in coarse-grained benchmarks.'}, {'title': 'Checkpoint Selection Strategy Reassessment',       
                             'description': 'The paper critically revisits the practice of selecting the last checkpoint during pretraining,    
                             advocating for a dynamic selection strategy that could yield better model performance.'}], 'methods': {'VSG':      
                             {'overview': 'Transforms triplets from scene graphs into structured captions, effectively treating them as         
                             additional image descriptions.', 'process': ['Sample K triplets from a scene graph.', 'Sort them based on spatial  
                             location of their subject entities.', 'Verbalise into a structured caption format.']}, 'MRC': {'overview':         
                             'Predicts the relationship between two entities by masking their visual context during training.', 'process':      
                             ['Encodes the subject and object entities from the scene graph triplet.', 'Masks their visual contexts.', 'Pools   
                             their final cross-modal representations and classifies using a two-layer MLP.']}}, 'results':                      
                             {'fine_grained_tasks': {'spatial_reasoning': {'result': 'REX-VLM achieved a state-of-the-art performance with      
                             improvements of +6.8/3.0 p.p. on VSR Dev/Test sets.', 'significance': 'Demonstrates the efficacy of models enhanced
                             with relational understanding.'}, 'VALSE_and_SVO_Probes': {'result': 'REX-VLM models posted gains of +1.7 p.p. on  
                             VALSE and +0.8 p.p. on SVO-Probes.', 'significance': 'Further validation of the benefits of incorporating          
                             structured visual relations.'}}, 'coarse_grained_tasks': {'Flickr30K_and_COCO': {'result': 'REX-VLM models         
                             maintained competitive performance (e.g., COCO Dev TR @ 1 with +2.8 p.p.) despite their focus on fine-grained      
                             tasks.', 'significance': 'Shows that the introduction of relational data does not compromise retrieval             
                             efficiency.'}}}, 'critical_reflections': {'strengths': ['The integration of structured visual information from     
                             scene graphs is insightful and adds a layer of interpretability to model behavior, especially for fine-grained     
                             tasks.', 'Thorough evaluation across various benchmarks strengthens the claims of the proposed methods’            
                             effectiveness.'], 'weaknesses': ['Dependence on a small dataset of human-annotated scene graphs poses scalability  
                             challenges, which could limit wider application or generalization of the model.', 'The decision to treat scene     
                             graph captions similarly to regular captions could introduce noise during training, affecting coarse retrieval     
                             tasks.']}, 'future_directions': [{'title': 'Scaling with Machine-Generated Data', 'description': 'Future work could
                             explore using machine-generated scene graphs to expand the dataset efficiently and enhance the model robustly.'},  
                             {'title': 'Cross-Domain Applicability', 'description': 'Further validation of the methods across diverse domains   
                             could uncover the extent of benefits from relational data integration.'}, {'title': 'Investigate Alternative       
                             Checkpoint Selection Strategies', 'description': 'A more comprehensive study on how checkpoint selection affects   
                             learning dynamics across different tasks could build upon the authors’ findings.'}]}                               
                                                                                                                                                
                             Paper ID: 2407.19474                                                                                               
                             Title: Visual Riddles: a Commonsense and World Knowledge Challenge for Large Vision and Language Models            
                             Keynote: {'title': 'Visual Riddles: a Commonsense and World Knowledge Challenge for Large Vision and Language      
                             Models', 'authors': ['Nitzan Bitton-Guetta', 'Aviv Slobodkin', 'Aviya Maimon', 'Eliya Habba', 'Royi Rassin',       
                             'Yonatan Bitton', 'Idan Szpektor', 'Amir Globerson', 'Yuval Elovici'], 'key_contributions': ['Introduction of the  
                             Visual Riddles benchmark consisting of 400 visual riddles to evaluate vision and language models.', 'Emphasis on   
                             the necessity for commonsense reasoning and world knowledge in interpreting complex visual scenarios.',            
                             'Demonstrates significant performance gaps between human and machine understanding in spatial and contextual       
                             reasoning.'], 'methods': {'data_collection': 'The Visual Riddles were hand-curated by a team experienced in image  
                             generation, using a variety of text-to-image models to create ambiguous scenarios requiring commonsense            
                             reasoning.', 'tasks': {'open_ended_vqa': 'Models answer open-ended questions based on a given image and question.',
                             'multiple_choice_vqa': 'Models select the correct answer from five options related to the riddle.',                
                             'automatic_evaluation': {'reference_based': 'Models assess candidates based on provided ground truth answers.',    
                             'reference_free': 'Models assess candidate answers without any ground truth information.'}}, 'evaluation_metrics': 
                             'Human evaluations were conducted for accuracy comparison against models, highlighting the performance gaps.'},    
                             'results': {'accuracy': {'humans': '82%', 'gemini_pro_1.5': '40%', 'GPT4': '34%', 'LLaVA-1.6-34B': '15%'},         
                             'observations': ['Existing models notably lag behind human judgment even with contextual hints.', 'The integration 
                             of auxiliary information significantly aids model performance in riddles.'], 'challenges_identified':              
                             'State-of-the-art models struggle with nuanced image interpretation and commonsense reasoning.'}, 'significance':  
                             'This benchmark demonstrates the critical need for improved reasoning capabilities within vision and language      
                             models, making it a valuable resource for further advancements in AI understanding of visual scenarios.',          
                             'critical_reflections': {'gaps': ['There exists a perceptible recognition and reasoning gap between model outputs  
                             and human capability.', 'Challenges persist in model performance especially in high-difficulty riddles requiring   
                             specialized knowledge.'], 'future_directions': ['Future research may focus on developing models that better utilize
                             visual cues and commonsense reasoning.', 'Exploration of automated dataset generation and improvement in model     
                             architectures specifically tailored for visual reasoning tasks.']}, 'TL;DR': 'The Visual Riddles benchmark presents
                             a novel challenge for vision and language models by testing their ability to interpret complex visual cues         
                             requiring commonsense reasoning, revealing significant performance gaps compared to human evaluation.', 'examples':
                             {'riddle_example': {'question': 'What is probably the gender of the cat?', 'answer': "Referencing the visual clue  
                             about the cat's fur color and the commonsense knowledge that calico cats are predominantly female."}},             
                             'observations': {'caption_quality': 'Human-generated captions tend to include critical visual details more         
                             effectively than model-generated captions, affecting the ability to answer riddles accurately.', 'tool_usage':     
                             'Models performed slightly better on riddles where external tools for verification might have been helpful,        
                             indicating a gap in self-sufficient reasoning.'}, 'datasets': {'url': 'https://visual-riddles.github.io/',         
                             'license': 'Apache License 2.0', 'dataset_info': 'The dataset is hand-curated, emphasizing quality challenges with 
                             a focus on cultural relevance and commonsense reasoning.'}}                                                        
                                                                                                                                                
                                                                                                                                                
                                                                                                                                                
                             - other relevant papers: Paper_id: 2507.01513                                                                      
                             Title: SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism                  
                             Abstract: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual   
                             reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal   
                             jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text            
                             Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs   
                             with LLMs'built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities,        
                             particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to     
                             text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To
                             bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass     
                             safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for     
                             inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens,      
                             without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we    
                             propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful      
                             tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional      
                             computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive  
                             evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating
                             jailbreak risks without compromising utility.                                                                      
                                                                                                                                                
                             Paper_id: 2409.18142                                                                                               
                             Title: A Survey on Multimodal Benchmarks: In the Era of Large AI Models                                            
                             Abstract: The rapid evolution of Multimodal Large Language Models (MLLMs) has brought substantial advancements in  
                             artificial intelligence, significantly enhancing the capability to understand and generate multimodal content.     
                             While prior studies have largely concentrated on model architectures and training methodologies, a thorough        
                             analysis of the benchmarks used for evaluating these models remains underexplored. This survey addresses this gap  
                             by systematically reviewing 211 benchmarks that assess MLLMs across four core domains: understanding, reasoning,   
                             generation, and application. We provide a detailed analysis of task designs, evaluation metrics, and dataset       
                             constructions, across diverse modalities. We hope that this survey will contribute to the ongoing advancement of   
                             MLLM research by offering a comprehensive overview of benchmarking practices and identifying promising directions  
                             for future work. An associated GitHub repository collecting the latest papers is available.                        
                                                                                                                                                
                             Paper_id: 2407.01599                                                                                               
                             Title: JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models    
                             Abstract: The rapid evolution of artificial intelligence (AI) through developments in Large Language Models (LLMs) 
                             and Vision-Language Models (VLMs) has brought significant advancements across various technological domains. While 
                             these models enhance capabilities in natural language processing and visual interactive tasks, their growing       
                             adoption raises critical concerns regarding security and ethical alignment. This survey provides an extensive      
                             review of the emerging field of jailbreaking--deliberately circumventing the ethical and operational boundaries of 
                             LLMs and VLMs--and the consequent development of defense mechanisms. Our study categorizes jailbreaks into seven   
                             distinct types and elaborates on defense strategies that address these vulnerabilities. Through this comprehensive 
                             examination, we identify research gaps and propose directions for future studies to enhance the security frameworks
                             of LLMs and VLMs. Our findings underscore the necessity for a unified perspective that integrates both jailbreak   
                             strategies and defensive solutions to foster a robust, secure, and reliable environment for the next generation of 
                             language models. More details can be found on our website: https://chonghan-chen.com/llm-jailbreak-zoo-survey/.    
                                                                                                                                                
                             Paper_id: 2511.17487                                                                                               
                             Title: Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small Multimodal Models         
                             Abstract: Scaling up multimodal models has enabled remarkable advances in visual understanding and reasoning, but  
                             practical demands call for smaller, efficient systems. In this work, we conduct a principled analysis of           
                             downscaling intelligence in multimodal models, examining how reduced large language model (LLM) capacity affects   
                             multimodal capabilities. Our initial findings reveal an interesting trend: LLM downscaling disproportionately      
                             affects visual capabilities, rather than abilities inherited from the LLM. We then examine whether this drop mainly
                             reflects the expected decline in visual reasoning or a more fundamental loss of perceptual abilities. Isolating the
                             effect of LLM downscaling on perception, we find performance still drops sharply, often matching or exceeding the  
                             impact on reasoning. To address this bottleneck, we introduce visual extraction tuning, which explicitly trains the
                             model to extract instruction-relevant visual details consistently across tasks. With these extracted visual        
                             details, we then apply step-by-step reasoning to generate answers. Together, these components form our             
                             Extract+Think approach, setting a new standard for efficiency and performance in this space.                       
                                                                                                                                                
                             Paper_id: 2407.21659                                                                                               
                             Title: Cross-modality Information Check for Detecting Jailbreaking in Multimodal Large Language Models             
                             Abstract: Multimodal Large Language Models (MLLMs) extend the capacity of LLMs to understand multimodal information
                             comprehensively, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have  
                             shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where       
                             malicious users can break the safety alignment of the target model and generate misleading and harmful answers.    
                             This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced 
                             by vision input. To enhance the security of MLLMs against jailbreak attacks, researchers have developed various    
                             defense techniques. However, these methods either require modifications to the model's internal structure or demand
                             significant computational resources during the inference phase. Multimodal information is a double-edged sword.    
                             While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by  
                             this, we propose Cross-modality Information DEtectoR (CIDER), a plug-and-play jailbreaking detector designed to    
                             identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and      
                             adversarial images. CIDER is independent of the target MLLMs and requires less computation cost. Extensive         
                             experimental results demonstrate the effectiveness and efficiency of CIDER, as well as its transferability to both 
                             white-box and black-box MLLMs.                                                                                     
                                                                                                                                                
                             Paper_id: 2410.04659                                                                                               
                             Title: ActiView: Evaluating Active Perception Ability for Multimodal Large Language Models                         
                             Abstract: Active perception, a crucial human capability, involves setting a goal based on the current understanding
                             of the environment and performing actions to achieve that goal. Despite significant efforts in evaluating          
                             Multimodal Large Language Models (MLLMs), active perception has been largely overlooked. To address this gap, we   
                             propose a novel benchmark named ActiView to evaluate active perception in MLLMs. We focus on a specialized form of 
                             Visual Question Answering (VQA) that eases and quantifies the evaluation yet challenging for existing MLLMs.       
                             Meanwhile, intermediate reasoning behaviors of models are also discussed. Given an image, we restrict the          
                             perceptual field of a model, requiring it to actively zoom or shift its perceptual field based on reasoning to     
                             answer the question successfully. We conduct extensive evaluation over 30 models, including proprietary and        
                             open-source models, and observe that restricted perceptual fields play a significant role in enabling active       
                             perception. Results reveal a significant gap in the active perception capability of MLLMs, indicating that this    
                             area deserves more attention. We hope that ActiView could help develop methods for MLLMs to understand multimodal  
                             inputs in more natural and holistic ways.                                                                          
                                                                                                                                                
                             Paper_id: 2510.27492                                                                                               
                             Title: ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning                        
                             Abstract: Multimodal reasoning requires iterative coordination between language and vision, yet it remains unclear 
                             what constitutes a meaningful interleaved chain of thought. We posit that text and image thoughts should function  
                             as complementary rather than isomorphic modalities that mutually advance reasoning. Guided by this principle, we   
                             build ThinkMorph, a unified model fine-tuned on approximately 24K high-quality interleaved reasoning traces        
                             spanning tasks with varying visual engagement. ThinkMorph learns to generate progressive text-image reasoning steps
                             that concretely manipulate visual content while maintaining coherent verbal logic. It delivers large gains on      
                             vision-centric benchmarks (averaging 34.7 percent over the base model) and generalizes to out-of-domain tasks,     
                             matching or surpassing larger and proprietary VLMs. Beyond performance, ThinkMorph exhibits emergent multimodal    
                             intelligence, including unseen visual manipulation skills, adaptive switching between reasoning modes, and better  
                             test-time scaling through diversified multimodal thoughts. These findings suggest promising directions for         
                             characterizing the emergent capabilities of unified models for multimodal reasoning.                               
                                                                                                                                                
                             Paper_id: 2509.25502                                                                                               
                             Title: Seeing Before Reasoning: A Unified Framework for Generalizable and Explainable Fake Image Detection         
                             Abstract: Detecting AI-generated images with multimodal large language models (MLLMs) has gained increasing        
                             attention, due to their rich world knowledge, common-sense reasoning, and potential for explainability. However,   
                             naively applying those MLLMs for detection often leads to suboptimal performance. We argue that the root of this   
                             failure lies in a fundamental mismatch: MLLMs are asked to reason about fakes before they can truly see them.      
                             First, they do not really see: existing MLLMs'vision encoders are primarily optimized for semantic-oriented        
                             recognition rather than the perception of low-level signals, leaving them insensitive to subtle forgery traces.    
                             Without access to reliable perceptual evidence, the model grounds its judgment on incomplete and limited visual    
                             observations. Second, existing finetuning data for detection typically uses narrow, instruction-style formats,     
                             which diverge sharply from the diverse, heterogeneous distributions seen in pretraining. In the absence of         
                             meaningful visual cues, the model therefore exploits these linguistic shortcuts, resulting in catastrophic         
                             forgetting of pretrained knowledge (even the basic dialogue capabilities). In response, we advocate for a new      
                             paradigm: seeing before reasoning. We propose that MLLMs should first be trained to perceive                       
                             artifacts-strengthening their artifact-aware visual perception-so that subsequent reasoning is grounded in actual  
                             observations. We therefore propose Forensic-Chat, a generalizable, explainable, and still-conversational (for      
                             multi-round dialogue) assistant for fake image detection. We also propose ExplainFake-Bench, a benchmark tailored  
                             for the evaluation of the MLLM's explainability for image forensics from five key aspects. Extensive experiments   
                             show its superiority of generalization and genuinely reliable explainability.                                      
                                                                                                                                                
                             Paper_id: 2306.13549                                                                                               
                             Title: A survey on multimodal large language models                                                                
                             Abstract: ABSTRACT Recently, the multimodal large language model (MLLM) represented by GPT-4V has been a new rising
                             research hotspot, which uses powerful large language models (LLMs) as a brain to perform multimodal tasks. The     
                             surprising emergent capabilities of the MLLM, such as writing stories based on images and optical character        
                             recognition–free math reasoning, are rare in traditional multimodal methods, suggesting a potential path to        
                             artificial general intelligence. To this end, both academia and industry have endeavored to develop MLLMs that can 
                             compete with or even outperform GPT-4V, pushing the limit of research at a surprising speed. In this paper, we aim 
                             to trace and summarize the recent progress of MLLMs. First, we present the basic formulation of the MLLM and       
                             delineate its related concepts, including architecture, training strategy and data, as well as evaluation. Then, we
                             introduce research topics about how MLLMs can be extended to support more granularity, modalities, languages and   
                             scenarios. We continue with multimodal hallucination and extended techniques, including multimodal in-context      
                             learning, multimodal chain of thought and LLM-aided visual reasoning. To conclude the paper, we discuss existing   
                             challenges and point out promising research directions.                                                            
                                                                                                                                                
                             Paper_id: 2408.08464                                                                                               
                             Title: $\textit{MMJ-Bench}$: A Comprehensive Study on Jailbreak Attacks and Defenses for Multimodal Large Language 
                             Models                                                                                                             
                             Abstract: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts, Multimodal    
                             Large Language Models (MLLMs), have shown exceptional performance in many real-world tasks. However, MLLMs face    
                             significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model's safety   
                             alignment to elicit harmful responses. The threat of jailbreak attacks on MLLMs arises from both the inherent      
                             vulnerabilities of LLMs and the multiple information channels that MLLMs process. While various attacks and        
                             defenses have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is    
                             evaluated on different dataset and metrics, making it impossible to compare the effectiveness of each method. To   
                             address this gap, we introduce \textit{MMJ-Bench}, a unified pipeline for evaluating jailbreak attacks and defense 
                             techniques for MLLMs. Through extensive experiments, we assess the effectiveness of various attack methods against 
                             SoTA MLLMs and evaluate the impact of defense mechanisms on both defense effectiveness and model utility for normal
                             tasks. Our comprehensive evaluation contribute to the field by offering a unified and systematic evaluation        
                             framework and the first public-available benchmark for MLLM jailbreak research. We also demonstrate several        
                             insightful findings that highlights directions for future studies.                                                 
                                                                                                                                                
                             Paper_id: 2409.02813                                                                                               
                             Title: MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark                                 
                             Abstract: This paper introduces MMMU-Pro, a robust version of the Massive Multi-discipline Multimodal Understanding
                             and Reasoning (MMMU) benchmark. MMMU-Pro rigorously assesses multimodal models' true understanding and reasoning   
                             capabilities through a three-step process based on MMMU: (1) filtering out questions answerable by text-only       
                             models, (2) augmenting candidate options, and (3) introducing a vision-only input setting where questions are      
                             embedded within images. This setting challenges AI to truly"see"and"read"simultaneously, testing a fundamental     
                             human cognitive skill of seamlessly integrating visual and textual information. Results show that model performance
                             is substantially lower on MMMU-Pro than on MMMU, ranging from 16.8% to 26.9% across models. We explore the impact  
                             of OCR prompts and Chain of Thought (CoT) reasoning, finding that OCR prompts have minimal effect while CoT        
                             generally improves performance. MMMU-Pro provides a more rigorous evaluation tool, closely mimicking real-world    
                             scenarios and offering valuable directions for future research in multimodal AI.                                   
                                                                                                                                                
                             Paper_id: 2306.13394                                                                                               
                             Title: MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models                              
                             Abstract: Multimodal Large Language Model (MLLM) relies on the powerful LLM to perform multimodal tasks, showing   
                             amazing emergent abilities in recent studies, such as writing poems based on an image. However, it is difficult for
                             these case studies to fully reflect the performance of MLLM, lacking a comprehensive evaluation. In this paper, we 
                             fill in this blank, presenting the first comprehensive MLLM Evaluation benchmark MME. It measures both perception  
                             and cognition abilities on a total of 14 subtasks. In order to avoid data leakage that may arise from direct use of
                             public datasets for evaluation, the annotations of instruction-answer pairs are all manually designed. The concise 
                             instruction design allows us to fairly compare MLLMs, instead of struggling in prompt engineering. Besides, with   
                             such an instruction, we can also easily carry out quantitative statistics. A total of 30 advanced MLLMs are        
                             comprehensively evaluated on our MME, which not only suggests that existing MLLMs still have a large room for      
                             improvement, but also reveals the potential directions for the subsequent model optimization. The data are released
                             at the project page https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation.           
                                                                                                                                                
                             Paper_id: a57bdbed3032070564efab6e39c197e27cf82d23                                                                 
                             Title: Words Over Pixels? Rethinking Vision in Multimodal Large Language Models                                    
                             Abstract: Multimodal Large Language Models (MLLMs) promise seamless integration of vision and language             
                             understanding. However, despite their strong performance, recent studies reveal that MLLMs often fail to           
                             effectively utilize visual information, frequently relying on textual cues instead. This survey provides a         
                             comprehensive analysis of the vision component in MLLMs, covering both application-level and architectural aspects.
                             We investigate critical challenges such as weak spatial reasoning, poor fine-grained visual perception, and        
                             suboptimal fusion of visual and textual modalities. Additionally, we explore limitations in current vision         
                             encoders, benchmark inconsistencies, and their implications for downstream tasks. By synthesizing recent           
                             advancements, we highlight key research opportunities to enhance visual understanding, improve cross-modal         
                             alignment, and develop more robust and efficient MLLMs. Our observations emphasize the urgent need to elevate      
                             vision to an equal footing with language, paving the path for more reliable and perceptually aware multimodal      
                             models.                                                                                                            
                                                                                                                                                
                             Paper_id: 2510.09361                                                                                               
                             Title: BLINK-Twice: You see, but do you observe? A Reasoning Benchmark on Visual Perception                        
                             Abstract: Recently, Multimodal Large Language Models (MLLMs) have made rapid progress, particularly in enhancing   
                             their reasoning capabilities. However, existing reasoning benchmarks still primarily assess language-based         
                             reasoning, often treating visual input as replaceable context. To address this gap, we introduce BLINK-Twice, a    
                             vision-centric reasoning benchmark grounded in challenging perceptual tasks. Instead of relying on external        
                             knowledge, our tasks require models to reason from visual content alone, shifting the focus from language-based to 
                             image-grounded reasoning. Compared to prior perception benchmarks, it moves beyond shallow perception ("see") and  
                             requires fine-grained observation and analytical reasoning ("observe"). BLINK-Twice integrates three core          
                             components: seven types of visual challenges for testing visual reasoning, natural adversarial image pairs that    
                             enforce reliance on visual content, and annotated reasoning chains for fine-grained evaluation of the reasoning    
                             process rather than final answers alone. We evaluate 20 leading MLLMs, including 12 foundation models and 8        
                             reasoning-enhanced models. BLINK-Twice poses a significant challenge to current models. While existing reasoning   
                             strategies in the language space-such as chain-of-thought or self-criticism can improve performance, they often    
                             result in unstable and redundant reasoning. We observe that repeated image observation improves performance across 
                             models, and active visual interaction, as demonstrated by models like o3, highlights the need for a new paradigm   
                             for vision reasoning. The dataset is publicly available at https://github.com/PicoTrex/BLINK-Twice                 
                                                                                                                                                
                             Paper_id: 2310.00653                                                                                               
                             Title: Reformulating Vision-Language Foundation Models and Datasets Towards Universal Multimodal Assistants        
                             Abstract: Recent Multimodal Large Language Models (MLLMs) exhibit impressive abilities to perceive images and      
                             follow open-ended instructions. The capabilities of MLLMs depend on two crucial factors: the model architecture to 
                             facilitate the feature alignment of visual modules and large language models; the multimodal instruction tuning    
                             datasets for human instruction following. (i) For the model architecture, most existing models introduce an        
                             external bridge module to connect vision encoders with language models, which needs an additional feature-alignment
                             pre-training. In this work, we discover that compact pre-trained vision language models can inherently serve as    
                             ``out-of-the-box'' bridges between vision and language. Based on this, we propose Muffin framework, which directly 
                             employs pre-trained vision-language models to act as providers of visual signals. (ii) For the multimodal          
                             instruction tuning datasets, existing methods omit the complementary relationship between different datasets and   
                             simply mix datasets from different tasks. Instead, we propose UniMM-Chat dataset which explores the                
                             complementarities of datasets to generate 1.1M high-quality and diverse multimodal instructions. We merge          
                             information describing the same image from diverse datasets and transforms it into more knowledge-intensive        
                             conversation data. Experimental results demonstrate the effectiveness of the Muffin framework and UniMM-Chat       
                             dataset. Muffin achieves state-of-the-art performance on a wide range of vision-language tasks, significantly      
                             surpassing state-of-the-art models like LLaVA and InstructBLIP. Our model and dataset are all accessible at        
                             https://github.com/thunlp/muffin.                                                                                  
                                                                                                                                                
                             Paper_id: 2404.12390                                                                                               
                             Title: BLINK: Multimodal Large Language Models Can See but Not Perceive                                            
                             Abstract: We introduce Blink, a new benchmark for multimodal language models (LLMs) that focuses on core visual    
                             perception abilities not found in other evaluations. Most of the Blink tasks can be solved by humans"within a      
                             blink"(e.g., relative depth estimation, visual correspondence, forensics detection, and multi-view reasoning).     
                             However, we find these perception-demanding tasks cast significant challenges for current multimodal LLMs because  
                             they resist mediation through natural language. Blink reformats 14 classic computer vision tasks into 3,807        
                             multiple-choice questions, paired with single or multiple images and visual prompting. While humans get 95.70%     
                             accuracy on average, Blink is surprisingly challenging for existing multimodal LLMs: even the best-performing      
                             GPT-4V and Gemini achieve accuracies of 51.26% and 45.72%, only 13.17% and 7.63% higher than random guessing,      
                             indicating that such perception abilities have not"emerged"yet in recent multimodal LLMs. Our analysis also        
                             highlights that specialist CV models could solve these problems much better, suggesting potential pathways for     
                             future improvements. We believe Blink will stimulate the community to help multimodal LLMs catch up with           
                             human-level visual perception.                                                                                     
                                                                                                                                                
                             Paper_id: 2308.02490                                                                                               
                             Title: MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities                                      
                             Abstract: We propose MM-Vet, an evaluation benchmark that examines large multimodal models (LMMs) on complicated   
                             multimodal tasks. Recent LMMs have shown various intriguing abilities, such as solving math problems written on the
                             blackboard, reasoning about events and celebrities in news images, and explaining visual jokes. Rapid model        
                             advancements pose challenges to evaluation benchmark development. Problems include: (1) How to systematically      
                             structure and evaluate the complicated multimodal tasks; (2) How to design evaluation metrics that work well across
                             question and answer types; and (3) How to give model insights beyond a simple performance ranking. To this end, we 
                             present MM-Vet, designed based on the insight that the intriguing ability to solve complicated tasks is often      
                             achieved by a generalist model being able to integrate different core vision-language (VL) capabilities. MM-Vet    
                             defines 6 core VL capabilities and examines the 16 integrations of interest derived from the capability            
                             combination. For evaluation metrics, we propose an LLM-based evaluator for open-ended outputs. The evaluator       
                             enables the evaluation across different question types and answer styles, resulting in a unified scoring metric. We
                             evaluate representative LMMs on MM-Vet, providing insights into the capabilities of different LMM system paradigms 
                             and models.                                                                                                        
                                                                                                                                                
                             Paper_id: 3417cfbcff54ae02ef20a61fd33da6f97807a9ef                                                                 
                             Title: MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models               
                             Abstract: As deep learning advances, Large Language Models (LLMs) and their multimodal counterparts,               
                             Vision-Language Models (VLMs), have shown exceptional performance in many real-world tasks. However, VLMs face     
                             significant security challenges, such as jailbreak attacks, where attackers attempt to bypass the model’s safety   
                             alignment to elicit harmful responses. The threat of jailbreak attacks on VLMs arises from both the inherent       
                             vulnerabilities of LLMs and the multiple information channels that VLMs process. While various attacks and defenses
                             have been proposed, there is a notable gap in unified and comprehensive evaluations, as each method is evaluated on
                             different dataset and metrics, making it impossible to compare the effectiveness of each method. To address this   
                             gap, we introduce MMJ-Bench, a unified pipeline for evaluating jailbreak attacks and defense techniques for VLMs.  
                             Through extensive experiments, we assess the effectiveness of various attack methods against SoTA VLMs and evaluate
                             the impact of defense mechanisms on both defense effectiveness and model utility for normal tasks. Our             
                             comprehensive evaluation contribute to the field by offering a unified and systematic evaluation framework and the 
                             first public-available benchmark for VLM jailbreak research. We also demonstrate several insightful findings that  
                             highlights directions for future studies.                                                                          
                                                                                                                                                
                             Paper_id: 2507.11932                                                                                               
                             Title: Hyperphantasia: A Benchmark for Evaluating the Mental Visualization Capabilities of Multimodal LLMs         
                             Abstract: Mental visualization, the ability to construct and manipulate visual representations internally, is a    
                             core component of human cognition and plays a vital role in tasks involving reasoning, prediction, and abstraction.
                             Despite the rapid progress of Multimodal Large Language Models (MLLMs), current benchmarks primarily assess passive
                             visual perception, offering limited insight into the more active capability of internally constructing visual      
                             patterns to support problem solving. Yet mental visualization is a critical cognitive skill in humans, supporting  
                             abilities such as spatial navigation, predicting physical trajectories, and solving complex visual problems through
                             imaginative simulation. To bridge this gap, we introduce Hyperphantasia, a synthetic benchmark designed to evaluate
                             the mental visualization abilities of MLLMs through four carefully constructed puzzles. Each task is procedurally  
                             generated and presented at three difficulty levels, enabling controlled analysis of model performance across       
                             increasing complexity. Our comprehensive evaluation of state-of-the-art models reveals a substantial gap between   
                             the performance of humans and MLLMs. Additionally, we explore the potential of reinforcement learning to improve   
                             visual simulation capabilities. Our findings suggest that while some models exhibit partial competence in          
                             recognizing visual patterns, robust mental visualization remains an open challenge for current MLLMs.              
                                                                                                                                                
                             Paper_id: 2306.09265                                                                                               
                             Title: LVLM-EHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models                            
                             Abstract: Large Vision-Language Models (LVLMs) have recently played a dominant role in multimodal vision-language  
                             learning. Despite the great success, it lacks a holistic evaluation of their efficacy. This paper presents a       
                             comprehensive evaluation of publicly available large multimodal models by building an LVLM evaluation Hub          
                             (LVLM-eHub). Our LVLM-eHub consists of 13 representative LVLMs such as InstructBLIP and LLaVA, which are thoroughly
                             evaluated by a quantitative capability evaluation and an online arena platform. The former evaluates five          
                             categories of multimodal capabilities of LVLMs such as visual question answering and object hallucination on 42    
                             in-domain text-related visual benchmarks, while the latter provides the user-level evaluation of LVLMs in an       
                             open-world question-answering scenario. The study investigates how specific features of LVLMs such as model        
                             configurations, modality alignment mechanisms, and training data affect the multimodal understanding. By conducting
                             a comprehensive comparison of these features on quantitative and arena evaluation, our study uncovers several      
                             innovative findings, which establish a fundamental framework for the development and evaluation of innovative      
                             strategies aimed at enhancing multimodal techniques.   
"""
# ================= 1. 基础工具函数 =================

def call_llm_api(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature
        )
        content = resp.choices[0].message.content
        print(f"   [LLM Response ({len(content)} chars)]")
        return content
    except Exception as e:
        print(f"Error calling API: {e}")
        return ""

def parse_llm_output(text: str) -> Union[Dict, List, None]:
    text = text.strip()
    # 尝试去除 Markdown 代码块标记 (```json ... ```)
    if "```" in text:
        text = re.sub(r'```json\s*|```', '', text).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # 寻找列表 [...]
            if '[' in text and ']' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                return json.loads(text[start:end])
            # 寻找对象 {...}
            elif '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                return json.loads(text[start:end])
        except Exception:
            pass
    
    print("Error: Failed to parse JSON/List from output.")
    print(f"Raw output: {text[:200]}...")
    return None

def apply_revision_to_text(original_text: str, revision_json: Dict[str, str]) -> str:
    """
    将revise 输出应用到text上的函数。
    """
    if revision_json.get("action") == "done":
        return original_text

    old_str = revision_json.get("originalText", "")
    new_str = revision_json.get("newText", "")

    if not old_str:
        print("Warning: 'originalText' is empty or do not have the key.")
        return original_text

    # 寻找精确匹配
    idx = original_text.find(old_str)
    if idx == -1:
        print(f"Error: Could not find exact substring to replace.\nTarget: {old_str[:50]}...")
        # 这里可以选择加入模糊匹配逻辑，但为了严谨暂时保持精确匹配失败则跳过
        return original_text
    
    # 执行替换
    new_text = original_text[:idx] + new_str + original_text[idx + len(old_str):]
    print(f"   >>> Applied revision: Replaced {len(old_str)} chars with {len(new_str)} chars.")
    return new_text

# ================= 2. 核心逻辑函数 =================

def run_reviewer(current_text: str, context_data: Dict) -> List[str]:
    """
    Reviewer功能的函数：读取文本，输出建议列表。
    """
    print("\n--- [Reviewer] analyzing text... ---")
    prompt = SECTION_REVIEW.format(
        current_section_length=len(current_text.split()),
        previous_section_text=context_data.get('prev', ""),
        next_section_text=context_data.get('next', ""),
        draft_text=current_text
    )
    
    response = call_llm_api([{"role": "user", "content": prompt}])
    suggestions = parse_llm_output(response)
    
    if isinstance(suggestions, list):
        print(f"   [Reviewer] Generated {len(suggestions)} suggestions.")
        return suggestions
    else:
        print("   [Reviewer] Failed to generate valid list.")
        return []

def run_reviser(current_text: str, suggestion: str, context_data: Dict) -> str:
    """
    Revise功能的函数：根据单条建议修改文本。
    """
    print(f"\n--- [Reviser] processing suggestion: {suggestion}... ---")
    prompt = SECTION_REVISE.format(
        topic="Multimodal Large Language Models", # 可配置
        text=current_text,
        citations=context_data.get('citations', ""),
        reviewer_suggestion=suggestion
    )

    response = call_llm_api([{"role": "user", "content": prompt}])
    parsed_json = parse_llm_output(response)

    if isinstance(parsed_json, dict):
        if parsed_json.get("action") == "done":
            print("   [Reviser] Decided no change needed.")
            return current_text
        elif parsed_json.get("action") == "replace":
            return apply_revision_to_text(current_text, parsed_json)
    
    print("   [Reviser] Invalid output or parsing failed.")
    return current_text

# ================= 3. 主程序 =================

def main():
    
    context_data = {
        "prev": "",
        "next": "",
        "citations": citations
    }

    current_text = original_text

    print("=== Start Iterative Revision Process ===")

    for i in range(MAX_OUTER_ITERATIONS):
        print(f"\n\n====== OUTER LOOP {i+1}/{MAX_OUTER_ITERATIONS} ======")
        
        # 1. 获取建议
        suggestions = run_reviewer(current_text, context_data)
        
        # 如果没有建议，或者建议包含 "done" (视模型具体输出而定)，则停止
        if not suggestions or len(suggestions) == 0 or(len(suggestions) == 1 and suggestions[0].lower() == "done"):
            print("Reviewer indicates completion. Exiting.")
            break

        # 2. 逐条执行建议
        print(f"Starting to apply {len(suggestions)} suggestions...")
        for idx, sug in enumerate(suggestions):
            # 将最新的 current_text 传入，实现串行修改
            new_text = run_reviser(current_text, sug, context_data)
            
            # 如果文本发生变化，更新 current_text
            if new_text != current_text:
                current_text = new_text
            else:
                print(f"   (Suggestion {idx+1} resulted in no change)")
        
        print(f"\n--- LOOP {i+1} REVISED TEXT --- \n")
        print(current_text)
        print(f"\n--- End of OUTER LOOP {i+1} --- \n")

    print("\n\n=== Final Result ===")
    print(current_text)

if __name__ == "__main__":
    main()