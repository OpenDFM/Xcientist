import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from utils.rich_logger import get_logger
from utils.utils import extract_json

logger = get_logger("AblationUtils")


class AblationDatabase:
    def __init__(self, db_path, embedding_model="BAAI/bge-large-en-v1.5"):
        self.logger = logger
        self.db_path = db_path
        
        try:
            self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
            self.embedding_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Loaded embedding model: {embedding_model} on {self.device}")
        except Exception as e:
            self.logger.warning(f"Failed to load model on GPU: {e}, falling back to CPU")
            self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
            self.embedding_model.to(torch.device('cpu'))
            self.device = torch.device('cpu')
        
        self._load_faiss_index()
        self._load_arxiv_mapping()
    
    def _load_faiss_index(self):
        import faiss
        title_index_path = os.path.join(self.db_path, 'faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin')
        if not os.path.exists(title_index_path):
            raise FileNotFoundError(f"FAISS index not found at: {title_index_path}")
        
        self.title_loaded_index = faiss.read_index(title_index_path)
        self.logger.info(f"Loaded FAISS index with {self.title_loaded_index.ntotal} vectors")
    
    def _load_arxiv_mapping(self):
        import json
        mapping_path = os.path.join(self.db_path, 'arxivid_to_index_abs.json')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Arxiv mapping not found at: {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            id_to_index = json.loads(f.read())
        
        self.id_to_index = {id: int(index) for id, index in id_to_index.items()}
        self.index_to_id = {int(index): id for id, index in id_to_index.items()}
        self.logger.info(f"Loaded arxiv mapping with {len(self.id_to_index)} entries")
    
    def get_embeddings(self, batch_text):
        batch_text = ['search_query: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text, convert_to_tensor=True)
        return embeddings.cpu().numpy().astype('float32')
    
    def search_papers(self, query_text, top_k=50):
        query_vector = self.get_embeddings([query_text])
        
        distances, indices = self.title_loaded_index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                paper_id = self.index_to_id.get(idx)
                if paper_id:
                    results.append({
                        'arxiv_id': paper_id,
                        'distance': float(distances[0][i])
                    })
        
        self.logger.info(f"Search for '{query_text}' returned {len(results)} papers")
        return results
    
    def search_papers_batch(self, queries, top_k=50):
        query_vectors = self.get_embeddings(queries)
        
        distances, indices = self.title_loaded_index.search(query_vectors, top_k)
        
        all_results = []
        for i in range(len(queries)):
            results = []
            for j, idx in enumerate(indices[i]):
                if idx != -1:
                    paper_id = self.index_to_id.get(idx)
                    if paper_id:
                        results.append({
                            'arxiv_id': paper_id,
                            'distance': float(distances[i][j])
                        })
            all_results.append(results)
        
        return all_results


def retrieve_papers_for_ablation(topic, db_path, num_papers=50):
    logger.info(f"Retrieving papers for ablation with topic: {topic}")
    
    try:
        db = AblationDatabase(db_path)
    except Exception as e:
        logger.error(f"Failed to initialize ablation database: {e}")
        raise
    
    results = db.search_papers(topic, top_k=num_papers)
    
    expanded_paper_ids = [r['arxiv_id'] for r in results]
    
    logger.info(f"Retrieved {len(expanded_paper_ids)} paper IDs for ablation")
    
    return expanded_paper_ids


SIMPLIFIED_OUTLINE_GENERATION = """You are a survey paper outline generator. Based on the given topic, generate a well-structured outline for a comprehensive survey paper.

Topic: {topic}

Generate a JSON outline with the following format (only titles, no descriptions):
{{
    "title": "Survey Title",
    "sections": [
        {{
            "title": "Section Title",
            "description": "",
            "subsections": [
                {{
                    "title": "Subsection Title",
                    "description": ""
                }}
            ]
        }}
    ]
}}

Requirements:
1. Generate 3-5 main sections covering different aspects of the topic
2. Each section should have 2-4 subsections
3. Use clear, descriptive titles that reflect the content
4. Return ONLY valid JSON, no explanations or markdown

Output JSON:"""

SIMPLIFIED_SUBSECTION_DRAFT = """You are a survey paper writer. Based on the paper keynotes and subsection title, write a comprehensive subsection.

Subsection Title: {subsection_title}
Topic: {topic}

Paper Keynotes (from related papers):
{keynotes}

Requirements:
1. Write a comprehensive subsection covering the topic
2. Reference papers ONLY using their titles in angle brackets, e.g., <Paper Title>
3. Include at least {min_citations} citations per subsection using the format <Paper Title>
4. Write in academic survey style with clear explanations
5. Combine information from multiple papers coherently
6. Do NOT add any additional citation formats like (Citations:) or References section
7. Do NOT number your citations

Write the subsection content:"""


def generate_simplified_outline(topic, chat_agent, temperature=0.5, max_retry=3):
    """Generate a simplified outline with only titles (no descriptions)"""
    from utils.utils import extract_json
    
    logger.info(f"Generating simplified outline for topic: {topic}")
    
    prompt = SIMPLIFIED_OUTLINE_GENERATION.format(topic=topic)
    
    def validate_outline(response, info_dict):
         """Validate that the response is a valid outline JSON"""
         outline = extract_json(response)
         if not outline:
             raise ValueError("Failed to parse JSON from response")
         if 'title' not in outline:
             raise ValueError("Outline missing 'title' field")
         if 'sections' not in outline:
             raise ValueError("Outline missing 'sections' field")
         if not isinstance(outline['sections'], list):
             raise ValueError("'sections' must be a list")
         for i, section in enumerate(outline['sections']):
             if 'title' not in section:
                 raise ValueError(f"Section {i} missing 'title' field")
             if 'subsections' not in section:
                 raise ValueError(f"Section {i} missing 'subsections' field")
             if not isinstance(section['subsections'], list):
                 raise ValueError(f"Section {i} 'subsections' must be a list")
             for j, subsection in enumerate(section['subsections']):
                 if 'title' not in subsection:
                     raise ValueError(f"Subsection {j} in section {i} missing 'title' field")
         return True, outline
    
    try:
        outline = chat_agent.remote_chat_with_retry(
            prompt=prompt,
            validate_fn=validate_outline,
            max_retry=max_retry,
            temperature=temperature
        )
        
        for section in outline.get('sections', []):
            if 'description' not in section:
                section['description'] = ""
            for subsection in section.get('subsections', []):
                if 'description' not in subsection:
                    subsection['description'] = ""
        
        logger.info(f"Successfully generated outline with {len(outline['sections'])} sections")
        return outline
        
    except Exception as e:
        logger.error(f"Failed to generate outline after {max_retry} retries: {e}")
        raise


def get_paper_keynotes(work_analyzer, paper_ids, top_k=50):
    """Get keynotes from the first k papers"""
    keynotes = []
    
    for i, paper_id in enumerate(paper_ids[:top_k]):
        try:
            keynote = work_analyzer.get_paper_keynote(paper_id)
            title = work_analyzer.work_collector.get_paper_title(paper_id)
            if keynote:
                keynotes.append({
                    'paper_id': paper_id,
                    'title': title,
                    'keynote': keynote
                })
        except Exception as e:
            logger.warning(f"Failed to get keynote for paper {paper_id}: {e}")
            continue
    
    return keynotes


def generate_simplified_survey(topic, collected_papers, work_analyzer, chat_agent, 
                               config, top_k_keynotes=50, min_citations=40):
    """Generate simplified survey without complex outline generation and paper assignment"""
    logger.info(f"Generating simplified survey for {len(collected_papers)} papers")
    
    outline = generate_simplified_outline(topic, chat_agent, 
                                         temperature=config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                                         max_retry=config.ModuleInfo.SurveyGenerator.outline_generation_max_retry)
    
    logger.info(f"Generated outline with {len(outline['sections'])} sections")
    
    keynotes = get_paper_keynotes(work_analyzer, collected_papers, top_k=top_k_keynotes)
    logger.info(f"Retrieved keynotes from {len(keynotes)} papers")
    
    keynote_text = ""
    for k in keynotes:
        keynote_text += f"\n\nPaper: <{k['title']}>\nKeynote: {k['keynote']}"
    
    subsection_prompts = []
    
    for section_idx, section in enumerate(outline.get('sections', [])):
        for subsection_idx, subsection in enumerate(section.get('subsections', [])):
            subsection_title = subsection.get('title', '')
            prompt = SIMPLIFIED_SUBSECTION_DRAFT.format(
                subsection_title=subsection_title,
                topic=topic,
                keynotes=keynote_text,
                min_citations=min_citations
            )
            subsection_prompts.append({
                'prompt': prompt,
                'section_idx': section_idx,
                'subsection_idx': subsection_idx,
                'section_title': section.get('title', ''),
                'subsection_title': subsection_title
            })
    
    logger.info(f"Generating {len(subsection_prompts)} subsections using batch chat...")
    prompt_list = [p['prompt'] for p in subsection_prompts]
    subsection_responses = chat_agent.batch_remote_chat(
        prompt_list,
        desc="Drafting survey subsections...",
        temperature=config.ModuleInfo.SurveyGenerator.subsection_draft_temperature
    )
    
    section_drafts = []
    for section_idx, section in enumerate(outline.get('sections', [])):
        section_title = section.get('title', '')
        section_content = f"## {section_title}\n\n"
        
        for subsection_idx, subsection in enumerate(section.get('subsections', [])):
            subsection_title = subsection.get('title', '')
            
            response_idx = None
            for i, p_info in enumerate(subsection_prompts):
                if p_info['section_idx'] == section_idx and p_info['subsection_idx'] == subsection_idx:
                    response_idx = i
                    break
            
            if response_idx is not None and subsection_responses[response_idx]:
                subsection_content = f"### {subsection_title}\n\n{subsection_responses[response_idx]}\n\n"
            else:
                subsection_content = f"### {subsection_title}\n\n[Content generation failed]\n\n"
            section_content += subsection_content
        
        section_drafts.append(section_content)
    
    outcome_draft = outline.get('title', topic + ' Survey') + "\n\n" + "\n\n".join(section_drafts)
    
    drafts = {
        "section_drafts": section_drafts,
        "full_draft": outcome_draft,
        "title": outline.get('title', topic + ' Survey'),
        "outline": outline
    }
    
    logger.info(f"Generated simplified survey draft with {len(section_drafts)} sections")
    
    return drafts
