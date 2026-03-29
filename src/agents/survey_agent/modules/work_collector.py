from typing import Dict, List
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.api_call import ChatAgent
from utils.config_utils import merge_with_default_survey_config
from utils.rich_logger import get_logger
from modules.paper_graph_retriever import PaperGraphRetriever
from modules.data_manager import DataManager
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch
import gc
from modules.pe import PAPER_RELATEDNESS_BASED_ON_TITLE_AND_ABSTRACT, SEED_PAPER_SELECTION
import diskcache as dc
from utils.utils import get_hash, extract_json
import hydra

class WorkCollector:
    def __init__(self, config, ignore_paper:List[str] = None, paper_graph_retriever = None, data_manager = None):
        self.config = config
        self.chat_agent = ChatAgent(config)
        self.logger = get_logger("WorkCollector")
        self.cache_path = self.config.BasicInfo.cache_path
        self.ignore_paper = set(ignore_paper) if ignore_paper is not None else set()

        # 初始化 DataManager（处理下载、解析、缓存等）
        self.data_manager = DataManager(config) if data_manager is None else data_manager
        self.paper_abstract_cache = self.data_manager.paper_abstract_cache

        ## local paper graph parameter
        self.expand_in_local_paper_graph = self.config.ModuleInfo.WorkCollector.expand_in_local_paper_graph
        self.advanced_filter_in_local_paper_graph_expansion = self.config.ModuleInfo.WorkCollector.advanced_filter_in_local_paper_graph_expansion
        if not self.expand_in_local_paper_graph:
            self.paper_graph_retriever = paper_graph_retriever
        else:
            self.paper_graph_retriever = PaperGraphRetriever(config)

        # load reference graph
        os.makedirs(self.cache_path, exist_ok=True)
        if os.path.exists(os.path.join(self.cache_path, "reference_graph.pkl")):
            with open(
                os.path.join(self.cache_path, "reference_graph.pkl"), "rb"
            ) as reader:
                self.reference_graph = pickle.load(reader)
        else:
            self.reference_graph = None

        # cache for paper relatedness
        self.relatedness_cache = dc.Cache(
            os.path.join(self.cache_path, "workcollector_relatedness_cache")
        )

        # embedding model for semantic similarity (lazy loading) - 委托给 data_manager
        self._embedding_model = None
        self._model_device = None
        self.graph_paper_ids = set()

    def _get_embedding_model(self):
        """Lazy load and cache the embedding model - 委托给 data_manager"""
        if self._embedding_model is not None:
            return self._embedding_model
        
        model_name = self.config.ModuleInfo.WorkCollector.sentence_transformer_model
        try:
            self._embedding_model = SentenceTransformer(model_name).cuda()
            self._model_device = "cuda"
        except Exception as e:
            if "out of memory" in str(e).lower():
                self.logger.error("Out of memory error detected. Using CPU instead.")
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    self.logger.warning("Failed to clear GPU cache.")
                    pass
                self._embedding_model = SentenceTransformer(model_name).cpu()
                self._model_device = "cpu"
            else:
                try:
                    self._embedding_model = SentenceTransformer(model_name).cpu()
                    self._model_device = "cpu"
                except Exception as e2:
                    self.logger.error(f"Failed to load SentenceTransformer model: {e2}")
                    raise e2
        
        return self._embedding_model

    # ========== 委托给 DataManager 的函数 ==========
    
    def filter_seed_papers(self, topic: str, papers: List[Dict], threshold: int = 4) -> List[Dict]:
        if not papers:
            return []

        self.logger.info(f"Filtering {len(papers)} candidate seed papers using LLM (Threshold >= {threshold})...")
        
        tasks = []
        for paper in papers:
            title = paper.get("title", "N/A")
            abstract = paper.get("abstract", "")
            
            prompt = SEED_PAPER_SELECTION.format(
                topic=topic,
                title=title,
                abstract=abstract if abstract else "Abstract not available."
            )
            tasks.append(prompt)

        responses = self.chat_agent.batch_remote_chat(
            tasks,
            temperature=0.0, 
            desc="Filtering Seed Papers"
        )

        valid_papers = []
        for paper, response in zip(papers, responses):
            try:
                result = extract_json(response)
                score = int(result.get("relevance_score", 0))
                reason = result.get("reason", "No reason provided")
                title = paper.get("title", "Unknown Title")

                if score >= threshold:
                    self.logger.info(f"✅ Accepted Seed Paper: [{score}] {title}")
                    if self.config.BasicInfo.debug:
                        self.logger.info(f"   Reason: {reason}")
                    valid_papers.append(paper)
                else:
                    self.logger.info(f"❌ Rejected Seed Paper: [{score}] {title} - {reason}")

            except Exception as e:
                self.logger.warning(f"Error parsing LLM response for paper {paper.get('title')}: {e}")
        
        self.logger.info(f"Seed paper filtering complete. Retained {len(valid_papers)}/{len(papers)} papers.")
        return valid_papers

    def collect_seed_papers(self, topic: str):
        """
        Collect related work based on the given topic.
        Returns a list of work items.
        """
        # version 1: use Semantic Scholar API to search papers
        fields = "title,externalIds,openAccessPdf,abstract"
        response = self.data_manager.semantic_scholar_api.search_papers(query=topic, fields=fields)

        if response is None:
            raise ValueError("Semantic Scholar search_papers failed after retries; aborting seed collection.")

        papers = response.get("data", [])
        self.logger.info(f"Found {len(papers)} papers related to topic: {topic}")

        if papers and self.config.ModuleInfo.WorkCollector.use_seed_filter_LLM:
            papers = self.filter_seed_papers(topic, papers, threshold=self.config.ModuleInfo.WorkCollector.LLM_seed_threshold)
        
        if not papers:
            self.logger.warning("No seed papers remained after LLM filtering! Please increase the seed paper number")
            return []

        valid_graph_seed_paper_ids = []
        if self.expand_in_local_paper_graph:
            self.logger.info("local paper graph enabled. Validating seed papers in paper graph...")
            for seed_paper in papers:

                if "ArXiv" in seed_paper.get("externalIds", {}):
                    paper_id = seed_paper["externalIds"]["ArXiv"]
                else:
                    paper_id = seed_paper.get("paperId")

                try:
                    title, abstract = self.data_manager.get_paper_title_abstract(paper_id)
                    results = self.paper_graph_retriever.search_by_paper_title(title)
                except Exception as e:
                    self.logger.error(f"Error occurs in seed paper validation in local paper graph mode: {e}")
                    continue
                if not results:
                    continue
                else:
                    valid_graph_seed_paper_ids.append(paper_id)
            if len(valid_graph_seed_paper_ids) == 0:
                raise ValueError("collected seed papers not in paper graph")
            self.logger.info(f"{len(valid_graph_seed_paper_ids)} seed papers in local paper graph, no need to download in advance, return...")
            self.graph_paper_ids.update(valid_graph_seed_paper_ids)
            return valid_graph_seed_paper_ids

        self.logger.info(
            f"Downloading and parsing up to {len(papers)} papers..."
        )
        valid_seed_papers_ids = self.data_manager.download_and_parse_papers(
            papers, limit=self.config.ModuleInfo.WorkCollector.max_seed_paper_num
        )
        valid_seed_papers_ids = [paper_id for paper_id in valid_seed_papers_ids if paper_id not in self.ignore_paper]
        self.graph_paper_ids.update(valid_seed_papers_ids)

        return valid_seed_papers_ids

    # ========== 直接委托给 DataManager 的函数 ==========
    
    def download_and_parse_papers(self, papers: list, limit: int = -1):
        """Download and parse papers - 委托给 DataManager"""
        return self.data_manager.download_and_parse_papers(papers, limit)

    def add_papers_abstracts_in_cache(self, papers: List[str], retry: int = 1):
        """Add paper abstracts to cache - 委托给 DataManager"""
        return self.data_manager.add_papers_abstracts_in_cache(papers, retry)

    def get_paper_title_abstract(self, paper_id: str, retry: int = 1):
        """Get paper title and abstract - 委托给 DataManager"""
        return self.data_manager.get_paper_title_abstract(paper_id, retry)

    def get_paper_title(self, paper_id: str, retry: int = 3):
        """Get paper title - 委托给 DataManager"""
        return self.data_manager.get_paper_title(paper_id, retry)

    def get_paper_raw_markdown(self, paper_id: str) -> str:
        """Get paper raw markdown - 委托给 DataManager"""
        return self.data_manager.get_paper_raw_markdown(paper_id)

    def get_paper_with_title(self, title: str):
        """Get paper with title - 委托给 DataManager"""
        return self.data_manager.get_paper_with_title(title)

    def get_paper_with_title_arxiv(self, title: str):
        """Get paper with title via arxiv - 委托给 DataManager"""
        return self.data_manager.get_paper_with_title_arxiv(title)

    def get_paper_with_title_semantic(self, title: str):
        """Get paper with title via semantic scholar - 委托给 DataManager"""
        return self.data_manager.get_paper_with_title_semantic(title)

    def get_paper_with_title_batch(self, titles: List[str]):
        """Get papers with titles in batch - 委托给 DataManager"""
        return self.data_manager.get_paper_with_title_batch(titles)

    def is_valid_abstract(self, abstract: str) -> bool:
        """Check if abstract is valid - 委托给 DataManager"""
        return self.data_manager.is_valid_abstract(abstract)

    # ========== 不委托给 DataManager 的函数 ==========
    
    def update_reference_graph(self, seed_paper_ids: List[str]):
        """
        Update the reference graph with new seed paper IDs.
        NOTE: All paper IDs are assumed to be Arxiv IDs first. If not found, fallback to Semantic Scholar Paper IDs.
        """
        self.logger.info("Updating reference graph...")
        if self.reference_graph is None:
            self.reference_graph = nx.DiGraph()

        def one_step(paper_id: str, visited, direction: str = "out"):
            if paper_id in visited:
                return
            visited.add(paper_id)

            related_papers = []
            if (
                paper_id not in self.reference_graph
                or (
                    direction == "out"
                    and self.reference_graph.out_degree(paper_id) == 0
                )
                or (direction == "in" and self.reference_graph.in_degree(paper_id) == 0)
            ):
                if "." in paper_id:
                    query_id = f"ARXIV:{paper_id}"
                else:
                    query_id = paper_id

                if direction == "out":
                    fields = "title,year,venue,abstract,authors,externalIds,references.title,references.externalIds,references.year,references.authors,references.abstract,references.venue"
                else:
                    fields = "title,year,venue,abstract,authors,externalIds,citations.title,citations.externalIds,citations.year,citations.authors,citations.abstract,citations.venue"

                try:
                    paper_detail = self.data_manager.semantic_scholar_api.get_paper_details(
                        query_id,
                        fields=fields,
                    )
                except Exception as e:
                    self.logger.error(f"Error fetching paper {paper_id} details from Semantic Scholar: {e}. Skipping.")
                    paper_detail = None

                if not paper_detail:
                    return

                _paper_id = paper_detail.get("externalIds", {}).get(
                    "ArXiv", paper_detail.get("paperId")
                )
                assert _paper_id == paper_id, "Paper ID mismatch!"
                if paper_id not in self.reference_graph:
                    self.reference_graph.add_node(
                        paper_id,
                        title=paper_detail.get("title"),
                        year=paper_detail.get("year", None),
                        authors=paper_detail.get("authors", None),
                        abstract=paper_detail.get("abstract", ""),
                        venue=paper_detail.get("venue", ""),
                    )

                relateds = paper_detail.get(
                    "references" if direction == "out" else "citations", []
                )
                if not relateds:
                    relateds = []

                for related in relateds:
                    if related.get("externalIds") is None:
                        continue
                    related_id = related.get("externalIds", {}).get(
                        "ArXiv", related.get("paperId")
                    )
                    if related_id not in self.reference_graph:
                        self.reference_graph.add_node(
                            related_id,
                            title=related.get("title"),
                            year=related.get("year", None),
                            authors=related.get("authors", None),
                            abstract=related.get("abstract", ""),
                            venue=related.get("venue", ""),
                        )

                    if direction == "out":
                        self.reference_graph.add_edge(paper_id, related_id)
                    else:
                        self.reference_graph.add_edge(related_id, paper_id)
                    related_papers.append(related_id)
            else:
                if direction == "out":
                    related_papers = list(self.reference_graph.successors(paper_id))
                else:
                    related_papers = list(self.reference_graph.predecessors(paper_id))
            return related_papers

        from tqdm import tqdm

        def traverse(direction="out"):
            current_papers = seed_paper_ids
            for _ in range(
                self.config.ModuleInfo.WorkCollector.reference_graph_depth
            ):
                next_papers = []
                visited = set()
                for paper_id in tqdm(current_papers):
                    related_papers = one_step(paper_id, visited, direction)
                    if related_papers:
                        next_papers.extend(related_papers)

                        if not self.config.ModuleInfo.WorkCollector.RAG_source_use_embedding_filter and \
                            not self.config.ModuleInfo.WorkCollector.RAG_source_use_LLM_filter:
                            self.graph_paper_ids.update(related_papers)
                current_papers = next_papers

        traverse(direction="out")
        traverse(direction="in")

        with open(os.path.join(self.cache_path, "reference_graph.pkl"), "wb") as writer:
            pickle.dump(self.reference_graph, writer)

        self.logger.info(
            f"Reference graph updated. Nodes: {self.reference_graph.number_of_nodes()}, Edges: {self.reference_graph.number_of_edges()}"
        )

    def compute_relatedness_scores_and_filter(
        self,
        seed_paper_ids: List[str],
    ) -> float:
        """Compute relatedness score between related papers and seed papers."""
        
        total = 0
        for seed_pid in seed_paper_ids:
            references = list(self.reference_graph.successors(seed_pid))
            citations = list(self.reference_graph.predecessors(seed_pid))
            total += len(references) + len(citations)

        self.logger.info(
            f"Total {total} related papers to compute relatedness scores for and filter."
        )

        def paper2text(paper_id: str) -> str:
            node_data = self.reference_graph.nodes[paper_id]
            title = node_data.get("title", "")
            abstract = node_data.get("abstract", "")
            return f"Title: {title}\nAbstract: {abstract}"

        model = self._get_embedding_model()

        seed_texts = [paper2text(pid) for pid in seed_paper_ids]
        seed_embeddings = model.encode(
            seed_texts,
            convert_to_tensor=True,
            batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
            show_progress_bar=True,
        )
        saved_papers = dict()
        for seed_pid in seed_paper_ids:
            saved_papers[seed_pid] = set()
            references = list(self.reference_graph.successors(seed_pid))
            citations = list(self.reference_graph.predecessors(seed_pid))
            related_pids = references + citations
            related_pids = [paper_id for paper_id in related_pids if paper_id not in self.ignore_paper]
            if len(related_pids) == 0:
                self.logger.warning(f"No related papers found for seed paper {seed_pid}, skipping relatedness computation.")
                continue
            related_texts = [paper2text(pid) for pid in related_pids]
            related_embeddings = model.encode(
                related_texts,
                convert_to_tensor=True,
                batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
                show_progress_bar=True,
            )
            cosine_scores = util.pytorch_cos_sim(
                seed_embeddings[seed_paper_ids.index(seed_pid)], related_embeddings
            )

            top_k = min(
                self.config.ModuleInfo.WorkCollector.related_work_top_k,
                len(related_pids),
            )
            top_results = torch.topk(cosine_scores, k=top_k)
            for score, idx in zip(top_results[0][0], top_results[1][0]):
                pid = related_pids[idx]
                sim_score = score.item()
                if (
                    sim_score
                    >= self.config.ModuleInfo.WorkCollector.related_work_threshold
                ):
                    saved_papers[seed_pid].add(pid)
                    if self.config.ModuleInfo.WorkCollector.RAG_source_use_embedding_filter and \
                        not self.config.ModuleInfo.WorkCollector.RAG_source_use_LLM_filter:
                        self.graph_paper_ids.add(pid)

            saved_papers[seed_pid].difference_update(seed_paper_ids)

        self.logger.info(
            f"Selected {len(set().union(*saved_papers.values()))} papers based on sentence-transformer relatedness scores with threshold {self.config.ModuleInfo.WorkCollector.related_work_threshold}, topk {self.config.ModuleInfo.WorkCollector.related_work_top_k}."
        )

        tasks = []
        for seed_pid in seed_paper_ids:
            related_pids = list(saved_papers[seed_pid])
            if not related_pids:
                continue

            seed_title = self.reference_graph.nodes[seed_pid].get("title", "")
            seed_abstract = self.reference_graph.nodes[seed_pid].get("abstract", "")

            for related_pid in related_pids:
                hash_key = get_hash(f"{seed_pid}||{related_pid}")
                if hash_key in self.relatedness_cache and "relevance_score" in self.relatedness_cache.get(hash_key, {}).keys():
                    self.logger.info(
                        f"Relatedness cache hit for seed {seed_pid} and related {related_pid}."
                    )
                else:
                    related_title = self.reference_graph.nodes[related_pid].get("title", "")
                    related_abstract = self.reference_graph.nodes[related_pid].get("abstract", "")

                    prompt = PAPER_RELATEDNESS_BASED_ON_TITLE_AND_ABSTRACT.format(
                        seed_title=seed_title,
                        seed_abstract=seed_abstract,
                        candidate_title=related_title,
                        candidate_abstract=related_abstract,
                    )
                    tasks.append((hash_key, seed_pid, related_pid, prompt))
        
        if tasks:
            prompts = [task[3] for task in tasks]
            responses = self.chat_agent.batch_remote_chat(
                prompts,
                temperature=self.config.ModuleInfo.WorkCollector.relatedness_temperature,
                desc="Computing relatedness scores",
            )
            for i, response in enumerate(responses):
                hash_key, seed_pid, related_pid, _ = tasks[i]
                try:
                    response = extract_json(response)
                    response["seed_paper_id"] = seed_pid
                    response["related_paper_id"] = related_pid
                    self.relatedness_cache[hash_key] = response
                except Exception as e:
                    self.logger.error(
                        f"Error processing relatedness response for seed {seed_pid} and related {related_pid}: {e}"
                    )
                    self.relatedness_cache[hash_key] = {
                        "relevance_score": 0.0,
                        "seed_paper_id": seed_pid,
                        "related_paper_id": related_pid,
                    }

        for seed_pid in seed_paper_ids:
            to_remove = set()
            for related_pid in saved_papers[seed_pid]:
                hash_key = get_hash(f"{seed_pid}||{related_pid}")
                relatedness_info = self.relatedness_cache.get(hash_key, {})
                relatedness_score = relatedness_info.get("relevance_score", 0.0)
                if (
                    relatedness_score
                    < self.config.ModuleInfo.WorkCollector.related_work_threshold_for_llm
                ):
                    to_remove.add(related_pid)
            saved_papers[seed_pid].difference_update(to_remove)

        if self.config.ModuleInfo.WorkCollector.RAG_source_use_LLM_filter and \
            not self.config.ModuleInfo.WorkCollector.RAG_source_downloadable_only:
            self.graph_paper_ids.update(set().union(*saved_papers.values()))

        self.logger.info(
            f"After LLM-based filtering, {len(set().union(*saved_papers.values()))} papers remain with threshold {self.config.ModuleInfo.WorkCollector.related_work_threshold_for_llm}."
        )
        return list(set().union(*saved_papers.values()))

    def expand_seed_papers_by_reference_and_citation(self, seed_paper_ids: List[str]):
        """Collect related papers based on the given seed paper IDs."""
        if self.expand_in_local_paper_graph and self.paper_graph_retriever:
            self.logger.info("Expanding seed papers by local paper graph...")
            return self.expand_and_filter_in_local_paper_graph(seed_paper_ids)
        
        self.update_reference_graph(seed_paper_ids)
        related_paper_ids = self.compute_relatedness_scores_and_filter(seed_paper_ids)
        
        for i in range(
            min(
                self.config.ModuleInfo.WorkCollector.log_related_work_num,
                len(related_paper_ids),
            )
            if self.config.ModuleInfo.WorkCollector.log_related_work_num > 0
            else len(related_paper_ids)
        ):
            pid = related_paper_ids[i]
            self.logger.info(
                f"Related Paper ID: {pid}, Title: {self.reference_graph.nodes[pid].get('title', 'N/A')}"
            )

        valid_expanded_paper_ids = self.data_manager.download_and_parse_papers(related_paper_ids)
        self.graph_paper_ids.update(valid_expanded_paper_ids)
        self.logger.info(f"valid RAG paper ids sources num: {len(self.graph_paper_ids)}")
        return valid_expanded_paper_ids

    def expand_and_filter_in_local_paper_graph(self, seed_paper_ids: List[str]):
        expanded_papers = self.expand_papers_by_local_paper_graph(seed_paper_ids)
        filtered_papers = self.filter_papers_local_paper_graph(expanded_papers, seed_paper_ids)
        return filtered_papers

    def expand_papers_by_local_paper_graph(self, seed_paper_ids: List[str]):
        if not self.expand_in_local_paper_graph or not self.paper_graph_retriever:
            self.logger.error("Error: expand_in_local_paper_graph False or paper_graph_retriever not initialized")
            raise ValueError("expand_in_local_paper_graph False or paper_graph_retriever not initialized")
        
        seed_paper_paper_graph_ids = []
        for seed_paper in seed_paper_ids:
            try:
                title, abstract = self.data_manager.get_paper_title_abstract(seed_paper)
            except Exception as e:
                self.logger.error(f"[Strange err: previous getting success] Error getting title and abstract for seed paper {seed_paper}: {e}. Skipping this seed paper for local graph expansion.")
                continue
            results = self.paper_graph_retriever.search_by_paper_title(title)
            if not results:
                self.logger.error(f"error out of expectation: fail to get seed paper {title} in paper graph (previous can)")
                raise ValueError(f"fail to get seed paper {title} in paper graph (previous can)")
            seed_paper_paper_graph_ids.append(results[0]["id"])
        
        expanded_papers = self.paper_graph_retriever.expand_nodes(seed_paper_paper_graph_ids, self.config.ModuleInfo.WorkCollector.reference_graph_depth)
        self.logger.info(f"expansion finished")
        return list(set(expanded_papers) - set(seed_paper_paper_graph_ids))

    def filter_papers_local_paper_graph(self, expanded_papers: List[str], seed_paper_ids: List[str]):
        self.logger.info(f"paper number: {len(expanded_papers)} before filter")
        expanded_papers_info = []
        for paper in expanded_papers:
            self.logger.info(f"retrieving paper_id_in_graph {paper} in graph...")
            paper_info = self.paper_graph_retriever.search_by_node_id(paper)
            if not paper_info:
                self.logger.info(f"fail to retrieve paper_id_in_graph {paper} in graph: return info None...")
                self.logger.info(f"return: {paper_info}")
                continue
            expanded_papers_info.append(paper_info[0])

        # Collect all titles that need to be queried
        valid_titles = []
        valid_paper_info_map = {}
        for paper_info in expanded_papers_info:
            paper_title = paper_info["paper_title"]
            if not paper_title:
                self.logger.warning(f"paper_title is empty")
                self.logger.warning(f"complete paper info: {paper_info}")
                continue
            valid_titles.append(paper_title)
            valid_paper_info_map[paper_title] = paper_info

        # Batch query papers by title
        self.logger.info(f"Batch querying {len(valid_titles)} papers by title...")
        batch_results = self.get_paper_with_title_batch(valid_titles)
        
        # Process batch results
        # get_paper_with_title_batch returns Dict[str, dict]: title -> paper_info
        expanded_papers_ids = []
        for paper_title in valid_titles:
            api_paper_info = batch_results.get(paper_title)
            if api_paper_info is None or not api_paper_info:
                self.logger.warning(f"{paper_title} cannot be retrieved from arxiv or semantic scholar")
                continue

            if api_paper_info.get("api_platform", "").lower() == "arxiv":
                paper_id = api_paper_info.get("paper_id", "")
            else:
                paper_id = api_paper_info.get("externalIds", {}).get(
                    "ArXiv", api_paper_info.get("paperId")
                )

            if paper_id is None or not paper_id:
                self.logger.warning(f"{paper_title} cannot be retrieved from arxiv or semantic scholar")
                continue
            expanded_papers_ids.append(paper_id)

        self.graph_paper_ids.update(expanded_papers_ids)
        self.logger.info(f"expansion in graph find {len(expanded_papers_ids)} valid papers (can be found in arxiv/semantic scholar)")
        
        if self.advanced_filter_in_local_paper_graph_expansion:
            self.logger.info(f"seed papers: {seed_paper_ids}")
            self.logger.info(f"papers before filter: {expanded_papers_ids}")
            
            def paper2text(paper_id: str) -> str:
                try:
                    title, abstract = self.data_manager.get_paper_title_abstract(paper_id)
                except Exception as e:
                    self.logger.error(f"Error getting title and abstract for paper {paper_id}: {e}. Using empty title and abstract for relatedness computation.")
                    title, abstract = "", ""
                return f"Title: {title}\nAbstract: {abstract}"

            model = self._get_embedding_model()

            seed_texts = [paper2text(pid) for pid in seed_paper_ids]
            seed_embeddings = model.encode(
                seed_texts,
                convert_to_tensor=True,
                batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
                show_progress_bar=True,
            )
            saved_papers = dict()
            for seed_pid in seed_paper_ids:
                saved_papers[seed_pid] = set()

                related_pids = [paper_id for paper_id in expanded_papers_ids if paper_id not in self.ignore_paper]
                if len(related_pids) == 0:
                    self.logger.warning(f"No related papers found for seed paper {seed_pid}, skipping relatedness computation.")
                    continue
                related_texts = [paper2text(pid) for pid in related_pids]
                related_embeddings = model.encode(
                    related_texts,
                    convert_to_tensor=True,
                    batch_size=self.config.ModuleInfo.WorkCollector.sentence_transformer_batch_size,
                    show_progress_bar=True,
                )
                cosine_scores = util.pytorch_cos_sim(
                    seed_embeddings[seed_paper_ids.index(seed_pid)], related_embeddings
                )

                top_k = min(
                    self.config.ModuleInfo.WorkCollector.related_work_top_k,
                    len(related_pids),
                )
                top_results = torch.topk(cosine_scores, k=top_k)
                for score, idx in zip(top_results[0][0], top_results[1][0]):
                    pid = related_pids[idx]
                    sim_score = score.item()
                    if (
                        sim_score
                        >= self.config.ModuleInfo.WorkCollector.related_work_threshold
                    ):
                        self.logger.info(f"{pid}: {seed_pid} embedding sim {sim_score} > {self.config.ModuleInfo.WorkCollector.related_work_threshold}")
                        saved_papers[seed_pid].add(pid)
                        if self.config.ModuleInfo.WorkCollector.RAG_source_use_embedding_filter and \
                            not self.config.ModuleInfo.WorkCollector.RAG_source_use_LLM_filter:
                            self.graph_paper_ids.add(pid)
                    else:
                        self.logger.info(f"{pid}: {seed_pid} embedding sim {sim_score} < {self.config.ModuleInfo.WorkCollector.related_work_threshold}")

                saved_papers[seed_pid].difference_update(seed_paper_ids)

            self.logger.info(
                f"Selected {len(set().union(*saved_papers.values()))} papers based on sentence-transformer relatedness scores with threshold {self.config.ModuleInfo.WorkCollector.related_work_threshold}, topk {self.config.ModuleInfo.WorkCollector.related_work_top_k}."
            )

            tasks = []
            for seed_pid in seed_paper_ids:
                related_pids = list(saved_papers[seed_pid])
                if not related_pids:
                    continue
                try:
                    seed_title, seed_abstract = self.data_manager.get_paper_title_abstract(seed_pid)
                except Exception as e:
                    self.logger.error(f"Error getting title and abstract for seed paper {seed_pid}: {e}. Skipping LLM-based relatedness computation for this seed.")
                    continue

                for related_pid in related_pids:
                    hash_key = get_hash(f"{seed_pid}||{related_pid}")
                    if hash_key in self.relatedness_cache and "relevance_score" in self.relatedness_cache.get(hash_key, {}).keys():
                        self.logger.info(
                            f"Relatedness cache hit for seed {seed_pid} and related {related_pid}."
                        )
                    else:
                        try:
                            related_title, related_abstract = self.data_manager.get_paper_title_abstract(related_pid)
                        except Exception as e:
                            self.logger.error(f"Error getting title and abstract for related paper {related_pid}: {e}. Skipping LLM-based relatedness computation for this pair.")
                            continue

                        prompt = PAPER_RELATEDNESS_BASED_ON_TITLE_AND_ABSTRACT.format(
                            seed_title=seed_title,
                            seed_abstract=seed_abstract,
                            candidate_title=related_title,
                            candidate_abstract=related_abstract,
                        )
                        tasks.append((hash_key, seed_pid, related_pid, prompt))
            
            if tasks:
                prompts = [task[3] for task in tasks]
                responses = self.chat_agent.batch_remote_chat(
                    prompts,
                    temperature=self.config.ModuleInfo.WorkCollector.relatedness_temperature,
                    desc="Computing relatedness scores",
                )
                for i, response in enumerate(responses):
                    hash_key, seed_pid, related_pid, _ = tasks[i]
                    try:
                        response = extract_json(response)
                        response["seed_paper_id"] = seed_pid
                        response["related_paper_id"] = related_pid
                        self.relatedness_cache[hash_key] = response
                    except Exception as e:
                        self.logger.error(
                            f"Error processing relatedness response for seed {seed_pid} and related {related_pid}: {e}"
                        )
                        self.relatedness_cache[hash_key] = {
                            "relevance_score": 0.0,
                            "seed_paper_id": seed_pid,
                            "related_paper_id": related_pid,
                        }

            for seed_pid in seed_paper_ids:
                to_remove = set()
                for related_pid in saved_papers[seed_pid]:
                    hash_key = get_hash(f"{seed_pid}||{related_pid}")
                    relatedness_info = self.relatedness_cache.get(hash_key, {})
                    relatedness_score = relatedness_info.get("relevance_score", 0.0)
                    if (
                        relatedness_score
                        < self.config.ModuleInfo.WorkCollector.related_work_threshold_for_llm
                    ):
                        to_remove.add(related_pid)
                saved_papers[seed_pid].difference_update(to_remove)

            if self.config.ModuleInfo.WorkCollector.RAG_source_use_LLM_filter and \
                not self.config.ModuleInfo.WorkCollector.RAG_source_downloadable_only:
                self.graph_paper_ids.update(set().union(*saved_papers.values()))

            self.logger.info(
                f"After LLM-based filtering, {len(set().union(*saved_papers.values()))} papers remain with threshold {self.config.ModuleInfo.WorkCollector.related_work_threshold_for_llm}."
            )
            expanded_papers_ids = list(set().union(*saved_papers.values()))
            self.logger.info(f"expanded papers after filter: {expanded_papers_ids}")

        return expanded_papers_ids


@hydra.main(config_path="../config", config_name="deep_survey_batch_xiaomi_fast", version_base=None)
def main(config):
    config = merge_with_default_survey_config(config)
    title = "Learning to Refine Source Representations for Neural Machine Translation"
    work_collector = WorkCollector(config)
    print(work_collector.get_paper_with_title(title))

if __name__ == "__main__":
    main()
