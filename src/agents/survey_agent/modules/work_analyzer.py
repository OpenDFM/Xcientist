from typing import List
import os
import pickle
from typing import Dict
import networkx as nx
from utils.api_call import ChatAgent, SemanticScholarAPI,ArxivAPI
from utils.utils import get_hash, extract_json
import diskcache as dc
import time
from modules.pe import (
    PAPER_DEEP_READING,
    PAPER_CLUSTERING,
    PROPOSE_QUESTIONS_FOR_CLUSTER,
    ANSWER_QUESTION_FOR_PAPERS,
    INTER_CLUSTER_ANALYSIS,
    PAPER_CLUSTERING_CREATING,
    PAPER_CLUSTERING_ASSIGNING,
    ERROR_FEEDBACK_PROMPT,
    PAPER_RELATIONSHIP_ANALYSIS,
    CLUSTER_TABLE_GENERATION
)
from modules.paper_graph_retriever import PaperGraphRetriever
import hdbscan
from sentence_transformers import SentenceTransformer
from utils.rich_logger import get_logger
import math
import requests
import xml.etree.ElementTree as ET
import copy

class WorkAnalyzer:
    def __init__(self, config, work_collector, paper_graph_retriever = None):
        self.config = config
        self.chat_agent = ChatAgent(config)
        self.semantic_scholar_api = SemanticScholarAPI(config)
        self.arxiv_api = ArxivAPI(config)
        self.logger = get_logger("WorkAnalyzer")
        self.work_collector = work_collector
        self.relation_analysis_graph = None
        self.relation_analysis_table = None

        self.cluster_fast_mode = self.config.ModuleInfo.WorkAnalyzer.cluster_assign_fast_mode
        self.use_graph_keynotes = self.config.ModuleInfo.WorkAnalyzer.use_local_paper_graph_keynotes
        self.use_ds_keynotes_when_graph_fail = self.config.ModuleInfo.WorkAnalyzer.use_ds_keynotes_when_graph_fail
        if self.use_graph_keynotes:
            if not paper_graph_retriever:
                self.paper_graph_retriever = PaperGraphRetriever(config)
            else:
                self.paper_graph_retriever = paper_graph_retriever

        # load reference graph
        self.cache_path = self.config.BasicInfo.cache_path
        os.makedirs(self.cache_path, exist_ok=True)
        if os.path.exists(os.path.join(self.cache_path, "reference_graph.pkl")):
            with open(
                os.path.join(self.cache_path, "reference_graph.pkl"), "rb"
            ) as reader:
                self.reference_graph = pickle.load(reader)
        else:
            self.reference_graph = None

        # cache for paper keynotes
        self.paper_keynote_cache = dc.Cache(
            os.path.join(self.cache_path, "paper_keynotes")
        )

        # caches for clustering and relationship graph/table
        self.cluster_cache = dc.Cache(
            os.path.join(self.cache_path, "cluster_cache")
        )
        self.relation_graph_cache = dc.Cache(
            os.path.join(self.cache_path, "relation_graph_cache")
        )

        self.paper_abstract_cache = self.work_collector.paper_abstract_cache

    def read_papers_and_write_keynotes(self, papers: List[str], retry: int = 1, ds_keynotes_fallback: bool = False):

        if self.config.ModuleInfo.WorkAnalyzer.abstract_only_mode:
            self.logger.info("Abstract-only mode enabled, fetching abstracts instead of deep reading and get keynotes.")
            return self.work_collector.add_papers_abstracts_in_cache(papers, retry=retry)

        if retry > self.config.ModuleInfo.WorkAnalyzer.paper_reading_max_retry:
            self.logger.error("Exceeded maximum retries for reading papers.")
            self.logger.error(f"Papers failed to read: {papers}")
            return papers

        if self.config.ModuleInfo.WorkAnalyzer.use_local_paper_graph_keynotes and not ds_keynotes_fallback:
            # extract information for baselines and empty node and write back to graph
            self.logger.info(f"getting keynotes in graph...")
            error_ids =  self.paper_graph_retriever.read_papers_and_write_keynotes(papers)
            if self.use_ds_keynotes_when_graph_fail and error_ids:
                self.logger.info(f"some paper not in graph, number: {len(error_ids)}, use previous methods")
                error_ids =  self.read_papers_and_write_keynotes(
                    papers = error_ids, 
                    retry = retry, 
                    ds_keynotes_fallback = True
                )
            return error_ids
        
        tasks = []
        err_papers = []
        for pid in papers:
            try:
                hash_id = get_hash(pid)
                if hash_id in self.paper_keynote_cache:
                    continue
                try:
                    paper_markdown_text = self.work_collector.get_paper_raw_markdown(pid)

                except Exception as e:
                    self.logger.error(f"Failed to get content for paper ID: {pid}: {e} in PAPER COMPREHENDING. Skipping this paper or use abstract based on config.")
                    if self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                        err = self.work_collector.add_papers_abstracts_in_cache([pid])
                        if hash_id in self.paper_abstract_cache and not err:
                            self.logger.info(f"Using abstract for paper ID: {pid} as keynote due to full text failure. Abstract length:{len(self.paper_abstract_cache[hash_id]['abstract'])}.")
                            continue
                    err_papers.append(pid)
                    continue

                max_ctx = self.config.APIInfo.llm_max_context_length
                overhead = self.config.ModuleInfo.WorkAnalyzer.llm_max_context_overhead_length_in_paper_reading
                allowed = max_ctx - overhead

                paper_markdown_text = self.chat_agent.truncate_text(pid, paper_markdown_text, allowed)

                prompt = PAPER_DEEP_READING.format(
                    paper_markdown_text=paper_markdown_text
                )

                tasks.append(
                    [
                        pid,
                        hash_id,
                        prompt,
                    ]
                )
            except Exception as e:
                self.logger.error(f"Error preparing paper {pid} for reading: {e} in PAPER COMPREHENDING")
                err_papers.append(pid)

        prompts = [task[2] for task in tasks]
        if not prompts and not err_papers:
            return []  # all papers are already processed

        try:
            responses = self.chat_agent.batch_remote_chat(
                prompts,
                temperature=self.config.ModuleInfo.WorkAnalyzer.paper_reading_temperature,
                desc="Reading papers",
            )
            if not responses:
                self.logger.error("No responses received from LLM during paper reading.")
                raise ValueError("No responses received from LLM during paper reading.")
        except Exception as e:
            self.logger.error(f'Error: {e} , during comprehending papers in getting response from LLM, retrying all...')
            return self.read_papers_and_write_keynotes(papers, retry=retry + 1)

        for i, response in enumerate(responses):
            try:
                pid, hash_id, _ = tasks[i]
                if not response:
                    self.logger.warning(f"Response is empty or None for paper {pid}. Marking as error.")
                    err_papers.append(pid)
                    continue
                if len(response.strip()) < 10:
                    self.logger.warning(f"Extracted keynote too short for paper {pid}. Marking as error.")
                    err_papers.append(pid)
                    continue
                keynote = extract_json(response)
                self.paper_keynote_cache[hash_id] = {
                    "paper_id": pid,
                    "keynote": keynote,
                }
                if self.config.BasicInfo.debug:
                    self.logger.info(f"paper ID {pid} keynote: {keynote}")
            except Exception as e:
                self.logger.warning(f"Error processing LLM's response for paper {tasks[i][0]}: {e}. First 500 char of response: {response[:500]}")
                err_papers.append(tasks[i][0])

        if err_papers:
            self.logger.info(f"Retrying {len(err_papers)} papers due to previous errors in keynotes generation...")
            return self.read_papers_and_write_keynotes(err_papers, retry=retry + 1)
        
        return []

    def generate_mla(self, paper_id: str):
        if (
            paper_id in self.reference_graph
            and "authors" in self.reference_graph[paper_id]
            and "year" in self.reference_graph[paper_id]
            and "venue" in self.reference_graph[paper_id]
            and "title" in self.reference_graph[paper_id]
        ):
            authors = self.reference_graph[paper_id].get("authors", [])
            title = self.reference_graph[paper_id].get("title", "")
            venue = self.reference_graph[paper_id].get("venue", "")
            year = self.reference_graph[paper_id].get("year", "")
        else:
            paper = None
            if "." in paper_id:  # arXiv IDs typically contain dots, e.g., "1706.03762"
                # Try arXiv API first for arXiv papers
                try:
                    query_id = "ARXIV:" + paper_id
                    paper = self.semantic_scholar_api.get_paper_details(
                        query_id, fields="title,year,venue,authors"
                    )
                except Exception as e:
                    self.logger.warning(f"semantic API failed for {paper_id}: {e}. Trying arXiv API.")
                    try:
                        paper = self.arxiv_api.get_paper_details(paper_id)
                    except Exception as e:
                        self.logger.error(f"arXiv API also failed for arXiv paper {paper_id}: {e}")
                        paper = None
            else:
                # For non-arXiv papers, try Semantic Scholar directly
                try:
                    paper = self.semantic_scholar_api.get_paper_details(
                        paper_id, fields="title,year,venue,authors"
                    )
                except Exception as e:
                    self.logger.error(f"Semantic Scholar failed for paper {paper_id}: {e}")
                    paper = None

            if not paper:
                self.logger.warning(f"Warning: Unable to fetch details for paper ID {paper_id}")
                raise ValueError("Unable to fetch paper details in mla generation")
            
            authors = paper.get("authors", [])
            title = paper.get("title", "")
            venue = paper.get("venue", "")
            year = paper.get("year", "")
            if authors and isinstance(authors[0], dict):
                authors = [a["name"] for a in authors]

        if len(authors) == 0:
            author_str = ""
        elif len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        else:
            author_str = ", ".join(authors[:-1]) + ", and " + authors[-1]

        citation = f'{author_str}. "{title}." *{venue}*, {year}'
        citation += "."
        return citation

    def get_paper_keynote(self, paper_id: str, ds_keynote_fallback: bool = False):
        """Get the keynote of a paper by its ID."""
        hash_id = get_hash(paper_id)

        if self.use_graph_keynotes and not ds_keynote_fallback:
            results, errors = self.paper_graph_retriever.get_paper_keynote([paper_id])
            if errors or None in results:
                if self.use_ds_keynotes_when_graph_fail:
                    return self.get_paper_keynote(paper_id, True)
                else:
                    raise ValueError(f"Failed to get keynote for paper ID {paper_id} in paper graph")
            return results

        if self.config.ModuleInfo.WorkAnalyzer.abstract_only_mode:
            try:
                title, abstract =  self.work_collector.get_paper_title_abstract(paper_id)
            except Exception as e:
                raise ValueError(f"Failed to get abstract for paper ID {paper_id}: {e}")
            return abstract

        elif not self.config.ModuleInfo.WorkAnalyzer.abstract_only_mode or ds_keynote_fallback:
            err = []
            if hash_id not in self.paper_keynote_cache or not self.config.ModuleInfo.WorkAnalyzer.cache_enabled:
                err = self.read_papers_and_write_keynotes([paper_id])

            if hash_id not in self.paper_keynote_cache or len(err) > 0:
                if self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                    self.logger.info(f"Using abstract for paper ID: {paper_id} as keynote due to full text failure.")
                    try:
                        title, abstract =  self.work_collector.get_paper_title_abstract(paper_id)
                    except Exception as e:
                        raise ValueError(f"Failed to get abstract for paper ID {paper_id}: {e}")
                    return abstract

            keynote_data = self.paper_keynote_cache[hash_id]["keynote"]
            return keynote_data

    def cluster_papers(self, papers: List[str]) -> List[Dict]:
        cached = self._load_cached_clusters(papers)
        if cached is not None:
            self.logger.info("Using cached clustering result.")
            return cached

        if self.config.ModuleInfo.WorkAnalyzer.clustering_in_steps:
            clusters = self.cluster_papers_in_steps(papers)
        else:
            clusters = self.cluster_papers_1_step(papers)

        self._store_cached_clusters(papers, clusters)
        return clusters

    def cluster_papers_1_step(self, papers: List[str]) -> List[Dict]:
        clusters = []
        num_batches = math.ceil(
            len(papers) / self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size
        )

        i = 0
        if self.config.BasicInfo.debug:
            self.logger.info(f'complete paper list for clustering: {papers}')
        while i < num_batches:
            valid = False
            for retry_time in range(
                self.config.ModuleInfo.WorkAnalyzer.paper_clustering_max_retry
            ):
                try:
                    self.logger.info(f"Clustering batch {i+1}/{num_batches}...")
                    batch = papers[
                        i
                        * self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size : (
                            i + 1
                        )
                        * self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size
                    ]
                    paper_keynotes = ""
                    for pid in batch:
                        keynote_json = self.get_paper_keynote(pid)
                        paper_keynotes += (
                            f"Paper ID: {pid}\nKeynote: {keynote_json}\n\n"
                        )

                    # self.logger.info(f"Clustering {len(batch)} papers...")
                    # self.logger.info(f'Clustering prompt: {PAPER_CLUSTERING.format(existing_clusters_json=clusters,new_batch_json=paper_keynotes,)}')
                    new_clusters = extract_json(
                        self.chat_agent.remote_chat(
                            PAPER_CLUSTERING.format(
                                existing_clusters_json=clusters,
                                new_batch_json=paper_keynotes,
                            ),
                            temperature=self.config.ModuleInfo.WorkAnalyzer.clustering_temperature,
                        )
                    )
                    self.validate_clusters(new_clusters, papers) #YZY MODIFY: from clusters
                    valid = True
                    break
                except Exception as e:
                    self.logger.warning(f"Error during clustering batch {i+1}: {e}. Retrying for {retry_time + 1}...")
            if not valid:
                raise ValueError("Clustering failed after maximum retries.")
            clusters = new_clusters
            if self.config.BasicInfo.debug:
                self.logger.info(f"CLUSTER after batch {i+1}: {new_clusters}")
            i += 1
        if self.config.BasicInfo.debug:
            self.logger.info(f"Final CLUSTER result: {clusters}")
        return clusters

    def cluster_papers_in_steps(self, papers: List[str]) -> List[Dict]:
        clusters = self._create_clusters(papers)
        clusters = self._assign_papers_to_clusters(papers, clusters)
        return clusters

    def _create_clusters(self, papers: List[str]) -> List[Dict]:
        clusters = []
        num_batches = math.ceil(
            len(papers) / self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_creation
        )

        # step 1: create clusters
        i = 0
        while i < num_batches:
            valid = False
            for retry_time in range(
                self.config.ModuleInfo.WorkAnalyzer.paper_clustering_creation_max_retry
            ):
                try:
                    self.logger.info(f"Clustering batch {i+1}/{num_batches}...")
                    batch = papers[
                        i
                        * self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_creation : (
                            i + 1
                        )
                        * self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_creation
                    ]
                    if self.config.BasicInfo.debug:
                        self.logger.info(f'complete paper list for clustering: {papers}')
                    paper_keynotes = ""
                    for pid in batch:
                        keynote_json = self.get_paper_keynote(pid)
                        paper_keynotes += (
                            f"Paper ID: {pid}\nKeynote: {keynote_json}\n\n"
                        )

                    new_clusters = extract_json(
                        self.chat_agent.remote_chat(
                            PAPER_CLUSTERING_CREATING.format(
                                existing_clusters_json=clusters,
                                new_batch_json=paper_keynotes,
                            ),
                            temperature=self.config.ModuleInfo.WorkAnalyzer.clustering_temperature,
                        )
                    )
                    valid = True
                    break
                except Exception as e:
                    self.logger.warning(f"Error during clustering batch creation {i+1}: {e}. Retrying for {retry_time + 1}...")
            if not valid:
                raise ValueError("Clustering failed after maximum retries.")
            clusters = new_clusters
            if self.config.BasicInfo.debug:
                self.logger.info(f"CLUSTER after batch {i+1}: {new_clusters}")
            i += 1
        if self.config.BasicInfo.debug:
            self.logger.info(f"Final CLUSTER result: {clusters}")
        return clusters

    def _validate_assignment(self, result, info_dict):
        # self.logger.info("outer validating....")
        error_conservatism_mode = info_dict.get("error_conservatism_mode", False)
        max_retry = info_dict.get("max_retry", 5)
        batches = info_dict.get("batches", [])
        papers = info_dict.get("papers", [])
        # Fixed: was "cluster" in info_dict but "clusters" in access - now using "cluster" which is what we pass
        clusters = info_dict.get("cluster", {})
        idx = info_dict.get("idx")
        retry_time = info_dict.get("retry_time", 0)


        omit_error = not error_conservatism_mode and retry_time >= max_retry - 1
        batch = batches[idx] if idx < len(batches) else []
        try:
            new_papers_dict = extract_json(result)
            if self._validate_new_papers_to_clusters(clusters, new_papers_dict, papers, batch, omit_error):
                return True, result
        except Exception as e:
            if omit_error:
                self.logger.warning(f"error in cluster assign: {e}, omit")
                # Fixed: return (val, result) format for retry
                return True, result
            else:
                self.logger.error(f"error in cluster assign: {e}")
                return False, result
        return True, result

    def _assign_papers_to_clusters(self, papers: List[str], clusters: List[dict]) -> List[Dict]:
        # step2: assign papers to clusters
        num_batches = math.ceil(
            len(papers) / self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_assignment
        )

        cluster_name_dict = {cluster["cluster_name"]: cluster for cluster in clusters}
        for cluster in cluster_name_dict.values():
            cluster["papers"] = []
        
        new_clusters = copy.deepcopy(cluster_name_dict)

        prompts = []
        batches = []
        # step 2: assign papers to clusters
        i = 0
        while i < num_batches:
            valid = False
            err_info = ""
            for retry_time in range(
                self.config.ModuleInfo.WorkAnalyzer.paper_clustering_assignment_max_retry
            ):
                try:
                    self.logger.info(f"Clustering batch {i+1}/{num_batches}...")
                    batch = papers[
                        i
                        * self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_assignment : (
                            i + 1
                        )
                        * self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_assignment
                    ]
                    if self.config.BasicInfo.debug:
                        self.logger.info(f'complete paper list for clustering: {papers}')
                    paper_keynotes = ""
                    for pid in batch:
                        keynote_json = self.get_paper_keynote(pid)
                        paper_keynotes += (
                            f"Paper ID: {pid}\nKeynote: {keynote_json}\n\n"
                        )

                    assign_prompt = PAPER_CLUSTERING_ASSIGNING.format(
                            clusters_json=cluster_name_dict,
                            batch_json=paper_keynotes,
                        ) + (ERROR_FEEDBACK_PROMPT.format(info = err_info) if err_info else "")


                    ### for fast mode, use batch chat out of he loop
                    if self.cluster_fast_mode:
                        self.logger.info(f"[CLUSTER FAST MODE DEBUG] use fast cluster")
                        valid = True
                        batches.append(batch)
                        prompts.append(assign_prompt)
                        break

                    new_papers_dict = extract_json(
                        self.chat_agent.remote_chat(
                            assign_prompt,
                            temperature=self.config.ModuleInfo.WorkAnalyzer.clustering_temperature,
                        )
                    )
                    if not self.config.BasicInfo.error_conservatism_mode and retry_time == self.config.ModuleInfo.WorkAnalyzer.paper_clustering_assignment_max_retry - 1:
                        self.logger.warning(f"max retry reached:{retry_time+1}/{self.config.ModuleInfo.WorkAnalyzer.paper_clustering_assignment_max_retry}, assigning papers directly to clusters and returning in PAPER CLUSTER PAPER ASSIGNING.")
                        new_clusters = self._validate_and_assign_new_papers_to_clusters(new_clusters, new_papers_dict, papers, batch, True)
                        
                    else:
                        new_clusters = self._validate_and_assign_new_papers_to_clusters(new_clusters, new_papers_dict, papers, batch, False)
                    valid = True
                    break
                except Exception as e:
                    self.logger.warning(f"Error during clustering batch assignment {i+1}: {e}. Retrying for {retry_time + 1}...")
                    err_info += f"Error during clustering batch {i+1}: {e}. \n"
            if not valid:
                raise ValueError("Clustering failed after maximum retries.")
            if self.config.BasicInfo.debug and not self.cluster_fast_mode:
                self.logger.info(f"CLUSTER after batch {i+1}: {new_clusters}")
            i += 1

        if self.cluster_fast_mode:
            info_dict = {
                "error_conservatism_mode": self.config.BasicInfo.error_conservatism_mode,
                "max_retry": self.config.ModuleInfo.WorkAnalyzer.paper_clustering_assignment_max_retry,
                "batches": batches,
                "papers": papers,
                "cluster": cluster_name_dict
            }

            results = self.chat_agent.batch_remote_chat_with_retry(prompts = prompts,
                                                                    validate_fn=self._validate_assignment,
                                                                    max_retry = self.config.ModuleInfo.WorkAnalyzer.paper_clustering_assignment_max_retry,
                                                                    desc = "assigning resultd in fast mode",
                                                                    temperature = self.config.ModuleInfo.WorkAnalyzer.clustering_temperature,
                                                                    info_dict = info_dict,
                                                                )
            
            for idx, result in enumerate(results):
                # self.logger.info(f"assigning result {idx}...")
                batch = batches[idx]
                new_papers_dict = extract_json(result)
                new_clusters = self._validate_and_assign_new_papers_to_clusters(new_clusters, new_papers_dict, papers, batch, True)

        if self.config.BasicInfo.debug:
            self.logger.info(f"Final CLUSTER result: {new_clusters}")

        new_clusters = list(new_clusters.values())
        return new_clusters

    def _validate_and_assign_new_papers_to_clusters(self, clusters: Dict[str, dict], new_papers: List[dict], valid_papers: List[str], necessary_papers: List[str], omit_err: bool = False) -> Dict[str, dict]:
        valid_papers = set(valid_papers)

        for paper in new_papers:
            if "clusters" not in paper or paper.get("clusters") is None:
                if omit_err:
                    self.logger.warning("Omitting this paper due to incomplete information in PAPER CLUSTER PAPER ASSIGNING: lack key clusters")
                    continue
                raise ValueError("Incomplete paper information in PAPER CLUSTER PAPER ASSIGNING: lack key clusters")

            assigned_clusters = paper.get("clusters")
            for assigned_cluster in assigned_clusters:
                if assigned_cluster in clusters:

                    if paper["id"] is None or paper["title"] is None or paper["tldr"] is None:
                        self.logger.error(f"Paper information incomplete for paper in cluster assignment: {paper}")
                        if omit_err:
                            self.logger.warning("Omitting this paper due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
                            continue
                        raise ValueError("Incomplete paper information in PAPER CLUSTER PAPER ASSIGNING.")

                    if paper["id"] not in valid_papers:
                        self.logger.error(f"Paper ID {paper['id']} not in original paper list during cluster assignment.")
                        if omit_err:
                            self.logger.warning("Omitting this paper due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
                            continue
                        raise ValueError("Invalid paper ID in PAPER CLUSTER PAPER ASSIGNING.")

                    clusters[assigned_cluster]["papers"].append({
                        "id": paper["id"],
                        "title": paper["title"],
                        "tldr": paper["tldr"]
                    })
                else:
                    self.logger.error(f"Assigned cluster {assigned_cluster} not found in existing clusters.")
                    if omit_err:
                        self.logger.warning(f"Omitting this fake cluster name:{assigned_cluster} due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
                        continue
                    raise ValueError("Invalid clustering assignment in PAPER CLUSTER PAPER ASSIGNING.")
                
        assigned_paper_ids = set(paper["id"] for paper in new_papers)
        unassigned = set(necessary_papers) - assigned_paper_ids

        if unassigned:
            self.logger.error(f"Papers in batch not assigned num: {len(unassigned)}")
            if not omit_err:
                raise ValueError("Missing papers in batch assignment.")
            else:
                self.logger.warning(f"Omitting this not assigned papers:{len(unassigned)} due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
        return clusters

    def _validate_new_papers_to_clusters(self, clusters: Dict[str, dict], new_papers: List[dict], valid_papers: List[str], necessary_papers: List[str], omit_err: bool = False) -> Dict[str, dict]:
        valid_papers = set(valid_papers)
        # self.logger.info("inner validating....")
        returned_papers = []

        for paper in new_papers:

            if "clusters" not in paper or paper.get("clusters") is None:
                self.logger.error("ERROR: incomplete information in PAPER CLUSTER PAPER ASSIGNING: lack key clusters or clusters empty")
                # self.logger.info(f"[CLUSTER FAST MODE DEBUG ERROR RESPONSE]: {new_papers}")
                if omit_err:
                    self.logger.warning("Omitting this paper due to incomplete information in PAPER CLUSTER PAPER ASSIGNING: lack key clusters or clusters empty")
                    continue
                raise ValueError("Incomplete paper information in PAPER CLUSTER PAPER ASSIGNING: lack key clusters")

            assigned_clusters = paper.get("clusters")
            
            for assigned_cluster in assigned_clusters:
                if assigned_cluster in clusters:

                    if paper["id"] is None or paper["title"] is None or paper["tldr"] is None:
                        self.logger.error(f"Paper information incomplete for paper in cluster assignment: {paper}")
                        # self.logger.info(f"[CLUSTER FAST MODE DEBUG ERROR RESPONSE]: {new_papers}")
                        if omit_err:
                            self.logger.warning("Omitting this paper due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
                            continue
                        raise ValueError("Incomplete paper information in PAPER CLUSTER PAPER ASSIGNING.")

                    if paper["id"] not in valid_papers:
                        self.logger.error(f"Paper ID {paper['id']} not in original paper list during cluster assignment.")
                        # self.logger.info(f"[CLUSTER FAST MODE DEBUG ERROR RESPONSE]: {new_papers}")
                        if omit_err:
                            self.logger.warning("Omitting this paper due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
                            continue
                        raise ValueError("Invalid paper ID in PAPER CLUSTER PAPER ASSIGNING.")
                else:
                    self.logger.error(f"Assigned cluster {assigned_cluster} not found in existing clusters.")
                    # self.logger.info(f"[CLUSTER FAST MODE DEBUG ERROR RESPONSE]: {new_papers}")
                    if omit_err:
                        self.logger.warning(f"Omitting this fake cluster name:{assigned_cluster} due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
                        continue
                    raise ValueError("Invalid clustering assignment in PAPER CLUSTER PAPER ASSIGNING.")

        assigned_paper_ids = set(paper["id"] for paper in new_papers)
        unassigned = set(necessary_papers) - assigned_paper_ids

        if unassigned:
            self.logger.error(f"Papers in batch not assigned num: {len(unassigned)}")
            # self.logger.info(f"[CLUSTER FAST MODE DEBUG ERROR RESPONSE]: {new_papers}")
            if not omit_err:
                raise ValueError("Missing papers in batch assignment.")
            else:
                self.logger.warning(f"Omitting this not assigned papers:{len(unassigned)} due to incomplete information in PAPER CLUSTER PAPER ASSIGNING.")
        self.logger.info("inner validate finish")
        return True
        

    def log_clusters(self, clusters: List[Dict]):
        paper_id_set = set()
        log_str = "Clustering Results:\n"
        for i, cluster in enumerate(clusters):
            log_str += f"Cluster {i+1}:\n"
            log_str += f"  Cluster Name: {cluster['cluster_name']}\n"
            log_str += f"  Summary: {cluster['summary']}\n"
            log_str += f"  Papers:\n"
            for paper in cluster["papers"]:
                log_str += f"    - ID: {paper['id']}\n"
                log_str += f"      Title: {paper['title']}\n"
                log_str += f"      TL;DR: {paper['tldr']}\n"
                paper_id_set.add(paper["id"])
            log_str += "\n"
            log_str += "-" * 40 + "\n\n"
        self.logger.info(log_str)
        self.logger.info(f"Total unique papers in clusters: {len(paper_id_set)}")
        if self.config.BasicInfo.debug:
            self.logger.info(f"all ref set: {paper_id_set}")

    def validate_clusters(self, clusters: List[Dict], papers: List[str]) -> List[Dict]:
        paper_id_set = set(papers)
        for cluster in clusters:
            for paper in cluster["papers"]:
                if paper["id"] not in paper_id_set:
                    self.logger.warning(
                        f"Paper ID {paper['id']} in cluster {cluster['cluster_name']} not in original paper list."
                    )
                    raise ValueError("Invalid clustering result.")

    def prepare_prompt_for_proposing_questions_for_cluster(self, cluster: List[str]):
        cluster_content = f"Paper keynotes:\n"
        for paper in cluster["papers"]:
            pid = paper["id"]
            keynote_json = self.get_paper_keynote(pid)
            cluster_content += f"Paper ID: {pid}\nKeynote: {keynote_json}\n\n"
        return PROPOSE_QUESTIONS_FOR_CLUSTER.format(cluster_content=cluster_content)

    def prepare_prompt_for_answering_questions_for_cluster(self, questions: List[Dict]):
        prompts = []
        for q in questions:
            question_text = q["question"]
            related_papers = q["related_papers"]
            related_papers_content = ""
            for pid in related_papers:
                keynote_json = self.get_paper_keynote(pid)
                related_papers_content += (
                    f"Paper ID: {pid}\nKeynote: {keynote_json}\n\n"
                )
            prompt = ANSWER_QUESTION_FOR_PAPERS.format(
                question=question_text, related_papers_content=related_papers_content
            )
            prompts.append(prompt)
        return prompts

    def intra_cluster_analysis(self, clusters: List[Dict], retry=1):
        try:
            # step 1: propose questions for each cluster
            prompts = []
            for cluster in clusters:
                prompt = self.prepare_prompt_for_proposing_questions_for_cluster(
                    cluster
                )
                prompts.append(prompt)

            # use fixed-length list to keep correspondence turn with clusters
            questions = [None] * len(clusters)
            step_1_clusters = clusters  # to avoid changing clusters during retry

            # turn mapping for retry
            indices = list(range(len(clusters)))

            # start question proposing with retry
            for _ in range(
                self.config.ModuleInfo.WorkAnalyzer.intra_clustering_probelm_proposing_max_retry
            ):
                valid = True
                try:
                    responses = self.chat_agent.batch_remote_chat(
                        prompts,
                        temperature=self.config.ModuleInfo.WorkAnalyzer.propose_question_temperature,
                        desc="Proposing questions for clusters",
                    )
                except Exception as e:
                    self.logger.error(f'Error: {e} , during proposing questions for clusters in getting response from LLM, retrying all...')
                    continue

                # self.logger.info(f"intra cluster analysis responses:\nTYPE:{type(responses)} | {responses}")

                err_prompts = []
                err_clusters = []
                err_indices = []
                err_infos = []

                # use current batch indices to map results back to original positions
                for i, response in enumerate(responses):
                    original_index = indices[i]

                    try:
                        cur_questions = extract_json(response)
                    except Exception as e:
                        self.logger.warning(f"Error extracting questions for cluster {original_index+1}: {e}. Retrying...")
                        err_prompts.append(prompts[i])    
                        err_clusters.append(step_1_clusters[i])
                        err_indices.append(original_index)
                        err_infos.append(f"Error extracting questions from your answer for cluster {original_index+1}: {e}. \n")
                        valid = False
                        continue
                        
                    omit_error = not self.config.BasicInfo.error_conservatism_mode and (_ + 1 == self.config.ModuleInfo.WorkAnalyzer.intra_clustering_probelm_proposing_max_retry)

                    response_valid, invalid_pid = self.validate_questions(cur_questions, step_1_clusters[i], omit_err=omit_error)
                    if response_valid:
                        questions[original_index] = cur_questions
                    else:
                        valid = False
                        self.logger.warning(f"Invalid questions proposed for cluster {original_index+1}. Retrying...")

                        err_prompts.append(prompts[i])    
                        err_clusters.append(step_1_clusters[i])
                        err_indices.append(original_index)
                        err_infos.append(f"Question related papers ID {invalid_pid} not valid. Make sure only cite provided paper.\n ")

                if valid:   # no error, break
                    break

                # next retry only for error cases
                prompts = err_prompts
                step_1_clusters = err_clusters
                indices = err_indices

                for i in range(len(prompts)):
                    prompts[i] += ERROR_FEEDBACK_PROMPT.format(info = err_infos[i])

            # if None occurs in questions after retries, raise error
            if any(q is None for q in questions):
                raise ValueError("Invalid question related papers after maximum retries.")

            # step 2: answer questions for each cluster
            prompts = self.prepare_prompt_for_answering_questions_for_cluster(
                [q for cluster_questions in questions for q in cluster_questions]
            )
            answers = self.chat_agent.batch_remote_chat(
                prompts,
                temperature=self.config.ModuleInfo.WorkAnalyzer.propose_question_temperature,
                desc="Answering questions for clusters",
            )

            # step 3: organize the Q&A
            for cluster_questions in questions:
                for q in cluster_questions:
                    q["answer"] = answers.pop(0)

            return questions
        except Exception as e:
            if (
                retry
                > self.config.ModuleInfo.WorkAnalyzer.intra_cluster_analysis_max_retry
            ):
                raise e
            return self.intra_cluster_analysis(clusters, retry=retry + 1)

    def validate_questions(self, questions: List[Dict], cluster: Dict, omit_err: bool = False):
        valid_paper_ids = {paper["id"] for paper in cluster["papers"]}
        err_id = []
        for q in questions:
            for pid in q["related_papers"]:
                if pid not in valid_paper_ids:
                    self.logger.warning(
                        f"Paper ID {pid} in question '{q['question']}' not in cluster papers."
                    )
                    if not omit_err:
                        return False, pid
                    else:
                        self.logger.warning(f"Omitting this invalid paper ID:{pid} in question due to omit_err mode.")
                        err_id.append(pid)
            if omit_err and err_id:
                q["related_papers"] = [pid for pid in q["related_papers"] if pid not in err_id]

        return True, ""

    def log_intra_cluster_analysis(self, analysis_results: List[List[Dict]]):
        log_str = "Intra-Cluster Analysis Results:\n"
        for i, cluster_results in enumerate(analysis_results):
            log_str += f"Cluster {i+1}:\n"
            for qa in cluster_results:
                log_str += f"  Question: {qa['question']}\n"
                log_str += f"  Related Papers: {', '.join(qa['related_papers'])}\n"
                log_str += f"  Answer: {qa['answer']}\n"
                log_str += "\n"
            log_str += "-" * 40 + "\n\n"
        self.logger.info(log_str)

    def inter_cluster_analysis(self, intra_analysis_results: List[List[Dict]]):
        cluster_analysis_content = ""
        for i, cluster_results in enumerate(intra_analysis_results):
            cluster_analysis_content += f"Group {i+1} Analysis:\n"
            for j, qa in enumerate(cluster_results):
                cluster_analysis_content += f"Question {j+1}: {qa['question']}\n"
                cluster_analysis_content += f"Answer: {j+1}: {qa['answer']}\n\n"
            cluster_analysis_content += "-" * 3 + "\n\n"

        prompt = INTER_CLUSTER_ANALYSIS.format(
            cluster_analysis_content=cluster_analysis_content
        )
        response = self.chat_agent.remote_chat(
            prompt,
            temperature=self.config.ModuleInfo.WorkAnalyzer.intra_cluster_analysis_temperature,
        )
        return response

    def log_inter_cluster_analysis(self, analysis_results):
        log_str = "Inter-Cluster Analysis Results:\n"
        log_str += analysis_results + "\n"
        self.logger.info(log_str)

    # ---------------------------- Relationship analysis ----------------------------
    def build_relationship_graphs(self, clusters: List[Dict]):
        """
        For each cluster, construct a directed graph over its papers and enrich edges with
        relation type/analysis from the LLM.

        Returns a dict: {cluster_name: nx.DiGraph}, where each edge has attributes
        {'type': str, 'analysis': str, 'raw': str}.
        """
        ref_graph: nx.DiGraph = self.work_collector.reference_graph
        if ref_graph is None:
            raise ValueError("Reference graph is not initialized; cannot build relationship graphs.")

        cached = self._load_cached_relation_graph(clusters)
        if cached is not None:
            self.logger.info("Using cached relationship graphs.")
            self.relation_analysis_graph = cached
            return cached

        results = {}
        pending_tasks = []  # collect all edge prompts for batched processing
        for cluster in clusters:
            cluster_name = cluster.get("cluster_name", "unknown_cluster")
            g = nx.DiGraph()

            # Collect node IDs in this cluster
            paper_ids = [p.get("id") for p in cluster.get("papers", []) if p.get("id")]
            g.add_nodes_from(paper_ids)

            for src in paper_ids:
                for dst in paper_ids:
                    if src == dst:
                        continue
                    if not ref_graph.has_edge(src, dst):
                        continue
                    if self.config.BasicInfo.debug:
                        self.logger.info(f"Analyzing relationship for edge {src}->{dst} in cluster {cluster_name}.")
                    # try:
                    src_title = ref_graph.nodes.get(src, {}).get("title", "")
                    dst_title = ref_graph.nodes.get(dst, {}).get("title", "")
                    src_keynote = self.get_paper_keynote(src)
                    dst_keynote = self.get_paper_keynote(dst)

                    prompt = PAPER_RELATIONSHIP_ANALYSIS.format(
                        src_title=src_title,
                        dist_title=dst_title,
                        src_keynote=src_keynote,
                        dist_keynote=dst_keynote,
                    )
                    pending_tasks.append({
                        "cluster_name": cluster_name,
                        "graph": g,
                        "src": src,
                        "dst": dst,
                        "prompt": prompt,
                    })
                    # except Exception as e:
                    #     self.logger.warning(f"Relation analysis failed for edge {src}->{dst}: {e}")
                    #     continue

            results[cluster_name] = g

        # Batch call LLM with retry
        temperature = getattr(
            self.config.ModuleInfo.WorkAnalyzer,
            "paper_relationship_temperature",
            0.3,
        )
        max_retry = getattr(
            self.config.ModuleInfo.WorkAnalyzer,
            "paper_relationship_max_retry",
            3,
        )

        tasks = pending_tasks
        for attempt in range(1, max_retry + 1):
            if not tasks:
                break
            prompts = [t["prompt"] for t in tasks]
            try:
                responses = self.chat_agent.batch_remote_chat(
                    prompts,
                    temperature=temperature,
                    desc=f"Relationship analysis attempt {attempt}/{max_retry}",
                )
            except Exception as e:
                self.logger.warning(f"batch_remote_chat failed on attempt {attempt}: {e}")
                continue

            next_tasks = []
            for task, resp in zip(tasks, responses or []):
                if resp is None:
                    next_tasks.append(task)
                    continue

                relation_type = "unspecified"
                analysis = resp
                try:
                    parsed = extract_json(resp)
                    if isinstance(parsed, dict):
                        relation_type = parsed.get("type") or parsed.get("relation") or relation_type
                        analysis = parsed.get("analysis") or parsed.get("description") or analysis
                except Exception:
                    pass

                try:
                    task["graph"].add_edge(
                        task["src"], task["dst"], type=relation_type, analysis=analysis, raw=resp
                    )
                    if self.config.BasicInfo.debug:
                        self.logger.info(f"GRAPH_DEBUG: Added edge {task['src']}->{task['dst']} with type '{relation_type}' in cluster {task['cluster_name']}.")
                except Exception as e:
                    self.logger.warning(f"Failed to add edge {task['src']}->{task['dst']}: {e}")
                    next_tasks.append(task)

            tasks = next_tasks

        # Drop edges that still failed after max retries
        for t in tasks:
            self.logger.warning(
                f"Dropping edge {t['src']}->{t['dst']} in cluster {t['cluster_name']} after {max_retry} retries."
            )
        self.relation_analysis_graph = results
        self._store_cached_relation_graph(clusters, results)
        self.logger.info(f"=====GRAPH_DEBUG=====")
        self.logger.info(f"{results}")
        self.logger.info(f"=====GRAPH END=====")
        return results

    def _get_relation_analysis_graph(self, relationship_graphs: Dict[str, nx.DiGraph] = None):
        """
        Convert relationship graphs to triples for downstream prompts.

        Returns a dict: {cluster_name: [ (src, relation, dst, analysis) ]}
        """
        if relationship_graphs is None:
            if self.relation_analysis_graph is None:
                raise ValueError("Relationship graphs not provided and not built yet.")
            relationship_graphs = self.relation_analysis_graph
        triples = {}
        for cluster_name, g in relationship_graphs.items():
            cluster_triples = []
            for u, v, data in g.edges(data=True):
                cluster_triples.append(
                    (
                        u,
                        data.get("type", "unspecified"),
                        v,
                        data.get("analysis", ""),
                    )
                )
            triples[cluster_name] = cluster_triples
        return triples

    def format_analysis_graph(self, relationship_graphs: Dict[str, nx.DiGraph] = None) -> str:
        """
        Call get_relation_analysis and format as a readable string per cluster.

        Output example:
        Cluster: C1
        - A --uses--> B | Because ...
        - ...
        """
        triples = self._get_relation_analysis_graph(relationship_graphs)
        parts: list[str] = []
        for cluster_name, items in triples.items():
            parts.append(f"Cluster: {cluster_name}")
            if not items:
                parts.append("  (no relations)")
                continue
            for src, rel, dst, analysis in items:
                rel_clean = rel or "unspecified"
                analysis_clean = analysis or ""
                parts.append(f"- {src} --{rel_clean}--> {dst} | {analysis_clean}")
        return "\n".join(parts)

    # ---------------------------- Cluster table generation ----------------------------
    def generate_cluster_tables(self, clusters: List[Dict]):
        """
        Build comparison tables for all clusters using CLUSTER_TABLE_GENERATION in batch.

        clusters: list of cluster dicts with keys 'cluster_name', 'papers' (list of dict with id/title),
        and optionally 'summary'.

        Returns a dict {cluster_name: table_json} parsed from the LLM responses.
        Raises if any cluster still fails after retries.
        """
        pending_tasks = []
        results = {}

        for cluster in clusters:
            cluster_name = cluster.get("cluster_name", "unknown_cluster")
            description = cluster.get("summary", "")

            paper_blocks = []
            for p in cluster.get("papers", []):
                pid = p.get("id") or ""
                title = p.get("title") or ""
                try:
                    keynote = self.get_paper_keynote(pid)
                except Exception as e:
                    self.logger.warning(f"Failed to get keynote for paper {pid}: {e}. Using empty keynote.")
                    keynote = ""
                paper_blocks.append(
                    f"Paper ID: {pid}\nPaper Title: {title}\nKeynote: {keynote}\n"
                )

            paper_content = "\n".join(paper_blocks)
            prompt = CLUSTER_TABLE_GENERATION.format(
                cluster_name=cluster_name,
                cluster_description=description,
                paper_content=paper_content,
            )

            pending_tasks.append({
                "cluster_name": cluster_name,
                "prompt": prompt,
            })

        max_retry = getattr(self.config.ModuleInfo.WorkAnalyzer, "cluster_table_max_retry", 3)
        temperature = getattr(self.config.ModuleInfo.WorkAnalyzer, "cluster_table_temperature", 0.3)

        tasks = pending_tasks
        last_err = None
        for attempt in range(1, max_retry + 1):
            if not tasks:
                break
            prompts = [t["prompt"] for t in tasks]
            try:
                responses = self.chat_agent.batch_remote_chat(
                    prompts,
                    temperature=temperature,
                    desc=f"Cluster table generation attempt {attempt}/{max_retry}",
                )
            except Exception as e:
                last_err = e
                self.logger.warning(f"batch_remote_chat failed on attempt {attempt}: {e}")
                continue

            next_tasks = []
            for task, resp in zip(tasks, responses or []):
                if resp is None:
                    next_tasks.append(task)
                    continue
                try:
                    parsed = extract_json(resp)
                    results[task["cluster_name"]] = parsed
                except Exception as e:
                    last_err = e
                    self.logger.warning(
                        f"Cluster table generation failed for {task['cluster_name']} on attempt {attempt}/{max_retry}: {e}"
                    )
                    next_tasks.append(task)

            tasks = next_tasks

        if tasks:
            failed = ", ".join([t["cluster_name"] for t in tasks])
            raise ValueError(
                f"Failed to generate cluster tables for: {failed} after {max_retry} retries: {last_err}"
            )

        self.relation_analysis_table = results
        return results

    def format_analysis_table(self, cluster_tables: Dict[str, Dict] = None):
        """Render all cluster tables into a markdown string."""
        if cluster_tables is None:
            if self.relation_analysis_table is None:
                raise ValueError("Cluster tables not provided and not generated yet.")
            cluster_tables = self.relation_analysis_table
        parts = []
        for cluster_name, table_json in cluster_tables.items():
            headers = ["Title"] + table_json.get("comparison_dimensions", [])
            md_table = "| " + " | ".join(headers) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

            for row in table_json.get("table_data", []):
                cols = [row.get("paper_title", "")]
                for dim in table_json.get("comparison_dimensions", []):
                    cols.append(row.get("columns", {}).get(dim, "N/A"))
                md_table += "| " + " | ".join(cols) + " |\n"

            parts.append(f"Cluster: {cluster_name}\n{md_table}")

        return "\n".join(parts)

    # ---------------------------- Cache helpers ----------------------------
    def _cache_task_key(self) -> str:
        topic = getattr(self.config.BasicInfo, "topic", "")
        cache_token = getattr(self.config.BasicInfo, "cache_token", "")
        return get_hash(f"{topic}|{cache_token}|{self.cache_path}")

    def _cluster_cache_key(self, papers: List[str]) -> str:
        sorted_ids = sorted(papers)
        return get_hash(f"clusters|{self._cache_task_key()}|{'|'.join(sorted_ids)}")

    def _relation_graph_cache_key(self, clusters: List[Dict]) -> str:
        cluster_repr = []
        for c in clusters:
            name = c.get("cluster_name", "")
            ids = sorted([p.get("id", "") for p in c.get("papers", [])])
            cluster_repr.append(f"{name}:{'|'.join(ids)}")
        return get_hash(f"relation_graph|{self._cache_task_key()}|{'||'.join(sorted(cluster_repr))}")

    def _prune_cache(self, cache: dc.Cache, max_entries: int):
        if max_entries is None or max_entries <= 0:
            return
        try:
            # exclude internal keys starting with '__'
            data_keys = [k for k in cache.iterkeys() if not (isinstance(k, str) and k.startswith("__"))]
            if len(data_keys) <= max_entries:
                return
            keyed = []
            for k in data_keys:
                try:
                    meta = cache.get(k, {}) or {}
                    ts = meta.get("ts", 0)
                except Exception:
                    ts = 0
                keyed.append((ts, k))
            keyed.sort()
            for _, k in keyed[: max(0, len(data_keys) - max_entries)]:
                try:
                    del cache[k]
                except Exception:
                    self.logger.warning(f"Failed to evict cache entry {k}")
        except Exception as e:
            self.logger.warning(f"Cache pruning skipped due to error: {e}")

    def _load_cached_clusters(self, papers: List[str]):
        if not getattr(self.config.ModuleInfo.WorkAnalyzer, "cluster_cache_enabled", True):
            return None
        key = self._cluster_cache_key(papers)
        if key in self.cluster_cache:
            try:
                payload = self.cluster_cache[key]
                return payload.get("data")
            except Exception as e:
                self.logger.warning(f"Failed to load cached clusters: {e}")
        return None

    def _store_cached_clusters(self, papers: List[str], clusters: List[Dict]):
        if not getattr(self.config.ModuleInfo.WorkAnalyzer, "cluster_cache_enabled", True):
            return
        key = self._cluster_cache_key(papers)
        self.cluster_cache[key] = {"data": clusters, "ts": time.time()}
        max_entries = getattr(self.config.ModuleInfo.WorkAnalyzer, "cluster_cache_max_entries", 5)
        self._prune_cache(self.cluster_cache, max_entries)

    def _load_cached_relation_graph(self, clusters: List[Dict]):
        if not getattr(self.config.ModuleInfo.WorkAnalyzer, "relation_graph_cache_enabled", True):
            return None
        key = self._relation_graph_cache_key(clusters)
        if key in self.relation_graph_cache:
            try:
                payload = self.relation_graph_cache[key]
                return payload.get("data")
            except Exception as e:
                self.logger.warning(f"Failed to load cached relationship graph: {e}")
        return None

    def _store_cached_relation_graph(self, clusters: List[Dict], graphs: Dict[str, nx.DiGraph]):
        if not getattr(self.config.ModuleInfo.WorkAnalyzer, "relation_graph_cache_enabled", True):
            return
        key = self._relation_graph_cache_key(clusters)
        self.relation_graph_cache[key] = {"data": graphs, "ts": time.time()}
        max_entries = getattr(self.config.ModuleInfo.WorkAnalyzer, "relation_graph_cache_max_entries", 5)
        self._prune_cache(self.relation_graph_cache, max_entries)


if __name__ == "__main__":
    root_ids = [
        # "arXiv:1706.03762",  # Transformer
        "fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5"
    ]
