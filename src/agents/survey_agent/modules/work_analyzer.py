from typing import List
import os
import pickle
from typing import Dict
from utils.api_call import ChatAgent, SemanticScholarAPI,ArxivAPI
from utils.utils import get_hash, extract_json
import diskcache as dc
from modules.pe import (
    PAPER_DEEP_READING,
    PAPER_CLUSTERING,
    PROPOSE_QUESTIONS_FOR_CLUSTER,
    ANSWER_QUESTION_FOR_PAPERS,
    INTRA_CLUSTER_ANALYSIS,
    PAPER_CLUSTERING_CREATING,
    PAPER_CLUSTERING_ASSIGNING,
    ERROR_FEEDBACK_PROMPT
)
import hdbscan
from sentence_transformers import SentenceTransformer
from utils.rich_logger import get_logger
import math
import requests
import xml.etree.ElementTree as ET

class WorkAnalyzer:
    def __init__(self, config, work_collector):
        self.config = config
        self.chat_agent = ChatAgent(config)
        self.semantic_scholar_api = SemanticScholarAPI(config)
        self.arxiv_api = ArxivAPI(config)
        self.logger = get_logger("WorkAnalyzer")
        self.work_collector = work_collector

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

        self.paper_abstract_cache = self.work_collector.paper_abstract_cache

    def read_papers_and_write_keynotes(self, papers: List[str], retry: int = 1):

        if self.config.ModuleInfo.WorkAnalyzer.abstract_only_mode:
            self.logger.info("Abstract-only mode enabled, fetching abstracts instead of deep reading and get keynotes.")
            return self.work_collector.add_papers_abstracts_in_cache(papers, retry=retry)

        if retry > self.config.ModuleInfo.WorkAnalyzer.paper_reading_max_retry:
            self.logger.error("Exceeded maximum retries for reading papers.")
            self.logger.error(f"Papers failed to read: {papers}")
            return papers
        
        tasks = []
        err_papers = []
        for pid in papers:
            try:
                hash_id = get_hash(pid)
                if hash_id in self.paper_keynote_cache:
                    continue
                try:
                    paper_markdown_text = self.get_paper_raw_markdown(pid)

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
        except Exception as e:
            self.logger.error(f'Error: {e} , during comprehending papers in getting response from LLM, retrying all...')
            return self.read_papers_and_write_keynotes(papers, retry=retry + 1)

        for i, response in enumerate(responses):
            try:
                pid, hash_id, _ = tasks[i]
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

    def get_paper_keynote(self, paper_id: str):
        """Get the keynote of a paper by its ID."""
        hash_id = get_hash(paper_id)

        if self.config.ModuleInfo.WorkAnalyzer.abstract_only_mode:
            try:
                title, abstract =  self.work_collector.get_paper_title_abstract(paper_id)
            except Exception as e:
                raise ValueError(f"Failed to get abstract for paper ID {paper_id}: {e}")
            return abstract

        else:
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

    def get_paper_raw_markdown(self, paper_id: str) -> str:
        md_path = os.path.join(f"{self.cache_path}/parsed_papers", paper_id, "auto", f"{paper_id}.md")
        if not os.path.exists(
            md_path
        ):
            if self.config.BasicInfo.debug:
                self.logger.info(f"Paper {paper_id} markdown not found in cache, re-downloading and parsing...")
            try:
                self.work_collector.download_and_parse_papers([paper_id])
            except Exception as e:
                self.logger.error(f"Failed to parse paper {paper_id}: {e}")
                raise e
        if not os.path.exists(md_path):
            self.logger.warning(f"Markdown still missing after parse: {md_path} or use abstract instead when 'use_abstract_when_full_text_fail' enabled.")
            raise ValueError("Markdown file missing in getting paper markdown")
        with open(
            md_path,
            "r",
            encoding="utf-8",
        ) as fr:
            paper_markdown_text = fr.read()
        return paper_markdown_text

    def cluster_papers(self, papers: List[str]) -> List[Dict]:
        if self.config.ModuleInfo.WorkAnalyzer.clustering_in_steps:
            return self.cluster_papers_in_steps(papers)
        else:
            return self.cluster_papers_1_step(papers)

    def cluster_papers_1_step(self, papers: List[str]) -> List[Dict]:
        clusters = []
        num_batches = math.ceil(
            len(papers) / self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size
        )

        i = 0
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
                    if self.config.BasicInfo.debug:
                        self.logger.info(f'complete paper list for clustering: {papers}')
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

    def _assign_papers_to_clusters(self, papers: List[str], clusters: List[dict]) -> List[Dict]:
        # step2: assign papers to clusters
        num_batches = math.ceil(
            len(papers) / self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size_in_assignment
        )

        cluster_name_dict = {cluster["cluster_name"]: cluster for cluster in clusters}
        for cluster in cluster_name_dict.values():
            cluster["papers"] = []
        
        new_clusters = cluster_name_dict

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

                    new_papers_dict = extract_json(
                        self.chat_agent.remote_chat(
                            assign_prompt,
                            temperature=self.config.ModuleInfo.WorkAnalyzer.clustering_temperature,
                        )
                    )
                    if not self.config.BasicInfo.error_conservatism_mode and retry_time == self.config.ModuleInfo.WorkAnalyzer.paper_clustering_assignment_max_retry - 1:
                        self.logger.warning("max retry reached, assigning papers directly to clusters and returning in PAPER CLUSTER PAPER ASSIGNING.")
                        new_clusters = self._validate_and_assign_new_papers_to_clusters(new_clusters, new_papers_dict, papers, batch, True)
                        
                    else:
                        new_clusters = self._validate_and_assign_new_papers_to_clusters(new_clusters, new_papers_dict, papers, batch, False)
                    valid = True
                    break
                except Exception as e:
                    self.logger.warning(f"Error during clustering batch {i+1}: {e}. Retrying for {retry_time + 1}...")
                    err_info += f"Error during clustering batch {i+1}: {e}. \n"
            if not valid:
                raise ValueError("Clustering failed after maximum retries.")
            if self.config.BasicInfo.debug:
                self.logger.info(f"CLUSTER after batch {i+1}: {new_clusters}")
            i += 1
        if self.config.BasicInfo.debug:
            self.logger.info(f"Final CLUSTER result: {new_clusters}")

        new_clusters = list(new_clusters.values())
        return new_clusters

    def _validate_and_assign_new_papers_to_clusters(self, clusters: Dict[str, dict], new_papers: List[dict], valid_papers: List[str], necessary_papers: List[str], omit_err: bool = False) -> Dict[str, dict]:
        valid_papers = set(valid_papers)

        for paper in new_papers:
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

        prompt = INTRA_CLUSTER_ANALYSIS.format(
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


if __name__ == "__main__":
    root_ids = [
        # "arXiv:1706.03762",  # Transformer
        "fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5"
    ]
