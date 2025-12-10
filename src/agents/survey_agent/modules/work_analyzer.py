from typing import List
import os
import pickle
from typing import Dict
from utils.api_call import ChatAgent, SemanticScholarAPI
from utils.utils import get_hash, extract_json
import diskcache as dc
from modules.pe import (
    PAPER_DEEP_READING,
    PAPER_CLUSTERING,
    PROPOSE_QUESTIONS_FOR_CLUSTER,
    ANSWER_QUESTION_FOR_PAPERS,
    INTRA_CLUSTER_ANALYSIS,
)
import hdbscan
from sentence_transformers import SentenceTransformer
from utils.rich_logger import get_logger
import math


class WorkAnalyzer:
    def __init__(self, config, work_collector):
        self.config = config
        self.chat_agent = ChatAgent(config)
        self.semantic_scholar_api = SemanticScholarAPI(config)
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

    def read_papers_and_write_keynotes(self, papers: List[str], retry: int = 1):
        if retry > self.config.ModuleInfo.WorkAnalyzer.paper_reading_max_retry:
            return
        try:
            tasks = []
            for pid in papers:
                hash_id = get_hash(pid)
                if hash_id in self.paper_keynote_cache:
                    continue

                with open(
                    os.path.join(
                        f"{self.cache_path}/parsed_papers", pid, "auto", f"{pid}.md"
                    ),
                    "r",
                    encoding="utf-8",
                ) as fr:
                    paper_markdown_text = fr.read()
                tasks.append(
                    [
                        pid,
                        hash_id,
                        PAPER_DEEP_READING.format(
                            paper_markdown_text=paper_markdown_text
                        ),
                    ]
                )

            prompts = [task[2] for task in tasks]
            if not prompts:
                return
            responses = self.chat_agent.batch_remote_chat(
                prompts,
                temperature=self.config.ModuleInfo.WorkAnalyzer.paper_reading_temperature,
                desc="Reading papers",
            )
            for i, response in enumerate(responses):
                pid, hash_id, _ = tasks[i]
                self.paper_keynote_cache[hash_id] = {
                    "paper_id": pid,
                    "keynote": extract_json(response),
                }
                if self.config.BasicInfo.debug:
                    self.logger.info(f"paper ID {pid} keynote: {extract_json(response)}")
        except Exception as e:
            self.read_papers_and_write_keynotes(papers, retry=retry + 1)

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
            if "." in paper_id:
                query_id = "ARXIV:" + paper_id
            else:
                query_id = paper_id
            paper = self.semantic_scholar_api.get_paper_details(
                query_id, fields="title,year,venue,authors"
            )
            # YZY DEBUG
            if not paper:
                print(f"Warning: Unable to fetch details for paper ID {paper_id}")
                return "Unknown Citation"

            authors = paper.get("authors", [])
            title = paper.get("title", "")
            venue = paper.get("venue", "")
            year = paper.get("year", "")
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
        if hash_id not in self.paper_keynote_cache:
            self.read_papers_and_write_keynotes([paper_id])
        keynote_data = self.paper_keynote_cache[hash_id]["keynote"]
        return keynote_data

    def get_paper_raw_markdown(self, paper_id: str) -> str:
        if not os.path.exists(
            os.path.join(
                f"{self.cache_path}/parsed_papers", paper_id, "auto", f"{paper_id}.md"
            )
        ):
            self.work_collector.download_and_parse_papers([paper_id])
        with open(
            os.path.join(
                f"{self.cache_path}/parsed_papers", paper_id, "auto", f"{paper_id}.md"
            ),
            "r",
            encoding="utf-8",
        ) as fr:
            paper_markdown_text = fr.read()
        return paper_markdown_text

    def cluster_papers(self, papers: List[str], retry=1) -> List[List[str]]:
        clusters = []
        num_batches = math.ceil(
            len(papers) / self.config.ModuleInfo.WorkAnalyzer.clustering_batch_size
        )

        i = 0
        while i < num_batches:
            valid = False
            for _ in range(
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

                    keynote_papers = []
                    paper_keynotes = ""
                    for pid in batch:
                        keynote_json = self.get_paper_keynote(pid)
                        paper_keynotes += (
                            f"Paper ID: {pid}\nKeynote: {keynote_json}\n\n"
                        )
                        keynote_papers.append(pid)

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
                    self.validate_clusters(new_clusters, keynote_papers, papers) #YZY MODIFY: from clusters
                    valid = True
                    break
                except Exception as e:
                    self.logger.warning("Retrying")
            if not valid:
                raise ValueError("Clustering failed after maximum retries.")
            clusters = new_clusters
            i += 1
        return new_clusters
        

    def log_clusters(self, clusters: List[Dict]):
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
            log_str += "\n"
            log_str += "-" * 40 + "\n\n"
        self.logger.info(log_str)

    def validate_clusters(self, clusters: List[Dict], keynote_papers: List[str], papers: List[str]) -> List[Dict]:
        paper_id_set = set(papers)
        keynote_id_set = set(keynote_papers) # YZY MODIFY
        for cluster in clusters:
            for paper in cluster["papers"]:
                if paper["id"] not in paper_id_set:
                    self.logger.warning(
                        f"Paper ID {paper['id']} in cluster {cluster['cluster_name']} not in original paper list."
                    )
                    if paper["id"] in keynote_id_set: # YZY MODIFY
                        self.logger.info(
                            f"Paper ID {paper['id']} found in keynote papers.")
                        continue
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

    def intra_cluster_analysis(self, clusters: List[List[str]], retry=1):
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
                responses = self.chat_agent.batch_remote_chat(
                    prompts,
                    temperature=self.config.ModuleInfo.WorkAnalyzer.propose_question_temperature,
                    desc="Proposing questions for clusters",
                )

                # self.logger.info(f"intra cluster analysis responses:\nTYPE:{type(responses)} | {responses}")

                err_prompts = []
                err_clusters = []
                err_indices = []

                # use current batch indices to map results back to original positions
                for i, response in enumerate(responses):
                    original_index = indices[i]
                    cur_questions = extract_json(response)
                    if (self.validate_questions(cur_questions, step_1_clusters[i])):
                        questions[original_index] = cur_questions
                    else:
                        valid = False
                        self.logger.warning(f"Invalid questions proposed for cluster {original_index+1}. Retrying...")

                        err_prompts.append(prompts[i])    
                        err_clusters.append(step_1_clusters[i])
                        err_indices.append(original_index)

                if valid:   # no error, break
                    break

                # next retry only for error cases
                prompts = err_prompts
                step_1_clusters = err_clusters
                indices = err_indices

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

    def validate_questions(self, questions: List[Dict], cluster: Dict):
        valid_paper_ids = {paper["id"] for paper in cluster["papers"]}
        for q in questions:
            for pid in q["related_papers"]:
                if pid not in valid_paper_ids:
                    self.logger.warning(
                        f"Paper ID {pid} in question '{q['question']}' not in cluster papers."
                    )
                    # raise ValueError("Invalid question related papers.")
                    return False
        return True

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
