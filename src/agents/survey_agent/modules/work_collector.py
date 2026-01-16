from typing import Dict, List
import os
from src.agents.survey_agent.utils.api_call import SemanticScholarAPI, ChatAgent, ArxivAPI
from src.agents.survey_agent.utils.rich_logger import get_logger
from src.agents.survey_agent.utils.mineru_utils import parse_doc
import requests
from contextlib import closing
import pickle
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import torch
from src.agents.survey_agent.modules.pe import PAPER_RELATEDNESS_BASED_ON_TITLE_AND_ABSTRACT
import diskcache as dc
from src.agents.survey_agent.utils.utils import get_hash, extract_json, is_valid_pdf

import gc

class WorkCollector:
    def __init__(self, config):
        self.config = config
        self.semantic_scholar_api = SemanticScholarAPI(config)
        self.cache_path = self.config.BasicInfo.cache_path
        self.chat_agent = ChatAgent(config)
        self.logger = get_logger("WorkCollector")
        self.paper_abstract_cache = dc.Cache(
            os.path.join(self.cache_path, "paper_abstracts")
        )
        self.graph_paper_ids = set()
        self.arxiv_api = ArxivAPI(config)

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

        # cache for paper relatedness
        self.relatedness_cache = dc.Cache(
            os.path.join(self.cache_path, "workcollector_relatedness_cache")
        )

    def collect_seed_papers(self, topic: str):
        """
        Collect related work based on the given topic.

        Returns a list of work items.
        """
        # version 1: use Semantic Scholar API to search papers
        fields = "title,externalIds,openAccessPdf"
        response = self.semantic_scholar_api.search_papers(query=topic, fields=fields)

        papers = response.get("data", [])
        self.logger.info(f"Found {len(papers)} papers related to topic: {topic}")
        self.logger.info(
            f"Downloading and parsing up to {self.config.ModuleInfo.WorkCollector.max_seed_paper_num} papers..."
        )
        valid_seed_papers_ids =  self.download_and_parse_papers(
            papers, limit=self.config.ModuleInfo.WorkCollector.max_seed_paper_num
        )
        self.graph_paper_ids.update(valid_seed_papers_ids)
        return valid_seed_papers_ids


    def download_and_parse_papers(self, papers: list, limit: int = -1):
        """
        Download and parse the papers given their IDs.

        Returns the parsed document.
        """

        valid_paper_ids = []
        valid_paper_paths = []
        valid_paper_paths_ids = []
        # step 1: download the paper PDF
        index = 0
        total = min(len(papers), limit) if limit > 0 else len(papers)
        while index < len(papers) and (limit < 0 or len(valid_paper_ids) < limit):
            paper = papers[index]
            index += 1

            download_urls = []
            is_arxiv = False
            if isinstance(paper, dict):
                if "ArXiv" in paper.get("externalIds", {}):
                    paper_id = paper["externalIds"]["ArXiv"]
                    is_arxiv = True
                else:
                    paper_id = paper.get("paperId")
                    is_arxiv = False

                if is_arxiv:
                    download_urls.append(f"https://export.arxiv.org/pdf/{paper_id}.pdf")
                    download_urls.append(f"https://arxiv.org/pdf/{paper_id}.pdf")

                elif paper.get("openAccessPdf", {}).get("url"):
                    download_urls.append(paper["openAccessPdf"]["url"])
                elif self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                    self.logger.info(f"No full text PDF found for paper {paper_id}, skipping download but keeping abstract.")
                    err = self.add_papers_abstracts_in_cache([paper_id])
                    if not err:
                        valid_paper_ids.append(paper_id)
                    continue
                else:
                    continue
                paper_title = paper.get("title", paper_id)
            elif isinstance(paper, str):
                is_arxiv = "." in paper
                if is_arxiv:
                    paper_id = paper
                    download_urls.append(f"https://export.arxiv.org/pdf/{paper_id}.pdf")
                    download_urls.append(f"https://arxiv.org/pdf/{paper_id}.pdf")
                else:
                    paper_id = paper
                    try:
                        paper = self.semantic_scholar_api.get_paper_details(
                            paper,
                            fields="title,externalIds,openAccessPdf",
                        )
                    except Exception as e:
                        self.logger.error(f"Error fetching paper {paper_id} details from Semantic Scholar: {e}. Skipping.")
                        continue
                        
                    if paper_id != paper.get("paperId"):
                        self.logger.warning(
                            f"Paper ID mismatch for {paper_id}, got {paper.get('paperId')} in DOWNLOAD AND PARSE PAPER, skipping."
                        )
                        continue
                    if not paper:
                        continue
                    try:
                    # YZY MODIFY: fix bug when paper is dict(str is meaning less)
                        if "ArXiv" in paper.get("externalIds", {}):
                            paper_id = paper["externalIds"]["ArXiv"]
                            is_arxiv = True
                        else:
                            paper_id = paper.get("paperId")
                            is_arxiv = False
                    except Exception as e:
                        self.logger.warning(f"Error getting paper ID: {e} with paper data: {paper}, skipping.")
                        continue
                    if is_arxiv:
                        download_urls.append(f"https://export.arxiv.org/pdf/{paper_id}.pdf")
                        download_urls.append(f"https://arxiv.org/pdf/{paper_id}.pdf")
                    elif paper.get("openAccessPdf", {}).get("url"):
                        download_urls.append(paper["openAccessPdf"]["url"])
                    elif self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                        self.logger.info(f"No full text PDF found for paper {paper_id}, skipping download but keeping abstract.")
                        err = self.add_papers_abstracts_in_cache([paper_id])
                        if not err:
                            valid_paper_ids.append(paper_id)
                        continue
                    else:
                        continue
                
                paper_title = self.reference_graph.nodes.get(paper_id, {}).get(
                    "title", paper_id
                )
            else:
                continue
            pdf_path = os.path.join(
                self.config.BasicInfo.cache_path,
                "pdf_papers",
                paper_id,
                f"{paper_id}.pdf",
            )

            if (
                not self.config.ModuleInfo.WorkCollector.download_safe_mode
                and os.path.exists(pdf_path)
            ):
                if is_valid_pdf(pdf_path):
                    if self.config.BasicInfo.debug:
                        self.logger.info(f"Cache Hit! Existing PDF at {pdf_path} is valid, skipping download.")
                    valid_paper_ids.append(paper_id)
                    if not os.path.exists(
                        os.path.join(
                            self.config.BasicInfo.cache_path, "parsed_papers", paper_id
                        )
                    ):
                        valid_paper_paths.append(pdf_path)
                        valid_paper_paths_ids.append(paper_id)
                    continue
                else:
                    try:
                        os.remove(pdf_path)
                    except OSError as e:
                        self.logger.error(f"Failed to delete invalid PDF {pdf_path}: {e}")
                        continue
                    self.logger.warning(f"Existing PDF at {pdf_path} is invalid, re-downloading.")
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

            for download_url in download_urls:
                downloaded = None
                downloaded = self._download_pdf_with_resume(
                    url = download_url,
                    filename = pdf_path,
                    title = f"{paper_title} ({index}/{total})",
                )
                if not downloaded or not is_valid_pdf(pdf_path):
                    self.logger.warning(
                        f"Failed to download valid paper {paper_id} from {download_url}, trying next URL if available."
                    )
                    continue
                else:
                    break

            if not downloaded:
                self.logger.warning(
                    f"Failed to download valid paper {paper_id} from {download_url}"
                )
                continue
            if not is_valid_pdf(pdf_path):
                self.logger.warning(
                    f"Existing PDF at {pdf_path} is invalid, deleting."
                )
                try:
                    os.remove(pdf_path)
                except OSError as e:
                    self.logger.error(f"Failed to delete invalid PDF {pdf_path}: {e}")
                    continue  # 或 return / raise
                continue

            valid_paper_ids.append(paper_id)
            # skip parsing if already parsed
            if not os.path.exists(
                os.path.join(
                    self.config.BasicInfo.cache_path, "parsed_papers", paper_id
                )
            ):
                valid_paper_paths.append(pdf_path)
                valid_paper_paths_ids.append(paper_id)

        # step 2: parse the downloaded PDFs
        self.logger.info(f"Parsing {len(valid_paper_paths)} downloaded papers...")
        if valid_paper_paths:
            try:
                parse_doc(
                    valid_paper_paths,
                    output_dir=os.path.join(
                        self.config.BasicInfo.cache_path, "parsed_papers"
                    ),
                    lang="en",
                )
            except Exception as e:
                for paper_path in valid_paper_paths:
                    try:
                        parse_doc(
                            [paper_path],
                            output_dir=os.path.join(
                                self.config.BasicInfo.cache_path, "parsed_papers"
                            ),
                            lang="en",
                        )
                    except Exception as e2:
                        self.logger.error(f"Failed to parse paper at {paper_path}: {e2}")
                        valid_paper_ids.remove(
                            valid_paper_paths_ids[
                                valid_paper_paths.index(paper_path)
                            ]
                        )

        return valid_paper_ids

    def _download_pdf_with_resume(self, url, filename, title, is_arxiv = False, chunk_size=1024 * 1024):
        temp_size = 0
        if os.path.exists(filename):
            temp_size = os.path.getsize(filename)
            
        headers = {"Range": f"bytes={temp_size}-"}
        if is_arxiv:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; arXiv-downloader/1.0; +https://1369700255@qq.com/)",
                "Range": f"bytes={temp_size}-"
            }


        if self.config.BasicInfo.debug:
            self.logger.info(f"[{title}] Resuming download from byte {temp_size} from {url}")

        import time

        try:
            with closing(requests.get(url, headers=headers, stream=True)) as resp:

                if resp.status_code not in (200, 206, 416):
                    self.logger.error(
                        f"[{title}] Could not download file: {resp.status_code}"
                    )
                    return False

                if resp.status_code == 416:
                    self.logger.info(f"[{title}] File already fully downloaded.")
                    return True
                self.logger.info(f"[{title}] Resuming download from byte {temp_size}")

                total_size = None
                if "Content-Range" in resp.headers:
                    total_size = int(resp.headers["Content-Range"].split("/")[-1])
                elif "Content-Length" in resp.headers:
                    total_size = int(resp.headers["Content-Length"]) + temp_size

                if total_size:
                    self.logger.info(f"[{title}] Total file size: {total_size} bytes")
                else:
                    self.logger.warning(
                        f"[{title}] Total file size unknown. Progress may not be shown."
                    )

                with open(filename, "ab") as f:
                    downloaded = temp_size

                    # For speed calculation
                    last_time = time.time()
                    last_downloaded = downloaded

                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # --- Speed Calculation ---
                            now = time.time()
                            elapsed = now - last_time
                            if elapsed > 0:
                                delta_bytes = downloaded - last_downloaded
                                speed_mb_s = delta_bytes / elapsed / 1024 / 1024
                            else:
                                speed_mb_s = 0.0

                            last_time = now
                            last_downloaded = downloaded
                            # -------------------------

                            if total_size:
                                percent = downloaded * 100 / total_size
                                self.logger.info(
                                    f"[{title}] {percent:.2f}% "
                                    f"({downloaded}/{total_size} bytes) "
                                    f"Speed: {speed_mb_s:.2f} MB/s"
                                )
                            else:
                                self.logger.info(
                                    f"[{title}] Downloaded {downloaded} bytes "
                                    f"Speed: {speed_mb_s:.2f} MB/s"
                                )

        except Exception as e:
            self.logger.error(f"[{title}] Error during download: {e}")
            return False

        self.logger.info(f"[{title}] Download completed → {filename}")
        return True

    def add_papers_abstracts_in_cache(self, papers: List[str], retry: int = 1):
        err_papers = []
        for pid in papers:
            hash_id = get_hash(pid)
            # already cached
            if hash_id in self.paper_abstract_cache:
                if self.is_valid_abstract(self.paper_abstract_cache[hash_id]["abstract"]):
                    continue

            # build query id for Semantic Scholar
            if "." in pid:
                query_id = f"ARXIV:{pid}"
            else:
                query_id = pid

            abstract = ""
            title = ""
            
            # fetch minimal metadata including abstract
            for attempt in range(retry):
                try:
                    paper = self.semantic_scholar_api.get_paper_details(
                        query_id, fields="abstract,title,externalIds"
                    )
                except Exception as e:
                    self.logger.warning(f"Error fetching paper {pid} details from Semantic Scholar: {e}. Retrying {attempt + 1}/{retry}...")
                    paper = None
                if not paper and "." in pid:
                    try:
                        self.logger.info(f"Trying arXiv API for paper {pid} as fallback.")
                        paper = self.arxiv_api.get_paper_details(pid)
                    except Exception as e:
                        self.logger.warning(f"Error fetching from arXiv for {pid}: {e}. Retrying {attempt + 1}/{retry}...")
                        paper = None

                if not paper:
                    continue
                    
                abstract = paper.get("abstract", "") or ""
                title = paper.get("title", "") or ""

                if not self.is_valid_abstract(abstract):
                    self.logger.warning(f"No abstract found for {pid}. or abstract too short, len: {len(abstract)}.")
                    err_papers.append(pid)
                    continue

                if self.config.BasicInfo.debug:
                    self.logger.info(f"Add abstract for {pid}: {len(abstract)} chars.")
                break
            

            # Store into the same cache structure used for keynotes so downstream code can use get_paper_keynote
            if self.is_valid_abstract(abstract):
                self.paper_abstract_cache[hash_id] = {
                    "paper_id": pid,
                    "abstract": abstract,
                    "title": title,
                }
            
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Cached abstract for {pid}: {len(abstract)} chars.")
            else:
                err_papers.append(pid)
                
        return err_papers

    def get_paper_title_abstract(self, paper_id: str, retry: int = 3):
        hash_id = get_hash(paper_id)
        # already cached
        if hash_id in self.paper_abstract_cache and self.is_valid_abstract(self.paper_abstract_cache[hash_id]['abstract']):
            if self.config.BasicInfo.debug:
                self.logger.info(f"Cache hit for paper {paper_id} abstract, len: {len(self.paper_abstract_cache[hash_id]['abstract'])}")
            return self.paper_abstract_cache[hash_id]['title'], self.paper_abstract_cache[hash_id]['abstract']

        self.add_papers_abstracts_in_cache([paper_id], retry=retry)
        if hash_id in self.paper_abstract_cache and self.is_valid_abstract(self.paper_abstract_cache[hash_id]['abstract']):
            if self.config.BasicInfo.debug:
                self.logger.info(f"Fetched and cached abstract for paper {paper_id}, len: {len(self.paper_abstract_cache[hash_id]['abstract'])}")
            return self.paper_abstract_cache[hash_id]['title'], self.paper_abstract_cache[hash_id]['abstract']
        else:
            raise ValueError(f"Failed to get valid abstract for paper ID {paper_id} after {retry} retries.")

    def get_paper_title(self, paper_id: str, retry: int = 3):
        hash_id = get_hash(paper_id)
        # already cached
        if hash_id in self.paper_abstract_cache:
            if self.config.BasicInfo.debug:
                self.logger.info(f"Cache hit for paper {paper_id} title, len: {len(self.paper_abstract_cache[hash_id]['title'])}")
            return self.paper_abstract_cache[hash_id]['title']

        self.add_papers_abstracts_in_cache([paper_id], retry=retry)
        if hash_id in self.paper_abstract_cache:
            if self.config.BasicInfo.debug:
                self.logger.info(f"Fetched and cached title for paper {paper_id}, len: {len(self.paper_abstract_cache[hash_id]['title'])}")
            return self.paper_abstract_cache[hash_id]['title']
        else:
            raise ValueError(f"Failed to get valid abstract for paper ID {paper_id} after {retry} retries.")

    def is_valid_abstract(self, abstract: str) -> bool:
        if not isinstance(abstract, str):
            return False
        
        if not abstract or abstract.strip() == "" or abstract == "abstract not found" or len(abstract) < 50:
            return False
        return True

    def update_reference_graph(self, seed_paper_ids: List[str]):
        """
        Update the reference graph with new seed paper IDs.

        NOTE: All paper IDs are assumed to be Arxiv IDs first. If not found, fallback to Semantic Scholar Paper IDs.
        """
        self.logger.info("Updating reference graph...")
        if self.reference_graph is None:
            # initialize empty graph
            self.reference_graph = nx.DiGraph()

        def one_step(paper_id: str, visited, direction: str = "out"):
            if paper_id in visited:
                return
            visited.add(paper_id)

            related_papers = []
            # expand graph in the given direction
            if (
                paper_id not in self.reference_graph
                or (
                    direction == "out"
                    and self.reference_graph.out_degree(paper_id) == 0
                )
                or (direction == "in" and self.reference_graph.in_degree(paper_id) == 0)
            ):
                # arxiv
                if "." in paper_id:
                    query_id = f"ARXIV:{paper_id}"
                else:
                    query_id = paper_id

                if direction == "out":
                    fields = "title,year,venue,abstract,authors,externalIds,references.title,references.externalIds,references.year,references.authors,references.abstract,references.venue"
                else:
                    fields = "title,year,venue,abstract,authors,externalIds,citations.title,citations.externalIds,citations.year,citations.authors,citations.abstract,citations.venue"

                try:
                    paper_detail = self.semantic_scholar_api.get_paper_details(
                        query_id,
                        fields=fields,
                    )
                except Exception as e:
                    self.logger.error(f"Error fetching paper {paper_id} details from Semantic Scholar: {e}. Skipping.")
                    paper_detail = None

                if not paper_detail:
                    return

                # add node of current paper
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

                # YZY MODIFY: handle bug when references/citations are None(cannot iterate)
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
            ):  # NOTE: +1
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

        # save graph
        with open(os.path.join(self.cache_path, "reference_graph.pkl"), "wb") as writer:
            pickle.dump(self.reference_graph, writer)

        self.logger.info(
            f"Reference graph updated. Nodes: {self.reference_graph.number_of_nodes()}, Edges: {self.reference_graph.number_of_edges()}"
        )

    def compute_relatedness_scores_and_filter(
        self,
        seed_paper_ids: List[str],
    ) -> float:
        """
        Compute relatedness score between related papers and seed papers.
        """

        total = 0
        for seed_pid in seed_paper_ids:
            references = list(self.reference_graph.successors(seed_pid))
            citations = list(self.reference_graph.predecessors(seed_pid))
            total += len(references) + len(citations)

        self.logger.info(
            f"Total {total} related papers to compute relatedness scores for and filter."
        )

        # step 1: sentence-transformer-embedding-based relatedness score computation
        def paper2text(paper_id: str) -> str:
            node_data = self.reference_graph.nodes[paper_id]
            title = node_data.get("title", "")
            abstract = node_data.get("abstract", "")
            return f"Title: {title}\nAbstract: {abstract}"

        try:
            model = SentenceTransformer(
                self.config.ModuleInfo.WorkCollector.sentence_transformer_model
            ).cuda()
        except Exception as e:
            if "out of memory" in str(e).lower():
                self.logger.error("Out of memory error detected. Using CPU instead.")
                try:
                    del model
                except Exception:
                    self.logger.warning("Failed to delete model from GPU memory.")
                    pass
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    self.logger.warning("Failed to clear GPU cache.")
                    pass

                model = SentenceTransformer(
                    self.config.ModuleInfo.WorkCollector.sentence_transformer_model
                ).cpu()

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

            # topk selection
            top_k = min(
                self.config.ModuleInfo.WorkCollector.related_work_top_k,
                len(related_pids),
            )
            top_results = torch.topk(cosine_scores, k=top_k)
            for score, idx in zip(top_results[0][0], top_results[1][0]):
                pid = related_pids[idx]
                sim_score = score.item()
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Relatedness score between seed paper {seed_pid} and related paper {pid}: {sim_score}, threshold: {self.config.ModuleInfo.WorkCollector.related_work_threshold}")
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

        # step 2: LLM-based relatedness score computation and filtering
        tasks = []
        for seed_pid in seed_paper_ids:
            related_pids = list(saved_papers[seed_pid])
            if not related_pids:
                continue

            seed_title = self.reference_graph.nodes[seed_pid].get("title", "")
            seed_abstract = self.reference_graph.nodes[seed_pid].get("abstract", "")

            for related_pid in related_pids:
                hash_key = get_hash(f"{seed_pid}||{related_pid}")
                if hash_key in self.relatedness_cache:
                    self.logger.info(
                        f"Relatedness cache hit for seed {seed_pid} and related {related_pid}."
                    )
                else:
                    related_title = self.reference_graph.nodes[related_pid].get(
                        "title", ""
                    )
                    related_abstract = self.reference_graph.nodes[related_pid].get(
                        "abstract", ""
                    )

                    prompt = PAPER_RELATEDNESS_BASED_ON_TITLE_AND_ABSTRACT.format(
                        seed_title=seed_title,
                        seed_abstract=seed_abstract,
                        candidate_title=related_title,
                        candidate_abstract=related_abstract,
                    )
                    tasks.append(
                        (hash_key, seed_pid, related_pid, prompt)
                    )  # store hash_key for caching
        if tasks:
            prompts = [task[3] for task in tasks]
            responses = self.chat_agent.batch_remote_chat(
                prompts,
                temperature=self.config.ModuleInfo.WorkCollector.relatedness_temperature,
                desc="Computing relatedness scores",
            )
            for i, response in enumerate(responses):
                hash_key, seed_pid, related_pid, _ = tasks[i]
                # save to cache
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
                    }  # default to 0.0 on error

        # filter based on LLM relatedness score
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
        """
        Collect related papers based on the given seed paper IDs.

        Returns a list of related work items.
        """
        # step 1: expand reference graph
        self.update_reference_graph(seed_paper_ids)

        # step 2: compute relatedness scores according to title + abstract and filter out low-relatedness papers
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

        valid_expanded_paper_ids = self.download_and_parse_papers(related_paper_ids)
        self.graph_paper_ids.update(valid_expanded_paper_ids)
        self.logger.info(f"valid RAG paper ids sources num: {len(self.graph_paper_ids)}")
        return valid_expanded_paper_ids
