"""
DataManager: 负责数据读写、缓存管理和API调用
从WorkCollector中分离出来，供其他模块（如PaperGraphRetriever）复用
"""
from typing import Any, Dict, List, Optional
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.api_call import SemanticScholarAPI, ArxivAPI
from utils.rich_logger import get_logger
from utils.mineru_utils import parse_doc
from utils.utils import get_hash, is_valid_pdf
import requests
from contextlib import closing
import diskcache as dc
import torch
import gc
import time
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils.gpu_utils import load_sentence_transformer_auto


class DataManager:
    """数据管理类：负责paper的下载、解析、缓存和API查询"""
    
    def __init__(self, config):
        self.config = config
        self.cache_path = self.config.BasicInfo.cache_path
        self.logger = get_logger("DataManager")
        
        # 初始化APIs
        self.semantic_scholar_api = SemanticScholarAPI(config)
        self.arxiv_api = ArxivAPI(config)
        
        # 初始化缓存
        self.paper_abstract_cache = dc.Cache(
            os.path.join(self.cache_path, "paper_abstracts")
        )
        
        # title to paper_id lookup cache (for filter_papers_local_paper_graph)
        self.title_lookup_cache = dc.Cache(
            os.path.join(self.cache_path, "title_lookup_cache")
        )
        
        # embedding model (lazy loading)
        self.embedding_model = None
        self._model_device = None

    def _get_embedding_model(self):
        """Lazy load and cache the embedding model."""
        if self.embedding_model is not None:
            return self.embedding_model
        
        model_name = self.config.ModuleInfo.WorkCollector.sentence_transformer_model
        try:
            self.embedding_model, self._model_device = load_sentence_transformer_auto(
                model_name,
                logger=self.logger,
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                self.logger.error("Out of memory error detected. Using CPU instead.")
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    self.logger.warning("Failed to clear GPU cache.")
                    pass
                self.embedding_model, self._model_device = load_sentence_transformer_auto(
                    model_name,
                    logger=self.logger,
                )
            else:
                try:
                    self.embedding_model, self._model_device = load_sentence_transformer_auto(
                        model_name,
                        logger=self.logger,
                    )
                except Exception as e2:
                    self.logger.error(f"Failed to load SentenceTransformer model: {e2}")
                    raise e2
        
        return self.embedding_model

    @staticmethod
    def _resolve_paper_reference_id(paper_ref: Any) -> str:
        """Normalize a paper lookup result into the cache/download paper id."""
        if isinstance(paper_ref, str):
            return paper_ref.strip()

        if not isinstance(paper_ref, dict):
            return ""

        external_ids = paper_ref.get("externalIds")
        if isinstance(external_ids, dict):
            for key in ("ArXiv", "arXiv", "ARXIV"):
                value = external_ids.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()

        for key in ("paperId", "paper_id", "id", "corpusId"):
            value = paper_ref.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        return ""

    def is_valid_abstract(self, abstract: str) -> bool:
        if not isinstance(abstract, str):
            return False
        if not abstract or abstract.strip() == "" or abstract == "abstract not found" or len(abstract) < 50:
            return False
        if len(abstract) < 300 and len(abstract) > 50:
            self.logger.warning(f"abstract too short, deug for safety: {abstract}")
        return True

    def add_papers_abstracts_in_cache(self, papers: List[str], retry: int = 1):
        """获取并缓存paper的摘要信息"""
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
                    abstract = paper.get("abstract", "") or ""
                    if not abstract:
                        raise ValueError(f"Failed to get abstract for {query_id} in semantic scholar, turn to arxiv")
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

    def get_paper_title_abstract(self, paper_id: str, retry: int = 1):
        """获取paper的title和abstract"""
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
        """获取paper的title"""
        hash_id = get_hash(paper_id)
        # already cached
        if hash_id in self.paper_abstract_cache:
            if self.config.BasicInfo.debug:
                self.logger.info(f"Cache hit for paper {paper_id} title, title: {self.paper_abstract_cache[hash_id]['title']}")
            return self.paper_abstract_cache[hash_id]['title']

        self.add_papers_abstracts_in_cache([paper_id], retry=retry)
        if hash_id in self.paper_abstract_cache:
            if self.config.BasicInfo.debug:
                self.logger.info(f"Fetched and cached title for paper {paper_id}, len: {len(self.paper_abstract_cache[hash_id]['title'])}")
            return self.paper_abstract_cache[hash_id]['title']
        else:
            raise ValueError(f"Failed to get valid abstract for paper ID {paper_id} after {retry} retries.")

    def _prepare_download_info(self, paper, reference_graph=None):
        """准备下载信息，返回(paper_id, download_urls, pdf_path, paper_title, is_arxiv)或None"""
        download_urls = []
        is_arxiv = False
        paper_id = None
        paper_title = None
        
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
            else:
                return None
            paper_title = paper.get("title", paper_id)
            
        elif isinstance(paper, str):
            is_arxiv = "." in paper
            if is_arxiv:
                paper_id = paper
                download_urls.append(f"https://export.arxiv.org/pdf/{paper_id}.pdf")
                download_urls.append(f"https://arxiv.org/pdf/{paper_id}.pdf")
                paper_title = paper_id
            else:
                return None
        else:
            return None
        
        pdf_path = os.path.join(
            self.cache_path,
            "pdf_papers",
            paper_id,
            f"{paper_id}.pdf",
        )
        
        return (paper_id, download_urls, pdf_path, paper_title, is_arxiv)

    def _download_single_paper(self, paper, index, total, reference_graph=None):
        """下载单个paper，返回(paper_id, pdf_path)或(None, None)"""
        # 准备下载信息
        info = self._prepare_download_info(paper, reference_graph)
        if info is None:
            return (None, None)
        
        paper_id, download_urls, pdf_path, paper_title, is_arxiv = info
        
        # 检查缓存
        if (
            not self.config.ModuleInfo.WorkCollector.download_safe_mode
            and os.path.exists(pdf_path)
        ):
            if is_valid_pdf(pdf_path):
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Cache Hit! Existing PDF at {pdf_path} is valid, skipping download.")
                return (paper_id, pdf_path)
            else:
                try:
                    os.remove(pdf_path)
                except OSError as e:
                    self.logger.error(f"Failed to delete invalid PDF {pdf_path}: {e}")
                self.logger.warning(f"Existing PDF at {pdf_path} is invalid, re-downloading.")
        
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        
        # 尝试下载
        downloaded = False
        for download_url in download_urls:
            downloaded = self._download_pdf_with_resume(
                url=download_url,
                filename=pdf_path,
                title=f"{paper_title} ({index}/{total})",
            )
            if not downloaded or not is_valid_pdf(pdf_path):
                self.logger.warning(
                    f"Failed to download valid paper {paper_id} from {download_url}, trying next URL if available."
                )
                continue
            else:
                break
        
        if not downloaded or not is_valid_pdf(pdf_path):
            self.logger.warning(f"Failed to download valid paper {paper_id}")
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
            return (None, None)
        
        return (paper_id, pdf_path)

    def _download_papers_serial(self, papers: list, limit: int = -1):
        """串行下载papers"""
        valid_paper_ids = []
        valid_paper_paths = []
        valid_paper_paths_ids = []
        
        reference_graph = self._get_reference_graph()
        
        index = 0
        total = min(len(papers), limit) if limit > 0 else len(papers)
        
        while index < len(papers) and (limit < 0 or len(valid_paper_ids) < limit):
            paper = papers[index]
            index += 1
            
            # 处理abstract_when_full_text_fail的情况
            if isinstance(paper, dict):
                paper_id = paper.get("externalIds", {}).get("ArXiv") or paper.get("paperId")
                if not paper.get("openAccessPdf", {}).get("url") and not ("ArXiv" in paper.get("externalIds", {})):
                    if self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                        self.logger.info(f"No full text PDF found for paper {paper_id}, skipping download but keeping abstract.")
                        err = self.add_papers_abstracts_in_cache([paper_id])
                        if not err:
                            valid_paper_ids.append(paper_id)
                        continue
                    else:
                        continue
            
            paper_id, pdf_path = self._download_single_paper(paper, index, total, reference_graph)
            
            if paper_id and pdf_path:
                valid_paper_ids.append(paper_id)
                if not os.path.exists(os.path.join(self.cache_path, "parsed_papers", paper_id)):
                    valid_paper_paths.append(pdf_path)
                    valid_paper_paths_ids.append(paper_id)
        
        return valid_paper_ids, valid_paper_paths, valid_paper_paths_ids

    def _download_papers_parallel(self, papers: list, limit: int = -1):
        """并行下载papers"""
        valid_paper_ids = []
        valid_paper_paths = []
        valid_paper_paths_ids = []
        
        reference_graph = self._get_reference_graph()
        
        # 过滤需要下载的papers
        download_tasks = []
        limit = limit if limit > 0 else len(papers)
        
        for idx, paper in enumerate(papers[:limit]):
            # 处理特殊情况
            if isinstance(paper, dict):
                paper_id = paper.get("externalIds", {}).get("ArXiv") or paper.get("paperId")
                if not paper.get("openAccessPdf", {}).get("url") and not ("ArXiv" in paper.get("externalIds", {})):
                    if self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                        self.logger.info(f"No full text PDF found for paper {paper_id}, skipping download but keeping abstract.")
                        err = self.add_papers_abstracts_in_cache([paper_id])
                        if not err:
                            valid_paper_ids.append(paper_id)
                        continue
                    else:
                        continue
            elif isinstance(paper, str) and "." not in paper:
                continue
            
            download_tasks.append((idx + 1, paper))
        
        if not download_tasks:
            return valid_paper_ids, valid_paper_paths, valid_paper_paths_ids
        
        total = len(download_tasks)
        max_workers = getattr(self.config.ModuleInfo.WorkCollector, 'download_parallel_workers', 8)
        
        self.logger.info(f"Starting parallel download with {max_workers} workers for {len(download_tasks)} papers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_single_paper, paper, idx, total, reference_graph): idx
                for idx, paper in download_tasks
            }
            
            for future in as_completed(futures):
                try:
                    paper_id, pdf_path = future.result()
                    if paper_id and pdf_path:
                        valid_paper_ids.append(paper_id)
                        if not os.path.exists(os.path.join(self.cache_path, "parsed_papers", paper_id)):
                            valid_paper_paths.append(pdf_path)
                            valid_paper_paths_ids.append(paper_id)
                except Exception as e:
                    self.logger.error(f"Error in parallel download: {e}")
        
        return valid_paper_ids, valid_paper_paths, valid_paper_paths_ids

    def _download_pdf_with_resume(self, url: str, filename: str, title: str, is_arxiv: bool = False, chunk_size: int = 1024 * 1024):
        """下载PDF文件，支持断点续传"""
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
            with closing(requests.get(url, headers=headers, stream=True, timeout=self.config.ModuleInfo.WorkCollector.download_timeout)) as resp:

                if resp.status_code not in (200, 206, 416):
                    self.logger.error(f"[{title}] Could not download file: {resp.status_code}")
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
                    self.logger.warning(f"[{title}] Total file size unknown. Progress may not be shown.")

                with open(filename, "ab") as f:
                    downloaded = temp_size

                    last_time = time.time()
                    last_downloaded = downloaded

                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            now = time.time()
                            elapsed = now - last_time
                            if elapsed > 0:
                                delta_bytes = downloaded - last_downloaded
                                speed_mb_s = delta_bytes / elapsed / 1024 / 1024
                            else:
                                speed_mb_s = 0.0

                            last_time = now
                            last_downloaded = downloaded

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

    def download_and_parse_papers(self, papers: list, limit: int = -1):
        """下载并解析papers"""
        # 检查是否启用并行下载
        use_parallel = getattr(self.config.ModuleInfo.WorkCollector, 'download_in_parallel', False)
        
        if use_parallel:
            self.logger.info("Using parallel download mode")
            valid_paper_ids, valid_paper_paths, valid_paper_paths_ids = self._download_papers_parallel(papers, limit)
        else:
            self.logger.info("Using serial download mode")
            valid_paper_ids, valid_paper_paths, valid_paper_paths_ids = self._download_papers_serial(papers, limit)

        # step 2: parse the downloaded PDFs
        self.logger.info(f"Parsing {len(valid_paper_paths)} downloaded papers...")
        if valid_paper_paths:
            try:
                parse_doc(
                    valid_paper_paths,
                    output_dir=os.path.join(self.cache_path, "parsed_papers"),
                    lang="en",
                )
            except Exception as e:
                for paper_path in valid_paper_paths:
                    try:
                        parse_doc(
                            [paper_path],
                            output_dir=os.path.join(self.cache_path, "parsed_papers"),
                            lang="en",
                        )
                    except Exception as e2:
                        self.logger.error(f"Failed to parse paper at {paper_path}: {e2}")
                        valid_paper_ids.remove(
                            valid_paper_paths_ids[valid_paper_paths.index(paper_path)]
                        )

        return valid_paper_ids

    def get_paper_raw_markdown(self, paper_id: Any) -> str:
        resolved_paper_id = self._resolve_paper_reference_id(paper_id)
        if not resolved_paper_id:
             raise ValueError("Paper id is empty or none when get_paper_raw_markdown")
        md_path = os.path.join(
            f"{self.cache_path}/parsed_papers",
            resolved_paper_id,
            "auto",
            f"{resolved_paper_id}.md",
        )
        if not os.path.exists(md_path):
            if self.config.BasicInfo.debug:
                self.logger.info(
                    f"Paper {resolved_paper_id} markdown not found in cache, re-downloading and parsing..."
                )
            try:
                self.download_and_parse_papers([paper_id if isinstance(paper_id, dict) else resolved_paper_id])
            except Exception as e:
                self.logger.error(f"Failed to parse paper {resolved_paper_id}: {e}")
                raise e
        if not os.path.exists(md_path):
            self.logger.warning(f"Markdown still missing after parse: {md_path}")
            raise ValueError("Markdown file missing in getting paper markdown")
        with open(md_path, "r", encoding="utf-8") as fr:
            paper_markdown_text = fr.read()
        return paper_markdown_text

    def get_paper_with_title_semantic(self, title: str):
        """通过Semantic Scholar API根据title搜索paper"""
        self.logger.info(f"Searching for paper with title: {title}")
        fields = "title,externalIds,openAccessPdf,abstract,authors,year,venue"
        search_results = []
        
        try:
            response = self.semantic_scholar_api.search_papers(query=title, fields=fields)
            if response and response.get("data"):
                search_results.extend(response["data"][:3])
                self.logger.info(f"Found {len(response['data'])} papers from Semantic Scholar")
        except Exception as e:
            self.logger.warning(f"Error searching Semantic Scholar for '{title}': {e}")
        
        if not search_results:
            self.logger.warning(f"No papers found for title: {title}")
            return None
        
        return self._select_best_paper_by_similarity(title, search_results)

    def get_paper_with_title_arxiv(self, title: str):
        """通过ArXiv API根据title搜索paper"""
        self.logger.info(f"Searching for paper with title: {title}")
        
        try:
            search_results = self.arxiv_api.search_papers_by_title(title)
            if search_results:
                self.logger.info(f"Found {len(search_results)} papers from arXiv for title: {title}")
                return self._select_best_paper_by_similarity(title, search_results)
        except Exception as e:
            self.logger.warning(f"Error searching arXiv for '{title}': {e}")
        
        self.logger.warning(f"No papers found for title: {title}")
        return None

    def get_paper_with_title(self, title: str):
        """根据title搜索paper，优先使用Semantic Scholar，失败则用ArXiv"""
        normalized = title.strip().lower()
        if normalized in self.title_lookup_cache:
            cached = self.title_lookup_cache[normalized]
            self.logger.info(f"Cache hit for title: {title[:50]}...")
            if cached["found"]:
                return cached["paper_info"]
            
        result = self.get_paper_with_title_semantic(title)
        if result:
            result["api_platform"] = "semantic"
        if not result:
            self.logger.info("Fail to retrieve out with semantic scholar, use arxiv instead")
            result = self.get_paper_with_title_arxiv(title)
            if result:
                result["api_platform"] = "semarxivantic"

        if result:
            self.title_lookup_cache[normalized] = {"found": True, "paper_info": result}
        return result

    def get_paper_with_title_batch(self, titles: List[str]):
        """
        批量根据title搜索paper，优先使用Semantic Scholar，失败则用ArXiv
        统一计算embedding和相似度，显著提升性能
        
        使用 title_lookup_cache 缓存查询结果，避免重复API调用。
        缓存格式：title (lowercase, stripped) -> {"found": bool, "paper_info": dict or None}
        
        Args:
            titles: 论文标题列表
            
        Returns:
            Dict[str, dict]: title -> paper_info 的映射，未找到的title对应None
        """
        if not titles:
            return {}
        
        self.logger.info(f"Batch searching for {len(titles)} papers...")
        
        # Normalize titles for cache lookup
        normalized_titles = {title: title.strip().lower() for title in titles}
        
        # Separate titles into cache hits and misses
        titles_to_query = []
        results = {}
        cache_misses = []
        
        for title in titles:
            normalized = normalized_titles[title]
            if normalized in self.title_lookup_cache:
                cached = self.title_lookup_cache[normalized]
                self.logger.info(f"Cache hit for title: {title[:50]}...")
                if cached["found"]:
                    results[title] = cached["paper_info"]
                # else:
                #     results[title] = None
            else:
                titles_to_query.append(title)
                cache_misses.append(normalized)
        
        self.logger.info(f"Cache hits: {len(titles) - len(titles_to_query)}, Cache misses: {len(titles_to_query)}")
        
        if not titles_to_query:
            return results
        
        # 批量搜索Semantic Scholar
        semantic_results = {}  # title -> list of search results
        for title in titles_to_query:
            try:
                fields = "title,externalIds,openAccessPdf,abstract,authors,year,venue"
                response = self.semantic_scholar_api.search_papers(query=title, fields=fields)
                time.sleep(3)
                if response and response.get("data"):
                    semantic_results[title] = response["data"][:5]  # 取前5个结果
            except Exception as e:
                self.logger.warning(f"Error searching Semantic Scholar for '{title}': {e}")
                semantic_results[title] = []
        
        # 批量搜索ArXiv作为fallback
        arxiv_results = {}  # title -> list of search results

        for title in titles_to_query:
            if semantic_results.get(title):
                arxiv_results[title] = []
                continue
            try:
                search_results = self.arxiv_api.search_papers_by_title(title)
                arxiv_results[title] = search_results[:3] if search_results else []
            except Exception as e:
                self.logger.warning(f"Error searching arXiv for '{title}': {e}")
                arxiv_results[title] = []
        
        # 批量计算embedding和相似度
        model = self._get_embedding_model()
        
        # 收集所有需要计算相似度的title
        all_titles_to_encode = []
        title_to_results = {}  # query_title -> [(paper_info, source), ...]
        
        for title in titles_to_query:
            title_to_results[title] = []

            for paper in semantic_results.get(title, [])[:3]:
                title_to_results[title].append((paper, "semantic"))

            for paper in arxiv_results.get(title, [])[:2]:
                title_to_results[title].append((paper, "arxiv"))
            
            # 收集所有需要encode的title
            all_titles_to_encode.append(title)
            for paper, _ in title_to_results[title]:
                all_titles_to_encode.append(paper.get("title", ""))
        
        # 批量编码所有title
        self.logger.info(f"Batch encoding {len(all_titles_to_encode)} titles...")
        all_embeddings = model.encode(
            all_titles_to_encode,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=True
        )
        
        # 解析embeddings
        embeddings_dict = {}
        idx = 0
        for title in all_titles_to_encode:
            embeddings_dict[title] = all_embeddings[idx]
            idx += 1
        
        # 批量计算相似度并选择最佳匹配，同时更新缓存
        for title in titles_to_query:
            normalized = normalized_titles[title]
            query_embedding = embeddings_dict[title]
            best_paper = None
            best_similarity = 0.0
            best_source = None
            
            for paper, source in title_to_results[title]:
                paper_title = paper.get("title", "")
                if not paper_title or paper_title not in embeddings_dict:
                    continue
                
                paper_embedding = embeddings_dict[paper_title]
                similarity = util.pytorch_cos_sim(query_embedding, paper_embedding).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_paper = paper
                    best_source = source
                
                if similarity > 0.95:
                    self.logger.info(f"Paper: {paper_title[:50]}... Similarity: {similarity:.4f}")
                
                if similarity == 1.0:
                    break
            
            if best_paper and best_similarity > 0.95:
                self.logger.info(f"Found matching paper for '{title}' with similarity {best_similarity:.4f}: {best_paper.get('title', 'N/A')}")
                best_paper["api_platform"] = best_source
                results[title] = best_paper
                # Cache the found result
                self.title_lookup_cache[normalized] = {"found": True, "paper_info": best_paper}
            else:
                self.logger.warning(f"No paper found with similarity > 0.95 for title: {title}")
                results[title] = None
                # Cache the "not found" result to avoid repeated lookups
                # self.title_lookup_cache[normalized] = {"found": False, "paper_info": None}
        
        return results

    def _select_best_paper_by_similarity(self, query_title: str, search_results: List[dict]):
        """根据embedding相似度选择最佳匹配的paper"""
        model = self._get_embedding_model()
        query_embedding = model.encode(query_title, convert_to_tensor=True)
        
        best_paper = None
        best_similarity = 0.0
        
        for paper in search_results[:3]:
            paper_title = paper.get("title", "")
            if not paper_title:
                continue
            
            paper_embedding = model.encode(paper_title, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, paper_embedding).item()
            self.logger.info(f"Paper: {paper_title[:50]}... Similarity: {similarity:.4f}")

            if similarity > 0.95 and similarity > best_similarity:
                best_similarity = similarity
                best_paper = paper
            if similarity == 1.0:
                best_paper = paper
                break
        
        if best_paper:
            self.logger.info(f"Found matching paper with similarity {best_similarity:.4f}: {best_paper.get('title', 'N/A')}")
            return best_paper
        else:
            self.logger.warning(f"No paper found with similarity > 0.95 for title: {query_title}")
            return None

    def _get_reference_graph(self):
        """获取reference graph（如果存在）"""
        import pickle
        ref_graph_path = os.path.join(self.cache_path, "reference_graph.pkl")
        if os.path.exists(ref_graph_path):
            with open(ref_graph_path, "rb") as reader:
                return pickle.load(reader)
        return None

    def clear_title_lookup_cache(self):
        """清除title lookup缓存"""
        self.title_lookup_cache.clear()
        self.logger.info("Title lookup cache cleared.")
        
    def get_title_lookup_cache_stats(self):
        """获取title lookup缓存的统计信息"""
        return {
            "size": len(self.title_lookup_cache),
            "cache_path": os.path.join(self.cache_path, "title_lookup_cache")
        }
