import requests
import time
import os
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import math
import argparse
import unicodedata
import yaml
import sys

from utils.read_yaml import load_config
from utils.api_call import ChatAgent

from utils.chat_utils import (
    load_prompt,
    cut_text_by_token,
    clean_chat_agent_format,
    load_file_as_string,
    sanitize_filename,
    save_result,
)

API_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = ",".join(
    [
        "title",
        "year",
        "venue",
        "url",
        "paperId",
        "isOpenAccess",
        "openAccessPdf",
        "externalIds",
        "authors",
    ]
)


class TopicRetriever:
    def __init__(self, config):
        self.api_config = config.APIInfo
        self.config = config.Modules.Retriever

        self.task_id = config.BasicInfo.task_id
        self.query = config.BasicInfo.topic
        self.max = self.config.max
        self.api_key = self.api_config.semantic_scholar_api_key

        self.output_path = f"{config.BasicInfo.output_path}/{self.task_id}"
        self.debug_output_path = f"{self.output_path}/references/pdfs"

        self.use_html = config.BasicInfo.use_html

        self.md_cache_path = config.BasicInfo.paper_cache_path
        self.pdf_cache_path = config.BasicInfo.pdf_cache_path
        self.html_cache_path = config.BasicInfo.html_cache_path
        self.prompt_path = config.BasicInfo.prompt_path

        self.keywords = []
        self.chat_agent = ChatAgent(config)
        self.use_kw = self.config.use_kw

    def make_session(self, timeout=30, retries=3, backoff_factor=0.5):
        sess = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        sess.mount("http://", HTTPAdapter(max_retries=retry))
        return sess

    def sanitize_string(self, v: str, replace_with="?") -> str:
        if not isinstance(v, str):
            v = str(v)
        v = unicodedata.normalize("NFKD", v)
        # replace char that latin-1 cannot encode
        out_chars = []
        for ch in v:
            try:
                ch.encode("latin-1")
                out_chars.append(ch)
            except UnicodeEncodeError:
                out_chars.append(replace_with)
        return "".join(out_chars)

    def get_keywords(self, query: str = None):
        # FIXME
        self.keywords = [query]
        # if query is None:
        #     query = self.query
        # kw_prompt = load_prompt(
        #     file_path=f"{self.prompt_path}/multi_stage/get_keywords.md",
        #     topic=self.query,
        # )
        # resp = self.chat_agent.remote_chat(text_content=kw_prompt)

        # self.keywords = [kw.strip() for kw in resp.split(",") if kw.strip()]

    # def safe_filename(self, s: str) -> str:
    #     keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    #     return "".join(c if c in keep else "_" for c in s)[:200]

    def download_file(self, session, url, dest_path, timeout=60):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                total = None
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    total = int(cl)

                with open(dest_path, "wb") as f:
                    if total:
                        pbar = tqdm(
                            total=total, unit="B", unit_scale=True, desc=dest_path.name
                        )
                    else:
                        pbar = tqdm(unit="B", unit_scale=True, desc=dest_path.name)

                    for chunk in r.iter_content(chunk_size=1 << 14):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                    pbar.close()

            if os.path.getsize(dest_path) < 2000:
                return False, f"file too small ({os.path.getsize(dest_path)} bytes)"
            with open(dest_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF-"):
                    return False, "not a PDF file (magic header mismatch)"
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def send_requirement(self, query=None, max_results=None, api_key=None):

        session = self.make_session()
        headers = {"User-Agent": "ss‐pdf‐downloader/1.0"}
        headers = {
            k: self.sanitize_string(v, replace_with="-") for k, v in headers.items()
        }
        if api_key:
            headers["x-api-key"] = api_key
        search_results = max_results * 3
        params = {
            "query": query,
            "limit": min(search_results, 300),
            "fields": DEFAULT_FIELDS,
            "openAccessPdf": True,  # filter for papers with open access PDF only
        }
        print(f"Searching for: {query!r} (max {max_results})")
        resp = session.get(API_SEARCH, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("data", [])
        if not results:
            print("No results found.")
            return
        return results

    def download_and_store(
        self,
        resp=None,
        out_dir=None,
        max_results=None,
        cache_dir=None,
        sleep_between_requests=1.0,
        max_retry=2,
    ):
        session = self.make_session()
        if resp is None:
            print("ERR: NO RESP from semantic scholar api, return")
            return

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        paper_retrieved_log = ""
        paper_list = []
        download_pdf = []

        for paper in resp:
            if count >= max_results:
                break
            title = paper.get("title") or f"paper_{count}"
            url_pdf = None
            semantic_id = paper.get("paperId")
            ex_id = paper.get("externalIds")
            arxiv_id = ex_id.get("ArXiv", None)
            paper_id = semantic_id if arxiv_id is None else arxiv_id

            if os.path.exists(f"{cache_dir}/{paper_id}.md"):
                print(f"Find {title} in cache, skip downloading")
                count += 1
                paper_list.append(paper_id)
                paper_retrieved_log += title + "\n"
                continue

            arxiv_ok = False
            if "ArXiv" in ex_id:
                arxiv_id = ex_id["ArXiv"]
                arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
                print("DEBUG url: ", arxiv_url)
                print(f"[{count+1}] Downloading PDF from arxiv: {title}")
                fname = arxiv_id + ".pdf"
                dest = out_dir / fname
                for try_times in range(max_retry):
                    arxiv_ok, msg = self.download_file(session, arxiv_url, dest)
                    if arxiv_ok:
                        break
                    else:
                        print(f"Fail, retrying {try_times} times...")
                if arxiv_ok:
                    print(f" Saved: {dest.name}")
                    paper_retrieved_log += title + "\n"
                    paper_list.append(paper_id)
                    download_pdf.append(dest)
                    count += 1
                else:
                    print(f" Failed download from arxiv: {msg}, try oap")
            if "ArXiv" not in ex_id or not arxiv_ok:
                oap = paper.get("openAccessPdf")
                if oap and isinstance(oap, dict):
                    url_pdf = oap.get("url")
                if not url_pdf:
                    # skip if no direct PDF link
                    print(f"[{count+1}] {title} → no openAccessPdf link, skip")
                    continue

                fname = semantic_id + ".pdf"
                dest = out_dir / fname
                print(f"[{count+1}] Downloading PDF from oap link: {title}")
                for try_times in range(max_retry):
                    ok, msg = self.download_file(session, url_pdf, dest)
                    if ok:
                        break
                    else:
                        print(f"Fail, retrying {try_times} times...")
                if ok:
                    print(f"    Saved: {dest.name}")
                    paper_retrieved_log += title + "\n"
                    paper_list.append(paper_id)
                    download_pdf.append(dest)
                    count += 1
                else:
                    print(f"    Failed download: {msg}")
                time.sleep(sleep_between_requests)

        # print(paper_retrieved_log)
        # with open(f"{self.output_path}/retrieved_paper.txt", "w") as f:
        #     f.write(paper_retrieved_log)
        print(f"Finished. Downloaded {count} PDFs into '{out_dir.resolve()}'")
        return paper_list, download_pdf

    def _looks_like_html(self, content_bytes: bytes) -> bool:
        try:
            start = content_bytes[:512].lower()
            return b"<html" in start or b"<!doctype html" in start or b"<body" in start
        except Exception:
            return False

    def download_arxiv_html_from_resp(
        self,
        resp=None,
        out_dir=None,
        max_results=None,
        cache_dir=None,
        timeout=20,
        sleep_between_requests=0.5,
        max_retry=2,
        min_size_bytes=1000,
    ):
        """
        - try arXiv HTML(/html/ID) first, then  ar5iv(/html/ID)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        session = self.make_session()

        # 安全的 ASCII user-agent
        headers = {"User-Agent": "ss-html-downloader/1.0"}

        saveds = []
        html_dest_paths = []
        attempted = 0

        for paper in resp:
            if len(saveds) >= max_results:
                break

            # externalIds may be missing or None
            ex = paper.get("externalIds") or {}
            arxiv_id = ex.get("ArXiv") or ex.get("arXiv")

            if not arxiv_id:
                continue

            if os.path.exists(f"{cache_dir}/{arxiv_id}.html"):
                print(f"Cache Hit for {arxiv_id}, skip downloading")
                saveds.append(arxiv_id)
                html_dest_paths.append(f"{cache_dir}/{arxiv_id}.html")
                continue

            # construct candidate urls (order = prefer official html -> ar5iv -> abs)
            candidates = [
                f"https://arxiv.org/html/{arxiv_id}",
                f"https://ar5iv.org/html/{arxiv_id}",
            ]

            title = paper.get("title") or arxiv_id
            saved_this = False

            for url in candidates:
                try:
                    attempted += 1
                    r = session.get(url, headers=headers, timeout=timeout)
                    # accept 200 only
                    if r.status_code != 200:
                        # some sites may redirect; requests auto-follows by default
                        # skip non-200
                        # print(f"skip {url} status {r.status_code}")
                        r.close()
                        continue

                    content = r.content
                    r.close()

                    # quick sanity checks: size and looks like html
                    if len(content) < min_size_bytes:
                        # too small, likely an error/placeholder page
                        continue
                    if not self._looks_like_html(content):
                        continue

                    # save
                    filename = arxiv_id + ".html"
                    dest = out_dir / filename
                    with open(dest, "wb") as f:
                        f.write(content)

                    saveds.append(arxiv_id)
                    html_dest_paths.append(str(dest))
                    print(f"[HTML] Saved {title} <- {url} -> {dest.name}")
                    saved_this = True
                    break
                except Exception as e:
                    # don't raise; log and try next candidate
                    print(f"[WARN] failed to fetch {url}: {e}")
                    time.sleep(0.1)
                    continue

            if not saved_this:
                print(f"[SKIP] Failed to obtain HTML for arXiv id {arxiv_id} ({title})")
            time.sleep(sleep_between_requests)

        print(
            f"Done: attempted {attempted} requests, saved {len(saveds)} html files to {out_dir}"
        )
        return saveds, html_dest_paths

    def semantic_search(
        self,
        query=None,
        out_dir=None,
        max_results=None,
        api_key=None,
        cache_dir=None,
        sleep_between_requests=1.0,
        max_retry=2,
    ):

        if query is None:
            query = self.query

        if out_dir is None and not self.use_html:
            out_dir = self.pdf_cache_path
        elif out_dir is None and self.use_html:
            out_dir = self.html_cache_path

        if max_results is None:
            max_results = self.max

        if api_key is None:
            api_key = self.api_key

        if cache_dir is None and not self.use_html:
            cache_dir = self.md_cache_path
        elif cache_dir is None and self.use_html:
            cache_dir = self.html_cache_path

        self.get_keywords(query)

        if self.use_kw:
            kw_query = ""
            for kw in self.keywords:
                kw_query += kw + ", "
            query = kw_query

        resp = self.send_requirement(query, max_results, api_key)
        print(resp)
        exit()
        if self.use_html:
            paper_list, download_pdf = self.download_arxiv_html_from_resp(
                resp, out_dir, max_results, cache_dir
            )
        else:
            paper_list, download_pdf = self.download_and_store(
                resp, out_dir, max_results, cache_dir
            )

        return paper_list, download_pdf


if __name__ == "__main__":
    # args = parse_arguments()
    config = load_config(Path("./config/deep_survey.yaml"))
    # QUERY = args.query
    # OUT_DIR = f"{args.base_dir}/outputs/pipeline/{args.task_id}/references"
    # MAX = args.max
    # API_KEY = args.api_key
    topicRetriever = TopicRetriever(config)

    # os.makedirs(OUT_DIR)
    # topicRetriever.search_and_download(QUERY, out_dir=f"{OUT_DIR}/pdfs", max_results=MAX, api_key=API_KEY, sleep_between_requests=1.0)
    paper_list, pdfs = topicRetriever.semantic_search()
    print("DEBUG pdf: ", paper_list)
    print("DEBUG paper list: ", pdfs)
