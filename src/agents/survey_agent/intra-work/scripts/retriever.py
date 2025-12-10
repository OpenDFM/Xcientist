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


API_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = ",".join([
    "title", "year", "venue", "url",
    "isOpenAccess", "openAccessPdf", "externalIds", "authors"
])

def parse_arguments():
    parser = argparse.ArgumentParser(description="argments for retrieve paper")

    parser.add_argument('-p', '--base_dir', type=str, default="/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work", help="input pdf path")
    parser.add_argument('-t', '--task_id', type=str, default="semantic_scholar_test", help="input pdf path")
    parser.add_argument('-q', '--query', type=str, default="AI automatic overview/survey generation", help="input pdf path")
    parser.add_argument('-k', '--api_key', type=str, default="1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ", help="output markdown path")
    parser.add_argument('-m', '--max', type=int, default=5, help="output markdown path")

    return parser.parse_args()


def make_session(timeout=30, retries=3, backoff_factor=0.5):
    sess = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429,500,502,503,504),
        allowed_methods=frozenset(['GET'])
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    return sess

def sanitize_string(v: str, replace_with='?') -> str:
    if not isinstance(v, str):
        v = str(v)
    v = unicodedata.normalize('NFKD', v)
    # replace char that latin-1 cannot encode
    out_chars = []
    for ch in v:
        try:
            ch.encode('latin-1')
            out_chars.append(ch)
        except UnicodeEncodeError:
            out_chars.append(replace_with)
    return ''.join(out_chars)

def safe_filename(s: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in s)[:200]

def download_file(session, url, dest_path, timeout=60):
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            total = None
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                total = int(cl)

            with open(dest_path, "wb") as f:
                if total:
                    pbar = tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name)
                else:
                    pbar = tqdm(unit='B', unit_scale=True, desc=dest_path.name)

                for chunk in r.iter_content(chunk_size=1<<14):
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

def search_and_download(query,
                        out_dir="papers",
                        max_results=10,
                        api_key=None,
                        sleep_between_requests=1.0,
                        max_retry=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    headers = {"User-Agent": "ss‐pdf‐downloader/1.0"}
    headers = {k: sanitize_string(v, replace_with='-') for k, v in headers.items()}
    if api_key:
        headers["x-api-key"] = api_key

    search_results = max_results*3
    params = {
        "query": query,
        "limit": min(search_results, 100),
        "fields": DEFAULT_FIELDS,
        "openAccessPdf": True  # filter for papers with open access PDF only
    }

    print(f"Searching for: {query!r} (max {max_results})")
    resp = session.get(API_SEARCH, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("data", [])
    if not results:
        print("No results found.")
        return

    count = 0
    for paper in results:
        # print("DEBUG: complete paper: ", paper)
        if count >= max_results:
            break
        title = paper.get("title") or f"paper_{count}"
        url_pdf = None
        ex_id = paper.get('externalIds')
        arxiv_ok = False
        if 'ArXiv' in ex_id:
            arxiv_id = ex_id['ArXiv']
            arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
            print("DEBUG url: ", arxiv_url)
            print(f"[{count+1}] Downloading PDF from arxiv: {title}")
            fname = safe_filename(title) + ".pdf"
            dest = out_dir / fname
            for try_times in range(max_retry):
                arxiv_ok, msg = download_file(session, arxiv_url, dest)
                if arxiv_ok:
                    break
                else:
                    print(f"Fail, retrying {try_times} times...")
            if arxiv_ok:
                print(f" Saved: {dest.name}")
                count += 1
            else:
                print(f" Failed download from arxiv: {msg}, try oap")
        if 'ArXiv' not in ex_id or not arxiv_ok:
            oap = paper.get("openAccessPdf")
            if oap and isinstance(oap, dict):
                url_pdf = oap.get("url")
            if not url_pdf:
                # skip if no direct PDF link
                print(f"[{count+1}] {title} → no openAccessPdf link, skip")
                continue

            fname = safe_filename(title) + ".pdf"
            dest = out_dir / fname
            print(f"[{count+1}] Downloading PDF from oap link: {title}")
            for try_times in range(max_retry):
                ok, msg = download_file(session, url_pdf, dest)
                if ok:
                    break
                else:
                    print(f"Fail, retrying {try_times} times...")
            if ok:
                print(f"    Saved: {dest.name}")
                count += 1
            else:
                print(f"    Failed download: {msg}")
            time.sleep(sleep_between_requests)

    print(f"Finished. Downloaded {count} PDFs into '{out_dir.resolve()}'")

if __name__ == "__main__":
    args = parse_arguments()
    QUERY = args.query
    OUT_DIR = f"{args.base_dir}/outputs/pipeline/{args.task_id}/references"
    MAX = args.max
    API_KEY = args.api_key

    os.makedirs(OUT_DIR)
    search_and_download(QUERY, out_dir=f"{OUT_DIR}/pdfs", max_results=MAX, api_key=API_KEY, sleep_between_requests=1.0)

