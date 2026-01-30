import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from pathlib import Path
from src.agents.survey_agent.utils.rich_logger import get_logger
import tiktoken
import xml.etree.ElementTree as ET


class ArxivAPI:
    def __init__(self, config):
        self.base_url = "http://export.arxiv.org/api/query"
        self.logger = get_logger("ArxivAPI")
        self.config = config

    def get_paper_details(self, paper_id: str):
        arxiv_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        paper = {}
        
        for retry_count in range(self.config.APIInfo.arxiv_api_max_retry):
            response = requests.get(arxiv_url, timeout=10)
            if response.status_code == 200:
                break
            else:
                self.logger.warning(f"arXiv API request failed for {paper_id}: {response.status_code}. Retrying {retry_count + 1}/3...")

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('.//atom:entry', ns)
            if entry is not None:
                paper["title"] = entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else ""
                paper["authors"] = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None]
                published = entry.find('atom:published', ns).text[:4] if entry.find('atom:published', ns) is not None else ""  # Extract year
                paper["venue"] = "arXiv"  # Default venue for arXiv
                paper["year"] = published
                self.logger.info(f"Fetched details from arXiv for {paper_id}")
            else:
                self.logger.warning(f"No entry found in arXiv response for {paper_id}")
                raise ValueError("No entry found in arXiv response in mla generation")
        else:
            self.logger.warning(f"arXiv API request failed for {paper_id}: {response.status_code}")
            raise ValueError("No entry found in arXiv response in mla generation")
        
        return paper

class SemanticScholarAPI:
    def __init__(self, config):
        self.headers = {"x-api-key": config.APIInfo.semantic_scholar_api_key}
        self.base_url = "http://api.semanticscholar.org/graph/v1"
        self.logger = get_logger("SemanticScholarAPI")
        self.config = config

    def _normalize_paper_id(self, paper_id: str) -> str:
        pid = str(paper_id or "").strip()
        if not pid:
            return pid
        lowered = pid.lower()
        if lowered.startswith("arxiv:"):
            return f"ARXIV:{pid.split(':', 1)[1].strip()}"
        # If it already contains a prefix (e.g., DOI:, CorpusId:), keep as-is.
        if ":" in pid:
            return pid
        # Normalize common arXiv IDs like 2505.11711 -> ARXIV:2505.11711
        if pid.count(".") == 1:
            left, right = pid.split(".", 1)
            if left.isdigit() and right.isdigit():
                return f"ARXIV:{pid}"
        return pid

    def search_papers(self, query: str, fields: str, retry_time: int = 0):
        max_retry = self.config.APIInfo.semantic_scholar_api_max_retry
        url = f"{self.base_url}/paper/search"
        params = {"query": query, "fields": fields}

        for attempt in range(retry_time, max_retry + 1):
            try:
                response = requests.get(
                    url, headers=self.headers, params=params, timeout=30
                )
            except requests.RequestException as e:
                self.logger.warning(
                    f"Semantic Scholar request failed ({attempt + 1}/{max_retry + 1}): {e}"
                )
                if attempt < max_retry:
                    time.sleep(min(5 * (attempt + 1), 30))
                    continue
                return None

            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                self.logger.info(
                    "Rate limit exceeded. Waiting 60 seconds before retrying..."
                )
                time.sleep(60)
                continue

            self.logger.error(
                f"Error occurs in search_papers. Status code: {response.status_code}"
            )
            if attempt < max_retry:
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            return None

    def get_paper_details(self, paper_id: str, fields: str, retry_time: int = 0):
        max_retry = self.config.APIInfo.semantic_scholar_api_max_retry
        normalized_id = self._normalize_paper_id(paper_id)
        url = f"{self.base_url}/paper/{normalized_id}?fields={fields}"

        for attempt in range(retry_time, max_retry + 1):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
            except requests.RequestException as e:
                self.logger.warning(
                    f"Semantic Scholar request failed ({attempt + 1}/{max_retry + 1}): {e}"
                )
                if attempt < max_retry:
                    time.sleep(min(5 * (attempt + 1), 30))
                    continue
                raise ValueError(
                    f"Failed to fetch paper details for {paper_id} from Semantic Scholar"
                )

            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                self.logger.info(
                    "Rate limit exceeded. Waiting 60 seconds before retrying..."
                )
                time.sleep(60)
                continue

            self.logger.error(
                "Error occurs in get_paper_details for %s. Status code: %s",
                normalized_id,
                response.status_code,
            )
            if attempt < max_retry:
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            raise ValueError(
                f"Failed to fetch paper details for {paper_id} from Semantic Scholar"
            )


class ChatAgent:
    Record_splitter = "||"
    Record_show_length = 200

    def __init__(self, config, use_different_api_for_judge=False) -> None:
        self.config = config
        self.remote_url = config.APIInfo.llm_api_base_url
        self.token = config.APIInfo.llm_api_key
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.batch_workers = config.APIInfo.batch_chat_agent_worker
        self.model_name = config.APIInfo.llm_model_name
        self.logger = get_logger("ChatAgent")

        if use_different_api_for_judge:
            self.logger.info("Using different LLM API key and URL for Judge module.")
            self.remote_url = config.ModuleInfo.Judge.judge_llm_api_base_url
            self.token = config.ModuleInfo.Judge.judge_llm_api_key
            self.model_name = config.ModuleInfo.Judge.model
            self.header = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=300),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def remote_chat(
        self,
        text_content: str,
        image_urls: list[str] = None,
        local_images: list[Path] = None,
        temperature: float = 0.5,
        debug: bool = False,
        model=None,
    ) -> str:
        """Chat with remote LLM, return result. Minimal logging; no file writes."""
        if model is None:
            model = self.model_name

        url = self.remote_url
        header = self.header
        messages = [{"role": "user", "content": text_content}]

        if image_urls:
            image_url_frame = [
                {"type": "image_url", "image_url": {"url": u}} for u in image_urls
            ]
            messages.append({"role": "user", "content": image_url_frame})

        payload = {"model": model, "messages": messages, "temperature": temperature}

        response = requests.post(url, headers=header, json=payload, timeout=90)

        if self.config.APIInfo.low_flow_mode:
            time.sleep(self.config.APIInfo.low_flow_latency)  # to reduce the API call frequency
            
        if response.status_code != 200:
            self.logger.error(
                f"chat response code: {response.status_code}\n{response.text}, retrying..."
            )
            try:
                self.logger.error(f"Error message: {res['choices'][0]['message']['content']}")
            except Exception:
                pass
            response.raise_for_status()
        try:
            res = response.json()
            res_text = res["choices"][0]["message"]["content"]
        except Exception as e:
            res_text = f"Error: {e}"
            self.logger.error(f"There is an error in remote_chat: {e}")

        if debug:
            return res_text, response
        return res_text

    def __remote_chat(
        self, index, content, temperature: float = 0.5, debug: bool = False
    ):
        model = self.model_name
        return index, self.remote_chat(
            text_content=content,
            image_urls=None,
            local_images=None,
            temperature=temperature,
            debug=debug,
            model=model,
        )

    def batch_remote_chat(
        self,
        prompt_l: list[str],
        desc: str = "batch_chating...",
        workers: int = None,
        temperature: float = 0.5,
        future_timeout: float = 300.0,
    ) -> list[str]:
        if workers is None:
            workers = self.batch_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_l = [
                executor.submit(self.__remote_chat, i, prompt_l[i], temperature)
                for i in range(len(prompt_l))
            ]
            res_l = ["no response"] * len(prompt_l)
            for future in tqdm(
                as_completed(future_l),
                desc=desc,
                total=len(future_l),
                dynamic_ncols=True,
            ):
                try:
                    i, resp = future.result(timeout=future_timeout)
                except Exception as e:
                    # Cancel the hanging future and record the error
                    future.cancel()
                    self.logger.warning(
                        f"batch_remote_chat future timed out or failed: {e}. Marking as timeout."
                    )
                    continue
                res_l[i] = resp
                if self.config.APIInfo.low_flow_mode:
                    time.sleep(self.config.APIInfo.low_flow_latency)  # to reduce the API call frequency
            for res in res_l:
                if res == "no response":
                    self.logger.warning(
                        f"Some batch_remote_chat tasks did not complete successfully."
                    )
                    raise ValueError("batch_remote_chat tasks Fail")
        return res_l

    def encode_with_fallback(self, text: str, model: str = "gpt-4o-mini"):
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text), enc

    def truncate_text(self, pid:str, text: str, allowed: int) -> str:

        tokens, enc = self.encode_with_fallback(text, model=self.config.APIInfo.llm_model_name)
        token_len = len(tokens)
        
        if token_len > allowed:
            self.logger.warning(f"Paper {pid} tokens={token_len}, truncate to {allowed}")
            if allowed < 1000:
                self.logger.warning(f"Allowed tokens {allowed} too small, need to debug!")
            tokens = tokens[:allowed]
            truncate_text = enc.decode(tokens)

            if(truncate_text[:3000] != text[:3000]):
                self.logger.warning(f"Truncation error for paper {pid}, fallback to approiximation.")
                approx_tokens = len(text) / 4  # 1 token ≈ 4 chars
                if approx_tokens > allowed:
                    new_char_len = int(allowed * 4)
                    self.logger.warning(
                        f"Paper {pid} markdown too long: ~{approx_tokens:.0f} tokens, "
                        f"truncating to ~{allowed}."
                    )
                    text = text[:new_char_len]
            else:
                text = truncate_text
        return text

    def estimate_tokens(self, text: str) -> int:
        tokens, enc = self.encode_with_fallback(text, model=self.config.APIInfo.llm_model_name)
        token_len = len(tokens)
        return token_len


if __name__ == "__main__":
    # # test LLM API call
    # test_prompt = "Explain the theory of relativity in simple terms."
    # # load config
    # from omegaconf import OmegaConf

    # config = OmegaConf.load("config/deep_survey.yaml")
    # chat_agent = ChatAgent(config)
    # response = chat_agent.remote_chat(test_prompt, temperature=0.7, debug=True)
    # print("LLM API Response:")
    # print(response)

    # test Semantic Scholar API call
    from omegaconf import OmegaConf

    config = OmegaConf.load("config/deep_survey.yaml")
    semantic_scholar_api = SemanticScholarAPI(config)
    # query = '"auto survey"'
    # fields = "title,externalIds,openAccessPdf"
    # response = semantic_scholar_api.search_papers(query=query, fields=fields)
    # print("Semantic Scholar API Response:")
    # print(response["data"][:10])
    paper_id = "ARXIV:2505.11711"
    fields = "title,year,abstract,authors,externalIds,citations"
    response = semantic_scholar_api.get_paper_details(paper_id=paper_id, fields=fields)
    print("Semantic Scholar API Paper Details Response:")
    print(response)
