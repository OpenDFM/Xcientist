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
from utils.rich_logger import get_logger
import tiktoken
import xml.etree.ElementTree as ET
from utils.utils import extract_json

import requests
import json
import time

class ArxivAPI:
    def __init__(self, config):
        self.base_url = "http://export.arxiv.org/api/query"
        self.logger = get_logger("ArxivAPI")
        self.config = config

    def get_paper_details(self, paper_id: str):
        arxiv_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        paper = {}
        
        for retry_count in range(self.config.APIInfo.arxiv_api_max_retry):
            response = requests.get(arxiv_url, timeout=120)
            if response.status_code == 200:
                break
            else:
                self.logger.warning(f"arXiv API request failed for {paper_id}: {response.status_code}. Retrying {retry_count + 1}/3...")
                if response.status_code == 429:
                    time.sleep(60)

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
                summary_el = entry.find('atom:summary', ns)
                paper["abstract"] = summary_el.text.strip() if summary_el is not None else ""
                self.logger.info(f"Fetched details from arXiv for {paper_id}")
            else:
                self.logger.warning(f"No entry found in arXiv response for {paper_id}")
                raise ValueError("No entry found in arXiv response in mla generation")
        else:
            self.logger.warning(f"arXiv API request failed for {paper_id}: {response.status_code}")
            raise ValueError("No entry found in arXiv response in mla generation")
        
        return paper

    def search_papers_by_title(self, title: str):
        """通过标题搜索arXiv论文"""
        import urllib.parse
        # 标题搜索使用 ti: 前缀
        search_query = f"ti:{urllib.parse.quote(title)}"
        arxiv_url = f"{self.base_url}?search_query={search_query}&start=0&max_results=10"
        
        papers = []
        for retry_count in range(self.config.APIInfo.arxiv_api_max_retry):
            response = requests.get(arxiv_url, timeout=120)
            if response.status_code == 200:
                break
            else:
                self.logger.warning(f"arXiv search request failed: {response.status_code}. Retrying {retry_count + 1}/3...")
                if response.status_code == 429:
                    time.sleep(60)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('.//atom:entry', ns):
                paper = {}
                # 提取 arXiv ID (从链接中提取)
                id_link = entry.find('atom:id', ns)
                if id_link is not None:
                    # 从 URL 中提取 arXiv ID，如 http://arxiv.org/abs/2301.00001v1
                    paper_id = id_link.text.split('/')[-1]
                    # 移除版本号
                    paper['paper_id'] = paper_id.split('v')[0] if 'v' in paper_id else paper_id
                paper["api_platform"] = "arxiv"
                paper["title"] = entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else ""
                paper["authors"] = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None]
                paper["year"] = entry.find('atom:published', ns).text[:4] if entry.find('atom:published', ns) is not None else ""
                paper["venue"] = "arXiv"
                summary_el = entry.find('atom:summary', ns)
                paper["abstract"] = summary_el.text.strip() if summary_el is not None else ""
                papers.append(paper)
            
            self.logger.info(f"Found {len(papers)} papers from arXiv for title: {title}")
        else:
            self.logger.warning(f"arXiv search request failed: {response.status_code}")
        
        return papers

class SemanticScholarAPI:
    def __init__(self, config):
        self.headers = {"x-api-key": config.APIInfo.semantic_scholar_api_key}
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.logger = get_logger("SemanticScholarAPI")
        self.config = config

    def search_papers(self, query: str, fields: str, retry_time: int = 0):
        """Search papers with bounded retries; log every attempt."""
        max_retry = self.config.APIInfo.semantic_scholar_api_max_retry
        url = f"{self.base_url}/paper/search"
        params = {"query": query, "fields": fields}

        resp = requests.get(url, headers=self.headers, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()

        if retry_time >= max_retry:
            self.logger.error(
                f"Error occurs in search_papers. Status code: {resp.status_code}, reached max retry {max_retry}."
            )
            return None

        self.logger.error(
            f"Error occurs in search_papers. Status code: {resp.status_code}, retrying {retry_time + 1}/{max_retry}..."
        )
        if resp.status_code == 429:
            self.logger.info("Rate limit exceeded. Waiting 60 seconds before retrying...")
            time.sleep(60)
            return self.search_papers(query, fields, retry_time + 1)
        else:
            time.sleep(min(5, 1 + retry_time))

        return self.search_papers(query, fields, retry_time+1)

    def get_paper_details(self, paper_id: str, fields: str, retry_time: int = 0):
        """Fetch paper details with bounded retries; raises on final failure."""
        max_retry = self.config.APIInfo.semantic_scholar_api_max_retry
        url = f"{self.base_url}/paper/{paper_id}?fields={fields}"
        resp = requests.get(url, headers=self.headers, timeout=60)

        if resp.status_code == 200:
            return resp.json()

        if retry_time >= max_retry:
            self.logger.error(
                f"Failed to fetch paper details for {paper_id} after {max_retry} retries. Status code: {resp.status_code}"
            )
            raise ValueError(f"Failed to fetch paper details for {paper_id} from Semantic Scholar")

        self.logger.error(
            f"Error occurs in get_paper_details. Status code: {resp.status_code}, retrying {retry_time + 1}/{max_retry}..."
        )
        if resp.status_code == 429:
            self.logger.info("Rate limit exceeded. Waiting 60 seconds before retrying...")
            time.sleep(60)
            return self.get_paper_details(paper_id, fields, retry_time + 1)
        else:
            time.sleep(min(5, 1 + retry_time))

        return self.get_paper_details(paper_id, fields, retry_time+1)


class TransientHTTPError(requests.RequestException):
    """Raised for retryable HTTP status codes."""
    pass

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
        stop=stop_after_attempt(10),
        wait=wait_exponential(min=1, max=300),
        retry=retry_if_exception_type((requests.RequestException, TransientHTTPError)),
    )
    def remote_chat(
        self,
        text_content: str,
        image_urls: list[str] = None,
        local_images: list[Path] = None,
        temperature: float = 0.5,
        debug: bool = False,
        model=None,
        max_output_tokens: int = 16000,
    ) -> str:
        """Chat with remote LLM, return result. Minimal logging; no file writes."""
        if model is None:
            model = self.model_name

        # Estimate input tokens and truncate if necessary to leave room for output.
        # Keep a hard input-side truncation cap at 390000 tokens while still
        # reserving the requested output budget.
        context_window = 406000
        hard_input_cap = 390000
        input_tokens, enc = self.encode_with_fallback(text_content, model=model)
        input_token_count = len(input_tokens)

        # Reserve space for output tokens
        max_input_tokens = min(context_window - max_output_tokens, hard_input_cap)

        if input_token_count > max_input_tokens:
            self.logger.warning(
                f"Input tokens ({input_token_count}) exceeds max allowed ({max_input_tokens}). "
                f"Truncating to fit context window."
            )
            truncated_tokens = input_tokens[:max_input_tokens]
            text_content = enc.decode(truncated_tokens)

        url = self.remote_url
        header = self.header
        messages = [{"role": "user", "content": text_content}]

        if image_urls:
            image_url_frame = [
                {"type": "image_url", "image_url": {"url": u}} for u in image_urls
            ]
            messages.append({"role": "user", "content": image_url_frame})

        # Determine if streaming is enabled
        use_stream = self.config.APIInfo.use_stream_mode
        
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": use_stream}
        
        # Enable stream=True in requests if streaming mode is on
        response = requests.post(url, headers=header, json=payload, timeout=getattr(self.config.APIInfo, "chat_timeout", 120), stream=use_stream)

        if self.config.APIInfo.low_flow_mode:
            time.sleep(self.config.APIInfo.low_flow_latency)

        # Handle Streaming Response
        if use_stream:
            response.raise_for_status()
            collected_content = []
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line[6:]
                        if json_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    collected_content.append(content)
                        except json.JSONDecodeError:
                            continue
            
            res_text = "".join(collected_content)
            if debug:
                return res_text, response
            return res_text

        # Handle Normal (Non-Streaming) Response - Original Logic
        try:
            resp_json = response.json()
        except Exception:
            resp_json = None

        msg = ""
        if isinstance(resp_json, dict):
            msg = str(resp_json.get("error", {}).get("message", ""))
        
        # Check for moderation blocks
        if "moderation block" in msg.lower() or "moderation" in msg.lower() or "Moderation Block" in response.text:
            self.logger.warning(f"Moderation blocked the prompt: {text_content}")
            raise ValueError("prompt blocked by moderation")

        retryable_status = {408, 429, 500, 502, 503, 504}
        if response.status_code in retryable_status:
            self.logger.error(f"chat response code: {response.status_code}, retrying...")
            # Note: You might need to import TransientHTTPError or use standard Exception if not defined
            raise requests.RequestException(f"Retryable status: {response.status_code}")
        
        if response.status_code != 200:
            self.logger.error(f"chat response code: {response.status_code}")
            response.raise_for_status()

        try:
            res = response.json()
            res_text = res["choices"][0]["message"]["content"]
            if not res_text:
                res_text = res["choices"][0]["message"].get("reasoning_content", None)
        except Exception as e:
            res_text = f"Error: {e}"
            self.logger.error(f"There is an error in remote_chat: {e}")
            raise e

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

    def _default_validate_fn(self, result: str, info_dict: dict = None) -> bool:
        """Default validation function that checks if the result is a non-empty string."""
        if not result or len(result) == 0:
            raise ValueError("Validation failed: Result is empty or not a string.")
        return True

    def remote_chat_with_retry(
        self,
        prompt: str,
        validate_fn: callable = None,
        max_retry: int = 5,
        temperature: float = 0.5,
        debug: bool = False,
        model=None,
        info_dict: dict = {},
    ) -> str:
        """
        Chat with remote LLM with retry logic for failed validations.
        
        Args:
            prompt: The prompt to send to the LLM
            validate_fn: A function that takes a result string and returns True if valid, 
                        raises ValueError/Exception if invalid. If None, no validation.
            max_retry: Maximum number of retry attempts
            temperature: Temperature for LLM
            debug: If True, return (response, response) tuple
            model: Model to use (defaults to self.model_name)
            
        Returns:
            The validated response string
            
        Raises:
            ValueError: If validation fails after max_retry attempts
        """
        if model is None:
            model = self.model_name
        if validate_fn is None:
            validate_fn = self._default_validate_fn

        info_dict["max_retry"] = max_retry
        for retry in range(max_retry):
            info_dict["retry_time"] = retry
            try:
                result = self.remote_chat(
                    text_content=prompt,
                    temperature=temperature,
                    debug=debug,
                    model=model,
                )
                
                # If no validation function, return directly
                if validate_fn is None:
                    return result
                
                # Validate the result
                val, result = validate_fn(result, info_dict)
                if not val:
                    raise ValueError("Validation failed for remote chat")
                return result
                
            except Exception as e:
                if retry < max_retry - 1:
                    self.logger.warning(
                        f"remote_chat_with_retry attempt {retry + 1}/{max_retry} failed: {e}. Retrying..."
                    )
                    if self.config.BasicInfo.debug:
                        self.logger.warning(f"return text: {result}...")
                    time.sleep(min(5, 1 + retry))  # Exponential backoff
                else:
                    self.logger.error(
                        f"remote_chat_with_retry failed after {max_retry} attempts: {e}"
                    )
                    if self.config.BasicInfo.debug:
                        self.logger.warning(f"return text: {result[:50]}...")
                    raise ValueError(
                        f"remote_chat_with_retry failed after {max_retry} attempts: {e}"
                    )
        
        # Should not reach here, but just in case
        raise ValueError(f"remote_chat_with_retry failed after {max_retry} retries")

    def batch_remote_chat(
        self,
        prompt_l: list[str],
        desc: str = "batch_chating...",
        workers: int = None,
        temperature: float = 0.5,
        future_timeout: float = 600.0,
    ) -> list[str]:
        if workers is None:
            workers = self.batch_workers
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_l = [
                executor.submit(self.__remote_chat, i, prompt_l[i], temperature)
                for i in range(len(prompt_l))
            ]
            res_l = [None] * len(prompt_l)
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
                if res is None:
                    self.logger.warning(
                        f"Some batch_remote_chat tasks did not complete successfully."
                    )
        return res_l

    def batch_remote_chat_with_retry(
        self,
        prompts: list[str],
        validate_fn: callable,
        max_retry: int = 5,
        desc: str = "batch_chating with retry...",
        workers: int = None,
        temperature: float = 0.5,
        future_timeout: float = 600.0,
        model: str = None,
        info_dict: dict = {},
    ) -> list[str]:
        """
        Batch remote chat with retry logic for failed validations.
        
        Args:
            prompts: List of prompts to send to the LLM
            validate_fn: A function that takes a result string and returns True if valid, 
                        raises ValueError/Exception if invalid
            max_retry: Maximum number of retry attempts
            desc: Description for progress bar
            workers: Number of parallel workers (defaults to self.batch_workers)
            temperature: Temperature for LLM
            future_timeout: Timeout for each future
            
        Returns:
            List of results in the same order as input prompts
            
        Raises:
            ValueError: If not all results pass validation after max_retry attempts
        """
        if workers is None:
            workers = self.batch_workers
        if model is None:
            model = self.model_name
        if validate_fn is None:
            validate_fn = self._default_validate_fn

        input_prompts = prompts.copy()
        input_indices = list(range(len(prompts)))
        all_results = [None] * len(prompts)
        finished = False
        info_dict["max_retry"] = max_retry
        
        for retry in range(max_retry):
            info_dict["retry_time"] = retry
            error_prompts = []
            error_indices = []
            
            # Call batch_remote_chat for current batch of prompts
            results = self.batch_remote_chat(
                input_prompts, 
                desc=f"{desc} (retry {retry + 1}/{max_retry})",
                workers=workers,
                temperature=temperature,
                future_timeout=future_timeout
            )
            
            # Validate each result
            for i in range(len(results)):
                info_dict["idx"] = input_indices[i]
                try:
                    # Validate the result using the provided validation function
                    val, result = validate_fn(results[i], info_dict)
                    if not val:
                        raise ValueError("Validation failed")
                    # If validation passes, store the result
                    all_results[input_indices[i]] = result
                except Exception as e:
                    self.logger.warning(f"Validation failed for prompt {input_indices[i]}: {e}")
                    if self.config.BasicInfo.debug:
                        self.logger.warning(f"return text: {results[i][:50]}...")
                    error_prompts.append(input_prompts[i])
                    error_indices.append(input_indices[i])
            
            # Check if all results are valid
            if len(error_indices) == 0 and len(error_prompts) == 0:
                finished = True
                break
            else:
                self.logger.info(
                    f"Validation failed for {len(error_prompts)}/{len(prompts)} prompts, "
                    f"retrying {retry + 1}/{max_retry}"
                )
                # Update for next retry - only process failed prompts
                input_prompts = error_prompts
                input_indices = error_indices
        
        if not finished:
            self.logger.error(
                f"batch_remote_chat_with_retry failed after {max_retry} retries. "
                f"Failed prompts: {len(error_prompts)}"
            )
            raise ValueError(
                f"batch_remote_chat_with_retry failed after {max_retry} retries. "
                f"{len(error_prompts)} prompts still failing validation."
            )
        
        return all_results

    def encode_with_fallback(self, text: str, model: str = "gpt-4o-mini"):
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text), enc

    def truncate_prompt(self, text: str, allowed: int, model: str = None) -> str:
        if model is None:
            model = self.model_name
        tokens, enc = self.encode_with_fallback(text, model=model)
        token_len = len(tokens)
        
        if token_len > allowed:
            self.logger.warning(f"Prompt tokens={token_len}, truncate to {allowed}")
            if allowed < 1000:
                self.logger.warning(f"Allowed tokens {allowed} too small, need to debug!")
            tokens = tokens[:allowed]
            truncate_text = enc.decode(tokens)

            if(truncate_text[:3000] != text[:3000]):
                self.logger.warning(f"Truncation error for prompt, fallback to approiximation.")
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
