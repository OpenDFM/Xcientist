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
from agents.survey_agent.utils.rich_logger import get_logger
import tiktoken

class SemanticScholarAPI:
    def __init__(self, config):
        self.headers = {"x-api-key": config.APIInfo.semantic_scholar_api_key}
        self.base_url = "http://api.semanticscholar.org/graph/v1"
        self.logger = get_logger("SemanticScholarAPI")

    def search_papers(self, query: str, fields: str):
        url = f"{self.base_url}/paper/search"
        params = {"query": query, "fields": fields}
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            self.logger.info(
                "Rate limit exceeded. Waiting 60 seconds before retrying..."
            )
            time.sleep(60)
            return self.search_papers(query, fields)
        else:
            self.logger.error(f"Status code: {response.status_code}")
            return None

    def get_paper_details(self, paper_id: str, fields: str):
        url = f"{self.base_url}/paper/{paper_id}?fields={fields}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            self.logger.info(
                "Rate limit exceeded. Waiting 60 seconds before retrying..."
            )
            time.sleep(60)
            return self.get_paper_details(paper_id, fields)
        else:
            self.logger.error(f"Status code: {response.status_code}")
            return None


class ChatAgent:
    Record_splitter = "||"
    Record_show_length = 200

    def __init__(self, config) -> None:
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

    @retry(
        stop=stop_after_attempt(30),
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

        if response.status_code != 200:
            self.logger.error(
                f"chat response code: {response.status_code}\n{response.text[:500]}, retrying..."
            )
            response.raise_for_status()
        try:
            res = response.json()
            res_text = res["choices"][0]["message"]["content"]
        except Exception as e:
            res_text = f"Error: {e}"
            self.logger.error(f"There is an error: {e}")

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
                i, resp = future.result()
                res_l[i] = resp
        return res_l

    import tiktoken

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
