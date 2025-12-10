
import fcntl
import requests
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from pathlib import Path

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from configs.config import (
    REMOTE_URL,
    LOCAL_URL,
    TOKEN,
    BASE_DIR,
    DEFAULT_CHATAGENT_MODEL,
    CHAT_AGENT_WORKERS,
)
from configs.config import OUTPUT_DIR

from configs.logger import get_logger
# from src.models.LLM.utils import encode_image
#from src.models.monitor.token_monitor import TokenMonitor

logger = get_logger("src.models.LLM.ChatAgent")
logger.debug(f"ChatAgent pid={os.getpid()}")


class ChatAgent:
    Cost_file = Path(f"{OUTPUT_DIR}/tmp/cost.txt")
    Request_stats_file = Path(f"{OUTPUT_DIR}/tmp/request_stats.txt")
    Record_splitter = "||"
    Record_show_length = 200

    def __init__(
        self,
        token_monitor: None = None,
        token: str = TOKEN,
        remote_url: str = REMOTE_URL,
        local_url: str = LOCAL_URL,
    ) -> None:
        self.remote_url = remote_url
        self.token = token
        self.local_url = local_url
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        self.batch_workers = CHAT_AGENT_WORKERS
        self.token_monitor = token_monitor

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
        model=DEFAULT_CHATAGENT_MODEL,
    ) -> str:
        """chat with remote LLM, return result."""
        url = self.remote_url
        header = self.header
        # text content
        messages = [{"role": "user", "content": text_content}]
        # insert image urls ----
        if (
            image_urls is not None
            and isinstance(image_urls, list)
            and len(image_urls) > 0
        ):
            image_url_frame = []
            for url_ in image_urls:
                image_url_frame.append(
                    {"type": "image_url", "image_url": {"url": url_}}
                )
            image_message_frame = {"role": "user", "content": image_url_frame}
            messages.append(image_message_frame)

        # insert local images ----
        # if (
        #     local_images is not None
        #     and isinstance(local_images, list)
        #     and len(local_images) > 0
        # ):
        #     local_image_frame = []
        #     for local_image in local_images:
        #         local_encoded_image = encode_image(local_image)
        #         local_image_frame.append(
        #             {
        #                 "type": "image_url",
        #                 "image_url": {
        #                     "url": f"data:image/jpeg;base64,{local_encoded_image}"
        #                 },
        #             }
        #         )
        #     image_message_frame = {"role": "user", "content": local_image_frame}
        #     messages.append(image_message_frame)

        payload = {"model": model, "messages": messages, "temperature": temperature}

        response = requests.post(url, headers=header, json=payload, timeout=90)

        if response.status_code != 200:
            logger.error(
                f"chat response code: {response.status_code}\n{response.text[:500]}, retrying..."
            )
            status_code = 0 if response.status_code != 200 else 1

            # 加了线程锁
            self.update_record(
                status_code=status_code,
                response_code=response.status_code,
                request=text_content,
                response=response.text,
            )
            response.raise_for_status()
        try:
            res = json.loads(response.text)
            res_text = res["choices"][0]["message"]["content"]
            # 更新总开销
            # token monitor
            if self.token_monitor:
                self.token_monitor.add_token(
                    model=model,
                    input_tokens=res["usage"]["prompt_tokens"],
                    output_tokens=res["usage"]["completion_tokens"],
                )
        except Exception as e:
            res_text = f"Error: {e}"
            logger.error(f"There is an error: {e}")

        status_code = 0 if response.status_code != 200 else 1
        self.update_record(
            status_code=status_code,
            response_code=response.status_code,
            request=text_content,
            response=res_text,
        )

        if debug:
            return res_text, response
        return res_text

    # map chat index
    def __remote_chat(
        self,
        index,
        content,
        temperature: float = 0.5,
        debug: bool = False,
        model=DEFAULT_CHATAGENT_MODEL,
    ):
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
        workers: int = CHAT_AGENT_WORKERS,
        temperature: float = 0.5,
    ) -> list[str]:
        """
        开启多线程进行对话
        """
        if workers is None:
            workers = self.batch_workers
        # 创建线程池
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交任务
            future_l = [
                executor.submit(self.__remote_chat, i, prompt_l[i], temperature)
                for i in range(len(prompt_l))
            ]
            # 领取任务结果
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

    @classmethod
    def update_record(
        cls, status_code: int, response_code: int, request: str, response: str
    ):
        "维护记录文件"
        content = (
            f"{status_code}{cls.Record_splitter}{response_code}{cls.Record_splitter}{request[: cls.Record_show_length]}{cls.Record_splitter}{response[: cls.Record_show_length]}".replace(
                "\n", ""
            )
            + "\n"
        )
        # 检查文件是否存在
        if not os.path.exists(cls.Request_stats_file):
            parent_dir = Path(cls.Request_stats_file).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            with open(cls.Request_stats_file, "w", encoding="utf-8") as fw:
                fcntl.flock(fw, fcntl.LOCK_EX)  # 加锁
                fw.write(content)
                logger.info(
                    f"record file {cls.Request_stats_file} did not exist, created and initialized with 0.0"
                )
                fcntl.flock(fw, fcntl.LOCK_UN)
        # 更新开销总计
        try:
            with open(cls.Request_stats_file, "a", encoding="utf-8") as fw:
                fcntl.flock(fw, fcntl.LOCK_EX)
                fw.write(content)
                fcntl.flock(fw, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to update cost: {e}")

    def local_chat(self, query, debug=False) -> str:
        """
        调用本地LLM进行推理, 保证端口已开启
        """
        query = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
            {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".format(query)

        payload = json.dumps(
            {
                "prompt": query,
                "temperature": 1.0,
                "max_tokens": 102400,
                "n": 1,
                # 可选的参数在这里：https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
            }
        )
        headers = {"Content-Type": "application/json"}
        res = requests.request("POST", self.local_url, headers=headers, data=payload)
        if res.status_code != 200:
            logger.info("chat response code: {}".format(res.status_code), query[:20])
            return "chat response code: {}".format(res.status_code)
        if debug:
            return res
        return res.json()["text"][0].replace(query, "")

    def __local_chat(self, index, query):
        return index, self.local_chat(query, debug=True)

    def batch_local_chat(self, query_l, worker=16, desc="bach local inferencing..."):
        """
        多线程本地推理
        """
        with ThreadPoolExecutor(max_workers=worker) as executor:
            # 提交任务
            future_l = [
                executor.submit(self.__local_chat, i, query_l[i])
                for i in range(len(query_l))
            ]
            # 领取任务结果
            res_l = ["no response"] * len(query_l)
            for future in tqdm(as_completed(future_l), desc=desc, total=len(future_l)):
                i, resp = future.result()
                res_l[i] = resp
        return res_l

    @staticmethod
    def show_request_stats():
        stats_file = ChatAgent.Request_stats_file
        logger.info(f"stats_file: {stats_file}")

        with stats_file.open("r", encoding="utf-8") as fr:
            succ_count = 0
            total_count = 0
            for line in fr:
                elements = line.strip().split(ChatAgent.Record_splitter)
                succ_count += int(elements[0])
                total_count += 1
            logger.info(f"请求成功率：{round(succ_count / total_count * 100, 2)}%")

    @staticmethod
    def clean_request_stats():
        stats_file = ChatAgent.Request_stats_file
        if stats_file.exists():
            logger.info(f"remove {stats_file}.")


if __name__ == "__main__":
    agent = ChatAgent()
    text_content = "图片里面有什么"

    # result = agent.remote_chat(text_content="今天天气怎么样",  model="gpt-4o")
    # print(result)
    #
    # image_urls = ["https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"]
    # result = agent.remote_chat( text_content=text_content, image_urls=image_urls, temperature=0.5, model="gpt-4o")
    # print(result)

    local_images = [f"{BASE_DIR}/resources/dummy_data/figs/dog_and_girl.jpeg"]
    result = agent.remote_chat(
        text_content=text_content,
        local_images=local_images,
        temperature=0.5,
        model="gpt-4o",
    )
    print(result)

    ChatAgent.show_request_stats()
