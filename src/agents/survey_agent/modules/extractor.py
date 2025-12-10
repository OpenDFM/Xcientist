import json
import os
import re
from pathlib import Path
from typing import Union
import argparse
import sys
import glob

from tqdm import tqdm

from typing import List, Tuple, Union, Dict, Any

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="argments for retrieve paper")

    parser.add_argument(
        "-p",
        "--base_dir",
        type=str,
        default="/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work",
        help="input pdf path",
    )
    parser.add_argument(
        "-t",
        "--task_id",
        type=str,
        default="semantic_scholar_test",
        help="input pdf path",
    )

    return parser.parse_args()


class DataCleaner:
    def __init__(self, config, papers: list[dict] = []):
        self.papers: list[dict] = papers

        self.config = config.Modules.Extractor

        self.base_dir = config.BasicInfo.base_dir
        self.prompt_path = config.BasicInfo.prompt_path
        self.paper_cache_path = config.BasicInfo.paper_cache_path
        self.schema_cache_path = config.BasicInfo.schema_cache_path
        self.extraction_cache_path = config.BasicInfo.extraction_cache_path

        self.output_path = config.BasicInfo.output_path
        self.task_id = config.BasicInfo.task_id

        self.max_md_length = self.config.max_md_length

        self.chat_agent = ChatAgent(config)

    def complete_title(self):
        for paper in tqdm(self.papers, desc="completing title..."):
            if "title" not in paper:
                paper["title"] = paper["md_text"].splitlines()[0].strip(" #")
                paper["title"] = paper["title"][:32]  # avoid too long title

    def complete_abstract(self):
        pattern = r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*"  # find "abstract" substring, with whitespace bettween letters.
        for paper in tqdm(self.papers, desc="completing abstract..."):
            if "abstract" in paper and len(paper["abstract"]) > 500:
                continue
            match = re.search(pattern, paper["md_text"], re.IGNORECASE)
            if match:
                index = match.start()
                paper["abstract"] = paper["md_text"][index : index + 2000]
            else:
                paper["abstract"] = paper["md_text"][:2000]

    def check_md_text_length(self):
        for paper in self.papers:
            if "md_text" not in paper:
                continue
            md_text = paper["md_text"]
            paper["md_text"] = cut_text_by_token(md_text, self.max_md_length)

    def __process_attri_response(self, res: str, paper_index: int):
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            print(
                f"[DEBUG:] get attri tree(multi_stage) for {self.papers[paper_index]['paper_id']}"
            )
            self.papers[paper_index]["attri"] = {**res_dic}
            return True
        except Exception as e:
            # with open("./prompt_log.txt", 'w') as f:
            #     f.write(res)
            print(
                f"Failed to process {self.papers[paper_index]['paper_id']}; The res: {res[:100]}; {e}"
            )  # logger,debug
            return False

    ### multi-stage func ###
    def __process_schema_response(self, res: str, paper_index: int):
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            print(
                f"[DEBUG:] get schema for {paper_index}: {self.papers[paper_index]['paper_id']}"
            )
            self.papers[paper_index]["schema"] = {**res_dic}
            return True
        except Exception as e:
            with open("./schema_log.txt", "w") as f:
                f.write(res)
            print(
                f"Failed to process {self.papers[paper_index]['paper_id']}; The res: {res[:100]}; {e}"
            )  # logger.debug
            return False

    ### multi_stage ####
    def save_papers_multi_stage(self, fix_schema: str = None) -> None:
        """save every cleaned paper."""
        filter_field = [
            "attri",
        ]
        for paper in self.papers:
            try:
                file_name = paper["paper_id"] + ".json"
                # file_name = sanitize_filename(file_name)
                # file_name = Path(f"{file_name}")
                file_path = os.path.join(self.extraction_cache_path, file_name)
                # save_dic = {key: paper.get(key, None) for key in filter_field}
                log_path = f"{self.output_path}/{self.task_id}/extractions/{file_name}"
                save_result(json.dumps(paper["attri"], indent=4), file_path)
                save_result(json.dumps(paper["attri"], indent=4), log_path)

                if fix_schema == None:
                    schema_file_name = paper["paper_id"] + "_schema.json"
                    schema_file_path = os.path.join(
                        self.schema_cache_path, schema_file_name
                    )
                    save_result(json.dumps(paper["schema"], indent=4), schema_file_path)
            except Exception as e:
                print(
                    f"There is an error when saving {file_path}. The error is: {e}"
                )  # logger.debug
        return self.papers

    #### end region ###

    ###### start region: multi-stage #######
    def get_schema_LLM_design_com(self):
        """extract attribute tree from paper"""
        print("[DEBUG:] no schemas provided. Generating...")
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            num = 0
            abstracts = ""
            for j, com_paper in enumerate(self.papers):
                if num < 3 and i != j:
                    # print("[DEBUG:] add abstract: ", com_paper["abstract"])
                    abstracts += com_paper["abstract"]
                    abstracts += "\n"
                    num += 1
                if num >= 3 or num >= len(self.papers) - 1:
                    break
            prompt = load_prompt(
                f"{self.prompt_path}/multi_stage/schema_LLM_compare.md",
                paper=paper["md_text"],
                abstracts=abstracts,
            )
            # with open(f"{self.base_dir}/prompt_log.txt", 'w') as file:
            #     file.write(prompt)
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = self.chat_agent.batch_remote_chat(
                prompts, desc="getting schema from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_schema_response(res, paper_index)
            ]
            cnt += 1

    def get_attri_LLM_design_multi_stage(
        self, prompt_file="attri_tree_LLM_multi_stage.md"
    ):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            prompt = load_prompt(
                f"{self.prompt_path}/multi_stage/{prompt_file}",
                paper=paper["md_text"],
                schema=paper["schema"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = self.chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )
            # with open(f"{self.base_dir}/tmp_log.txt", 'w') as f:
            #     f.write(prompts[0])

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def offline_proc_LLM_design_multi_stage_batch(
        self, paper_list: list, fix_schema_path: str = None
    ) -> Dict[str, Any]:
        outcomes = {}
        md_texts = []
        for paper_id in paper_list:
            print(f"DEBUG: processing {paper_id}")
            target_cache = f"{self.extraction_cache_path}/{paper_id}.json"
            if os.path.exists(target_cache):
                print(f"DEBUG: cache hit for {paper_id}")
                with open(Path(target_cache), "r") as f:
                    outcomes[paper_id] = json.load(f)
            else:
                print(f"DEBUG: cache miss for {paper_id}")
                try:
                    with open(
                        Path(f"{self.paper_cache_path}/{paper_id}/auto/{paper_id}.md"),
                        "r",
                    ) as f:
                        md_texts.append((f.read(), paper_id))
                except FileNotFoundError:
                    print(f"[LOG:] Paper file not found: {paper_id}")

        print(f"[DEBUG:] load .md from{self.paper_cache_path}")

        self.papers = [
            {"md_text": md_text, "paper_id": paper_id} for md_text, paper_id in md_texts
        ]
        for paper in self.papers:
            print(f"DEBUG: {paper['paper_id']}")

        self.complete_title()
        self.complete_abstract()
        self.check_md_text_length()

        if fix_schema_path == None:
            self.get_schema_LLM_design_com()
        else:
            print(f"[DEBUG:] schemas provided at {fix_schema_path}. Loading...")
            for paper in self.papers:
                with open(
                    f"{fix_schema_path}/{paper['paper_id']}_schema.json",
                    encoding="utf-8",
                ) as f:
                    paper["schema"] = f.read()

        self.get_attri_LLM_design_multi_stage(
            prompt_file="attri_tree_LLM_multi_stage v1.3.md"
        )

        self.save_papers_multi_stage(fix_schema=fix_schema_path)
        for paper in self.papers:
            outcomes[paper_id] = paper["attri"]
        print(
            f"=========={len(self.papers)} remain after cleaning. =========="
        )  # logger.info
        return outcomes


if __name__ == "__main__":
    config = load_config(Path("./config/deep_survey.yaml"))
    # topicRetriever = TopicRetriever(config)

    paper_list = [
        "1907.10863",
        "1907.10864",
        "07a040081ddf04927695d4326bfb2dfd9996bd43",
    ]
    dc = DataCleaner(config)
    dc.offline_proc_LLM_design_multi_stage_batch(paper_list=paper_list)
    print(len(dc.papers))
