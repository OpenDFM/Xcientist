import json
import os
import re
from pathlib import Path
from typing import Union
import argparse
import sys
import glob

from tqdm import tqdm
from ChatAgent import ChatAgent


current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

from configs.config import BASE_DIR, CHAT_AGENT_WORKERS, MD_TEXT_LENGTH
from configs.config import OUTPUT_DIR
from configs.logger import get_logger

from utils import cut_text_by_token, load_prompt
# from src.models.monitor.time_monitor import TimeMonitor
from utils import (
    clean_chat_agent_format,
    load_file_as_string,
    sanitize_filename,
    save_result,
)

logger = get_logger("src.modules.preprocessor.DataCleaner")

def parse_arguments():
    parser = argparse.ArgumentParser(description="argments for retrieve paper")

    parser.add_argument('-p', '--base_dir', type=str, default="/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work", help="input pdf path")
    parser.add_argument('-t', '--task_id', type=str, default="semantic_scholar_test", help="input pdf path")

    return parser.parse_args()

class DataCleaner:
    def __init__(self, papers: list[dict] = []):
        self.papers: list[dict] = papers
        self.chat_agent_workers = CHAT_AGENT_WORKERS

    def load_json_dir(self, json_path_dir: Path):
        """load papers from json directory."""
        papers = []
        cnt_total = 0
        for file in os.listdir(json_path_dir):
            if file.endswith(".json"):
                p = os.path.join(json_path_dir, file)
                dic = json.loads(load_file_as_string(p))
                if (
                    "md_text" in dic
                ):  # Only consider those with `md_text` as available papers.
                    papers.append(dic)
                cnt_total += 1
        logger.info(f"Find {len(papers)} out of {cnt_total} papers available.")
        self.papers = papers

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

    def complete_bib(self, bib_file_save_path: str):
        """Not only complete the bib_name, also need to save all bibnames into a references.bib file"""
        var_name_i = 0
        bib_all = []
        remove_non_ascii_chars = (
            lambda input_string: input_string.replace(",", "")
            .encode("ascii", "ignore")
            .decode("ascii")
        )

        for paper in tqdm(self.papers, desc="completing bibname..."):
            if "reference" in paper:
                bib_name = paper["reference"].splitlines()[0].split("{")[1].strip(",")
                new_bib_name = remove_non_ascii_chars(bib_name)

                paper["bib_name"] = new_bib_name
                paper["reference"] = paper["reference"].replace(bib_name, new_bib_name)
            else:
                title = remove_non_ascii_chars(paper["title"])
                bib_name = "".join([c for c in title if not c.isspace()][:10]) + str(
                    var_name_i
                )
                var_name_i += 1
                bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"

                paper["reference"] = bib_tex
                paper["bib_name"] = bib_name

            bib_all.append(paper["reference"])

        save_result("\n".join(bib_all), bib_file_save_path)

    def check_md_text_length(self):
        for paper in self.papers:
            if "md_text" not in paper:
                continue
            md_text = paper["md_text"]
            paper["md_text"] = cut_text_by_token(md_text, MD_TEXT_LENGTH)

    def __process_paper_type_response(self, res: str, paper_index: int):
        kinds = ["method", "benchmark", "theory", "survey"]
        for k in kinds:
            if k in res.lower():
                self.papers[paper_index]["paper_type"] = k
                print(f"[DEBUG:] get {self.papers[paper_index]['title']} reference type: {k}")
                return True
        logger.error(
            f"failed to extract papertype of {self.papers[paper_index]['title']}"
        )
        logger.error(f"The response from gpt is {res}")
        return False

    def get_paper_type(self, chat_agent: ChatAgent):
        """complete the paper type field with chatgpt."""
        # load prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            abstract = paper["abstract"]
            prompt = load_prompt(
                f"{BASE_DIR}/prompts/paper_type_classification.md",
                abstract=abstract,
            )
            prompts_and_index.append([prompt, i])
        # batch_chat
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(prompts, desc="getting paper type...")
            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_paper_type_response(res, paper_index)
            ]
            cnt += 1

    def __process_attri_response(self, res: str, paper_index: int):
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            print(f"[DEBUG:] get attri tree(multi_stage) for {self.papers[paper_index]['path'].stem}")
            self.papers[paper_index]["attri"] = {**res_dic}
            return True
        except Exception as e:
            # with open("./prompt_log.txt", 'w') as f:
            #     f.write(res)
            logger.debug(
                f"Failed to process {self.papers[paper_index]['path'].stem}; The res: {res[:100]}; {e}"
            )
            return False
    ### multi-stage func ###
    def __process_schema_response(self, res: str, paper_index: int):
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            print(f"[DEBUG:] get schema for {self.papers[paper_index]['path'].stem}")
            self.papers[paper_index]["schema"] = {**res_dic}
            return True
        except Exception as e:
            with open("./schema_log.txt", 'w') as f:
                f.write(res)
            logger.debug(
                f"Failed to process {self.papers[paper_index]['path'].stem}; The res: {res[:100]}; {e}"
            )
            return False
    ### end ###

    def get_attri(self, chat_agent: ChatAgent):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            # 根据 paper_type 加载对应的 prompt
            paper_type = paper["paper_type"].lower()
            prompt = load_prompt(
                f"{BASE_DIR}/prompts/attri_tree_for_{paper_type}.md",
                paper=paper["md_text"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def save_papers(
        self, save_dir: Union[str, Path], file_name_attr: str = "title"
    ) -> None:
        """save every cleaned paper."""
        filter_field = [
            "from",
            "scholar_id",
            "detail_id",
            "title",
            "bib_name",
            "paper_type",
            "attri",
            "mount_outline",
            "similarity_score",
            "image",
        ]
        for paper in self.papers:
            try:
                file_name = paper[file_name_attr] + ".json"
                file_name = sanitize_filename(file_name)
                file_path = os.path.join(save_dir, file_name)
                save_dic = {key: paper.get(key, None) for key in filter_field}
                save_result(json.dumps(save_dic, indent=4), file_path)
            except Exception as e:
                logger.error(
                    f"There is an error when saving {file_path}. The error is: {e}"
                )
        return self.papers

    ### multi_stage ####
    def save_papers_multi_stage(
        self, save_dir: Union[str, Path], file_name_attr: str = "title", fix_schema: str = None
    ) -> None:
        """save every cleaned paper."""
        filter_field = [
            "attri",
        ]
        for paper in self.papers:
            try:
                file_name = paper['path'].stem + ".json"
                # file_name = sanitize_filename(file_name)
                file_name = Path(f"attris/{file_name}")
                file_path = os.path.join(save_dir, file_name)
                # save_dic = {key: paper.get(key, None) for key in filter_field}
                save_result(json.dumps(paper["attri"], indent=4), file_path)
                
                if fix_schema == None:
                    schema_file_name = paper['path'].stem + "_schema.json"
                    # schema_file_name = sanitize_filename(schema_file_name)
                    schema_file_name = Path(f"schemas/{schema_file_name}")
                    schema_file_path = os.path.join(save_dir, schema_file_name)
                    save_result(json.dumps(paper["schema"], indent=4), schema_file_path)
            except Exception as e:
                logger.error(
                    f"There is an error when saving {file_path}. The error is: {e}"
                )
        return self.papers
    ### end ###

    def offline_proc(self, task_id: str, ref_path: str) -> None:
        ref_data_path = Path(ref_path)
        print(f"[DEBUG:] loading .md from{ref_data_path}")
        md_texts = [p.read_text() for p in ref_data_path.glob("*.md") if p.is_file()]
        self.papers = [{"md_text": md_text} for md_text in md_texts]

        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        self.complete_bib(bib_file_path)

        self.check_md_text_length()
        chat_agent = ChatAgent()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

    ### new functions ###

    ##### start region: LLM design #####
    def get_attri_LLM_design_com(self, chat_agent: ChatAgent):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            num = 0
            abstracts = ""
            for j, com_paper in enumerate(self.papers):
                if num < 3 and i != j:
                    #print("[DEBUG:] add abstract: ", com_paper["abstract"])
                    abstracts += com_paper["abstract"]
                    abstracts += "\n"
                    num += 1
                if num >= 3:
                    break
            prompt = load_prompt(
                f"{BASE_DIR}/prompts/attri_tree_LLM_compare.md",
                paper=paper["md_text"],
                abstracts = abstracts
            )
            # with open(f"{BASE_DIR}/prompt_log.txt", 'w') as file:
            #     file.write(prompt)
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def get_attri_LLM_design(self, chat_agent: ChatAgent, com: bool = False):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            prompt = load_prompt(
                f"{BASE_DIR}/prompts/attri_tree_LLM_test_design.md",
                paper=paper["md_text"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def offline_proc_LLM_design(self, task_id: str, ref_path: str, com: bool = False) -> None:
        ref_data_path = Path(ref_path)
        print(f"[DEBUG:] loading .md from{ref_data_path}")
        md_texts = [p.read_text() for p in ref_data_path.glob("*.md") if p.is_file()]
        self.papers = [{"md_text": md_text} for md_text in md_texts]

        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(f"{OUTPUT_DIR}/single_stage_test/{task_id}/latex/references.bib")
        self.complete_bib(bib_file_path)

        self.check_md_text_length()
        chat_agent = ChatAgent()
        if(not com):
            self.get_attri_LLM_design(chat_agent=chat_agent)
        else:
            self.get_attri_LLM_design_com(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/single_stage_test/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")
    #### end region ###


    ###### start region: multi-stage #######
    def get_schema_LLM_design_com(self, chat_agent: ChatAgent):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        print("[DEBUG:] no schemas provided. Generating...")
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            num = 0
            abstracts = ""
            for j, com_paper in enumerate(self.papers):
                if num < 3 and i != j:
                    #print("[DEBUG:] add abstract: ", com_paper["abstract"])
                    abstracts += com_paper["abstract"]
                    abstracts += "\n"
                    num += 1
                if num >= 3 or num >= len(self.papers) - 1:
                    break
            prompt = load_prompt(
                f"{BASE_DIR}/prompts/multi_stage/schema_LLM_compare.md",
                paper=paper["md_text"],
                abstracts = abstracts
            )
            # with open(f"{BASE_DIR}/prompt_log.txt", 'w') as file:
            #     file.write(prompt)
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting schema from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_schema_response(res, paper_index)
            ]
            cnt += 1
    
    def get_attri_LLM_design_multi_stage(self, chat_agent: ChatAgent, prompt_file = "attri_tree_LLM_multi_stage.md"):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            prompt = load_prompt(
                f"{BASE_DIR}/prompts/multi_stage/{prompt_file}",
                paper=paper["md_text"],
                schema=paper["schema"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )
            # with open(f"{BASE_DIR}/tmp_log.txt", 'w') as f:
            #     f.write(prompts[0])

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def offline_proc_LLM_design_multi_stage(self, task_id: str, ref_path: str, fix_schema_path: str = None, args: argparse.Namespace = None) -> None:
        ref_data_path = Path(ref_path)
        print(f"[DEBUG:] loading .md from{ref_data_path}")
        md_texts = [
            (p.read_text(encoding="utf-8"), p)
            for p in ref_data_path.rglob("*.md")
            if p.is_file()
        ]
        self.papers = [{"md_text": md_text, "path": path} for md_text, path in md_texts]

        self.complete_title()
        self.complete_abstract()
        # bib_file_path = Path(f"{OUTPUT_DIR}/multi_stage_test/{task_id}/latex/references.bib")
        # self.complete_bib(bib_file_path)

        self.check_md_text_length()
        chat_agent = ChatAgent()

        if fix_schema_path == None:
            self.get_schema_LLM_design_com(chat_agent=chat_agent)
        else:
            print(f"[DEBUG:] schemas provided at {fix_schema_path}. Loading...")
            for paper in self.papers:
                with open(f"{fix_schema_path}/{paper['path'].stem}.json", encoding="utf-8") as f:
                    paper["schema"] = f.read()

        self.get_attri_LLM_design_multi_stage(chat_agent=chat_agent, prompt_file = "attri_tree_LLM_multi_stage v1.3.md")


        save_path = Path(f"{args.base_dir}/outputs/pipeline/{args.task_id}/extract")
        self.save_papers_multi_stage(save_dir=save_path, fix_schema = fix_schema_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")


# python -m src.modules.preprocessor.data_cleaner
if __name__ == "__main__":
    args = parse_arguments()
    print(args.base_dir)
    dc = DataCleaner()
    dc.offline_proc_LLM_design_multi_stage(args.task_id,f"{args.base_dir}/outputs/pipeline/{args.task_id}/references/mds", None, args)
    print(len(dc.papers))
