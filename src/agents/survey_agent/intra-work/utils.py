import base64
import os
from pathlib import Path

import tiktoken

from configs.config import CUT_WORD_LENGTH
from configs.logger import get_logger


##utils2##
import json
import logging
import re
import ast
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Union, Dict

logger = get_logger("src.modules.LLM.utils")


def load_prompt(file_path: Path, **kwargs):
    """读取prompt模板"""
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            return f.read().format(**kwargs)
    else:
        logger.error(f"Prompt template not found at {file_path}")
        return ""


def cut_text_by_token(text: str, max_tokens: int, model: str = "gpt-4o-mini"):
    """Cut text by token num."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        encoded_text = encoding.encode(text)
        cut_text = encoding.decode(encoded_text[:max_tokens])
    except Exception as e:
        logger.error(e)
        cut_text = text[: CUT_WORD_LENGTH * max_tokens]
    return cut_text



##utils2
def sanitize_json_escapes(text: str) -> str:
    latex_chars = "%&$ _{}#~^"
    for ch in latex_chars.split():
        pattern = re.compile(rf'(?<!\\)\\{re.escape(ch)}')
        text = pattern.sub(r'\\\\' + ch, text)

    # 通用规则：把单反斜杠且后继不是 JSON 合法转义字符的，换成双反斜杠
    # 这里的合法首字符集合： "  \  /  b f n r t u
    general_pattern = re.compile(r'(?<!\\)\\(?!["\\/bfnrtu])')
    text = general_pattern.sub(r'\\\\', text)

    return text

def clean_chat_agent_format(content: str):
    Clean_patten = re.compile(pattern=r"```(json|latex)?", flags=re.DOTALL)
    content = re.sub(Clean_patten, "", content)
    content = sanitize_json_escapes(content)
    return content

def load_file_as_string(path: Union[str, Path]) -> str:
    if isinstance(path, str):
        with open(path, "r", encoding="utf-8") as fr:
            return fr.read()
    elif isinstance(path, Path):
        with path.open("r", encoding="utf-8") as fr:
            return fr.read()
    else:
        raise ValueError(path)

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\\/:"*?<>|]', "_", filename)

def save_result(result: str, path: Union[str, Path]) -> None:
    """save a string to a file, if the prefix dir doesn't exit, create them.

    Args:
        result (str): string waiting to be saved.
        path (str): where to save this string.
    """
    if isinstance(path, str):
        path = Path(path)
    directory = path.parent
    # 如果目录不存在，则创建目录
    if not directory.exists():
        directory.mkdir(exist_ok=True, parents=True)
    # 写入文件
    with path.open("w", encoding="utf-8") as fw:
        fw.write(result)