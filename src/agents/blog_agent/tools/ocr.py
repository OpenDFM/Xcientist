#!/usr/bin/env python3
"""
OCR + DeepEraser 工具 - 识别图片文字并去除
"""

import os
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR
import warnings

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# 加载配置
from blog_agent.config import load_config
_config = load_config()
_deeperaser_config = _config.get("deeperaser", {})
_DEFAULT_USE_CUDA = _deeperaser_config.get("use_cuda", False)
_DEFAULT_MODEL = _config.get("model", "MiniMax-M2.5")

# 获取 API 配置
_minimax_config = _config.get("minimax", {})
_default_api_key = _minimax_config.get("api_key", "")
_default_base_url = _minimax_config.get("base_url", "https://api.minimaxi.com/v1")

# 屏蔽警告
warnings.filterwarnings("ignore")

# 导入 DeepEraser
from blog_agent.utils.deeperaser import remove_text


def remove_text_from_image(
    workspace_dir: str,
    input_image: str,
    output_image: str,
    mask_image: str = None,
    use_cuda: bool = None,
    ocr_lang: str = "ch",
    text_det_unclip_ratio: float = 1.6,
    humancheckimg: str = None,
) -> dict:
    """
    OCR 识别文字区域，生成掩码，然后用 DeepEraser 去除文字

    Args:
        workspace_dir: 工作目录路径
        input_image: 输入图片文件名
        output_image: 输出图片文件名
        mask_image: 掩码图片文件名 (可选)
        use_cuda: 是否使用 GPU
        ocr_lang: OCR 语言
        text_det_unclip_ratio: 文字检测扩展比例
        humancheckimg: 人工检查图片文件名 (可选)，用于输出标注了OCR置信度和LLM拼写检查结果的图片

    Returns:
        dict: {"success": bool, "input_path": str, "mask_path": str, "output_path": str, "text_count": int, "error": str or None, "humancheck_path": str or None}
    """
    input_path = os.path.join(workspace_dir, input_image)
    output_path = os.path.join(workspace_dir, output_image)

    if not os.path.exists(input_path):
        return {
            "success": False,
            "input_path": input_path,
            "mask_path": None,
            "output_path": output_path,
            "text_count": 0,
            "error": f"输入文件不存在: {input_path}",
        }

    # 使用默认值
    if use_cuda is None:
        use_cuda = _DEFAULT_USE_CUDA
    if mask_image:
        mask_path = os.path.join(workspace_dir, mask_image)
    else:
        mask_path = os.path.join(workspace_dir, "_auto_mask.png")

    # 人工检查图片路径，默认使用 checkimg.png
    if humancheckimg is None:
        humancheckimg = "checkimg.png"
    humancheck_path = os.path.join(workspace_dir, humancheckimg)

    print("[OCR] 正在使用 OCR 识别文字并生成掩码...")
    _generate_mask_from_ocr(
        mask_path,
        input_path,
        ocr_lang=ocr_lang,
        text_det_unclip_ratio=text_det_unclip_ratio,
        humancheck_path=humancheck_path,
    )

    # 使用 DeepEraser 去除文字
    try:
        remove_text(
            input_image_path=input_path,
            mask_image_path=mask_path,
            output_image_path=output_path,
            use_cuda=use_cuda,
        )
        return {
            "success": True,
            "input_path": input_path,
            "mask_path": mask_path,
            "output_path": output_path,
            "text_count": -1,
            "error": None,
            "humancheck_path": humancheck_path,
        }
    except Exception as e:
        return {
            "success": False,
            "input_path": input_path,
            "mask_path": mask_path,
            "output_path": output_path,
            "text_count": -1,
            "error": str(e),
            "humancheck_path": humancheck_path,
        }


def _generate_mask_from_ocr(
    mask_path: str,
    image_path: str,
    ocr_lang: str = "ch",
    text_det_unclip_ratio: float = 1.6,
    humancheck_path: str = None,
) -> None:
    """使用 PaddleOCR 识别文字并生成掩码图"""
    ocr = PaddleOCR(
        lang=ocr_lang,
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
        device="cpu",
        text_det_unclip_ratio=text_det_unclip_ratio,
    )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    result = ocr.predict(image_path)

    # 存储文字信息用于后续处理
    text_data = []  # (text, poly, score)

    for page in result:
        texts = page.get('rec_texts', [])     # 识别出的文本列表
        polys = page.get('dt_polys', [])     # 对应的坐标框列表
        scores = page.get('rec_scores', [])   # 置信度分数

        for text, poly, score in zip(texts, polys, scores):
            # poly 是一个 numpy 数组，包含了 4 个顶点的坐标
            print(f"内容: {text} | 置信度: {score:.2f}")
            text_data.append((text, poly, score))

    text_count = 0
    # 只处理置信度>=0.6的文字区域（低于0.6的忽略）
    valid_items = [(text, poly, score) for text, poly, score in text_data if score >= 0.6]
    text_count = len(valid_items)

    for _, poly, _ in valid_items:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    print(f"[OCR] 识别到 {text_count} 处文字区域（置信度>=0.6）")
    cv2.imwrite(mask_path, mask)

    # 如果需要生成人工检查图片
    if humancheck_path:
        _generate_humancheck_image(image, text_data, humancheck_path)


def _generate_humancheck_image(
    image: np.ndarray,
    text_data: list,
    output_path: str,
) -> None:
    """生成人工检查图片，标注OCR置信度和LLM拼写检查结果"""
    import copy

    # 复制原图用于绘制
    check_image = copy.deepcopy(image)

    # 过滤掉置信度<0.6的文字（既不框也不检查）
    valid_text_data = [(text, poly, score) for text, poly, score in text_data if score >= 0.6]

    # 分离中置信度(0.6<=score<0.9)和高置信度(>=0.9)的文字
    mid_conf_items = [(text, poly, score) for text, poly, score in valid_text_data if score < 0.9]
    high_conf_items = [(text, poly, score) for text, poly, score in valid_text_data if score >= 0.9]

    print(f"[HumanCheck] 中置信度(0.6-0.9): {len(mid_conf_items)} 处")
    print(f"[HumanCheck] 高置信度(>=0.9): {len(high_conf_items)} 处，需要LLM检查拼写")

    # 用红框标注中置信度文字
    for _, poly, _ in mid_conf_items:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(check_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # 用LLM一次性检查所有高置信度文字的拼写
    misspelled_indices = []
    if high_conf_items:
        try:
            from openai import OpenAI

            # 构建待检查的文本列表
            texts_list = "\n".join([f"{idx}: {text}" for idx, (text, _, _) in enumerate(high_conf_items)])

            # 构建 prompt
            prompt = f"""You are a spelling checker for academic/scientific text. The text may contain technical terms, function names, module names, variable names, code snippets, or mathematical expressions.

Your task is to ONLY check for:
1. Obvious spelling errors (typos in normal words)
2. OCR recognition errors (garbled characters, strange symbols that don't belong)

Ignore:
- Technical terms, function names, module names, variable names
- Code snippets, mathematical expressions
- Acronyms and abbreviations

Now please check the following text list and return the indices of problematic items:

{texts_list}

Return format: Only output the indices of problematic items, separated by commas, e.g., "2,5,10"
If nothing is wrong, return "none" """

            # 使用 OpenAI SDK 调用
            client = OpenAI(api_key=_default_api_key, base_url=_default_base_url)
            response = client.chat.completions.create(
                model=_DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            result = response.choices[0].message.content
            print(f"[HumanCheck] LLM返回: {result}")

            # 解析返回的编号
            if result and result.lower() != "none":
                try:
                    misspelled_indices = [int(x.strip()) for x in result.split(",") if x.strip().isdigit()]
                except:
                    print(f"[HumanCheck] 解析LLM返回失败: {result}")
        except Exception as e:
            print(f"[HumanCheck] LLM调用失败: {e}")

        # 用黄框标注拼写错误的文字
        for idx in misspelled_indices:
            if 0 <= idx < len(high_conf_items):
                _, poly, _ = high_conf_items[idx]
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(check_image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    cv2.imwrite(output_path, check_image)
    print(f"[HumanCheck] 已保存到: {output_path}")


def remove_text_batch(
    workspace_dir: str,
    image_list: list,
    use_cuda: bool = False,
) -> list:
    """批量处理多张图片"""
    results = []

    for img_file in image_list:
        name, ext = os.path.splitext(img_file)
        output_file = f"{name}_cleaned{ext}"

        result = remove_text_from_image(
            workspace_dir=workspace_dir,
            input_image=img_file,
            output_image=output_file,
            use_cuda=use_cuda,
        )
        results.append(result)

    return results
