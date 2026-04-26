
#!/usr/bin/env python3
"""
Illustrate: 合并图片生成和OCR文字去除

Usage:
    from illustrate import illustrate
    illustrate(method_file="graph1.md", output_dir="./output", only_gen_img=False)
"""

import os
from pathlib import Path
from typing import Optional

# Import gengraph and ocr
from .gengraph import generate_figure
from .ocr import remove_text_from_image


def illustrate(
    method_file: str,
    output_dir: str,
    output_filename: str = "figure.png",
    only_gen_img: bool = False,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_reference_image: bool = False,
    reference_image_path: Optional[str] = None,
    use_cuda: bool = None,
    ocr_lang: str = "ch",
    text_det_unclip_ratio: float = 1.6,
) -> dict:
    """
    生成学术风格图片，可选是否进行OCR文字去除

    Args:
        method_file: 包含 method 文本的文件路径
        output_dir: 输出目录
        output_filename: 输出文件名 (默认: figure.png)
        only_gen_img: True=只生成图片, False=生成图片+OCR清理
        provider: API Provider (默认: xiai)
        api_key: API Key (默认: 配置的 key)
        model: 模型名称 (默认: provider 的默认值)
        use_reference_image: 是否使用参考图
        reference_image_path: 参考图路径
        use_cuda: 是否使用 GPU (OCR用)
        ocr_lang: OCR 语言
        text_det_unclip_ratio: 文字检测扩展比例

    Returns:
        dict: {
            "success": bool,
            "figure_path": str,  # 生成的图片路径
            "mask_path": str or None,  # OCR mask路径
            "cleaned_path": str or None,  # 清理后图片路径
            "check_path": str or None,  # 检查图片路径
            "error": str or None
        }
    """
    result = {
        "success": False,
        "figure_path": "",
        "mask_path": "",
        "cleaned_path": "",
        "check_path": "",
        "error": "",
    }

    try:
        # Step 1: Generate figure
        figure_path = generate_figure(
            method_file=method_file,
            output_dir=output_dir,
            output_filename=output_filename,
            provider=provider,
            api_key=api_key,
            model=model,
            use_reference_image=use_reference_image,
            reference_image_path=reference_image_path,
        )
        result["figure_path"] = figure_path

        # Step 2: OCR text removal (if not only_gen_img)
        if not only_gen_img:
            input_image = os.path.basename(figure_path)
            output_image = input_image.replace(".png", "_cleaned.png")
            mask_image = input_image.replace(".png", "_mask.png")
            check_image = input_image.replace(".png", "_check.png")

            ocr_result = remove_text_from_image(
                workspace_dir=output_dir,
                input_image=input_image,
                output_image=output_image,
                mask_image=mask_image,
                use_cuda=use_cuda,
                ocr_lang=ocr_lang,
                text_det_unclip_ratio=text_det_unclip_ratio,
                humancheckimg=check_image,
            )

            if ocr_result.get("success"):
                result["mask_path"] = os.path.join(output_dir, mask_image)
                result["cleaned_path"] = os.path.join(output_dir, output_image)
                result["check_path"] = os.path.join(output_dir, check_image)
                result["success"] = True
            else:
                result["error"] = str(ocr_result.get("error") or "")
                result["success"] = True  # Figure generated OK
        else:
            result["success"] = True

    except Exception as e:
        result["error"] = str(e) if e else ""

    return result
