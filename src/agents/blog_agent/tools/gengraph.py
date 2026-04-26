#!/usr/bin/env python3
"""
从 paper method 文本生成学术风格图片（步骤一）

Usage:
    from gengraph import generate_figure
    generate_figure(method_file="paper_method.txt", output_dir="./output")
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# 从 config 加载 gengraph 配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from blog_agent.config import load_config

_config = load_config()
_gengraph_config = _config.get("gengraph", {})
_providers_config = _gengraph_config.get("providers", {})

DEFAULT_PROVIDER = _gengraph_config.get("provider", "xiai")
DEFAULT_API_KEY = _gengraph_config.get("api_key", "")

# 获取当前 provider 的配置
_current_provider_config = _providers_config.get(DEFAULT_PROVIDER, {})
DEFAULT_BASE_URL = _current_provider_config.get("base_url", "https://api.xi-ai.cn/v1")
DEFAULT_MODEL = _current_provider_config.get("model", "gemini-3-pro-image-preview")




# ============================================================================
# 图像生成 LLM 调用接口
# ============================================================================

def call_llm_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """统一的图像生成 LLM 调用接口"""
    if provider == "bianxie" or provider == "xiai":
        return _call_bianxie_image_generation(prompt, api_key, model, base_url, reference_image)
    if provider == "gemini":
        return _call_gemini_image_generation(prompt, api_key, model, reference_image)
    return _call_openrouter_image_generation(prompt, api_key, model, base_url, reference_image)


def _call_bianxie_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 OpenAI SDK 调用 Bianxie/XiAI 图像生成接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        if reference_image is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            buf = io.BytesIO()
            reference_image.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
            messages = [{"role": "user", "content": message_content}]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = completion.choices[0].message.content if completion and completion.choices else None
        if not content:
            return None

        # 处理 base64 格式
        pattern = r'data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)'
        match = re.search(pattern, content)
        if match:
            image_base64 = match.group(2)
            image_data = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_data))

        # 处理 URL 格式
        url_pattern = r'!\[.*?\]\((https?://[^\)]+)\)'
        url_match = re.search(url_pattern, content)
        if url_match:
            image_url = url_match.group(1)
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))

        return None
    except Exception as e:
        print(f"[Bianxie/XiAI] 图像生成 API 调用失败: {e}")
        raise


def _call_openrouter_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 OpenRouter 图像生成接口"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        if reference_image is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            buf = io.BytesIO()
            reference_image.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
            messages = [{"role": "user", "content": message_content}]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = completion.choices[0].message.content if completion and completion.choices else None
        if not content:
            return None

        url_pattern = r'!\[.*?\]\((https?://[^\)]+)\)'
        match = re.search(url_pattern, content)
        if match:
            image_url = match.group(1)
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))

        return None
    except Exception as e:
        print(f"[OpenRouter] 图像生成 API 调用失败: {e}")
        raise


def _call_gemini_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """使用 Google Gemini 图像生成接口"""
    try:
        import google.genai as genai

        client = genai.Client(api_key=api_key)

        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            image_size="4K",
        )

        if response and hasattr(response, 'generated_images') and response.generated_images:
            first_image = response.generated_images[0]
            if hasattr(first_image, 'image') and first_image.image:
                return first_image.image
            elif hasattr(first_image, 'bytes') and first_image.bytes:
                return Image.open(io.BytesIO(first_image.bytes))

        return None
    except ImportError:
        print("[Gemini] google-genai 未安装，请运行: pip install google-genai")
        raise
    except Exception as e:
        print(f"[Gemini] 图像生成 API 调用失败: {e}")
        raise


# ============================================================================
# 主函数
# ============================================================================

def generate_figure(
    method_file: str,
    output_dir: str,
    output_filename: str = "figure.png",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_reference_image: bool = False,
    reference_image_path: Optional[str] = None,
) -> str:
    """
    从 paper method 文本生成学术风格图片

    Args:
        method_file: 包含 method 文本的文件路径
        output_dir: 输出目录
        output_filename: 输出文件名 (默认: figure.png)
        provider: API Provider (默认: xiai)
        api_key: API Key (默认: 配置的 key)
        model: 模型名称 (默认: provider 的默认值)
        use_reference_image: 是否使用参考图
        reference_image_path: 参考图路径

    Returns:
        生成的图片路径
    """
    # 读取 method text
    method_path = Path(method_file)
    if not method_path.exists():
        raise FileNotFoundError(f"文件不存在: {method_file}")

    with open(method_path, "r", encoding="utf-8") as f:
        method_text = f.read()

    # 获取配置
    provider = provider or DEFAULT_PROVIDER
    api_key = api_key or DEFAULT_API_KEY

    # 从 providers 配置中获取当前 provider 的 base_url 和 model
    provider_config = _providers_config.get(provider, {})
    base_url = provider_config.get("base_url", DEFAULT_BASE_URL)
    model = model or provider_config.get("model", DEFAULT_MODEL)

    # 参考图
    reference_image = None
    if use_reference_image and reference_image_path:
        reference_image = Image.open(reference_image_path)

    # 构建 prompt
    if use_reference_image and reference_image:
        prompt = f"""Generate a figure to visualize the method described below.

You should closely imitate the visual (artistic) style of the reference figure I provide, focusing only on aesthetic aspects, NOT on layout or structure.

Specifically, match:
- overall visual tone and mood
- illustration abstraction level
- line style
- color usage
- shading style
- icon and shape style
- arrow and connector aesthetics
- typography feel

The content structure, number of components, and layout may differ freely.
Only the visual style should be consistent.

The goal is that the figure looks like it was drawn by the same illustrator using the same visual design language as the reference figure.

Below is the method section of the paper:
{method_text}"""
    else:
#        prompt = f"""Generate a professional academic journal style figure for the paper below so as to visualize the method it proposes, below is the method section of this paper:

#{method_text}

#The figure should be engaging and using academic journal style with cute characters.
#"""
        '''prompt=fGenerate a professional academic journal style figure to visualize the proposed method from the paper below. The figure should be both engaging and visually appealing, incorporating cute characters to illustrate concepts.

The method section of the paper is provided here:
{method_text}

---
**Figure Generation Guidelines:**

1.  **Clarity & Readability:** Ensure all text within the figure is exceptionally clear, distinct, and easy to read. Prefer simple, sans-serif fonts that are well-separated from graphic elements.
2.  **Background Design for Clean Editing:**
    * **Focus on Solid or Smooth Gradients:** Use clean, unobtrusive backgrounds. Prioritize solid colors, or minimal, soft abstract shapes.
    * **FORBIDDEN: Grid Background:** NEVER use any grid, graph paper, or dot matrix background. Use solid color or gradient only.
    * **Avoid Busy Patterns:** Explicitly avoid complex patterns, checkerboard designs, intricate textures, or any repeating grid structures in the background. The background should be uniform enough to facilitate easy text removal and re-insertion without visual artifacts.
    * **Optimal Contrast:** Ensure there is high contrast between the text and its immediate background for maximum legibility and OCR accuracy.
3.  **Engaging Visuals:** Integrate "cute characters" and illustrative elements that creatively explain the method, not just decorate it. The overall aesthetic should be suitable for a modern academic journal.
4.  **Composition for Accessibility:** Arrange elements and text boxes with sufficient internal spacing. Text should not be tightly "boxed in" by lines or other graphical elements, allowing for easy OCR bounding box generation.
'''
        prompt=f'''Act as a professional scientific illustrator specializing in "Graphical Abstracts." Your goal is to visualize the methodology described below into a high-quality, academic journal-style figure.

### METHODOLOGY TO VISUALIZE:
#{method_text}

### STRUCTURAL & LOGICAL GUIDELINES (Based on Academic Best Practices):
1. **Insight-Driven Logic**: Do not just draw a flowchart; visualize the "psychological model" of the method. Create a clear "Entry Point" for the reader and ensure a consistent information flow (strictly Left-to-Right or Top-to-Bottom).
2. **The "Rule of 3 and 5"**:
   - Use a maximum of 5 colors (prefer a professional, grayscale-friendly palette like Viridis or a soft academic pastel scheme).
   - Use a maximum of 3 distinct types of arrows to differentiate between 'data flow,' 'causality,' and 'loops.'
   - Avoid redundant or decorative arrows. Only use arrows when they represent actual data flow or causality.
3. **Simplicity**: Remove non-essential variables. Focus on the core mechanism to minimize cognitive load.
4. **Alignment**: Ensure all elements are well-aligned with consistent spacing. (Note: "alignment" refers to layout, NOT background pattern.)
5. **FORBIDDEN**: Grid Background: NEVER use any grid, graph paper, or dot matrix background.
### AESTHETIC STYLE:
- **Tone**: Professional, clean, and minimalist academic style (similar to Nature or Cell journals).
- **Characters**: Incorporate "Cute Characters" (e.g., minimalist, stylized mascot-style agents, robots, or personified icons) to represent the actors or processing units within the method. The characters should be clean, flat-design, and not distracting from the scientific logic.
- **Rendering**: Flat 2D vector art, clean lines, no 3D shadows, consistent line weights.

### FINAL OUTPUT SPECIFICATION:
- A high-resolution, vector-style graphical abstract. 
- Ensure all labels are legible and symbols are standardized. 
- The figure must be self-explanatory, allowing a researcher to understand the method's "Insight" at a single glance.'''

    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"发送请求到: {base_url}")

    # 调用 API
    img = call_llm_image_generation(
        prompt=prompt,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        reference_image=reference_image,
    )

    if img is None:
        raise Exception('API 响应中没有找到图片')

    # 保存图片
    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        img.save(str(output_path), format='PNG')
    except TypeError:
        img.save(str(output_path))
        with Image.open(str(output_path)) as normalized:
            normalized.save(str(output_path), format='PNG')

    print(f"图片已保存: {output_path}")
    return str(output_path)
