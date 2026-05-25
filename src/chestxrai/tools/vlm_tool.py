"""
VLM Review Tool — multimodal visual inspection of chest X-rays and Grad-CAM heatmaps.

Provider is chosen via environment variables (see llm_config.py):
  VLM_PROVIDER = ollama | openai | anthropic | gemini   (default: ollama)
  VLM_MODEL    = llava:7b | gpt-4o | claude-haiku-4-5-20251001 | gemini-2.5-flash  (default: llava:7b)
"""

import os
import glob
import base64
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "..", "gradcam_outputs")

_PROMPT = (
    "You are a senior radiologist reviewing a chest X-ray alongside Grad-CAM attention maps "
    "produced by a DenseNet121 classifier (NIH ChestX-ray14, 14 pathology classes).\n\n"
    "Analyse the image(s) and report:\n"
    "1. ANATOMICAL FINDINGS — heart size, lung fields, costophrenic angles, visible "
    "consolidation / effusion / opacity / hyperinflation.\n"
    "2. GRAD-CAM VALIDATION — do the highlighted regions map to anatomically correct "
    "locations for the predicted pathology?\n"
    "3. IMAGE QUALITY — positioning, exposure, artefacts that may affect interpretation.\n"
    "4. VISUAL IMPRESSION — your single most clinically concerning observation.\n\n"
    "Be concise and precise. Use standard radiology terminology. "
    "Output only the interpretation — no preamble."
)


class VLMToolInput(BaseModel):
    image_path: str = Field(..., description="Absolute path to the chest X-ray image file")


class VLMReviewTool(BaseTool):
    name: str = "vlm_image_review"
    description: str = (
        "Visually inspect a chest X-ray and its Grad-CAM attention heatmaps using "
        "a multimodal vision model. Confirms whether DenseNet predictions align with "
        "visible anatomical features. Supports Ollama (LLaVA), OpenAI, and Anthropic "
        "via the VLM_PROVIDER / VLM_MODEL environment variables. "
        "Input: absolute path to the chest X-ray image."
    )
    args_schema: Type[BaseModel] = VLMToolInput

    def _run(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return f"[VLMReviewTool] Image not found: {image_path}"

        provider = os.getenv("VLM_PROVIDER", "ollama").lower().strip()
        model    = os.getenv("VLM_MODEL",    "llava:7b")

        # Collect images: original + up to 2 top GradCAM overlays
        image_paths = [image_path]
        basename    = os.path.splitext(os.path.basename(image_path))[0]
        heatmaps    = sorted(glob.glob(os.path.join(GRADCAM_DIR, f"{basename}_*.png")))[:2]
        image_paths.extend(p for p in heatmaps if os.path.exists(p))

        if provider == "ollama":
            return _call_ollama(model, image_paths)
        if provider == "openai":
            return _call_openai(model, image_paths)
        if provider == "anthropic":
            return _call_anthropic(model, image_paths)
        if provider == "gemini":
            return _call_gemini(model, image_paths)
        if provider == "openrouter":
            return _call_openrouter(model, image_paths)

        return f"[VLMReviewTool] Unknown VLM_PROVIDER '{provider}'. Use: ollama, openai, anthropic, gemini, openrouter."


# ── Provider implementations ───────────────────────────────────

def _call_ollama(model: str, image_paths: list[str]) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    images   = [_b64(p) for p in image_paths]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _PROMPT, "images": images}],
        "stream": False,
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return f"[VLMReviewTool] Ollama unreachable at {base_url} — is 'ollama serve' running?"
    except requests.exceptions.Timeout:
        return "[VLMReviewTool] Vision model timed out after 120 s."
    except Exception as e:
        return f"[VLMReviewTool] Ollama error: {e}"


def _call_openai(model: str, image_paths: list[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[VLMReviewTool] OPENAI_API_KEY not set."

    content = []
    for i, p in enumerate(image_paths):
        label   = "Original chest X-ray" if i == 0 else f"Grad-CAM overlay {i}"
        ext     = _ext(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:{ext};base64,{_b64(p)}"}})
        content.append({"type": "text", "text": label})
    content.append({"type": "text", "text": _PROMPT})

    try:
        import openai
        client   = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
        )
        return response.choices[0].message.content
    except ImportError:
        return "[VLMReviewTool] 'openai' package not installed. Run: pip install openai"
    except Exception as e:
        return f"[VLMReviewTool] OpenAI error: {e}"


def _call_anthropic(model: str, image_paths: list[str]) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[VLMReviewTool] ANTHROPIC_API_KEY not set."

    content = []
    for i, p in enumerate(image_paths):
        label      = "Original chest X-ray" if i == 0 else f"Grad-CAM overlay {i}"
        media_type = _ext(p)
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": _b64(p)}})
        content.append({"type": "text", "text": label})
    content.append({"type": "text", "text": _PROMPT})

    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model, max_tokens=600,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
    except ImportError:
        return "[VLMReviewTool] 'anthropic' package not installed. Run: pip install anthropic"
    except Exception as e:
        return f"[VLMReviewTool] Anthropic error: {e}"


def _call_openrouter(model: str, image_paths: list[str]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[VLMReviewTool] OPENROUTER_API_KEY (or OPENAI_API_KEY) not set."

    content = []
    for i, p in enumerate(image_paths):
        label = "Original chest X-ray" if i == 0 else f"Grad-CAM overlay {i}"
        ext   = _ext(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:{ext};base64,{_b64(p)}"}})
        content.append({"type": "text", "text": label})
    content.append({"type": "text", "text": _PROMPT})

    try:
        import openai
        client   = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=600,
        )
        return response.choices[0].message.content
    except ImportError:
        return "[VLMReviewTool] 'openai' package not installed. Run: pip install openai"
    except Exception as e:
        return f"[VLMReviewTool] OpenRouter error: {e}"


def _call_gemini(model: str, image_paths: list[str]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[VLMReviewTool] GEMINI_API_KEY not set."

    try:
        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)

        parts = []
        for i, p in enumerate(image_paths):
            label = "Original chest X-ray" if i == 0 else f"Grad-CAM overlay {i}"
            parts.append(label)
            parts.append(Image.open(p))
        parts.append(_PROMPT)

        response = model_obj.generate_content(parts)
        return response.text
    except ImportError:
        return (
            "[VLMReviewTool] 'google-generativeai' package not installed. "
            "Run: pip install google-generativeai"
        )
    except Exception as e:
        return f"[VLMReviewTool] Gemini error: {e}"


# ── Helpers ────────────────────────────────────────────────────

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _ext(path: str) -> str:
    s = os.path.splitext(path)[1].lower().lstrip(".")
    return f"image/{'jpeg' if s in ('jpg', 'jpeg') else s}"
