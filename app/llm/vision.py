"""Short descriptions of figures by a vision model (optional).

Set VISION_MODEL to any image-capable model on an OpenAI-compatible API: a free online tier
(see .env.example) or a local Ollama model. VISION_BASE_URL / VISION_API_KEY may point at a
different provider than the answer model; left empty they reuse LLM_BASE_URL / LLM_API_KEY.

The description is added to the figure's search text, so questions like "what did the chest
X-ray show" can find it. With no vision model, figures are found by their caption alone. A
failure here (including an exhausted free quota) never stops ingestion: the figure keeps its caption.
Large images are shrunk first: online APIs limit image size, and smaller images use less quota.
"""

import base64
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import logger

PROMPT = (
    "You are indexing a figure from a medical research article so it can be found by search. "
    "In at most 4 sentences, describe what the figure shows: its type (chart, X-ray, photo, "
    "flowchart...), what is measured or depicted, and the main finding visible. Do not guess "
    "beyond what is visible. Caption for context: {caption}"
)
_MIME = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
MAX_SIDE = 1280  # pixels
MAX_BYTES = 1_500_000


@lru_cache
def _client():
    from openai import OpenAI

    s = get_settings()
    return OpenAI(
        base_url=s.vision_base_url or s.llm_base_url,
        api_key=s.vision_api_key or s.llm_api_key or "not-needed",
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )


def _prepare(image: bytes, ext: str) -> tuple[bytes, str]:
    """JPEG/PNG under the size limits: many APIs refuse WebP/GIF or very large images.
    If conversion fails the original is sent."""
    if ext in ("jpg", "jpeg", "png") and len(image) <= MAX_BYTES:
        return image, ext
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(image)) as img:
            img = img.convert("RGB")  # drops transparency, converts CMYK/palette images
            img.thumbnail((MAX_SIDE, MAX_SIDE))  # keeps the aspect ratio
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=85)
        return out.getvalue(), "jpg"
    except Exception as exc:
        logger.debug(f"Image conversion failed ({exc}); sending the original")
        return image, ext


def describe_figure(image: bytes, ext: str, caption: str) -> str | None:
    s = get_settings()
    if not s.vision_model or s.llm_provider == "mock":
        return None
    image, ext = _prepare(image, ext.lower().lstrip("."))
    data_url = f"data:image/{_MIME.get(ext, 'png')};base64,{base64.b64encode(image).decode()}"
    try:
        resp = _client().chat.completions.create(
            model=s.vision_model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT.format(caption=caption[:600])},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return text[:1200] or None
    except Exception as exc:
        logger.warning(f"Figure description failed ({exc}); using the caption only")
        return None
