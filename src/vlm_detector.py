"""
vlm_detector.py - Calls the Volcengine ARK API for visual grounding.
Converts a camera frame into a base64 image, sends it to the model,
and parses the returned <bbox> coordinates.
"""
import base64
import re
import logging
import os
import cv2
import numpy as np
from typing import Optional, Tuple

from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

from .config import get_config

load_dotenv()

logger = logging.getLogger(__name__)

# Regex to find ONE bbox (first match wins)
_BBOX_RE = re.compile(r"<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>")


def _frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV BGR frame as JPEG base64 string."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _norm_to_pixel(
    x_min: int, y_min: int, x_max: int, y_max: int,
    frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    """
    Convert normalized [0-999] coordinates to absolute pixel coordinates.
    API returns coords normalised relative to a 1000×1000 grid.
    """
    px_x1 = int(x_min * frame_w / 1000)
    py_y1 = int(y_min * frame_h / 1000)
    px_x2 = int(x_max * frame_w / 1000)
    py_y2 = int(y_max * frame_h / 1000)
    # Clamp to frame bounds
    px_x1 = max(0, min(frame_w - 1, px_x1))
    py_y1 = max(0, min(frame_h - 1, py_y1))
    px_x2 = max(0, min(frame_w - 1, px_x2))
    py_y2 = max(0, min(frame_h - 1, py_y2))
    return px_x1, py_y1, px_x2, py_y2


class VLMDetector:
    """
    Wraps the ARK visual grounding API.

    detect(frame, description) -> (x, y, w, h) in pixels, or None if not found.
    """

    def __init__(self):
        cfg = get_config()
        api_cfg = cfg["api"]
        api_key = os.getenv("ARK_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ARK_API_KEY not found. Make sure it is set in your .env file."
            )
        self._model = api_cfg["model"]
        self._timeout = api_cfg.get("timeout", 30)
        self._client = Ark(
            api_key=api_key,
            base_url=api_cfg["base_url"],
            timeout=self._timeout,
        )

    def detect(
        self,
        frame: np.ndarray,
        description: str,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Ask the VLM to locate `description` in `frame`.

        Returns:
            (x, y, w, h) — top-left corner and size in pixels, or None.
        """
        h, w = frame.shape[:2]
        b64 = _frame_to_base64(frame)

        prompt = (
            "请在图像中找到\u201c" + description + "\u201d，"
            "输出其 bounding box 坐标，格式为 <bbox>x_min y_min x_max y_max</bbox>，"
            "坐标归一化到 1000×1000。"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }]
            )
        except Exception as e:
            logger.error("ARK API call failed: %s", e)
            return None

        content = response.choices[0].message.content or ""
        logger.debug("VLM response: %s", content)

        match = _BBOX_RE.search(content)
        if not match:
            logger.warning("No <bbox> found in response: %s", content)
            return None

        x_min, y_min, x_max, y_max = [int(v) for v in match.groups()]
        px_x1, py_y1, px_x2, py_y2 = _norm_to_pixel(x_min, y_min, x_max, y_max, w, h)

        box_w = px_x2 - px_x1
        box_h = py_y2 - py_y1
        if box_w <= 0 or box_h <= 0:
            logger.warning("Degenerate bbox returned: %s", match.group())
            return None

        logger.info("Detected '%s' at (%d,%d,%d,%d)", description, px_x1, py_y1, box_w, box_h)
        return px_x1, py_y1, box_w, box_h
