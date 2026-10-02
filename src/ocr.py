from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.preprocessing import ProcessedImage


@dataclass
class OCRResult:
    text: str
    lines: list[str]
    mean_confidence: float | None


class OCRUnavailableError(RuntimeError):
    pass


def create_ocr_engine() -> Any:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as error:
        raise OCRUnavailableError(
            "RapidOCR is unavailable. Install the project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error
    return RapidOCR()


def extract_text(processed: ProcessedImage, engine: Any | None = None) -> OCRResult:
    ocr_engine = engine if engine is not None else create_ocr_engine()
    result, _ = ocr_engine(processed.enhanced)
    if not result:
        return OCRResult("", [], None)

    lines: list[str] = []
    confidences: list[float] = []
    for item in result:
        if len(item) < 2:
            continue
        text = str(item[1]).strip()
        if text:
            lines.append(text)
            if len(item) > 2:
                try:
                    confidences.append(float(item[2]))
                except (TypeError, ValueError):
                    pass

    confidence = sum(confidences) / len(confidences) if confidences else None
    return OCRResult("\n".join(lines), lines, confidence)