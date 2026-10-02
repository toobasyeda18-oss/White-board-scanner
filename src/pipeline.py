from dataclasses import dataclass
from typing import Any

from src.ocr import OCRResult, extract_text
from src.preprocessing import ProcessedImage, preprocess_image


@dataclass
class ScanResult:
    image: ProcessedImage
    ocr: OCRResult


def scan_image(source: bytes | bytearray | str, engine: Any | None = None) -> ScanResult:
    processed = preprocess_image(source)
    ocr_result = extract_text(processed, engine=engine)
    return ScanResult(processed, ocr_result)