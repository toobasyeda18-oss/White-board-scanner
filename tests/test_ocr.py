import numpy as np

from src.ocr import extract_text
from src.preprocessing import ProcessedImage


class FakeOCREngine:
    def __call__(self, image):
        assert image.ndim == 2
        return [([[0, 0], [10, 0], [10, 5], [0, 5]], "Read this", 0.9)], 0.01


def test_extract_text_reads_lines_and_confidence() -> None:
    image = np.zeros((20, 30), dtype=np.uint8)
    processed = ProcessedImage(image, image, image, image, False)

    result = extract_text(processed, engine=FakeOCREngine())

    assert result.text == "Read this"
    assert result.lines == ["Read this"]
    assert result.mean_confidence == 0.9