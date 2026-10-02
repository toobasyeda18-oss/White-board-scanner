from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

from src.config import RAW_DATA_DIR
from src.dataset import DatasetRecord, scan_dataset
from src.logging_config import configure_logging
from src.ocr import create_ocr_engine, extract_text
from src.preprocessing import preprocess_image


logger = logging.getLogger(__name__)


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_value in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_value in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + (reference_value != hypothesis_value),
                )
            )
        previous = current
    return previous[-1]


def _normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = re.findall(r"\w+", _normalized_text(reference), flags=re.UNICODE)
    hypothesis_words = re.findall(r"\w+", _normalized_text(hypothesis), flags=re.UNICODE)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _edit_distance(reference_words, hypothesis_words) / len(reference_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = list("".join(_normalized_text(reference).split()))
    hypothesis_chars = list("".join(_normalized_text(hypothesis).split()))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return _edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def evaluate_records(
    records: list[DatasetRecord], engine: Any | None = None
) -> dict[str, Any]:
    ocr_engine = engine if engine is not None else create_ocr_engine()
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for record in records:
        if not record.valid:
            skipped.append({"record_id": record.record_id, "reason": "; ".join(record.issues)})
            continue
        try:
            processed = preprocess_image(record.image_path)
            prediction = extract_text(processed, ocr_engine)
        except Exception as error:
            logger.exception("OCR evaluation failed for %s", record.record_id)
            skipped.append({"record_id": record.record_id, "reason": str(error)})
            continue

        items.append(
            {
                "record_id": record.record_id,
                "wer": word_error_rate(record.transcript, prediction.text),
                "cer": character_error_rate(record.transcript, prediction.text),
                "mean_confidence": prediction.mean_confidence,
            }
        )

    return {
        "evaluated_images": len(items),
        "skipped_images": len(skipped),
        "mean_wer": sum(item["wer"] for item in items) / len(items) if items else None,
        "mean_cer": sum(item["cer"] for item in items) / len(items) if items else None,
        "items": items,
        "skipped": skipped,
    }


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate OCR against folder-local transcripts.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    arguments = parser.parse_args()

    scan = scan_dataset(arguments.raw_dir)
    for issue in scan.issues:
        logger.warning("Dataset note: %s", issue)
    report = evaluate_records(scan.records)
    report["dataset_issues"] = scan.issues
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()