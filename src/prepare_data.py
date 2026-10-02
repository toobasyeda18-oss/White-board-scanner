from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

import cv2

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.dataset import scan_dataset
from src.logging_config import configure_logging
from src.preprocessing import preprocess_image


logger = logging.getLogger(__name__)


def prepare_dataset(raw_dir: Path = RAW_DATA_DIR, output_dir: Path = PROCESSED_DATA_DIR) -> dict:
    scan = scan_dataset(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    valid_records = [record for record in scan.records if record.valid]

    for record in scan.records:
        preview_path = ""
        detected = False
        if record.valid:
            try:
                processed = preprocess_image(record.image_path)
                preview_file = previews_dir / f"{record.record_id}.png"
                if not cv2.imwrite(str(preview_file), processed.enhanced):
                    raise OSError(f"Could not write processed preview: {preview_file}")
                preview_path = preview_file.relative_to(output_dir).as_posix()
                detected = processed.whiteboard_detected
            except Exception as error:
                record.valid = False
                record.issues.append(f"Preprocessing failed: {error}")
                logger.exception("Preprocessing failed for %s", record.record_id)

        rows.append(
            {
                "record_id": record.record_id,
                "source_image": record.image_path.relative_to(raw_dir).as_posix(),
                "transcript_file": record.transcript_path.relative_to(raw_dir).as_posix(),
                "transcript_characters": len(record.transcript),
                "width": record.width,
                "height": record.height,
                "whiteboard_detected": detected,
                "valid": record.valid,
                "issues": "; ".join(record.issues),
                "processed_image": preview_path,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    fieldnames = list(rows[0]) if rows else [
        "record_id", "source_image", "transcript_file", "transcript_characters",
        "width", "height", "whiteboard_detected", "valid", "issues", "processed_image",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dimensions = [(row.width, row.height) for row in valid_records if row.width and row.height]
    report = {
        "raw_directory": str(raw_dir),
        "paired_records": len(scan.records),
        "valid_records": sum(record.valid for record in scan.records),
        "invalid_records": sum(not record.valid for record in scan.records),
        "image_width_median": statistics.median(width for width, _ in dimensions) if dimensions else None,
        "image_height_median": statistics.median(height for _, height in dimensions) if dimensions else None,
        "issues": scan.issues,
        "manifest": manifest_path.name,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Validate and preprocess the raw whiteboard dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    arguments = parser.parse_args()
    report = prepare_dataset(arguments.raw_dir, arguments.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()