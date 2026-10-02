from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from src.config import RAW_DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.preprocessing import load_image


logger = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    record_id: str
    image_path: Path
    transcript_path: Path
    transcript: str
    width: int | None
    height: int | None
    valid: bool
    issues: list[str]


@dataclass
class DatasetScan:
    records: list[DatasetRecord]
    issues: list[str]


def read_transcript(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        logger.warning("Transcript is not UTF-8; decoding with Windows-1252: %s", path)
        return path.read_text(encoding="cp1252").strip()


def _normalized_label(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def scan_dataset(raw_dir: Path = RAW_DATA_DIR) -> DatasetScan:
    if not raw_dir.exists():
        return DatasetScan([], [f"Dataset directory does not exist: {raw_dir}"])

    directories = sorted({path.parent for path in raw_dir.rglob("*") if path.is_file()})
    records: list[DatasetRecord] = []
    scan_issues: list[str] = []

    for directory in directories:
        image_paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        transcript_paths = sorted(path for path in directory.glob("*.txt") if path.is_file())
        if not image_paths and not transcript_paths:
            continue
        relative_directory = directory.relative_to(raw_dir).as_posix()
        if len(image_paths) != 1 or len(transcript_paths) != 1:
            issue = (
                f"{relative_directory}: expected one image and one transcript; "
                f"found {len(image_paths)} image(s) and {len(transcript_paths)} transcript(s)."
            )
            scan_issues.append(issue)
            logger.warning(issue)
            continue

        image_path = image_paths[0]
        transcript_path = transcript_paths[0]
        record_issues: list[str] = []
        width: int | None = None
        height: int | None = None
        try:
            image = load_image(image_path)
            height, width = image.shape[:2]
        except ValueError as error:
            record_issues.append(str(error))

        try:
            transcript = read_transcript(transcript_path)
        except (OSError, UnicodeError) as error:
            transcript = ""
            record_issues.append(f"Unable to read transcript: {error}")
        if not transcript:
            record_issues.append("Transcript is empty.")

        record_id = relative_directory.replace("/", "__") or image_path.stem
        records.append(
            DatasetRecord(
                record_id,
                image_path,
                transcript_path,
                transcript,
                width,
                height,
                not record_issues,
                record_issues,
            )
        )

    labels: dict[str, list[str]] = {}
    for record in records:
        if record.transcript:
            labels.setdefault(_normalized_label(record.transcript), []).append(record.record_id)
    for record_ids in labels.values():
        if len(record_ids) > 1:
            issue = f"Identical transcript contents found for: {', '.join(record_ids)}."
            scan_issues.append(issue)
            logger.warning(issue)

    course_numbers = {
        int(match.group(1))
        for record in records
        if (match := re.search(r"course(\d+)", record.record_id, re.IGNORECASE))
    }
    if course_numbers:
        missing = sorted(set(range(min(course_numbers), max(course_numbers) + 1)) - course_numbers)
        if missing:
            scan_issues.append(f"Course-number gaps in available folders: {', '.join(map(str, missing))}.")

    if not records:
        scan_issues.append("No unambiguous image/transcript pairs were found.")
    return DatasetScan(records, scan_issues)