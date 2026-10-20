import hashlib
import json
from pathlib import Path


def save_json(data, output_path):
    """Write a JSON artifact with stable formatting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    return output_path


def calculate_sha256(file_path):
    """Calculate the SHA-256 fingerprint of a file's bytes."""
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_fingerprint(csv_path, output_path, verification_code):
    """Save the dataset fingerprint and student verification code."""
    fingerprint = calculate_sha256(csv_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        f"Verification code: {verification_code}\n"
        f"Dataset: {Path(csv_path).name}\n"
        f"SHA-256: {fingerprint}\n",
        encoding="utf-8",
    )

    return fingerprint