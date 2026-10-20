
"""Reproducible entry point for AI Assignment 2.

Musanze Harvest Intelligence Service
Verification code format: AI-A2-LLLL
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_ARTIFACTS = [
    "artifacts/data_fingerprint.txt",
    "artifacts/regression_metrics.json",
    "artifacts/classification_metrics.json",
    "artifacts/clustering_report.json",
    "artifacts/test_predictions.csv",
    "artifacts/plots/residuals.png",
    "artifacts/plots/confusion_matrix.png",
    "artifacts/plots/clusters.png",
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Musanze Harvest Intelligence pipeline."
    )
    parser.add_argument(
        "--student-id",
        required=True,
        help="Institutional student username.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Last four digits of the registration number.",
    )
    return parser.parse_args()


def validate_identity(student_id, seed):
    """Validate the individual seed and create the verification code."""
    if "@" not in student_id:
        raise ValueError(
            "Provide your institutional username, e.g. user@ines.ac.rw."
        )

    username = student_id.split("@", 1)[0]
    digits = "".join(character for character in username if character.isdigit())

    if len(digits) < 4:
        raise ValueError(
            "The username must contain the registration digits."
        )

    expected_seed = int(digits[-4:])

    if seed < 0 or seed > 9999:
        raise ValueError("Seed must be between 0000 and 9999.")

    if seed != expected_seed:
        raise ValueError(
            f"Incorrect seed. The last four username digits give "
            f"{expected_seed}, but you supplied {seed}."
        )

    return f"AI-A2-{seed:04d}"


def run_script(script_name, *args):
    """Execute a Python script and stop if it fails."""
    script_path = ROOT / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required script does not exist: {script_path}"
        )

    print(f"\n>>> Running {script_name}", flush=True)

    subprocess.run(
        [sys.executable, str(script_path), *map(str, args)],
        cwd=ROOT,
        check=True,
    )


def verify_artifacts():
    """Check that the required output artifacts exist."""
    missing = [
        artifact
        for artifact in REQUIRED_ARTIFACTS
        if not (ROOT / artifact).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Required artifacts are missing:\n- "
            + "\n- ".join(missing)
        )

    fingerprint_path = ROOT / "artifacts" / "data_fingerprint.txt"
    dataset_files = list((ROOT / "data").glob("*.csv"))

    if not dataset_files:
        raise RuntimeError("No generated CSV dataset was found in data/.")

    print("\nGenerated dataset files:")
    for dataset in dataset_files:
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        print(f"  {dataset.relative_to(ROOT)}")
        print(f"  SHA-256: {digest}")

    print("\nDataset fingerprint file:")
    print(fingerprint_path.read_text(encoding="utf-8").strip())

    print("\nAll required artifact files exist.")


def main():
    """Run dataset generation, training, and artifact verification."""
    args = parse_args()

    try:
        verification_code = validate_identity(
            args.student_id,
            args.seed,
        )

        print("=" * 60)
        print("MUSANZE HARVEST INTELLIGENCE SERVICE")
        print("=" * 60)
        print(f"Student ID: {args.student_id}")
        print(f"Random seed: {args.seed}")
        print(f"Verification code: {verification_code}")

        # The existing generator accepts --seed only.
        run_script(
            "data/generate_data.py",
            "--seed",
            args.seed,
        )

        # Run the existing project training/reporting script.
        run_script("check_models.py")

        verify_artifacts()

        print("\nPipeline completed successfully.")

    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\nPIPELINE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
