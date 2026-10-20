
"""Command-line prediction for Musanze Harvest Intelligence."""

import argparse
import json
import sys
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError

from app import FarmInput, predict_farm

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Predict harvest yield and farm production risk."
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to a JSON file containing one farm record.",
    )
    args = parser.parse_args()

    input_path = Path(args.json)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    try:
        with input_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, dict):
            raise ValueError("The JSON root must be an object.")

        farm = FarmInput.model_validate(raw_data)
        result = predict_farm(farm)
        print(json.dumps(result, indent=2))

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as exc:
        print(
            json.dumps(
                {"error": "Invalid farm input", "details": exc.errors()},
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )
        sys.exit(2)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except HTTPException as exc:
        print(f"ERROR: {exc.detail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
