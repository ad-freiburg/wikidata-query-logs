import argparse
import json
import re
from csv import DictReader
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote_plus

from tqdm import tqdm


def clean(sparql: str) -> str:
    """Normalize whitespace in a SPARQL query."""
    return re.sub(r"\s+", " ", sparql, flags=re.DOTALL).strip()


def prepare_file(
    in_file: Path,
    out_files: dict[str, TextIO],
    seen: set[str],
) -> tuple[int, int]:
    """Process a single input file and write unique queries to output files."""
    num_total = 0
    num_duplicate = 0

    with open(in_file, "r") as f:
        reader = DictReader(
            f,
            delimiter="\t",
            fieldnames=["sparql", "timestamp", "source", "user_agent"],
        )

        for row in reader:
            if row["source"] not in out_files:
                continue

            sparql = clean(unquote_plus(row["sparql"]))
            num_total += 1
            if sparql in seen:
                num_duplicate += 1
                continue

            seen.add(sparql)
            out_files[row["source"]].write(json.dumps(sparql) + "\n")

    return num_total, num_duplicate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Wikidata query logs for processing"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input TSV files containing Wikidata query logs",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to save prepared JSONL files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["organic"],
        help="Data splits to prepare (corresponding to 'source' column in the logs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output files",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Whether to show progress bars",
    )

    args = parser.parse_args()

    print(f"Preparing {len(args.files)} input files for splits: {args.splits}")

    out_files: dict[str, TextIO] = {}
    for split in args.splits:
        out_path = args.output_dir / f"{split}.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"  - Skipping {split}: output file already exists")
            continue
        out_files[split] = open(out_path, "w")

    if not out_files:
        print("\nNo files to process")
        return

    num_total = 0
    num_duplicate = 0
    seen: set[str] = set()

    for file in tqdm(
        args.files,
        desc="Processing files",
        leave=False,
        disable=not args.progress,
    ):
        total, duplicate = prepare_file(file, out_files, seen)
        num_total += total
        num_duplicate += duplicate

    for f in out_files.values():
        f.close()

    num_unique = num_total - num_duplicate
    print(f"\nResults:")
    print(f"  - Total queries: {num_total:,}")
    print(f"  - Unique queries: {num_unique:,}")
    print(f"  - Duplicates: {num_duplicate:,} ({num_duplicate / max(num_total, 1):.1%})")
    print("\nDone!")


if __name__ == "__main__":
    main()
