import argparse
import json
import os
import re
from csv import DictReader
from typing import TextIO
from urllib.parse import unquote_plus

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        help="Input TSV files containing Wikidata query logs",
    )
    parser.add_argument(
        "output_dir",
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
    return parser.parse_args()


def clean(sparql: str) -> str:
    return re.sub(r"\s+", " ", sparql, flags=re.DOTALL).strip()


def prepare_file(
    in_file: str,
    out_files: dict[str, TextIO],
    seen: set[str],
) -> tuple[int, int]:
    num_total = 0
    num_duplicate = 0

    reader = DictReader(
        open(in_file, "r"),
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


def prepare(args: argparse.Namespace):
    out_files = {}
    for split in args.splits:
        out_file = os.path.join(args.output_dir, f"{split}.jsonl")
        if os.path.exists(out_file) and not args.overwrite:
            print(f"Output file for {split} in {args.output_dir} already exist")
            continue
        out_files[split] = open(out_file, "w")

    if not out_files:
        return

    num_total = 0
    num_duplicate = 0
    seen = set()

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

    print(
        f"{num_duplicate:,} / {num_total:,} duplicates "
        f"({num_duplicate / max(num_total, 1):.1%}) for "
        f"splits {args.splits}"
    )


if __name__ == "__main__":
    prepare(parse_args())
