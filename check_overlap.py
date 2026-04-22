"""
Detect duplicate or near-duplicate SPARQL queries between two JSONL files.

Usage:
    python check_overlap.py FILE_A FILE_B [--level LEVEL] [--remove-from-b OUTPUT]

FILE_A is the reference set (e.g. train.jsonl); FILE_B is the set to check
(e.g. val.jsonl or a benchmark test set). Overlapping entries are reported by
their `id` field (or line index if no id is present).

Matching levels:
  exact       Whitespace-normalized sparql string must match character-for-character.
  template    Same predicate pattern, entities/vars/literals replaced
              (normalize_properties=False). Recommended for clean train/test splits.
  structural  Everything normalized including properties — same abstract graph shape.

Samples that fail to parse (or have a missing/empty sparql field) are excluded
from both the overlap computation and the --remove-from-b output.
"""

import argparse
import json
import re

from tqdm import tqdm

from sparql_statistics import normalize_tree, tree_to_sparql
from utils import load_sparql_parser, parse_sparql


def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj:
                obj["_line"] = i
            samples.append(obj)
    return samples


def sample_id(s: dict) -> str:
    return s.get("id", f"line:{s.get('_line', '?')}")


def normalize_whitespace(sparql: str) -> str:
    return re.sub(r"\s+", " ", sparql).strip()


LEVELS = ["exact", "template", "structural"]


def get_key(sparql: str, level: str, parser) -> str | None:
    if not sparql:
        return None
    if level == "exact":
        return normalize_whitespace(sparql)
    try:
        tree = parse_sparql(sparql, parser)
        norm = normalize_tree(tree, normalize_properties=(level == "structural"))
        return tree_to_sparql(norm)
    except Exception:
        return None


def build_index(
    samples: list[dict], level: str, parser
) -> tuple[dict[str, list[str]], int]:
    index: dict[str, list[str]] = {}
    failures = 0
    for s in tqdm(samples, desc="Indexing FILE_A", leave=False):
        sparql = s.get("sparql", "")
        key = get_key(sparql, level, parser)
        if key is None:
            failures += 1
            continue
        index.setdefault(key, []).append(sample_id(s))
    return index, failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a", metavar="FILE_A", help="Reference JSONL (e.g. train)")
    ap.add_argument("file_b", metavar="FILE_B", help="JSONL to check (e.g. val/test)")
    ap.add_argument(
        "--level",
        choices=LEVELS,
        default="template",
        help="Matching level (default: template)",
    )
    ap.add_argument(
        "--remove-from-b",
        metavar="OUTPUT",
        help="Write FILE_B minus overlapping entries to OUTPUT",
    )
    args = ap.parse_args()

    print(f"Loading parser...", flush=True)
    parser = load_sparql_parser()

    print(f"Loading {args.file_a}...", flush=True)
    samples_a = load_jsonl(args.file_a)
    print(f"Loading {args.file_b}...", flush=True)
    samples_b = load_jsonl(args.file_b)

    index_a, failures_a = build_index(samples_a, args.level, parser)

    overlapping_ids: list[str] = []
    overlapping_set: set[int] = set()
    failed_set: set[int] = set()

    for i, s in enumerate(tqdm(samples_b, desc="Checking FILE_B", leave=False)):
        sparql = s.get("sparql", "")
        key = get_key(sparql, args.level, parser)
        if key is None:
            failed_set.add(i)
            continue
        if key in index_a:
            overlapping_ids.append(sample_id(s))
            overlapping_set.add(i)

    parseable_b = len(samples_b) - len(failed_set)
    overlap_pct = 100 * len(overlapping_ids) / max(parseable_b, 1)

    # Summary
    print()
    print(f"FILE_A : {args.file_a}")
    print(f"FILE_B : {args.file_b}")
    print(f"Level  : {args.level}")
    print()
    print(f"  FILE_A samples      : {len(samples_a):>7}")
    print(f"  FILE_A parse errors : {failures_a:>7}")
    print(f"  FILE_B samples      : {len(samples_b):>7}")
    print(f"  FILE_B parse errors : {len(failed_set):>7}")
    print(f"  Overlapping in B    : {len(overlapping_ids):>7}  ({overlap_pct:.1f}% of parseable)")

    if overlapping_ids:
        print()
        print("Overlapping IDs in FILE_B:")
        for oid in overlapping_ids:
            print(f"  {oid}")

    if args.remove_from_b:
        kept = [
            s for i, s in enumerate(samples_b)
            if i not in overlapping_set and i not in failed_set
        ]
        with open(args.remove_from_b, "w") as out:
            for s in kept:
                s.pop("_line", None)
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
        print()
        print(
            f"Wrote {len(kept)} samples "
            f"(removed {len(overlapping_set)} overlapping, "
            f"{len(failed_set)} parse errors) to {args.remove_from_b}"
        )


if __name__ == "__main__":
    main()
