# SPARQL Overlap Analysis (02-02-26 release vs. other benchmarks)

Overlap between **WDQL one-per-cluster train set (02-02-26, 82,661 samples)** and
the test sets of other Wikidata KGQA benchmarks, computed with `check_overlap.py`.

Numbers are % of the benchmark test set that overlaps with the WDQL train set.

## WDQL train vs benchmark test sets

| Benchmark | Test size | exact | template | structural |
|-----------|----------:|------:|---------:|-----------:|
| wwq | 1,431 | 0% | 13.1% | 76.1% |
| qald7 | 50 | 0% | 14.0% | 54.0% |
| spinach | 165 | 0% | 2.7% | 6.7% |
| qawiki | 518 | 0% | 3.1% | 27.2% |
| qald10 | 394 | 0% | 1.5% | 35.0% |
| lcquad2 | 6,028 | 0% | 0.1% | 14.1% |
| simplequestions | 5,622 | 0% | 65.0% | 100% |

## Benchmark-internal (own train vs own test)

For reference, the overlap within each benchmark's own train/test split.

| Benchmark | Test size | exact | template | structural |
|-----------|----------:|------:|---------:|-----------:|
| wwq | 1,431 | 21.9% | 92.5% | 97.8% |
| qald7 | 50 | 2.0% | 12.0% | 52.0% |
| qald10 | 394 | 0% | 6.1% | 33.8% |
| lcquad2 | 6,028 | 2.5% | 67.2% | 100.0% |
| simplequestions | 5,622 | 12.5% | 100% | 100% |
| wdql-one-per-cluster (02-02-26) | 10,333 | 1.4% | 52.1% | 62.5% |

## Matching levels

| Level | What is normalized | What it catches |
|-------|--------------------|-----------------|
| **exact** | Whitespace only | Literal string duplicates |
| **template** | Entities, variables, literals (predicates kept) | Same query template, different entities |
| **structural** | Everything including predicates | Same abstract graph shape |
