import json
import sys
from collections import Counter

from tqdm import tqdm

from utils import load_sparql_parser, parse_sparql

# Wikidata IRI base prefixes, sorted by length (longest first) for correct matching
WIKIDATA_IRI_PREFIXES = [
    ("http://www.wikidata.org/prop/direct-normalized/", "wdtn:"),
    ("http://www.wikidata.org/prop/statement/value-normalized/", "psn:"),
    ("http://www.wikidata.org/prop/qualifier/value-normalized/", "pqn:"),
    ("http://www.wikidata.org/prop/reference/value-normalized/", "prn:"),
    ("http://www.wikidata.org/prop/direct/", "wdt:"),
    ("http://www.wikidata.org/prop/statement/", "ps:"),
    ("http://www.wikidata.org/prop/qualifier/", "pq:"),
    ("http://www.wikidata.org/prop/reference/", "pr:"),
    ("http://www.wikidata.org/entity/", "wd:"),
    ("http://www.wikidata.org/prop/", "p:"),
]


def get_iri_prefix(iri: str) -> str | None:
    """Get the Wikidata prefix for an IRI, or None if not a Wikidata IRI."""
    for base, prefix in WIKIDATA_IRI_PREFIXES:
        if iri.startswith(base):
            return prefix
    return None


def collect_iris(tree: dict | list, iris_by_prefix: dict[str, set[str]]) -> None:
    """Recursively collect Wikidata IRIs from the parse tree, grouped by prefix."""
    if isinstance(tree, dict):
        name = tree.get("name")
        value = tree.get("value")

        if name == "IRIREF" and value:
            # Full IRI like <http://...>, strip brackets
            iri = value[1:-1]
            prefix = get_iri_prefix(iri)
            if prefix:
                iris_by_prefix[prefix].add(iri)
        elif name == "PNAME_LN" and value:
            # Prefixed name like wd:Q42 or wdt:P31
            for base, prefix in WIKIDATA_IRI_PREFIXES:
                if value.startswith(prefix):
                    # Expand to full IRI
                    local_name = value[len(prefix) :]
                    iri = base + local_name
                    iris_by_prefix[prefix].add(iri)
                    break

        for child in tree.get("children", []):
            collect_iris(child, iris_by_prefix)
    elif isinstance(tree, list):
        for item in tree:
            collect_iris(item, iris_by_prefix)


def collect_present_nodes(tree: dict | list, present: set[str]) -> None:
    """Recursively collect node names present in the parse tree."""
    if isinstance(tree, dict):
        if "name" in tree:
            present.add(tree["name"])
        if "children" in tree:
            for child in tree["children"]:
                collect_present_nodes(child, present)
    elif isinstance(tree, list):
        for item in tree:
            collect_present_nodes(item, present)


def count_node_occurrences(tree: dict | list, node_name: str) -> int:
    """Recursively count occurrences of a specific node name in the parse tree."""
    count = 0
    if isinstance(tree, dict):
        if tree.get("name") == node_name:
            count += 1
        if "children" in tree:
            for child in tree["children"]:
                count += count_node_occurrences(child, node_name)
    elif isinstance(tree, list):
        for item in tree:
            count += count_node_occurrences(item, node_name)
    return count


def main() -> None:
    parser = load_sparql_parser()

    # Statistics aggregators
    total_queries = 0
    json_errors = 0
    sparql_parse_errors = 0

    # Count queries containing each construct (presence)
    construct_presence: Counter[str] = Counter()

    # Track triple pattern counts per query
    triple_pattern_counts: list[int] = []

    # Track queries using advanced SPARQL constructs
    advanced_constructs = {"UNION", "MINUS", "OPTIONAL", "GROUP", "|", "/", "PathMod", "SubSelect"}
    queries_with_advanced = 0

    # Track queries using any property path feature
    path_constructs = {"|", "/", "PathMod"}
    queries_with_paths = 0

    # Track IRIs by prefix
    iris_by_prefix: dict[str, set[str]] = {
        prefix: set() for _, prefix in WIKIDATA_IRI_PREFIXES
    }

    # Constructs we're interested in (terminals from sparql.l and rules from sparql.y)
    constructs_of_interest = [
        # Query types (terminals)
        ("SELECT", "SELECT query"),
        ("CONSTRUCT", "CONSTRUCT query"),
        ("DESCRIBE", "DESCRIBE query"),
        ("ASK", "ASK query"),
        # Graph pattern keywords (terminals)
        ("UNION", "UNION"),
        ("MINUS", "MINUS"),
        ("OPTIONAL", "OPTIONAL"),
        ("FILTER", "FILTER"),
        ("SERVICE", "SERVICE"),
        ("BIND", "BIND"),
        ("VALUES", "VALUES"),
        ("EXISTS", "EXISTS"),
        # Solution modifier keywords (terminals)
        ("GROUP", "GROUP BY"),
        ("HAVING", "HAVING"),
        ("ORDER", "ORDER BY"),
        ("LIMIT", "LIMIT"),
        ("OFFSET", "OFFSET"),
        ("DISTINCT", "DISTINCT"),
        # Aggregate keywords (terminals)
        ("COUNT", "COUNT"),
        ("SUM", "SUM"),
        ("MIN", "MIN"),
        ("MAX", "MAX"),
        ("AVG", "AVG"),
        ("SAMPLE", "SAMPLE"),
        ("GROUP_CONCAT", "GROUP_CONCAT"),
        # Property paths (literal tokens)
        ("|", "Path alternative (|)"),
        ("/", "Path sequence (/)"),
        ("PathMod", "Path modifier (?, *, +)"),
        # Subqueries (rule)
        ("SubSelect", "Subquery"),
    ]

    for line in tqdm(sys.stdin, desc="Processing queries", unit=" queries"):
        line = line.strip()
        if not line:
            continue

        try:
            query = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", file=sys.stderr)
            json_errors += 1
            continue

        total_queries += 1

        try:
            tree = parse_sparql(query, parser)
            present: set[str] = set()
            collect_present_nodes(tree, present)

            for node in present:
                construct_presence[node] += 1

            # Check for advanced constructs
            if present & advanced_constructs:
                queries_with_advanced += 1

            # Check for property paths
            if present & path_constructs:
                queries_with_paths += 1

            # Count triple patterns
            triple_count = count_node_occurrences(tree, "TriplesSameSubjectPath")
            triple_pattern_counts.append(triple_count)

            # Collect IRIs by prefix
            collect_iris(tree, iris_by_prefix)

        except Exception as e:
            print(f"SPARQL parse error: {e}", file=sys.stderr)
            sparql_parse_errors += 1

    # Print results
    successfully_parsed = total_queries - sparql_parse_errors

    print(f"\n{'=' * 60}")
    print("SPARQL Query Statistics")
    print(f"{'=' * 60}")
    print(f"\nTotal queries: {total_queries}")
    print(f"JSON decode errors: {json_errors}")
    print(f"SPARQL parse errors: {sparql_parse_errors}")
    print(f"Successfully parsed: {successfully_parsed}")

    if successfully_parsed > 0:
        adv_pct = queries_with_advanced / successfully_parsed * 100
        print(f"Queries with advanced constructs: {queries_with_advanced} ({adv_pct:.1f}%)")
        path_pct = queries_with_paths / successfully_parsed * 100
        print(f"Queries with property paths: {queries_with_paths} ({path_pct:.1f}%)")
        # Triple pattern statistics
        print(f"\n{'-' * 60}")
        print("Triple Pattern Statistics:")
        print(f"{'-' * 60}")
        total_triples = sum(triple_pattern_counts)
        avg_triples = total_triples / len(triple_pattern_counts)
        min_triples = min(triple_pattern_counts)
        max_triples = max(triple_pattern_counts)
        sorted_counts = sorted(triple_pattern_counts)
        median_triples = sorted_counts[len(sorted_counts) // 2]
        print(f"  Total triple patterns: {total_triples}")
        print(f"  Average per query: {avg_triples:.2f}")
        print(f"  Median per query: {median_triples}")
        print(f"  Min per query: {min_triples}")
        print(f"  Max per query: {max_triples}")

        # Distribution
        triple_distribution: Counter[int] = Counter(triple_pattern_counts)
        print("\n  Distribution (count: queries):")
        for count in sorted(triple_distribution.keys())[:15]:
            num_queries = triple_distribution[count]
            pct = num_queries / successfully_parsed * 100
            print(f"    {count}: {num_queries} ({pct:.1f}%)")
        if len(triple_distribution) > 15:
            print(f"    ... ({len(triple_distribution) - 15} more)")

        # Construct presence
        print(f"\n{'-' * 60}")
        print("Construct Presence (queries containing at least one):")
        print(f"{'-' * 60}")

        for node_name, display_name in constructs_of_interest:
            count = construct_presence.get(node_name, 0)
            pct = count / successfully_parsed * 100
            print(f"  {display_name}: {count} ({pct:.1f}%)")

        # Unique IRIs by prefix
        print(f"\n{'-' * 60}")
        print("Unique Wikidata IRIs by Prefix:")
        print(f"{'-' * 60}")

        total_iris = sum(len(iris) for iris in iris_by_prefix.values())
        print(f"  Total unique IRIs: {total_iris:,}")
        for _, prefix in WIKIDATA_IRI_PREFIXES:
            count = len(iris_by_prefix[prefix])
            if count > 0:
                print(f"  {prefix} {count:,}")


if __name__ == "__main__":
    main()
