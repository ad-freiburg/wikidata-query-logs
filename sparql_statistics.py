import json
import sys
from collections import Counter

from tqdm import tqdm

from utils import load_sparql_parser, parse_sparql

# Wikidata IRI base prefixes, sorted by length (longest first) for correct matching
WIKIDATA_IRI_PREFIXES = [
    ("http://www.wikidata.org/prop/statement/value-normalized/", "psn:"),
    ("http://www.wikidata.org/prop/qualifier/value-normalized/", "pqn:"),
    ("http://www.wikidata.org/prop/reference/value-normalized/", "prn:"),
    ("http://www.wikidata.org/prop/direct-normalized/", "wdtn:"),
    ("http://www.wikidata.org/prop/statement/value/", "psv:"),
    ("http://www.wikidata.org/prop/qualifier/value/", "pqv:"),
    ("http://www.wikidata.org/prop/reference/value/", "prv:"),
    ("http://www.wikidata.org/prop/statement/", "ps:"),
    ("http://www.wikidata.org/prop/qualifier/", "pq:"),
    ("http://www.wikidata.org/prop/reference/", "pr:"),
    ("http://www.wikidata.org/entity/statement/", "wds:"),
    ("http://www.wikidata.org/prop/novalue/", "wdno:"),
    ("http://www.wikidata.org/prop/direct/", "wdt:"),
    ("http://www.wikidata.org/wiki/Special:EntityData/", "wdata:"),
    ("http://www.wikidata.org/reference/", "wdref:"),
    ("http://www.wikidata.org/entity/", "wd:"),
    ("http://www.wikidata.org/value/", "wdv:"),
    ("http://www.wikidata.org/prop/", "p:"),
    ("http://wikiba.se/ontology#", "wikibase:"),
]

# Default prefix map (standard Wikidata prefixes, as predefined by WDQS)
DEFAULT_PREFIX_MAP = {prefix: base for base, prefix in WIKIDATA_IRI_PREFIXES}


WIKIDATA_BASE = "http://www.wikidata.org/"


def get_wikidata_prefix(iri: str) -> str | None:
    """
    Get the canonical Wikidata prefix for an IRI, or None if not a Wikidata IRI.

    Returns "other:" for Wikidata IRIs that don't match any known prefix.
    """
    for base, prefix in WIKIDATA_IRI_PREFIXES:
        if iri.startswith(base):
            return prefix
    # Check if it's still a Wikidata IRI (but with unknown prefix)
    if iri.startswith(WIKIDATA_BASE):
        return "other:"
    return None


def extract_prefix_declarations(tree: dict | list) -> dict[str, str]:
    """
    Extract PREFIX declarations from the parse tree.

    Returns a dict mapping prefix (e.g., "wd:") to base IRI (e.g., "http://www.wikidata.org/entity/").
    Includes default Wikidata prefixes, which can be overridden by explicit declarations.
    """
    # Start with default Wikidata prefixes
    prefix_map: dict[str, str] = DEFAULT_PREFIX_MAP.copy()

    def visit(node: dict | list) -> None:
        if isinstance(node, dict):
            # Look for PrefixDecl which contains PNAME_NS and IRIREF
            if node.get("name") == "PrefixDecl":
                pname_ns = None
                iri_ref = None
                for child in node.get("children", []):
                    if isinstance(child, dict):
                        if child.get("name") == "PNAME_NS":
                            pname_ns = child.get("value")  # e.g., "wd:"
                        elif child.get("name") == "IRIREF":
                            iri_ref = child.get("value")  # e.g., "<http://...>"
                if pname_ns and iri_ref:
                    # Strip angle brackets from IRI
                    base_iri = iri_ref[1:-1]
                    prefix_map[pname_ns] = base_iri

            for child in node.get("children", []):
                visit(child)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(tree)
    return prefix_map


def collect_iris(
    tree: dict | list,
    iris_by_prefix: dict[str, set[str]],
    query_prefix_map: dict[str, str],
) -> None:
    """
    Recursively collect Wikidata IRIs from the parse tree, grouped by canonical prefix.

    Uses query_prefix_map to resolve prefixed names (PNAME_LN) to full IRIs,
    then classifies them by Wikidata prefix.
    """
    if isinstance(tree, dict):
        name = tree.get("name")
        value = tree.get("value")

        if name == "IRIREF" and value:
            # Full IRI like <http://...>, strip brackets
            iri = value[1:-1]
            prefix = get_wikidata_prefix(iri)
            if prefix:
                iris_by_prefix[prefix].add(iri)
        elif name == "PNAME_LN" and value:
            # Prefixed name like wd:Q42 or custom:Q42
            # Find the prefix part (everything up to and including the colon)
            colon_idx = value.find(":")
            if colon_idx != -1:
                query_prefix = value[: colon_idx + 1]  # e.g., "wd:" or "custom:"
                local_name = value[colon_idx + 1 :]  # e.g., "Q42"

                # Look up the base IRI from the query's PREFIX declarations
                base_iri = query_prefix_map.get(query_prefix)
                if base_iri:
                    # Expand to full IRI
                    full_iri = base_iri + local_name
                    # Classify by canonical Wikidata prefix
                    canonical_prefix = get_wikidata_prefix(full_iri)
                    if canonical_prefix:
                        iris_by_prefix[canonical_prefix].add(full_iri)

        for child in tree.get("children", []):
            collect_iris(child, iris_by_prefix, query_prefix_map)
    elif isinstance(tree, list):
        for item in tree:
            collect_iris(item, iris_by_prefix, query_prefix_map)


STRING_LITERAL_NODES = {
    "STRING_LITERAL1",
    "STRING_LITERAL2",
    "STRING_LITERAL_LONG1",
    "STRING_LITERAL_LONG2",
}
NUMERIC_LITERAL_NODES = {
    "INTEGER",
    "DECIMAL",
    "DOUBLE",
    "INTEGER_POSITIVE",
    "INTEGER_NEGATIVE",
    "DECIMAL_POSITIVE",
    "DECIMAL_NEGATIVE",
    "DOUBLE_POSITIVE",
    "DOUBLE_NEGATIVE",
}


def collect_literals(tree: dict | list, literals: set[str]) -> None:
    """Recursively collect unique literals (both string and numeric) from the parse tree.

    Normalizes literal values by stripping quotes from string literals so that:
    - 400, "400", '400', and "400"^^xsd:integer all become 400
    - "hello", 'hello', and "hello"@en all become hello
    """
    if isinstance(tree, dict):
        name = tree.get("name")
        value = tree.get("value")
        if name in STRING_LITERAL_NODES and value:
            # Strip quotes from string literals (single, double, or triple)
            # Handle """ ''' " ' in that order
            normalized_value = value
            for quote in ['"""', "'''", '"', "'"]:
                if normalized_value.startswith(quote) and normalized_value.endswith(
                    quote
                ):
                    normalized_value = normalized_value[len(quote) : -len(quote)]
                    break
            literals.add(normalized_value)
        elif name in NUMERIC_LITERAL_NODES and value:
            literals.add(value)
        for child in tree.get("children", []):
            collect_literals(child, literals)
    elif isinstance(tree, list):
        for item in tree:
            collect_literals(item, literals)


def _extract_string_from_subtree(tree: dict | list) -> str | None:
    """Find and return the first STRING_LITERAL value in a subtree, or None."""
    if isinstance(tree, dict):
        if tree.get("name") in STRING_LITERAL_NODES:
            return tree.get("value")
        for child in tree.get("children", []):
            result = _extract_string_from_subtree(child)
            if result is not None:
                return result
    elif isinstance(tree, list):
        for item in tree:
            result = _extract_string_from_subtree(item)
            if result is not None:
                return result
    return None


def _has_lang_builtin(tree: dict | list) -> bool:
    """Check if a subtree contains a LANG BuiltInCall."""
    if isinstance(tree, dict):
        if tree.get("name") == "BuiltInCall":
            children = tree.get("children", [])
            if (
                children
                and isinstance(children[0], dict)
                and children[0].get("name") == "LANG"
            ):
                return True
        for child in tree.get("children", []):
            if _has_lang_builtin(child):
                return True
    elif isinstance(tree, list):
        for item in tree:
            if _has_lang_builtin(item):
                return True
    return False


def _strip_literal_quotes(value: str) -> str:
    """Strip surrounding quotes from a string literal value."""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def collect_languages(tree: dict | list, languages: set[str]) -> None:
    """
    Recursively collect language identifiers from:
    1. LANGTAG nodes on language-tagged literals (e.g., @en)
    2. LANGMATCHES(LANG(?x), "en*") calls - second argument
    3. LANG(?x) = "en" comparisons in RelationalExpression
    """
    if isinstance(tree, dict):
        name = tree.get("name")
        value = tree.get("value")

        # 1. LANGTAG: @en -> en
        if name == "LANGTAG" and value:
            lang = value.lstrip("@").lower()
            if lang:
                languages.add(lang)

        # 2. BuiltInCall with LANGMATCHES: extract second Expression child
        if name == "BuiltInCall":
            children = tree.get("children", [])
            if (
                children
                and isinstance(children[0], dict)
                and children[0].get("name") == "LANGMATCHES"
            ):
                expressions = [
                    c
                    for c in children
                    if isinstance(c, dict) and c.get("name") == "Expression"
                ]
                if len(expressions) >= 2:
                    lang_str = _extract_string_from_subtree(expressions[1])
                    if lang_str:
                        lang = _strip_literal_quotes(lang_str).rstrip("*").lower()
                        if lang:
                            languages.add(lang)

        # 3. RelationalExpression with ComparisonOp(=) and LANG on one side
        if name == "RelationalExpression":
            children = tree.get("children", [])
            has_eq = any(
                isinstance(c, dict)
                and c.get("name") == "ComparisonOp"
                and any(
                    isinstance(gc, dict) and gc.get("value") == "="
                    for gc in c.get("children", [])
                )
                for c in children
            )
            if has_eq:
                num_exprs = [
                    c
                    for c in children
                    if isinstance(c, dict) and c.get("name") == "NumericExpression"
                ]
                if len(num_exprs) == 2:
                    if _has_lang_builtin(num_exprs[0]):
                        lang_str = _extract_string_from_subtree(num_exprs[1])
                    elif _has_lang_builtin(num_exprs[1]):
                        lang_str = _extract_string_from_subtree(num_exprs[0])
                    else:
                        lang_str = None
                    if lang_str:
                        lang = _strip_literal_quotes(lang_str).lower()
                        if lang:
                            languages.add(lang)

        for child in tree.get("children", []):
            collect_languages(child, languages)
    elif isinstance(tree, list):
        for item in tree:
            collect_languages(item, languages)


def normalize_tree(
    tree: dict | list,
    normalize_properties: bool = False,
    var_map: dict[str, str] | None = None,
    entity_map: dict[str, str] | None = None,
    property_map: dict[str, str] | None = None,
    literal_map: dict[str, str] | None = None,
) -> dict | list:
    """
    Normalize a SPARQL parse tree by replacing:
    - Variables with ?var0, ?var1, ... (preserving binding structure)
    - Entity IRIs (wd:Qxxx) with wd:E0, wd:E1, ...
    - Literals (both string and numeric) with "lit0", "lit1", ... (same value gets same placeholder)
    - Optionally property IRIs with wdt:P0, p:P0, etc.
    - Language tags and datatypes are removed entirely

    The same original value always maps to the same placeholder within a query,
    preserving the structure (e.g., if ?x appears twice, both become ?var0).

    Returns a new normalized tree (does not modify original).
    """
    if var_map is None:
        var_map = {}
    if entity_map is None:
        entity_map = {}
    if property_map is None:
        property_map = {}
    if literal_map is None:
        literal_map = {}

    # Property prefixes (for normalization)
    property_prefixes = {
        "wdt:",
        "p:",
        "ps:",
        "pq:",
        "pr:",
        "psv:",
        "pqv:",
        "prv:",
        "psn:",
        "pqn:",
        "prn:",
        "wdtn:",
        "wdno:",
    }
    # Entity prefixes
    entity_prefixes = {"wd:", "s:", "ref:", "v:"}

    if isinstance(tree, dict):
        name = tree.get("name")
        value = tree.get("value")
        new_node: dict = {"name": name}

        # Skip PREFIX declarations entirely - they don't affect structural equivalence
        if name == "PrefixDecl":
            return new_node

        if name == "VAR1" or name == "VAR2":
            # Variable like ?x or $x - same variable always gets same placeholder
            if value not in var_map:
                var_map[value] = f"?var{len(var_map)}"
            new_node["value"] = var_map[value]
        elif name == "PNAME_LN" and value:
            # Prefixed name like wd:Q42 or wdt:P31
            colon_idx = value.find(":")
            if colon_idx != -1:
                prefix = value[: colon_idx + 1]
                local = value[colon_idx + 1 :]

                if prefix in entity_prefixes:
                    # Entity IRI - same entity always gets same placeholder
                    if value not in entity_map:
                        entity_map[value] = f"{prefix}E{len(entity_map)}"
                    new_node["value"] = entity_map[value]
                elif prefix in property_prefixes:
                    if normalize_properties:
                        # Property IRI - same property always gets same placeholder
                        if value not in property_map:
                            property_map[value] = f"{prefix}P{len(property_map)}"
                        new_node["value"] = property_map[value]
                    else:
                        # Keep property IRIs as-is
                        new_node["value"] = value
                else:
                    # Other prefixed names - keep as-is
                    new_node["value"] = value
            else:
                new_node["value"] = value
        elif name == "IRIREF" and value:
            # Full IRI like <http://...>
            iri = value[1:-1]  # Strip angle brackets
            wd_prefix = get_wikidata_prefix(iri)
            if wd_prefix:
                # Convert to prefixed form for normalization
                for base, prefix in WIKIDATA_IRI_PREFIXES:
                    if iri.startswith(base):
                        local = iri[len(base) :]
                        prefixed = f"{prefix}{local}"

                        if prefix in entity_prefixes:
                            # Same entity always gets same placeholder
                            if prefixed not in entity_map:
                                entity_map[prefixed] = f"{prefix}E{len(entity_map)}"
                            new_node["value"] = f"<{entity_map[prefixed]}>"
                        elif prefix in property_prefixes:
                            if normalize_properties:
                                # Same property always gets same placeholder
                                if prefixed not in property_map:
                                    property_map[prefixed] = (
                                        f"{prefix}P{len(property_map)}"
                                    )
                                new_node["value"] = f"<{property_map[prefixed]}>"
                            else:
                                new_node["value"] = value
                        else:
                            new_node["value"] = value
                        break
                else:
                    new_node["value"] = value
            else:
                # Non-Wikidata IRI - keep as-is
                new_node["value"] = value
        elif name in (
            "STRING_LITERAL1",
            "STRING_LITERAL2",
            "STRING_LITERAL_LONG1",
            "STRING_LITERAL_LONG2",
        ):
            # String literal - same value always gets same placeholder
            if value not in literal_map:
                literal_map[value] = f'"lit{len(literal_map)}"'
            new_node["value"] = literal_map[value]
        elif name in (
            "INTEGER",
            "DECIMAL",
            "DOUBLE",
            "INTEGER_POSITIVE",
            "INTEGER_NEGATIVE",
            "DECIMAL_POSITIVE",
            "DECIMAL_NEGATIVE",
            "DOUBLE_POSITIVE",
            "DOUBLE_NEGATIVE",
        ):
            # Numeric literal - same value always gets same placeholder
            if value not in literal_map:
                literal_map[value] = f'"lit{len(literal_map)}"'
            new_node["value"] = literal_map[value]
        elif value is not None:
            new_node["value"] = value

        if "children" in tree:
            # Filter out LANGTAG and datatype nodes (IRIREF following ^^)
            filtered_children = []
            skip_next = False
            for i, child in enumerate(tree["children"]):
                if skip_next:
                    skip_next = False
                    continue
                # Skip LANGTAG nodes entirely
                if isinstance(child, dict) and child.get("name") == "LANGTAG":
                    continue
                # Skip ^^ operator and following IRIREF (datatype annotation)
                if isinstance(child, dict) and child.get("value") == "^^":
                    # Skip this and the next child (the datatype IRI)
                    if i + 1 < len(tree["children"]):
                        skip_next = True
                    continue
                filtered_children.append(child)

            new_node["children"] = [
                normalize_tree(
                    child,
                    normalize_properties,
                    var_map,
                    entity_map,
                    property_map,
                    literal_map,
                )
                for child in filtered_children
            ]

        return new_node
    elif isinstance(tree, list):
        return [
            normalize_tree(
                item,
                normalize_properties,
                var_map,
                entity_map,
                property_map,
                literal_map,
            )
            for item in tree
        ]
    else:
        return tree


def tree_to_sparql(tree: dict | list) -> str:
    """
    Convert a parse tree to SPARQL by extracting all terminal values.
    This provides a human-readable representation of the (normalized) query.
    Skips PREFIX declarations as they don't affect structural equivalence.
    """
    terminals = []

    def collect_terminals(node: dict | list) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            value = node.get("value")
            children = node.get("children", [])

            # Skip PREFIX declarations entirely
            if name == "PrefixDecl":
                return

            if value and not children:
                # Terminal node - add its value
                terminals.append(value)

            # Recurse into children
            for child in children:
                collect_terminals(child)
        elif isinstance(node, list):
            for item in node:
                collect_terminals(item)

    collect_terminals(tree)
    return " ".join(terminals)


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
    advanced_constructs = {
        "UNION",
        "MINUS",
        "OPTIONAL",
        "GROUP",
        "|",
        "/",
        "PathMod",
        "SubSelect",
    }
    queries_with_advanced = 0

    # Track queries using any property path feature
    path_constructs = {"|", "/", "PathMod"}
    queries_with_paths = 0

    # Track IRIs by prefix (including "other:" for unknown Wikidata prefixes)
    iris_by_prefix: dict[str, set[str]] = {
        prefix: set() for _, prefix in WIKIDATA_IRI_PREFIXES
    }
    iris_by_prefix["other:"] = set()

    # Track unique normalized query patterns
    unique_patterns_keep_props: set[str] = (
        set()
    )  # Normalize entities/literals, keep properties
    unique_patterns_norm_props: set[str] = (
        set()
    )  # Normalize everything including properties

    # Track unique literals (global across all queries)
    all_literals: set[str] = set()

    # Track languages: per-query sets, then aggregate into a Counter
    language_counter: Counter[str] = Counter()  # How many queries use each language
    queries_with_language: int = 0

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

            # Extract PREFIX declarations and collect IRIs
            query_prefix_map = extract_prefix_declarations(tree)
            collect_iris(tree, iris_by_prefix, query_prefix_map)

            # Collect literals
            collect_literals(tree, all_literals)

            # Collect languages (per-query set, then update global counter)
            query_languages: set[str] = set()
            collect_languages(tree, query_languages)
            if query_languages:
                queries_with_language += 1
                for lang in query_languages:
                    language_counter[lang] += 1

            # Normalize and track unique patterns
            # Mode 1: Keep properties, normalize entities/variables/literals
            normalized_keep_props = normalize_tree(tree, normalize_properties=False)
            normalized_sparql_keep_props = tree_to_sparql(normalized_keep_props)
            unique_patterns_keep_props.add(normalized_sparql_keep_props)

            # Mode 2: Normalize everything including properties
            normalized_norm_props = normalize_tree(tree, normalize_properties=True)
            normalized_sparql_norm_props = tree_to_sparql(normalized_norm_props)
            unique_patterns_norm_props.add(normalized_sparql_norm_props)

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
        print(
            f"Queries with advanced constructs: {queries_with_advanced} ({adv_pct:.1f}%)"
        )
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
        # Show "other:" count if any unknown Wikidata prefixes were found
        other_count = len(iris_by_prefix["other:"])
        if other_count > 0:
            print(f"  other: {other_count:,}")
            # Extract unique namespaces from "other" IRIs
            other_namespaces: dict[str, int] = {}
            for iri in iris_by_prefix["other:"]:
                # Extract namespace (everything up to last / or #)
                last_sep = max(iri.rfind("/"), iri.rfind("#"))
                if last_sep > 0:
                    namespace = iri[: last_sep + 1]
                    other_namespaces[namespace] = other_namespaces.get(namespace, 0) + 1
            print("    Namespaces in 'other':")
            for ns, count in sorted(other_namespaces.items(), key=lambda x: -x[1]):
                print(f"      {ns} ({count})")

        # Unique query patterns
        print(f"\n{'-' * 60}")
        print("Unique Query Patterns (after normalization):")
        print(f"{'-' * 60}")

        n_keep = len(unique_patterns_keep_props)
        n_norm = len(unique_patterns_norm_props)
        pct_keep = n_keep / successfully_parsed * 100
        pct_norm = n_norm / successfully_parsed * 100

        print(f"  Keeping properties (normalize entities/vars/literals):")
        print(f"    Unique patterns: {n_keep:,} ({pct_keep:.1f}% of queries)")
        print(f"  Normalizing properties (normalize everything):")
        print(f"    Unique patterns: {n_norm:,} ({pct_norm:.1f}% of queries)")

        # Literal statistics
        print(f"\n{'-' * 60}")
        print("Unique Literals:")
        print(f"{'-' * 60}")
        n_total_literals = len(all_literals)
        print(f"  Total unique literals: {n_total_literals:,}")

        # Language statistics
        print(f"\n{'-' * 60}")
        print("Language Usage:")
        print(f"{'-' * 60}")
        lang_pct = queries_with_language / successfully_parsed * 100
        print(f"  Queries using language: {queries_with_language:,} ({lang_pct:.1f}%)")
        if language_counter:
            print(f"  Language distribution (by number of queries):")
            for lang, count in language_counter.most_common():
                pct = count / successfully_parsed * 100
                print(f"    {lang}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
