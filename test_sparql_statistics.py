import pytest

from sparql_statistics import (
    SPARQL_CONSTRUCTS,
    WIKIDATA_IRI_PREFIXES,
    _extract_iris,
    _extract_triples,
    _get_where_body,
    _jaccard,
    collect_iris,
    collect_languages,
    collect_literals,
    collect_present_nodes,
    count_node_occurrences,
    extract_prefix_declarations,
    get_wikidata_prefix,
    is_advanced_query,
    normalize_tree,
    remove_label_service,
    remove_lang_filters,
    remove_rdfs_label_triples,
    tree_to_sparql,
)
from utils import load_sparql_parser, parse_sparql


@pytest.fixture(scope="module")
def parser():
    """Load the SPARQL parser once for all tests."""
    return load_sparql_parser()


def get_present_nodes(query: str, parser) -> set[str]:
    """Helper to parse a query and return present node names."""
    tree = parse_sparql(query, parser)
    present: set[str] = set()
    collect_present_nodes(tree, present)
    return present


class TestQueryTypes:
    def test_select_query(self, parser):
        present = get_present_nodes("SELECT ?x WHERE { ?x ?p ?o }", parser)
        assert "SELECT" in present
        assert "CONSTRUCT" not in present
        assert "ASK" not in present

    def test_ask_query(self, parser):
        present = get_present_nodes("ASK { ?x ?p ?o }", parser)
        assert "ASK" in present
        assert "SELECT" not in present

    def test_construct_query(self, parser):
        present = get_present_nodes("CONSTRUCT { ?x ?p ?o } WHERE { ?x ?p ?o }", parser)
        assert "CONSTRUCT" in present
        assert "SELECT" not in present


class TestGraphPatterns:
    def test_union(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { { ?x <http://a> ?y } UNION { ?x <http://b> ?y } }",
            parser,
        )
        assert "UNION" in present

    def test_no_union(self, parser):
        present = get_present_nodes("SELECT ?x WHERE { ?x <http://a> ?y }", parser)
        assert "UNION" not in present

    def test_minus(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y MINUS { ?x <http://b> ?z } }", parser
        )
        assert "MINUS" in present

    def test_optional(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y OPTIONAL { ?x <http://b> ?z } }",
            parser,
        )
        assert "OPTIONAL" in present

    def test_filter(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y FILTER(?y > 10) }", parser
        )
        assert "FILTER" in present

    def test_bind(self, parser):
        present = get_present_nodes(
            "SELECT ?x ?z WHERE { ?x <http://a> ?y BIND(?y + 1 AS ?z) }", parser
        )
        assert "BIND" in present

    def test_values(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { VALUES ?x { <http://a> <http://b> } ?x ?p ?o }", parser
        )
        assert "VALUES" in present

    def test_service(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { SERVICE <http://endpoint> { ?x ?p ?o } }", parser
        )
        assert "SERVICE" in present

    def test_exists(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y FILTER EXISTS { ?x <http://b> ?z } }",
            parser,
        )
        assert "EXISTS" in present


class TestSolutionModifiers:
    def test_group_by(self, parser):
        present = get_present_nodes(
            "SELECT ?x (COUNT(?y) AS ?c) WHERE { ?x <http://a> ?y } GROUP BY ?x",
            parser,
        )
        assert "GROUP" in present

    def test_having(self, parser):
        present = get_present_nodes(
            "SELECT ?x (COUNT(?y) AS ?c) WHERE { ?x <http://a> ?y } GROUP BY ?x HAVING (COUNT(?y) > 1)",
            parser,
        )
        assert "HAVING" in present

    def test_order_by(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y } ORDER BY ?x", parser
        )
        assert "ORDER" in present

    def test_limit(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y } LIMIT 10", parser
        )
        assert "LIMIT" in present

    def test_offset(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a> ?y } OFFSET 5", parser
        )
        assert "OFFSET" in present

    def test_distinct(self, parser):
        present = get_present_nodes(
            "SELECT DISTINCT ?x WHERE { ?x <http://a> ?y }", parser
        )
        assert "DISTINCT" in present


class TestAggregates:
    def test_count(self, parser):
        present = get_present_nodes(
            "SELECT (COUNT(?x) AS ?c) WHERE { ?x ?p ?o }", parser
        )
        assert "COUNT" in present

    def test_sum(self, parser):
        present = get_present_nodes(
            "SELECT (SUM(?y) AS ?s) WHERE { ?x <http://a> ?y }", parser
        )
        assert "SUM" in present

    def test_avg(self, parser):
        present = get_present_nodes(
            "SELECT (AVG(?y) AS ?a) WHERE { ?x <http://a> ?y }", parser
        )
        assert "AVG" in present

    def test_min_max(self, parser):
        present = get_present_nodes(
            "SELECT (MIN(?y) AS ?mi) (MAX(?y) AS ?ma) WHERE { ?x <http://a> ?y }",
            parser,
        )
        assert "MIN" in present
        assert "MAX" in present


class TestPropertyPaths:
    def test_simple_predicate_no_path_operators(self, parser):
        """Simple predicate should NOT trigger path operators."""
        present = get_present_nodes("SELECT ?x WHERE { ?x <http://a> ?y }", parser)
        assert "|" not in present
        assert "/" not in present
        assert "PathMod" not in present

    def test_path_sequence(self, parser):
        """Path sequence with / should be detected."""
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x <http://a>/<http://b> ?y }", parser
        )
        assert "/" in present
        assert "|" not in present

    def test_path_alternative(self, parser):
        """Path alternative with | should be detected."""
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x (<http://a>|<http://b>) ?y }", parser
        )
        assert "|" in present
        assert "/" not in present

    def test_path_modifier_star(self, parser):
        """Path modifier * should be detected."""
        present = get_present_nodes("SELECT ?x WHERE { ?x <http://a>* ?y }", parser)
        assert "PathMod" in present

    def test_path_modifier_plus(self, parser):
        """Path modifier + should be detected."""
        present = get_present_nodes("SELECT ?x WHERE { ?x <http://a>+ ?y }", parser)
        assert "PathMod" in present

    def test_path_modifier_question(self, parser):
        """Path modifier ? should be detected."""
        present = get_present_nodes("SELECT ?x WHERE { ?x <http://a>? ?y }", parser)
        assert "PathMod" in present

    def test_complex_path(self, parser):
        """Complex path with multiple operators."""
        present = get_present_nodes(
            "SELECT ?x WHERE { ?x (<http://a>|<http://b>)/<http://c>* ?y }", parser
        )
        assert "|" in present
        assert "/" in present
        assert "PathMod" in present


class TestSubquery:
    def test_subquery(self, parser):
        present = get_present_nodes(
            "SELECT ?x WHERE { { SELECT ?x WHERE { ?x <http://a> ?y } } ?x <http://b> ?z }",
            parser,
        )
        assert "SubSelect" in present

    def test_no_subquery(self, parser):
        present = get_present_nodes("SELECT ?x WHERE { ?x <http://a> ?y }", parser)
        assert "SubSelect" not in present


def count_triple_patterns(query: str, parser) -> int:
    """Helper to parse a query and return the number of triple patterns."""
    tree = parse_sparql(query, parser)
    return count_node_occurrences(tree, "TriplesSameSubjectPath")


class TestTriplePatternCounting:
    def test_single_triple(self, parser):
        """A query with one triple pattern."""
        count = count_triple_patterns("SELECT ?x WHERE { ?x <http://a> ?y }", parser)
        assert count == 1

    def test_two_triples(self, parser):
        """A query with two triple patterns."""
        count = count_triple_patterns(
            "SELECT ?x WHERE { ?x <http://a> ?y . ?y <http://b> ?z }", parser
        )
        assert count == 2

    def test_three_triples(self, parser):
        """A query with three triple patterns."""
        count = count_triple_patterns(
            "SELECT ?x WHERE { ?x <http://a> ?y . ?y <http://b> ?z . ?z <http://c> ?w }",
            parser,
        )
        assert count == 3

    def test_triple_in_optional(self, parser):
        """Triples inside OPTIONAL should be counted."""
        count = count_triple_patterns(
            "SELECT ?x WHERE { ?x <http://a> ?y OPTIONAL { ?x <http://b> ?z } }",
            parser,
        )
        assert count == 2

    def test_triples_in_union(self, parser):
        """Triples in both sides of UNION should be counted."""
        count = count_triple_patterns(
            "SELECT ?x WHERE { { ?x <http://a> ?y } UNION { ?x <http://b> ?z } }",
            parser,
        )
        assert count == 2

    def test_triples_in_subquery(self, parser):
        """Triples in subquery should be counted."""
        count = count_triple_patterns(
            "SELECT ?x WHERE { { SELECT ?x WHERE { ?x <http://a> ?y } } ?x <http://b> ?z }",
            parser,
        )
        assert count == 2

    def test_no_triples_values_only(self, parser):
        """VALUES clause without triple patterns."""
        count = count_triple_patterns(
            "SELECT ?x WHERE { VALUES ?x { <http://a> <http://b> } }", parser
        )
        assert count == 0

    def test_complex_query_multiple_triples(self, parser):
        """Complex query with multiple patterns in different clauses."""
        count = count_triple_patterns(
            """SELECT ?x WHERE {
                ?x <http://a> ?y .
                ?y <http://b> ?z
                OPTIONAL { ?z <http://c> ?w }
                FILTER(?w > 10)
            }""",
            parser,
        )
        assert count == 3


class TestGetWikidataPrefix:
    def test_entity_iri(self):
        """Entity IRI should return wd: prefix."""
        assert get_wikidata_prefix("http://www.wikidata.org/entity/Q42") == "wd:"

    def test_prop_direct_iri(self):
        """prop/direct IRI should return wdt: prefix."""
        assert get_wikidata_prefix("http://www.wikidata.org/prop/direct/P31") == "wdt:"

    def test_prop_statement_iri(self):
        """prop/statement IRI should return ps: prefix."""
        assert (
            get_wikidata_prefix("http://www.wikidata.org/prop/statement/P31") == "ps:"
        )

    def test_prop_statement_normalized_iri(self):
        """prop/statement/value-normalized should return psn: (not ps:)."""
        assert (
            get_wikidata_prefix(
                "http://www.wikidata.org/prop/statement/value-normalized/P31"
            )
            == "psn:"
        )

    def test_prop_qualifier_iri(self):
        """prop/qualifier IRI should return pq: prefix."""
        assert (
            get_wikidata_prefix("http://www.wikidata.org/prop/qualifier/P585") == "pq:"
        )

    def test_prop_qualifier_normalized_iri(self):
        """prop/qualifier/value-normalized should return pqn: (not pq:)."""
        assert (
            get_wikidata_prefix(
                "http://www.wikidata.org/prop/qualifier/value-normalized/P585"
            )
            == "pqn:"
        )

    def test_statement_iri(self):
        """entity/statement IRI should return s: prefix (not wd:)."""
        assert (
            get_wikidata_prefix("http://www.wikidata.org/entity/statement/Q42-abc123")
            == "wds:"
        )

    def test_non_wikidata_iri(self):
        """Non-Wikidata IRIs should return None."""
        assert get_wikidata_prefix("http://www.w3.org/2000/01/rdf-schema#label") is None
        assert get_wikidata_prefix("http://schema.org/name") is None


def get_iris_by_prefix(query: str, parser) -> dict[str, set[str]]:
    """Helper to parse a query and return collected IRIs by prefix."""
    tree = parse_sparql(query, parser)
    iris_by_prefix: dict[str, set[str]] = {
        prefix: set() for _, prefix in WIKIDATA_IRI_PREFIXES
    }
    iris_by_prefix["other:"] = set()
    query_prefix_map = extract_prefix_declarations(tree)
    collect_iris(tree, iris_by_prefix, query_prefix_map)
    return iris_by_prefix


class TestCollectIris:
    def test_wd_entity(self, parser):
        """wd: entities should be collected under wd: prefix."""
        iris = get_iris_by_prefix("SELECT ?x WHERE { wd:Q42 ?p ?x }", parser)
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]

    def test_wdt_property(self, parser):
        """wdt: properties should be collected under wdt: prefix."""
        iris = get_iris_by_prefix("SELECT ?x WHERE { ?s wdt:P31 ?x }", parser)
        assert "http://www.wikidata.org/prop/direct/P31" in iris["wdt:"]

    def test_multiple_prefixes(self, parser):
        """IRIs with different prefixes should be collected separately."""
        iris = get_iris_by_prefix("SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }", parser)
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]
        assert "http://www.wikidata.org/prop/direct/P31" in iris["wdt:"]

    def test_full_iri(self, parser):
        """Full Wikidata IRIs should be collected."""
        iris = get_iris_by_prefix(
            "SELECT ?x WHERE { <http://www.wikidata.org/entity/Q42> ?p ?x }", parser
        )
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]

    def test_non_wikidata_iri_ignored(self, parser):
        """Non-Wikidata IRIs should be ignored."""
        iris = get_iris_by_prefix(
            "SELECT ?x WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#label> ?x }",
            parser,
        )
        total_iris = sum(len(s) for s in iris.values())
        assert total_iris == 0

    def test_same_iri_deduplicated_within_prefix(self, parser):
        """Same IRI appearing twice should be deduplicated within its prefix."""
        iris = get_iris_by_prefix(
            "SELECT ?x WHERE { wd:Q42 ?p ?x . wd:Q42 ?q ?y }", parser
        )
        assert len(iris["wd:"]) == 1
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]

    def test_psn_not_matched_by_ps(self, parser):
        """psn: should not be incorrectly matched by ps: prefix."""
        iris = get_iris_by_prefix(
            "SELECT ?x WHERE { ?s ps:P31 ?x . ?s psn:P569 ?y }", parser
        )
        assert "http://www.wikidata.org/prop/statement/P31" in iris["ps:"]
        assert (
            "http://www.wikidata.org/prop/statement/value-normalized/P569"
            in iris["psn:"]
        )
        assert len(iris["ps:"]) == 1
        assert len(iris["psn:"]) == 1

    def test_statement_prefix(self, parser):
        """s: prefix should be collected under s: (not wd:)."""
        iris = get_iris_by_prefix(
            "PREFIX s: <http://www.wikidata.org/entity/statement/> "
            "SELECT ?x WHERE { ?s s:Q42-abc123 ?x }",
            parser,
        )
        assert "http://www.wikidata.org/entity/statement/Q42-abc123" in iris["wds:"]
        assert len(iris["wd:"]) == 0


class TestExtractPrefixDeclarations:
    def test_single_prefix(self, parser):
        """Single PREFIX declaration should be extracted."""
        tree = parse_sparql(
            "PREFIX wd: <http://www.wikidata.org/entity/> SELECT ?x WHERE { ?x ?p ?o }",
            parser,
        )
        prefix_map = extract_prefix_declarations(tree)
        assert prefix_map.get("wd:") == "http://www.wikidata.org/entity/"

    def test_multiple_prefixes(self, parser):
        """Multiple PREFIX declarations should be extracted."""
        tree = parse_sparql(
            "PREFIX wd: <http://www.wikidata.org/entity/> "
            "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
            "SELECT ?x WHERE { ?x ?p ?o }",
            parser,
        )
        prefix_map = extract_prefix_declarations(tree)
        assert prefix_map.get("wd:") == "http://www.wikidata.org/entity/"
        assert prefix_map.get("wdt:") == "http://www.wikidata.org/prop/direct/"

    def test_custom_prefix_name(self, parser):
        """Custom prefix names should be extracted."""
        tree = parse_sparql(
            "PREFIX entity: <http://www.wikidata.org/entity/> "
            "SELECT ?x WHERE { ?x ?p ?o }",
            parser,
        )
        prefix_map = extract_prefix_declarations(tree)
        assert prefix_map.get("entity:") == "http://www.wikidata.org/entity/"

    def test_no_prefixes(self, parser):
        """Query without PREFIX declarations should return default Wikidata prefixes."""
        tree = parse_sparql("SELECT ?x WHERE { ?x ?p ?o }", parser)
        prefix_map = extract_prefix_declarations(tree)
        # Should contain default Wikidata prefixes
        assert "wd:" in prefix_map
        assert "wdt:" in prefix_map
        assert prefix_map["wd:"] == "http://www.wikidata.org/entity/"


class TestCustomPrefixDeclarations:
    def test_custom_entity_prefix(self, parser):
        """Custom prefix for wd: namespace should be resolved correctly."""
        iris = get_iris_by_prefix(
            "PREFIX entity: <http://www.wikidata.org/entity/> "
            "SELECT ?x WHERE { entity:Q42 ?p ?x }",
            parser,
        )
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]

    def test_custom_property_prefix(self, parser):
        """Custom prefix for wdt: namespace should be resolved correctly."""
        iris = get_iris_by_prefix(
            "PREFIX prop: <http://www.wikidata.org/prop/direct/> "
            "SELECT ?x WHERE { ?s prop:P31 ?x }",
            parser,
        )
        assert "http://www.wikidata.org/prop/direct/P31" in iris["wdt:"]

    def test_mixed_standard_and_custom_prefixes(self, parser):
        """Standard and custom prefixes in same query should both work."""
        iris = get_iris_by_prefix(
            "PREFIX entity: <http://www.wikidata.org/entity/> "
            "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
            "SELECT ?x WHERE { entity:Q42 wdt:P31 ?x }",
            parser,
        )
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]
        assert "http://www.wikidata.org/prop/direct/P31" in iris["wdt:"]

    def test_prefix_not_declared_not_collected(self, parser):
        """Prefixed names without matching PREFIX declaration should not be collected."""
        iris = get_iris_by_prefix(
            "SELECT ?x WHERE { undeclared:Q42 ?p ?x }",
            parser,
        )
        # undeclared:Q42 has no PREFIX declaration, so it shouldn't be resolved
        total_iris = sum(len(s) for s in iris.values())
        assert total_iris == 0

    def test_swapped_wd_wdt_prefixes(self, parser):
        """Swapped prefix names should still resolve to correct canonical prefixes."""
        # Here we declare wd: to point to prop/direct (normally wdt:)
        # and wdt: to point to entity (normally wd:)
        iris = get_iris_by_prefix(
            "PREFIX wd: <http://www.wikidata.org/prop/direct/> "
            "PREFIX wdt: <http://www.wikidata.org/entity/> "
            "SELECT ?x WHERE { wdt:Q42 wd:P31 ?x }",
            parser,
        )
        # wdt:Q42 resolves to entity/Q42, should be classified as wd:
        assert "http://www.wikidata.org/entity/Q42" in iris["wd:"]
        # wd:P31 resolves to prop/direct/P31, should be classified as wdt:
        assert "http://www.wikidata.org/prop/direct/P31" in iris["wdt:"]

    def test_unknown_wikidata_prefix_counted_as_other(self, parser):
        """Unknown Wikidata IRIs should be counted under 'other:'."""
        iris = get_iris_by_prefix(
            "SELECT ?x WHERE { <http://www.wikidata.org/unknown/namespace/X123> ?p ?x }",
            parser,
        )
        assert "http://www.wikidata.org/unknown/namespace/X123" in iris["other:"]
        # Should still be counted in total
        total_iris = sum(len(s) for s in iris.values())
        assert total_iris == 1


def normalize_query(query: str, parser, normalize_properties: bool = False) -> str:
    """Helper to parse, normalize, and convert query to normalized SPARQL."""
    tree = parse_sparql(query, parser)
    normalized = normalize_tree(tree, normalize_properties=normalize_properties)
    return tree_to_sparql(normalized)


def assert_queries_normalize_same(
    queries: list[str], parser, normalize_properties: bool = False
) -> None:
    """Assert that all queries in the list normalize to the same SPARQL form."""
    normalized = [normalize_query(q, parser, normalize_properties) for q in queries]
    first = normalized[0]
    for i, norm in enumerate(normalized[1:], 1):
        assert norm == first, (
            f"Query {i + 1} normalized differently than Query 1:\n"
            f"Query 1: {queries[0]}\n"
            f"Query {i + 1}: {queries[i]}\n"
            f"Normalized 1: {first}\n"
            f"Normalized {i + 1}: {norm}"
        )


def assert_queries_normalize_different(
    query1: str, query2: str, parser, normalize_properties: bool = False
) -> None:
    """Assert that two queries normalize to different SPARQL forms."""
    norm1 = normalize_query(query1, parser, normalize_properties)
    norm2 = normalize_query(query2, parser, normalize_properties)
    assert norm1 != norm2, (
        f"Queries normalized to the same form but should be different:\n"
        f"Query 1: {query1}\n"
        f"Query 2: {query2}\n"
        f"Normalized: {norm1}"
    )


def assert_normalizes_to(
    input_query: str,
    expected_normalized_sparql: str,
    parser,
    normalize_properties: bool = False,
) -> None:
    """Assert that input query normalizes to the expected SPARQL form."""
    tree = parse_sparql(input_query, parser)
    normalized = normalize_tree(tree, normalize_properties=normalize_properties)
    normalized_sparql = tree_to_sparql(normalized)

    assert normalized_sparql == expected_normalized_sparql, (
        f"Query did not normalize as expected:\n"
        f"Input:    {input_query}\n"
        f"Expected: {expected_normalized_sparql}\n"
        f"Got:      {normalized_sparql}"
    )


class TestNormalizeTreeVariables:
    def test_single_variable_normalized(self, parser):
        """A single variable should be normalized to ?var0."""
        assert_normalizes_to(
            "SELECT ?x WHERE { ?x ?p ?o }",
            "SELECT ?var0 WHERE { ?var0 ?var1 ?var2 }",
            parser,
        )

    def test_multiple_variables_get_different_placeholders(self, parser):
        """Different variables should get different placeholders."""
        assert_normalizes_to(
            "SELECT ?x ?y WHERE { ?x ?p ?y }",
            "SELECT ?var0 ?var1 WHERE { ?var0 ?var2 ?var1 }",
            parser,
        )

    def test_same_variable_same_placeholder(self, parser):
        """Same variable appearing twice should get same placeholder."""
        assert_normalizes_to(
            "SELECT ?x WHERE { ?x <http://a> ?y . ?x <http://b> ?z }",
            "SELECT ?var0 WHERE { ?var0 <http://a> ?var1 . ?var0 <http://b> ?var2 }",
            parser,
        )


class TestNormalizeTreeEntities:
    def test_wd_entity_normalized(self, parser):
        """wd: entities should be normalized to wd:E0, wd:E1, etc."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 ?var0 }",
            parser,
        )

    def test_same_entity_same_placeholder(self, parser):
        """Same entity appearing twice should get same placeholder."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x . wd:Q42 wdt:P27 ?y }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 ?var0 . wd:E0 wdt:P27 ?var1 }",
            parser,
        )

    def test_different_entities_different_placeholders(self, parser):
        """Different entities should get different placeholders."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 wd:Q5 }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 wd:E1 }",
            parser,
        )

    def test_statement_entity_normalized(self, parser):
        """s: (statement) entities should be normalized (PREFIX declarations ignored)."""
        assert_normalizes_to(
            "PREFIX s: <http://www.wikidata.org/entity/statement/> "
            "SELECT ?x WHERE { ?s s:Q42-abc123 ?x }",
            "SELECT ?var0 WHERE { ?var1 s:E0 ?var0 }",
            parser,
        )


class TestNormalizeTreeLiterals:
    def test_string_literal_normalized(self, parser):
        """Different string VALUES should normalize to same pattern (same structure)."""
        assert_queries_normalize_same(
            [
                'SELECT ?x WHERE { ?x <http://a> "Albert Einstein" }',
                'SELECT ?x WHERE { ?x <http://a> "Marie Curie" }',
            ],
            parser,
        )

    def test_integer_literal_normalized(self, parser):
        """Different numeric VALUES should normalize to same pattern (same structure)."""
        assert_queries_normalize_same(
            [
                "SELECT ?x WHERE { ?x <http://a> 42 }",
                "SELECT ?x WHERE { ?x <http://a> 100 }",
            ],
            parser,
        )

    def test_same_string_literal_same_placeholder(self, parser):
        """Same string literal appearing twice in same query uses same placeholder."""
        # Both "hello" twice should normalize the same regardless of actual value
        q1 = 'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "hello" }'
        q2 = 'SELECT ?x WHERE { ?x <http://a> "world" . ?x <http://b> "world" }'
        assert_queries_normalize_same([q1, q2], parser)

    def test_different_string_literals_different_placeholders(self, parser):
        """Different string literals in same query should create different pattern than same literals."""
        q1 = 'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "world" }'
        q2 = 'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "hello" }'
        assert_queries_normalize_different(q1, q2, parser)

    def test_same_numeric_literal_same_placeholder(self, parser):
        """Same numeric literal appearing twice in same query uses same placeholder."""
        # Both numbers appearing twice should normalize the same regardless of actual value
        q1 = "SELECT ?x WHERE { ?x <http://a> 110.4 . ?x <http://b> 110.4 }"
        q2 = "SELECT ?x WHERE { ?x <http://a> 42.0 . ?x <http://b> 42.0 }"
        assert_queries_normalize_same([q1, q2], parser)

    def test_different_numeric_literals_different_placeholders(self, parser):
        """Different numeric literals in same query should create different pattern."""
        q1 = "SELECT ?x WHERE { ?x <http://a> 110.4 . ?x <http://b> 34 }"
        q2 = "SELECT ?x WHERE { ?x <http://a> 110.4 . ?x <http://b> 110.4 }"
        assert_queries_normalize_different(q1, q2, parser)

    def test_mixed_literals_different_placeholders(self, parser):
        """Mixed string and numeric literals use unified sequence."""
        # All literals (string and numeric) use same "lit0", "lit1", ... sequence
        assert_normalizes_to(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> 42 . ?x <http://c> "world" }',
            'SELECT ?var0 WHERE { ?var0 <http://a> "lit0" . ?var0 <http://b> "lit1" . ?var0 <http://c> "lit2" }',
            parser,
        )

    def test_string_literal_with_and_without_datatype(self, parser):
        """String literal with/without datatype should normalize the same."""
        assert_queries_normalize_same(
            [
                'SELECT ?x WHERE { ?x <http://a> "hello" }',
                'SELECT ?x WHERE { ?x <http://a> "hello"^^<http://www.w3.org/2001/XMLSchema#string> }',
            ],
            parser,
        )

    def test_numeric_literal_with_and_without_datatype(self, parser):
        """Numeric literal with/without datatype should normalize the same."""
        assert_queries_normalize_same(
            [
                'SELECT ?x WHERE { ?x <http://a> "123" }',
                'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#integer> }',
                'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#decimal> }',
            ],
            parser,
        )

    def test_raw_number_and_typed_string_normalize_same(self, parser):
        """Raw numeric literal and string with numeric datatype should normalize the same."""
        assert_queries_normalize_same(
            [
                "SELECT ?x WHERE { ?x <http://a> 400 }",
                'SELECT ?x WHERE { ?x <http://a> "400"^^<http://www.w3.org/2001/XMLSchema#integer> }',
            ],
            parser,
        )


class TestNormalizeTreeProperties:
    def test_properties_kept_by_default(self, parser):
        """Properties should NOT be normalized when normalize_properties=False."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 ?var0 }",
            parser,
            normalize_properties=False,
        )

    def test_properties_normalized_when_enabled(self, parser):
        """Properties should be normalized when normalize_properties=True."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P0 ?var0 }",
            parser,
            normalize_properties=True,
        )

    def test_same_property_same_placeholder(self, parser):
        """Same property appearing twice should get same placeholder."""
        assert_normalizes_to(
            "SELECT ?x ?y WHERE { wd:Q42 wdt:P31 ?x . wd:Q5 wdt:P31 ?y }",
            "SELECT ?var0 ?var1 WHERE { wd:E0 wdt:P0 ?var0 . wd:E1 wdt:P0 ?var1 }",
            parser,
            normalize_properties=True,
        )

    def test_different_properties_different_placeholders(self, parser):
        """Different properties should get different placeholders."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x . wd:Q42 wdt:P27 ?y }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P0 ?var0 . wd:E0 wdt:P1 ?var1 }",
            parser,
            normalize_properties=True,
        )

    def test_different_property_prefixes_shared_counter(self, parser):
        """Different property prefixes share a counter (p:P31 -> p:P0, ps:P31 -> ps:P1)."""
        assert_normalizes_to(
            "SELECT ?x WHERE { ?s p:P31 ?stmt . ?stmt ps:P31 ?x }",
            "SELECT ?var0 WHERE { ?var1 p:P0 ?var2 . ?var2 ps:P1 ?var0 }",
            parser,
            normalize_properties=True,
        )


class TestTreeToCanonicalString:
    def test_deterministic_output(self, parser):
        """Same query should always produce same canonical string."""
        query = "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }"
        canonical1 = normalize_query(query, parser)
        canonical2 = normalize_query(query, parser)
        assert canonical1 == canonical2

    def test_different_queries_different_output(self, parser):
        """Different queries should produce different canonical strings."""
        canonical1 = normalize_query("SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }", parser)
        canonical2 = normalize_query("SELECT ?x WHERE { wd:Q42 wdt:P27 ?x }", parser)
        # With properties kept, these should be different
        assert canonical1 != canonical2


class TestQueryPatternDeduplication:
    def test_same_structure_different_entities_same_pattern(self, parser):
        """Queries with same structure but different entities should have same normalized pattern."""
        canonical1 = normalize_query("SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }", parser)
        canonical2 = normalize_query("SELECT ?x WHERE { wd:Q5 wdt:P31 ?x }", parser)
        # Same pattern (same property), just different entity
        assert canonical1 == canonical2

    def test_same_structure_different_properties_different_pattern(self, parser):
        """Queries with different properties should have different patterns (when keeping properties)."""
        canonical1 = normalize_query(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            parser,
            normalize_properties=False,
        )
        canonical2 = normalize_query(
            "SELECT ?x WHERE { wd:Q42 wdt:P27 ?x }",
            parser,
            normalize_properties=False,
        )
        assert canonical1 != canonical2

    def test_same_structure_different_properties_same_pattern_when_normalized(
        self, parser
    ):
        """Queries with different properties should have same pattern when normalizing properties."""
        canonical1 = normalize_query(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            parser,
            normalize_properties=True,
        )
        canonical2 = normalize_query(
            "SELECT ?x WHERE { wd:Q5 wdt:P27 ?x }",
            parser,
            normalize_properties=True,
        )
        assert canonical1 == canonical2

    def test_different_structure_different_pattern(self, parser):
        """Queries with different structure should have different patterns."""
        canonical1 = normalize_query("SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }", parser)
        canonical2 = normalize_query(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x . ?x wdt:P31 ?y }",
            parser,
        )
        assert canonical1 != canonical2

    def test_variable_names_dont_affect_pattern(self, parser):
        """Different variable names should not affect the pattern."""
        canonical1 = normalize_query("SELECT ?x WHERE { ?x wdt:P31 ?y }", parser)
        canonical2 = normalize_query("SELECT ?a WHERE { ?a wdt:P31 ?b }", parser)
        assert canonical1 == canonical2

    def test_entity_values_dont_affect_pattern(self, parser):
        """Different entity values should not affect the pattern."""
        canonical1 = normalize_query("SELECT ?x WHERE { wd:Q42 wdt:P31 wd:Q5 }", parser)
        canonical2 = normalize_query(
            "SELECT ?x WHERE { wd:Q100 wdt:P31 wd:Q200 }", parser
        )
        assert canonical1 == canonical2


def get_literals(query: str, parser) -> set[str]:
    """Helper: returns set of all literals for a query."""
    tree = parse_sparql(query, parser)
    literals: set[str] = set()
    collect_literals(tree, literals)
    return literals


def get_languages(query: str, parser) -> set[str]:
    """Helper: returns set of languages found in a query."""
    tree = parse_sparql(query, parser)
    languages: set[str] = set()
    collect_languages(tree, languages)
    return languages


class TestCollectLiterals:
    def test_string_literal(self, parser):
        """String literals should have quotes stripped."""
        literals = get_literals('SELECT ?x WHERE { ?x <http://a> "hello" }', parser)
        assert "hello" in literals
        assert len(literals) == 1

    def test_integer_literal(self, parser):
        literals = get_literals("SELECT ?x WHERE { ?x <http://a> 42 }", parser)
        assert "42" in literals
        assert len(literals) == 1

    def test_decimal_literal(self, parser):
        literals = get_literals("SELECT ?x WHERE { ?x <http://a> 3.14 }", parser)
        assert "3.14" in literals

    def test_multiple_literals_deduplicated(self, parser):
        """Same literal appearing twice should be counted once."""
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "hello" }', parser
        )
        assert len(literals) == 1

    def test_multiple_different_literals(self, parser):
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "world" . ?x <http://c> 42 }',
            parser,
        )
        # 2 strings + 1 number = 3 total
        assert len(literals) == 3
        assert "hello" in literals
        assert "world" in literals
        assert "42" in literals

    def test_language_tagged_string_collected(self, parser):
        """The string part of a lang-tagged literal should be collected without quotes."""
        literals = get_literals('SELECT ?x WHERE { ?x <http://a> "hello"@en }', parser)
        assert "hello" in literals

    def test_typed_literal_string_collected(self, parser):
        """The string part of a typed literal (e.g. xsd:date) should be collected without quotes."""
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01"^^<http://www.w3.org/2001/XMLSchema#date> }',
            parser,
        )
        assert "2024-01-01" in literals

    def test_numeric_and_string_same_value_deduplicated(self, parser):
        """400, "400", and "400"^^xsd:integer should all be counted as one literal."""
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> 400 . ?x <http://b> "400" . ?x <http://c> "400"^^<http://www.w3.org/2001/XMLSchema#integer> }',
            parser,
        )
        assert len(literals) == 1
        assert "400" in literals

    def test_no_literals(self, parser):
        literals = get_literals("SELECT ?x WHERE { ?x <http://a> ?y }", parser)
        assert len(literals) == 0


class TestCollectLanguages:
    def test_langtag(self, parser):
        """Language from LANGTAG on a literal."""
        langs = get_languages('SELECT ?x WHERE { ?x <http://a> "hello"@en }', parser)
        assert "en" in langs

    def test_langtag_case_insensitive(self, parser):
        """LANGTAG should be lowercased."""
        langs = get_languages('SELECT ?x WHERE { ?x <http://a> "hello"@EN }', parser)
        assert "en" in langs

    def test_langtag_with_sublang(self, parser):
        """Language subtag like en-US should be preserved (lowercased)."""
        langs = get_languages('SELECT ?x WHERE { ?x <http://a> "hello"@en-US }', parser)
        assert "en-us" in langs

    def test_lang_filter_equality(self, parser):
        """Language from FILTER(LANG(?x) = "en")."""
        langs = get_languages(
            'SELECT ?x WHERE { ?x <http://a> ?label . FILTER(LANG(?label) = "en") }',
            parser,
        )
        assert "en" in langs

    def test_langmatches(self, parser):
        """Language from LANGMATCHES(LANG(?x), "en*") - should strip trailing *."""
        langs = get_languages(
            'SELECT ?x WHERE { ?x <http://a> ?label . FILTER(LANGMATCHES(LANG(?label), "en*")) }',
            parser,
        )
        assert "en" in langs

    def test_langmatches_exact(self, parser):
        """LANGMATCHES without wildcard."""
        langs = get_languages(
            'SELECT ?x WHERE { ?x <http://a> ?label . FILTER(LANGMATCHES(LANG(?label), "fr")) }',
            parser,
        )
        assert "fr" in langs

    def test_multiple_languages(self, parser):
        """Multiple language sources in same query."""
        langs = get_languages(
            "SELECT ?x ?y WHERE { "
            '?x <http://a> "hello"@en . '
            '?x <http://b> ?label . FILTER(LANG(?label) = "de") '
            "}",
            parser,
        )
        assert "en" in langs
        assert "de" in langs

    def test_no_language(self, parser):
        """Query without any language constructs."""
        langs = get_languages("SELECT ?x WHERE { ?x <http://a> ?y }", parser)
        assert len(langs) == 0


class TestAdvancedConstructs:
    """Test the classification of queries as advanced or basic."""

    # Test basic queries (not advanced)
    def test_simple_select_not_advanced(self, parser):
        """Simple SELECT query should NOT be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x ?p ?o }", parser)
        assert not is_advanced_query(tree)

    def test_select_with_filter_not_advanced(self, parser):
        """SELECT with FILTER should NOT be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { ?x <http://a> ?y FILTER(?y > 10) }", parser
        )
        assert not is_advanced_query(tree)

    def test_select_with_filter_lang_not_advanced(self, parser):
        """SELECT with FILTER using LANG function should NOT be classified as advanced."""
        tree = parse_sparql(
            'SELECT ?x WHERE { ?x rdfs:label ?label . FILTER(LANG(?label) = "en") }',
            parser,
        )
        assert not is_advanced_query(tree)

    def test_select_with_order_not_advanced(self, parser):
        """SELECT with ORDER BY should NOT be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x <http://a> ?y } ORDER BY ?x", parser)
        assert not is_advanced_query(tree)

    def test_select_with_limit_not_advanced(self, parser):
        """SELECT with LIMIT should NOT be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x <http://a> ?y } LIMIT 10", parser)
        assert not is_advanced_query(tree)

    def test_select_with_offset_not_advanced(self, parser):
        """SELECT with OFFSET should NOT be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x <http://a> ?y } OFFSET 5", parser)
        assert not is_advanced_query(tree)

    def test_select_with_distinct_not_advanced(self, parser):
        """SELECT DISTINCT should NOT be classified as advanced."""
        tree = parse_sparql("SELECT DISTINCT ?x WHERE { ?x <http://a> ?y }", parser)
        assert not is_advanced_query(tree)

    def test_select_with_all_basic_features_not_advanced(self, parser):
        """SELECT with all basic features should NOT be classified as advanced."""
        tree = parse_sparql(
            "SELECT DISTINCT ?x WHERE { ?x <http://a> ?y FILTER(?y > 10) } ORDER BY ?x LIMIT 10 OFFSET 5",
            parser,
        )
        assert not is_advanced_query(tree)

    def test_ask_query_not_advanced(self, parser):
        """ASK query should NOT be classified as advanced."""
        tree = parse_sparql("ASK { ?x ?p ?o }", parser)
        assert not is_advanced_query(tree)

    def test_construct_query_not_advanced(self, parser):
        """CONSTRUCT query should NOT be classified as advanced."""
        tree = parse_sparql("CONSTRUCT { ?x ?p ?o } WHERE { ?x ?p ?o }", parser)
        assert not is_advanced_query(tree)

    # Test advanced queries
    def test_union_is_advanced(self, parser):
        """Query with UNION should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { { ?x <http://a> ?y } UNION { ?x <http://b> ?y } }",
            parser,
        )
        assert is_advanced_query(tree)

    def test_optional_is_advanced(self, parser):
        """Query with OPTIONAL should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { ?x <http://a> ?y OPTIONAL { ?x <http://b> ?z } }",
            parser,
        )
        assert is_advanced_query(tree)

    def test_minus_is_advanced(self, parser):
        """Query with MINUS should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { ?x <http://a> ?y MINUS { ?x <http://b> ?z } }",
            parser,
        )
        assert is_advanced_query(tree)

    def test_group_by_is_advanced(self, parser):
        """Query with GROUP BY should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x (COUNT(?y) AS ?c) WHERE { ?x <http://a> ?y } GROUP BY ?x",
            parser,
        )
        assert is_advanced_query(tree)

    def test_having_is_advanced(self, parser):
        """Query with HAVING should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x (COUNT(?y) AS ?c) WHERE { ?x <http://a> ?y } GROUP BY ?x HAVING (COUNT(?y) > 1)",
            parser,
        )
        assert is_advanced_query(tree)

    def test_count_is_advanced(self, parser):
        """Query with COUNT aggregate should be classified as advanced."""
        tree = parse_sparql("SELECT (COUNT(?x) AS ?c) WHERE { ?x ?p ?o }", parser)
        assert is_advanced_query(tree)

    def test_sum_is_advanced(self, parser):
        """Query with SUM aggregate should be classified as advanced."""
        tree = parse_sparql("SELECT (SUM(?y) AS ?s) WHERE { ?x <http://a> ?y }", parser)
        assert is_advanced_query(tree)

    def test_bind_is_advanced(self, parser):
        """Query with BIND should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x ?z WHERE { ?x <http://a> ?y BIND(?y + 1 AS ?z) }", parser
        )
        assert is_advanced_query(tree)

    def test_values_is_advanced(self, parser):
        """Query with VALUES should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { VALUES ?x { <http://a> <http://b> } ?x ?p ?o }",
            parser,
        )
        assert is_advanced_query(tree)

    def test_property_path_sequence_is_advanced(self, parser):
        """Query with property path sequence (/) should be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x <http://a>/<http://b> ?y }", parser)
        assert is_advanced_query(tree)

    def test_property_path_alternative_is_advanced(self, parser):
        """Query with property path alternative (|) should be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x (<http://a>|<http://b>) ?y }", parser)
        assert is_advanced_query(tree)

    def test_property_path_modifier_is_advanced(self, parser):
        """Query with property path modifier (*, +, ?) should be classified as advanced."""
        tree = parse_sparql("SELECT ?x WHERE { ?x <http://a>* ?y }", parser)
        assert is_advanced_query(tree)

    def test_subquery_is_advanced(self, parser):
        """Query with subquery should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { { SELECT ?x WHERE { ?x <http://a> ?y } } ?x <http://b> ?z }",
            parser,
        )
        assert is_advanced_query(tree)

    def test_service_is_advanced(self, parser):
        """Query with SERVICE should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { SERVICE <http://endpoint> { ?x ?p ?o } }", parser
        )
        assert is_advanced_query(tree)

    def test_exists_is_advanced(self, parser):
        """Query with EXISTS should be classified as advanced."""
        tree = parse_sparql(
            "SELECT ?x WHERE { ?x <http://a> ?y FILTER EXISTS { ?x <http://b> ?z } }",
            parser,
        )
        assert is_advanced_query(tree)


class TestNormalizeLanguageTags:
    def test_language_tag_removed(self, parser):
        """Language tags should be removed during normalization."""
        canonical1 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@en }', parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@de }', parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" }', parser
        )
        # All three should be identical after normalization
        assert canonical1 == canonical2 == canonical3, (
            f"Language tags not properly normalized:\n"
            f"@en: {canonical1}\n"
            f"@de: {canonical2}\n"
            f"no tag: {canonical3}"
        )

    def test_multiple_language_tags_removed(self, parser):
        """Multiple language tags should all be removed."""
        canonical1 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@en . ?x <http://b> "world"@fr }',
            parser,
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@de . ?x <http://b> "world"@es }',
            parser,
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "world" }', parser
        )
        assert canonical1 == canonical2 == canonical3, (
            f"Multiple language tags not properly normalized:\n"
            f"en/fr: {canonical1}\n"
            f"de/es: {canonical2}\n"
            f"no tags: {canonical3}"
        )

    def test_language_subtag_removed(self, parser):
        """Language subtags like en-US should be removed."""
        canonical1 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@en-US }', parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@en-GB }', parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@en }', parser
        )
        canonical4 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" }', parser
        )
        assert canonical1 == canonical2 == canonical3 == canonical4, (
            f"Language subtags not properly normalized:\n"
            f"en-US: {canonical1}\n"
            f"en-GB: {canonical2}\n"
            f"en: {canonical3}\n"
            f"no tag: {canonical4}"
        )


class TestNormalizeDatatypes:
    def test_datatype_removed(self, parser):
        """Datatypes should be removed during normalization."""
        canonical1 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01"^^<http://www.w3.org/2001/XMLSchema#date> }',
            parser,
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01"^^<http://www.w3.org/2001/XMLSchema#string> }',
            parser,
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01" }', parser
        )
        assert canonical1 == canonical2 == canonical3, (
            f"Datatypes not properly normalized:\n"
            f"date: {canonical1}\n"
            f"string: {canonical2}\n"
            f"no type: {canonical3}"
        )

    def test_numeric_datatype_removed(self, parser):
        """Numeric datatypes should be removed."""
        canonical1 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#integer> }',
            parser,
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#decimal> }',
            parser,
        )
        canonical3 = normalize_query('SELECT ?x WHERE { ?x <http://a> "123" }', parser)
        assert canonical1 == canonical2 == canonical3, (
            f"Numeric datatypes not properly normalized:\n"
            f"integer: {canonical1}\n"
            f"decimal: {canonical2}\n"
            f"no type: {canonical3}"
        )

    def test_multiple_datatypes_removed(self, parser):
        """Multiple datatypes in same query should all be removed."""
        canonical1 = normalize_query(
            "SELECT ?x WHERE { "
            '?x <http://a> "2024"^^<http://www.w3.org/2001/XMLSchema#gYear> . '
            '?x <http://b> "42"^^<http://www.w3.org/2001/XMLSchema#integer> '
            "}",
            parser,
        )
        canonical2 = normalize_query(
            "SELECT ?x WHERE { "
            '?x <http://a> "2024"^^<http://www.w3.org/2001/XMLSchema#string> . '
            '?x <http://b> "42"^^<http://www.w3.org/2001/XMLSchema#decimal> '
            "}",
            parser,
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024" . ?x <http://b> "42" }', parser
        )
        assert canonical1 == canonical2 == canonical3, (
            f"Multiple datatypes not properly normalized:\n"
            f"gYear/int: {canonical1}\n"
            f"string/dec: {canonical2}\n"
            f"no types: {canonical3}"
        )


class TestNormalizeLanguageAndDatatype:
    def test_language_and_datatype_both_removed(self, parser):
        """Queries with both language tags and datatypes should normalize the same."""
        canonical1 = normalize_query(
            "SELECT ?x WHERE { "
            '?x <http://a> "hello"@en . '
            '?x <http://b> "2024"^^<http://www.w3.org/2001/XMLSchema#gYear> '
            "}",
            parser,
        )
        canonical2 = normalize_query(
            "SELECT ?x WHERE { "
            '?x <http://a> "hello"@de . '
            '?x <http://b> "2024"^^<http://www.w3.org/2001/XMLSchema#string> '
            "}",
            parser,
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "2024" }', parser
        )
        assert canonical1 == canonical2 == canonical3, (
            f"Language and datatypes not properly normalized:\n"
            f"en/gYear: {canonical1}\n"
            f"de/string: {canonical2}\n"
            f"no tags: {canonical3}"
        )


class TestNormalizeWikidataQueries:
    def test_wikidata_query_with_language(self, parser):
        """Real Wikidata query with language tags should normalize properly."""
        canonical1 = normalize_query(
            "SELECT ?itemLabel WHERE { "
            "wd:Q42 rdfs:label ?itemLabel . "
            'FILTER(LANG(?itemLabel) = "en") '
            "}",
            parser,
        )
        canonical2 = normalize_query(
            "SELECT ?itemLabel WHERE { "
            "wd:Q5 rdfs:label ?itemLabel . "
            'FILTER(LANG(?itemLabel) = "de") '
            "}",
            parser,
        )
        # These should be the same pattern (different entity, different language string)
        assert canonical1 == canonical2, (
            f"Wikidata queries with language filters not properly normalized:\n"
            f"Q42/en: {canonical1}\n"
            f"Q5/de: {canonical2}"
        )

    def test_wikidata_query_with_language_tag_on_literal(self, parser):
        """Wikidata query with language-tagged literals should normalize."""
        canonical1 = normalize_query(
            "SELECT ?x WHERE { "
            'wd:Q42 rdfs:label "Albert Einstein"@en . '
            "wd:Q42 wdt:P31 ?x "
            "}",
            parser,
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { wd:Q5 rdfs:label "Mensch"@de . wd:Q5 wdt:P31 ?x }',
            parser,
        )
        # Same structure, different entities and language tags
        assert canonical1 == canonical2, (
            f"Wikidata queries with language-tagged literals not properly normalized:\n"
            f"Q42/@en: {canonical1}\n"
            f"Q5/@de: {canonical2}"
        )


# ── WDQL comparison helpers ───────────────────────────────────────────────────


class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        # |{2}| / |{1, 2, 3, 4}| = 1/4
        assert _jaccard({1, 2}, {2, 3, 4}) == pytest.approx(1 / 4)

    def test_one_subset_of_other(self):
        # |{1}| / |{1, 2}| = 1/2
        assert _jaccard({1}, {1, 2}) == pytest.approx(1 / 2)

    def test_both_empty(self):
        assert _jaccard(set(), set()) == 1.0

    def test_one_empty(self):
        assert _jaccard(set(), {1}) == 0.0
        assert _jaccard({1}, set()) == 0.0


def _extract_triples_from_query(
    query: str, parser, normalize_literals: bool = False
) -> set[str]:
    """Parse a query and extract its triple strings."""
    tree = parse_sparql(query, parser)
    body = _get_where_body(tree)
    assert body is not None
    return _extract_triples(body, normalize_literals=normalize_literals)


class TestGetWhereBody:
    def test_returns_group_graph_pattern(self, parser):
        tree = parse_sparql("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser)
        body = _get_where_body(tree)
        assert body is not None
        assert body.get("name") == "GroupGraphPattern"

    def test_ask_query(self, parser):
        tree = parse_sparql("ASK { ?x wdt:P31 wd:Q5 }", parser)
        body = _get_where_body(tree)
        assert body is not None
        assert body.get("name") == "GroupGraphPattern"

    def test_no_where_clause(self, parser):
        # DESCRIBE without WHERE has no WhereClause
        tree = parse_sparql("DESCRIBE wd:Q42", parser)
        body = _get_where_body(tree)
        assert body is None


class TestRemoveLabelService:
    def test_removes_wikibase_label_service_by_iri(self, parser):
        query = (
            "SELECT ?item ?itemLabel WHERE { "
            "?item wdt:P31 wd:Q5 . "
            "SERVICE <http://wikiba.se/ontology#label> { "
            '  <http://www.bigdata.com/rdf#serviceParam> wikibase:language "en" '
            "} "
            "}"
        )
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped, had = remove_label_service(body)
        assert had is True
        # SERVICE node should still exist but have no children
        from sparql_statistics import _find_all_nodes

        services = _find_all_nodes(stripped, "ServiceGraphPattern")
        assert all("children" not in s for s in services)

    def test_removes_wikibase_label_service_by_pname(self, parser):
        query = (
            "SELECT ?item ?itemLabel WHERE { "
            "?item wdt:P31 wd:Q5 . "
            "SERVICE wikibase:label { "
            '  bd:serviceParam wikibase:language "en" '
            "} "
            "}"
        )
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped, had = remove_label_service(body)
        assert had is True
        from sparql_statistics import _find_all_nodes

        services = _find_all_nodes(stripped, "ServiceGraphPattern")
        assert all("children" not in s for s in services)

    def test_keeps_non_label_service(self, parser):
        query = "SELECT ?x WHERE { SERVICE <http://other.endpoint/> { ?x ?p ?o } }"
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped, had = remove_label_service(body)
        assert had is False
        from sparql_statistics import _find_all_nodes

        services = _find_all_nodes(stripped, "ServiceGraphPattern")
        assert len(services) == 1
        assert "children" in services[0]

    def test_no_service_unchanged(self, parser):
        query = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped, had = remove_label_service(body)
        assert had is False
        from sparql_statistics import _find_all_nodes

        services = _find_all_nodes(stripped, "ServiceGraphPattern")
        assert len(services) == 0


class TestRemoveRdfsLabelTriples:
    def test_removes_rdfs_label_triple_by_iri(self, parser):
        query = (
            "SELECT ?item ?label WHERE { "
            "?item wdt:P31 wd:Q5 . "
            "?item <http://www.w3.org/2000/01/rdf-schema#label> ?label "
            "}"
        )
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped = remove_rdfs_label_triples(body)
        from sparql_statistics import _find_all_nodes

        triples = _find_all_nodes(stripped, "TriplesSameSubjectPath")
        # Only the wdt:P31 triple should remain (with children)
        with_children = [t for t in triples if "children" in t]
        assert len(with_children) == 1

    def test_removes_rdfs_label_triple_by_pname(self, parser):
        query = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
            "SELECT ?item ?label WHERE { "
            "?item wdt:P31 wd:Q5 . "
            "?item rdfs:label ?label "
            "}"
        )
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped = remove_rdfs_label_triples(body)
        from sparql_statistics import _find_all_nodes

        triples = _find_all_nodes(stripped, "TriplesSameSubjectPath")
        with_children = [t for t in triples if "children" in t]
        assert len(with_children) == 1

    def test_keeps_multi_predicate_triple_with_rdfs_label(self, parser):
        # A triple with multiple predicates (p1 ; rdfs:label) should NOT be removed
        query = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
            "SELECT ?item ?name ?label WHERE { "
            "?item wdt:P31 wd:Q5 ; rdfs:label ?label "
            "}"
        )
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped = remove_rdfs_label_triples(body)
        from sparql_statistics import _find_all_nodes

        triples = _find_all_nodes(stripped, "TriplesSameSubjectPath")
        with_children = [t for t in triples if "children" in t]
        assert len(with_children) == 1

    def test_no_rdfs_label_unchanged(self, parser):
        query = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        tree = parse_sparql(query, parser)
        body = _get_where_body(tree)
        stripped = remove_rdfs_label_triples(body)
        from sparql_statistics import _find_all_nodes

        triples = _find_all_nodes(stripped, "TriplesSameSubjectPath")
        with_children = [t for t in triples if "children" in t]
        assert len(with_children) == 1


class TestExtractTriples:
    def test_single_triple(self, parser):
        triples = _extract_triples_from_query(
            "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser
        )
        assert len(triples) == 1

    def test_two_triples(self, parser):
        triples = _extract_triples_from_query(
            "SELECT ?x ?y WHERE { ?x wdt:P31 wd:Q5 . ?x wdt:P27 ?y }", parser
        )
        assert len(triples) == 2

    def test_variables_normalized_to_placeholder(self, parser):
        # ?item and ?x should both become ?_ so two structurally identical
        # queries with different var names produce the same triple strings
        t1 = _extract_triples_from_query(
            "SELECT ?item WHERE { ?item wdt:P31 wd:Q5 }", parser
        )
        t2 = _extract_triples_from_query("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser)
        assert t1 == t2

    def test_entities_and_properties_kept(self, parser):
        # Entities and properties must not be normalized (only vars are)
        t1 = _extract_triples_from_query("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser)
        t2 = _extract_triples_from_query(
            "SELECT ?x WHERE { ?x wdt:P31 wd:Q42 }", parser
        )
        assert t1 != t2

    def test_different_properties_differ(self, parser):
        t1 = _extract_triples_from_query("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser)
        t2 = _extract_triples_from_query("SELECT ?x WHERE { ?x wdt:P27 wd:Q5 }", parser)
        assert t1 != t2

    def test_literals_kept_without_normalization(self, parser):
        t1 = _extract_triples_from_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" }',
            parser,
            normalize_literals=False,
        )
        t2 = _extract_triples_from_query(
            'SELECT ?x WHERE { ?x <http://a> "world" }',
            parser,
            normalize_literals=False,
        )
        assert t1 != t2

    def test_literals_collapsed_with_normalization(self, parser):
        t1 = _extract_triples_from_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" }', parser, normalize_literals=True
        )
        t2 = _extract_triples_from_query(
            'SELECT ?x WHERE { ?x <http://a> "world" }', parser, normalize_literals=True
        )
        assert t1 == t2


def _iri_jaccard(raw: str, clean: str, parser) -> float:
    """Convenience: parse two queries and return IRI Jaccard of their WHERE bodies."""
    raw_tree = parse_sparql(raw, parser)
    clean_tree = parse_sparql(clean, parser)
    raw_body = _get_where_body(raw_tree)
    clean_body = _get_where_body(clean_tree)
    assert raw_body is not None and clean_body is not None
    clean_prefix_map = extract_prefix_declarations(clean_tree)
    return _jaccard(
        _extract_iris(raw_body),
        _extract_iris(clean_body, prefix_map=clean_prefix_map),
    )


class TestExtractIRIs:
    def test_full_iri_extracted(self, parser):
        tree = parse_sparql(
            "SELECT ?x WHERE { ?x <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> }",
            parser,
        )
        iris = _extract_iris(_get_where_body(tree))
        assert "<http://www.wikidata.org/prop/direct/P31>" in iris
        assert "<http://www.wikidata.org/entity/Q5>" in iris

    def test_pname_expanded_with_prefix_map(self, parser):
        # wdt:P31 and wd:Q5 should expand to full IRIs when prefix_map is provided
        tree = parse_sparql("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser)
        prefix_map = extract_prefix_declarations(tree)
        iris = _extract_iris(_get_where_body(tree), prefix_map=prefix_map)
        assert "<http://www.wikidata.org/prop/direct/P31>" in iris
        assert "<http://www.wikidata.org/entity/Q5>" in iris

    def test_pname_not_expanded_without_prefix_map(self, parser):
        # Without prefix_map, PNAME_LN nodes are skipped
        tree = parse_sparql("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }", parser)
        iris = _extract_iris(_get_where_body(tree))
        assert len(iris) == 0

    def test_custom_prefix_expanded(self, parser):
        query = (
            "PREFIX myprop: <http://www.wikidata.org/prop/direct/> "
            "SELECT ?x WHERE { ?x myprop:P31 <http://www.wikidata.org/entity/Q5> }"
        )
        tree = parse_sparql(query, parser)
        prefix_map = extract_prefix_declarations(tree)
        iris = _extract_iris(_get_where_body(tree), prefix_map=prefix_map)
        assert "<http://www.wikidata.org/prop/direct/P31>" in iris
        assert "<http://www.wikidata.org/entity/Q5>" in iris

    def test_literals_and_variables_excluded(self, parser):
        tree = parse_sparql('SELECT ?x WHERE { ?x <http://a> "hello" }', parser)
        iris = _extract_iris(_get_where_body(tree))
        assert iris == {"<http://a>"}


class TestIRIJaccard:
    def test_identical_iris(self, parser):
        # Raw uses full IRIs, clean uses prefixes for the same IRIs → Jaccard = 1.0
        raw = "SELECT ?x WHERE { ?x <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> }"
        clean = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        assert _iri_jaccard(raw, clean, parser) == 1.0

    def test_clean_adds_one_iri(self, parser):
        # raw has {P31, Q5}, clean has {P31, Q5, P27, Q183} → 2/4 = 1/2
        raw = "SELECT ?x WHERE { ?x <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> }"
        clean = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 . ?x wdt:P27 wd:Q183 }"
        assert _iri_jaccard(raw, clean, parser) == pytest.approx(2 / 4)

    def test_completely_different_iris(self, parser):
        raw = "SELECT ?x WHERE { ?x <http://www.wikidata.org/prop/direct/P18> <http://www.wikidata.org/entity/Q5> }"
        clean = "SELECT ?x WHERE { ?x wdt:P31 wd:Q42 }"
        assert _iri_jaccard(raw, clean, parser) == 0.0

    def test_literals_ignored(self, parser):
        # Literals don't affect IRI Jaccard — only the predicate IRI matters
        raw = 'SELECT ?x WHERE { ?x <http://www.wikidata.org/prop/direct/P2509> "anon_123" }'
        clean = 'SELECT ?x WHERE { ?x wdt:P2509 "real value" }'
        assert _iri_jaccard(raw, clean, parser) == 1.0

    def test_label_service_stripped_matches_rdfs_label_stripped(self, parser):
        raw = (
            "SELECT ?item ?itemLabel WHERE { "
            "?item <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> . "
            'SERVICE wikibase:label { bd:serviceParam wikibase:language "en" } '
            "}"
        )
        clean = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
            "SELECT ?item ?itemLabel WHERE { "
            "?item wdt:P31 wd:Q5 . "
            "?item rdfs:label ?itemLabel "
            "}"
        )
        raw_tree = parse_sparql(raw, parser)
        clean_tree = parse_sparql(clean, parser)
        raw_body = _get_where_body(raw_tree)
        clean_body = _get_where_body(clean_tree)
        assert raw_body is not None and clean_body is not None
        clean_prefix_map = extract_prefix_declarations(clean_tree)
        raw_stripped, had_label_service = remove_label_service(raw_body)
        clean_stripped = (
            remove_rdfs_label_triples(clean_body) if had_label_service else clean_body
        )
        j = _jaccard(
            _extract_iris(raw_stripped),
            _extract_iris(clean_stripped, prefix_map=clean_prefix_map),
        )
        assert j == 1.0


def _construct_jaccard(raw: str, clean: str, parser) -> float:
    """Parse raw and clean queries and return Jaccard on SPARQL_CONSTRUCTS node sets.
    Mirrors main(): strips label service from raw; conditionally strips rdfs:label
    triples and lang filters from clean only if raw had a label service."""
    t_raw = parse_sparql(raw, parser)
    t_clean = parse_sparql(clean, parser)
    t_raw_stripped, had_label_service = remove_label_service(t_raw)
    t_clean_stripped = (
        remove_lang_filters(remove_rdfs_label_triples(t_clean))
        if had_label_service
        else t_clean
    )
    c_raw: set[str] = set()
    c_clean: set[str] = set()
    collect_present_nodes(t_raw_stripped, c_raw)
    collect_present_nodes(t_clean_stripped, c_clean)
    return _jaccard(c_raw & SPARQL_CONSTRUCTS, c_clean & SPARQL_CONSTRUCTS)


class TestConstructJaccard:
    def test_identical_query_structure(self, parser):
        # Both have {SELECT} → Jaccard = 1/1 = 1.0
        q1 = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        q2 = "SELECT ?y WHERE { ?y wdt:P31 wd:Q42 }"
        assert _construct_jaccard(q1, q2, parser) == 1.0

    def test_optional_added_reduces_jaccard(self, parser):
        # raw has {SELECT}, clean has {SELECT, OPTIONAL}
        # intersection=1, union=2 → 1/2
        raw = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        clean = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 OPTIONAL { ?x wdt:P18 ?img } }"
        assert _construct_jaccard(raw, clean, parser) == pytest.approx(1 / 2)

    def test_partial_construct_overlap(self, parser):
        # raw has {SELECT, FILTER}, clean has {SELECT, FILTER, OPTIONAL}
        # intersection=2, union=3 → 2/3
        raw = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 FILTER(?x != wd:Q1) }"
        clean = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 OPTIONAL { ?x wdt:P18 ?img } FILTER(?x != wd:Q1) }"
        assert _construct_jaccard(raw, clean, parser) == pytest.approx(2 / 3)

    def test_ask_vs_select(self, parser):
        # raw has {SELECT}, clean has {ASK} → intersection=0, union=2 → 0/2 = 0.0
        raw = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 }"
        clean = "ASK { ?x wdt:P31 wd:Q5 }"
        assert _construct_jaccard(raw, clean, parser) == 0.0


class TestRemoveLangFilters:
    def test_langmatches_filter_stripped(self, parser):
        q = "SELECT ?x ?label WHERE { ?x wdt:P31 wd:Q5 . FILTER(LANGMATCHES(LANG(?label), \"en\")) }"
        tree = parse_sparql(q, parser)
        stripped = remove_lang_filters(tree)
        present: set[str] = set()
        collect_present_nodes(stripped, present)
        assert "FILTER" not in present
        assert "LANG" not in present
        assert "LANGMATCHES" not in present
        # regular triple pattern still present
        assert "SELECT" in present

    def test_lang_eq_filter_stripped(self, parser):
        q = "SELECT ?x ?label WHERE { ?x wdt:P31 wd:Q5 . FILTER(LANG(?label) = \"en\") }"
        tree = parse_sparql(q, parser)
        stripped = remove_lang_filters(tree)
        present: set[str] = set()
        collect_present_nodes(stripped, present)
        assert "FILTER" not in present
        assert "LANG" not in present

    def test_non_lang_filter_kept(self, parser):
        q = "SELECT ?x WHERE { ?x wdt:P31 wd:Q5 FILTER(?x != wd:Q1) }"
        tree = parse_sparql(q, parser)
        stripped = remove_lang_filters(tree)
        present: set[str] = set()
        collect_present_nodes(stripped, present)
        assert "FILTER" in present

    def test_mixed_filters_only_lang_stripped(self, parser):
        # One lang filter + one value filter: only lang filter removed
        q = (
            "SELECT ?x ?label WHERE { "
            "?x wdt:P31 wd:Q5 "
            "FILTER(LANGMATCHES(LANG(?label), \"en\")) "
            "FILTER(?x != wd:Q1) }"
        )
        tree = parse_sparql(q, parser)
        stripped = remove_lang_filters(tree)
        present: set[str] = set()
        collect_present_nodes(stripped, present)
        assert "FILTER" in present  # value filter remains
        assert "LANGMATCHES" not in present

    def test_service_stripped_matches_lang_filter_stripped(self, parser):
        # raw has SERVICE wikibase:label, clean has FILTER(LANGMATCHES(LANG(...)))
        # After stripping both, construct Jaccard should be 1.0
        raw = (
            "SELECT ?x ?xLabel WHERE { "
            "?x wdt:P31 wd:Q5 . "
            "SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" } }"
        )
        clean = (
            "SELECT ?x ?label WHERE { "
            "?x wdt:P31 wd:Q5 . "
            "FILTER(LANGMATCHES(LANG(?label), \"en\")) }"
        )
        assert _construct_jaccard(raw, clean, parser) == 1.0
