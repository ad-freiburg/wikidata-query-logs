import pytest

from sparql_statistics import (
    WIKIDATA_IRI_PREFIXES,
    collect_iris,
    collect_languages,
    collect_literals,
    collect_present_nodes,
    count_node_occurrences,
    extract_prefix_declarations,
    get_wikidata_prefix,
    normalize_tree,
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
        assert get_wikidata_prefix("http://www.wikidata.org/prop/statement/P31") == "ps:"

    def test_prop_statement_normalized_iri(self):
        """prop/statement/value-normalized should return psn: (not ps:)."""
        assert (
            get_wikidata_prefix("http://www.wikidata.org/prop/statement/value-normalized/P31")
            == "psn:"
        )

    def test_prop_qualifier_iri(self):
        """prop/qualifier IRI should return pq: prefix."""
        assert get_wikidata_prefix("http://www.wikidata.org/prop/qualifier/P585") == "pq:"

    def test_prop_qualifier_normalized_iri(self):
        """prop/qualifier/value-normalized should return pqn: (not pq:)."""
        assert (
            get_wikidata_prefix("http://www.wikidata.org/prop/qualifier/value-normalized/P585")
            == "pqn:"
        )

    def test_statement_iri(self):
        """entity/statement IRI should return s: prefix (not wd:)."""
        assert get_wikidata_prefix("http://www.wikidata.org/entity/statement/Q42-abc123") == "s:"

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
        assert "http://www.wikidata.org/prop/statement/value-normalized/P569" in iris["psn:"]
        assert len(iris["ps:"]) == 1
        assert len(iris["psn:"]) == 1

    def test_statement_prefix(self, parser):
        """s: prefix should be collected under s: (not wd:)."""
        iris = get_iris_by_prefix(
            "PREFIX s: <http://www.wikidata.org/entity/statement/> "
            "SELECT ?x WHERE { ?s s:Q42-abc123 ?x }",
            parser,
        )
        assert "http://www.wikidata.org/entity/statement/Q42-abc123" in iris["s:"]
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


def assert_queries_normalize_same(queries: list[str], parser, normalize_properties: bool = False) -> None:
    """Assert that all queries in the list normalize to the same SPARQL form."""
    normalized = [normalize_query(q, parser, normalize_properties) for q in queries]
    first = normalized[0]
    for i, norm in enumerate(normalized[1:], 1):
        assert norm == first, (
            f"Query {i+1} normalized differently than Query 1:\n"
            f"Query 1: {queries[0]}\n"
            f"Query {i+1}: {queries[i]}\n"
            f"Normalized 1: {first}\n"
            f"Normalized {i+1}: {norm}"
        )


def assert_queries_normalize_different(query1: str, query2: str, parser, normalize_properties: bool = False) -> None:
    """Assert that two queries normalize to different SPARQL forms."""
    norm1 = normalize_query(query1, parser, normalize_properties)
    norm2 = normalize_query(query2, parser, normalize_properties)
    assert norm1 != norm2, (
        f"Queries normalized to the same form but should be different:\n"
        f"Query 1: {query1}\n"
        f"Query 2: {query2}\n"
        f"Normalized: {norm1}"
    )


def assert_normalizes_to(input_query: str, expected_normalized_sparql: str, parser, normalize_properties: bool = False) -> None:
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
            parser
        )

    def test_multiple_variables_get_different_placeholders(self, parser):
        """Different variables should get different placeholders."""
        assert_normalizes_to(
            "SELECT ?x ?y WHERE { ?x ?p ?y }",
            "SELECT ?var0 ?var1 WHERE { ?var0 ?var2 ?var1 }",
            parser
        )

    def test_same_variable_same_placeholder(self, parser):
        """Same variable appearing twice should get same placeholder."""
        assert_normalizes_to(
            "SELECT ?x WHERE { ?x <http://a> ?y . ?x <http://b> ?z }",
            "SELECT ?var0 WHERE { ?var0 <http://a> ?var1 . ?var0 <http://b> ?var2 }",
            parser
        )


class TestNormalizeTreeEntities:
    def test_wd_entity_normalized(self, parser):
        """wd: entities should be normalized to wd:E0, wd:E1, etc."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 ?var0 }",
            parser
        )

    def test_same_entity_same_placeholder(self, parser):
        """Same entity appearing twice should get same placeholder."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x . wd:Q42 wdt:P27 ?y }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 ?var0 . wd:E0 wdt:P27 ?var1 }",
            parser
        )

    def test_different_entities_different_placeholders(self, parser):
        """Different entities should get different placeholders."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 wd:Q5 }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 wd:E1 }",
            parser
        )

    def test_statement_entity_normalized(self, parser):
        """s: (statement) entities should be normalized (PREFIX declarations ignored)."""
        assert_normalizes_to(
            "PREFIX s: <http://www.wikidata.org/entity/statement/> "
            "SELECT ?x WHERE { ?s s:Q42-abc123 ?x }",
            "SELECT ?var0 WHERE { ?var1 s:E0 ?var0 }",
            parser
        )


class TestNormalizeTreeLiterals:
    def test_string_literal_normalized(self, parser):
        """Different string VALUES should normalize to same pattern (same structure)."""
        assert_queries_normalize_same([
            'SELECT ?x WHERE { ?x <http://a> "Albert Einstein" }',
            'SELECT ?x WHERE { ?x <http://a> "Marie Curie" }',
        ], parser)

    def test_integer_literal_normalized(self, parser):
        """Different numeric VALUES should normalize to same pattern (same structure)."""
        assert_queries_normalize_same([
            "SELECT ?x WHERE { ?x <http://a> 42 }",
            "SELECT ?x WHERE { ?x <http://a> 100 }",
        ], parser)

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
            parser
        )

    def test_string_literal_with_and_without_datatype(self, parser):
        """String literal with/without datatype should normalize the same."""
        assert_queries_normalize_same([
            'SELECT ?x WHERE { ?x <http://a> "hello" }',
            'SELECT ?x WHERE { ?x <http://a> "hello"^^<http://www.w3.org/2001/XMLSchema#string> }',
        ], parser)

    def test_numeric_literal_with_and_without_datatype(self, parser):
        """Numeric literal with/without datatype should normalize the same."""
        assert_queries_normalize_same([
            'SELECT ?x WHERE { ?x <http://a> "123" }',
            'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#integer> }',
            'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#decimal> }',
        ], parser)

    def test_raw_number_and_typed_string_normalize_same(self, parser):
        """Raw numeric literal and string with numeric datatype should normalize the same."""
        assert_queries_normalize_same([
            'SELECT ?x WHERE { ?x <http://a> 400 }',
            'SELECT ?x WHERE { ?x <http://a> "400"^^<http://www.w3.org/2001/XMLSchema#integer> }',
        ], parser)


class TestNormalizeTreeProperties:
    def test_properties_kept_by_default(self, parser):
        """Properties should NOT be normalized when normalize_properties=False."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P31 ?var0 }",
            parser,
            normalize_properties=False
        )

    def test_properties_normalized_when_enabled(self, parser):
        """Properties should be normalized when normalize_properties=True."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P0 ?var0 }",
            parser,
            normalize_properties=True
        )

    def test_same_property_same_placeholder(self, parser):
        """Same property appearing twice should get same placeholder."""
        assert_normalizes_to(
            "SELECT ?x ?y WHERE { wd:Q42 wdt:P31 ?x . wd:Q5 wdt:P31 ?y }",
            "SELECT ?var0 ?var1 WHERE { wd:E0 wdt:P0 ?var0 . wd:E1 wdt:P0 ?var1 }",
            parser,
            normalize_properties=True
        )

    def test_different_properties_different_placeholders(self, parser):
        """Different properties should get different placeholders."""
        assert_normalizes_to(
            "SELECT ?x WHERE { wd:Q42 wdt:P31 ?x . wd:Q42 wdt:P27 ?y }",
            "SELECT ?var0 WHERE { wd:E0 wdt:P0 ?var0 . wd:E0 wdt:P1 ?var1 }",
            parser,
            normalize_properties=True
        )

    def test_different_property_prefixes_shared_counter(self, parser):
        """Different property prefixes share a counter (p:P31 -> p:P0, ps:P31 -> ps:P1)."""
        assert_normalizes_to(
            "SELECT ?x WHERE { ?s p:P31 ?stmt . ?stmt ps:P31 ?x }",
            "SELECT ?var0 WHERE { ?var1 p:P0 ?var2 . ?var2 ps:P1 ?var0 }",
            parser,
            normalize_properties=True
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

    def test_same_structure_different_properties_same_pattern_when_normalized(self, parser):
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
        canonical2 = normalize_query("SELECT ?x WHERE { wd:Q100 wdt:P31 wd:Q200 }", parser)
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
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> "hello" }', parser
        )
        assert 'hello' in literals
        assert len(literals) == 1

    def test_integer_literal(self, parser):
        literals = get_literals(
            "SELECT ?x WHERE { ?x <http://a> 42 }", parser
        )
        assert "42" in literals
        assert len(literals) == 1

    def test_decimal_literal(self, parser):
        literals = get_literals(
            "SELECT ?x WHERE { ?x <http://a> 3.14 }", parser
        )
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
        assert 'hello' in literals
        assert 'world' in literals
        assert '42' in literals

    def test_language_tagged_string_collected(self, parser):
        """The string part of a lang-tagged literal should be collected without quotes."""
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> "hello"@en }', parser
        )
        assert 'hello' in literals

    def test_typed_literal_string_collected(self, parser):
        """The string part of a typed literal (e.g. xsd:date) should be collected without quotes."""
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01"^^<http://www.w3.org/2001/XMLSchema#date> }',
            parser,
        )
        assert '2024-01-01' in literals

    def test_numeric_and_string_same_value_deduplicated(self, parser):
        """400, "400", and "400"^^xsd:integer should all be counted as one literal."""
        literals = get_literals(
            'SELECT ?x WHERE { ?x <http://a> 400 . ?x <http://b> "400" . ?x <http://c> "400"^^<http://www.w3.org/2001/XMLSchema#integer> }',
            parser,
        )
        assert len(literals) == 1
        assert '400' in literals

    def test_no_literals(self, parser):
        literals = get_literals(
            "SELECT ?x WHERE { ?x <http://a> ?y }", parser
        )
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
            'SELECT ?x ?y WHERE { '
            '?x <http://a> "hello"@en . '
            '?x <http://b> ?label . FILTER(LANG(?label) = "de") '
            '}',
            parser,
        )
        assert "en" in langs
        assert "de" in langs

    def test_no_language(self, parser):
        """Query without any language constructs."""
        langs = get_languages(
            "SELECT ?x WHERE { ?x <http://a> ?y }", parser
        )
        assert len(langs) == 0


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
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello"@de . ?x <http://b> "world"@es }',
            parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "world" }',
            parser
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
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01"^^<http://www.w3.org/2001/XMLSchema#string> }',
            parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024-01-01" }',
            parser
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
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "123"^^<http://www.w3.org/2001/XMLSchema#decimal> }',
            parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "123" }',
            parser
        )
        assert canonical1 == canonical2 == canonical3, (
            f"Numeric datatypes not properly normalized:\n"
            f"integer: {canonical1}\n"
            f"decimal: {canonical2}\n"
            f"no type: {canonical3}"
        )

    def test_multiple_datatypes_removed(self, parser):
        """Multiple datatypes in same query should all be removed."""
        canonical1 = normalize_query(
            'SELECT ?x WHERE { '
            '?x <http://a> "2024"^^<http://www.w3.org/2001/XMLSchema#gYear> . '
            '?x <http://b> "42"^^<http://www.w3.org/2001/XMLSchema#integer> '
            '}',
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { '
            '?x <http://a> "2024"^^<http://www.w3.org/2001/XMLSchema#string> . '
            '?x <http://b> "42"^^<http://www.w3.org/2001/XMLSchema#decimal> '
            '}',
            parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "2024" . ?x <http://b> "42" }',
            parser
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
            'SELECT ?x WHERE { '
            '?x <http://a> "hello"@en . '
            '?x <http://b> "2024"^^<http://www.w3.org/2001/XMLSchema#gYear> '
            '}',
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { '
            '?x <http://a> "hello"@de . '
            '?x <http://b> "2024"^^<http://www.w3.org/2001/XMLSchema#string> '
            '}',
            parser
        )
        canonical3 = normalize_query(
            'SELECT ?x WHERE { ?x <http://a> "hello" . ?x <http://b> "2024" }',
            parser
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
            'SELECT ?itemLabel WHERE { '
            'wd:Q42 rdfs:label ?itemLabel . '
            'FILTER(LANG(?itemLabel) = "en") '
            '}',
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?itemLabel WHERE { '
            'wd:Q5 rdfs:label ?itemLabel . '
            'FILTER(LANG(?itemLabel) = "de") '
            '}',
            parser
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
            'SELECT ?x WHERE { '
            'wd:Q42 rdfs:label "Albert Einstein"@en . '
            'wd:Q42 wdt:P31 ?x '
            '}',
            parser
        )
        canonical2 = normalize_query(
            'SELECT ?x WHERE { '
            'wd:Q5 rdfs:label "Mensch"@de . '
            'wd:Q5 wdt:P31 ?x '
            '}',
            parser
        )
        # Same structure, different entities and language tags
        assert canonical1 == canonical2, (
            f"Wikidata queries with language-tagged literals not properly normalized:\n"
            f"Q42/@en: {canonical1}\n"
            f"Q5/@de: {canonical2}"
        )
