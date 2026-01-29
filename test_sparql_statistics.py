import pytest

from sparql_statistics import (
    WIKIDATA_IRI_PREFIXES,
    collect_iris,
    collect_present_nodes,
    count_node_occurrences,
    get_iri_prefix,
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


class TestGetIriPrefix:
    def test_entity_iri(self):
        """Entity IRI should return wd: prefix."""
        assert get_iri_prefix("http://www.wikidata.org/entity/Q42") == "wd:"

    def test_prop_direct_iri(self):
        """prop/direct IRI should return wdt: prefix."""
        assert get_iri_prefix("http://www.wikidata.org/prop/direct/P31") == "wdt:"

    def test_prop_statement_iri(self):
        """prop/statement IRI should return ps: prefix."""
        assert get_iri_prefix("http://www.wikidata.org/prop/statement/P31") == "ps:"

    def test_prop_statement_normalized_iri(self):
        """prop/statement/value-normalized should return psn: (not ps:)."""
        assert (
            get_iri_prefix("http://www.wikidata.org/prop/statement/value-normalized/P31")
            == "psn:"
        )

    def test_prop_qualifier_iri(self):
        """prop/qualifier IRI should return pq: prefix."""
        assert get_iri_prefix("http://www.wikidata.org/prop/qualifier/P585") == "pq:"

    def test_prop_qualifier_normalized_iri(self):
        """prop/qualifier/value-normalized should return pqn: (not pq:)."""
        assert (
            get_iri_prefix("http://www.wikidata.org/prop/qualifier/value-normalized/P585")
            == "pqn:"
        )

    def test_non_wikidata_iri(self):
        """Non-Wikidata IRIs should return None."""
        assert get_iri_prefix("http://www.w3.org/2000/01/rdf-schema#label") is None
        assert get_iri_prefix("http://schema.org/name") is None


def get_iris_by_prefix(query: str, parser) -> dict[str, set[str]]:
    """Helper to parse a query and return collected IRIs by prefix."""
    tree = parse_sparql(query, parser)
    iris_by_prefix: dict[str, set[str]] = {
        prefix: set() for _, prefix in WIKIDATA_IRI_PREFIXES
    }
    collect_iris(tree, iris_by_prefix)
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
