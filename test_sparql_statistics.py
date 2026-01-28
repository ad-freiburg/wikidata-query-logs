import pytest

from sparql_statistics import collect_present_nodes, count_node_occurrences
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
        present = get_present_nodes(
            "CONSTRUCT { ?x ?p ?o } WHERE { ?x ?p ?o }", parser
        )
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
