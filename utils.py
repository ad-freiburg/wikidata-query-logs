from importlib import resources

from grammar_utils.parse import LR1Parser


def validate_sample(data: dict | None) -> tuple[bool, str]:
    """
    Check if a sample is valid according to VALIDITY_RULES.md

    Args:
        data: Parsed JSON data from a sample file

    Returns:
        (is_valid, reason) - True if valid with reason "valid",
                            False with specific reason if invalid

    Invalid reasons:
        - null_json: The JSON is null or empty
        - not_a_sample (request failed): HTTP error body, the model never ran
        - not_a_sample (truncated generation): an intermediate record, not an output
        - error_field_not_null: The error field is not null
        - output_field_null: The output field is missing or null
        - type_is_cancel: output.type equals "cancel"
        - no_questions: output.questions field is missing or empty
        - sparql_execution_failed: formatted result contains "SPARQL execution failed"
        - empty_result: formatted result contains "Got no rows" or "Got 0 rows"
    """
    # Check if JSON is null or empty
    if data is None:
        return False, "null_json"

    # A file can also hold an HTTP error body or an intermediate record, since /run
    # returns the last record generate() yielded. Neither carries an "error" field,
    # so both need excluding here or they count as model failures.
    record_type = data.get("type")
    if record_type is None:
        return False, "not_a_sample (request failed)"
    if record_type != "output":
        return False, "not_a_sample (truncated generation)"

    # Check if error field is not null
    err = data.get("error")
    if err is not None:
        return False, f"error_field_not_null (reason={err['reason']})"

    # Check if output field exists and is not null
    if "output" not in data or data["output"] is None:
        return False, "output_field_null"

    output_data = data["output"]

    # Explicit cancel - sample is not usable
    if output_data.get("type") == "cancel":
        return False, "type_is_cancel"

    # Check if questions field exists and has content
    if "questions" not in output_data or not output_data["questions"]:
        return False, "no_questions"

    # Check if formatted result indicates SPARQL execution failure
    formatted = output_data.get("sparql_result", output_data.get("formatted", ""))
    if formatted and "SPARQL execution failed" in formatted:
        return False, "sparql_execution_failed (execution)"

    if formatted and "Error executing SPARQL query over" in formatted:
        return False, "sparql_execution_failed (preprocessing)"

    # Check if result is empty; runs up to Qwen3-Next-80B-A3B say "Got no rows"
    if formatted and ("Got no rows" in formatted or "Got 0 rows" in formatted):
        return False, "empty_result"

    return True, "valid"


def load_sparql_grammar() -> tuple[str, str]:
    """
    Load SPARQL grammar and lexer from package resources.

    Returns:
        A tuple containing the SPARQL grammar and lexer as strings.
    """
    sparql_grammar = resources.read_text("grasp.sparql.grammar", "sparql.y")
    sparql_lexer = resources.read_text("grasp.sparql.grammar", "sparql.l")
    return sparql_grammar, sparql_lexer


def load_sparql_parser() -> LR1Parser:
    """
    Load and return an LR1Parser for SPARQL.

    Returns:
        An LR1Parser instance for SPARQL.
    """
    sparql_grammar, sparql_lexer = load_sparql_grammar()
    return LR1Parser(sparql_grammar, sparql_lexer)


def parse_sparql(query: str, parser: LR1Parser) -> dict:
    """
    Parse a SPARQL query using the provided LR1Parser.

    Args:
        query: The SPARQL query string to parse.
        parser: An LR1Parser instance for SPARQL.

    Returns:
        The parse tree as a dictionary.
    """
    return parser.parse(query, skip_empty=True)
