"""
Utility functions for working with Wikidata query-SPARQL samples.
"""


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
        - error_field_not_null: The error field is not null
        - output_field_null: The output field is missing or null
        - type_is_cancel: output.type equals "cancel"
        - no_questions: output.questions field is missing or empty
        - sparql_execution_failed: formatted result contains "SPARQL execution failed"
        - empty_result: formatted result contains "Got 0 rows"
    """
    # Check if JSON is null or empty
    if data is None:
        return False, "null_json"

    # Check if error field is not null
    err = data.get("error")
    if err is not None:
        return False, f"error_field_not_null (reason={err['reason']})"

    # Check if output field exists and is not null
    if "output" not in data or data["output"] is None:
        return False, "output_field_null"

    output_data = data["output"]

    # Explicit cancel is not a fail
    if output_data.get("type") == "cancel":
        return True, "cancelled"

    # Check if questions field exists and has content
    if "questions" not in output_data or not output_data["questions"]:
        return False, "no_questions"

    # Check if formatted result indicates SPARQL execution failure
    formatted = output_data.get("sparql_result", output_data.get("formatted", ""))
    if formatted and "SPARQL execution failed" in formatted:
        return False, "sparql_execution_failed (execution)"

    if formatted and "Error executing SPARQL query over" in formatted:
        return False, "sparql_execution_failed (preprocessing)"

    # Check if result is empty (Got 0 rows)
    if formatted and "Got no rows" in formatted:
        return False, "empty_result"

    return True, "valid"
