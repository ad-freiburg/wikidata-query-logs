# Sample Validity Rules

This document defines the criteria for determining whether a sample in the dataset is valid or invalid.

## Valid Sample

A sample is considered **valid** if it meets ALL of the following criteria:

1. ✅ The JSON is not null (can be parsed successfully)
2. ✅ The `error` field is `null`
3. ✅ The `output` field exists and is not `null`
4. ✅ `output.type` is NOT `"cancel"`
5. ✅ `output.questions` field exists and contains at least one question
6. ✅ The `formatted` result does NOT contain `"SPARQL execution failed"` (if present)
7. ✅ The `formatted` result does NOT contain `"Got not rows"` (if present)

## Invalid Sample Categories

A sample is considered **invalid** if ANY of the following is true:

### 1. Null or Parse Errors
- **null_json**: The JSON file is null or empty
- **json_decode_error**: The JSON cannot be parsed

### 2. Error Field
- **error_field_not_null**: The `error` field is not null (indicates generation failure)

### 3. Missing Output
- **output_field_null**: The `output` field is missing or null

### 4. Cancelled Generation
- **type_is_cancel**: `output.type` equals `"cancel"` (generation was cancelled)

### 5. Missing Questions
- **no_questions**: `output.questions` field is missing or empty

### 6. Execution Failures
- **sparql_execution_failed**: The `formatted` result contains "SPARQL execution failed"
- **empty_result**: The `formatted` result contains "Got no rows"

## JSON Structure

Expected structure:
```json
{
  "type": "output",
  "task": "wikidata-query-logs",
  "output": {
    "formatted": "...",
    "sparql": "SELECT ...",
    "questions": ["Question 1", "Question 2", "..."],
    "type": "answer"
  },
  "elapsed": 69.94,
  "error": null,
  "messages": [...]
}
```

## Notes

- The validity checks can be computed on-demand in the visualization app
- For embedding generation, we only need samples with non-empty `questions` field
- SPARQL execution errors and empty results don't prevent embedding generation, but should be flagged in visualization
