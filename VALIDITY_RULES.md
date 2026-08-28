# Sample Validity Rules

This document defines the criteria for determining whether a sample in the dataset is valid or invalid.

## Valid Sample

A sample is considered **valid** if it meets ALL of the following criteria:

1. ✅ The JSON is not null (can be parsed successfully)
2. ✅ The record is a finished generation, i.e. `type` is exactly `"output"`
3. ✅ The `error` field is `null`
4. ✅ The `output` field exists and is not `null`
5. ✅ `output.type` is NOT `"cancel"`
6. ✅ `output.questions` field exists and contains at least one question
7. ✅ The `formatted` result does NOT contain `"SPARQL execution failed"` (if present)
8. ✅ The `formatted` result does NOT contain `"Got no rows"` or `"Got 0 rows"` (if present)

## Invalid Sample Categories

A sample is considered **invalid** if ANY of the following is true:

### 1. Null or Parse Errors
- **null_json**: The JSON file is null or empty
- **json_decode_error**: The JSON cannot be parsed

### 1b. Not a Generation Attempt

The run script saves the body of `POST /run` verbatim, and `/run` returns the last
record `generate()` yielded — the final output only if it ran to completion. So a
file may hold something else entirely. Neither case carries an `error` field, so
both are indistinguishable from a model failure unless `type` is checked first, and
both must be excluded from the sample count rather than counted as failures:

- **not_a_sample (request failed)**: no `type` field, just a `{"detail": ...}` HTTP
  error body (usually the input query failed to parse).
- **not_a_sample (truncated generation)**: a `type` other than `"output"`, i.e. an
  intermediate record (`"input"`, `"system"`, `"model"`, `"tool"`). Test
  `type == "output"` rather than enumerating these — which one appears depends only
  on how far generation had got.

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
- **empty_result**: The `formatted` result contains "Got no rows" or "Got 0 rows"

> Note: GRASP runs up to Qwen3-Next-80B-A3B report an empty result as
> "Got no rows", later ones as "Got 0 rows". Both spellings must be checked, or
> empty results silently pass validation.

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
