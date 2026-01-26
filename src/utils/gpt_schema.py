def gpt_json_schema():
    """Defines the JSON schema for evidence classification using GPT."""
    json_schema = {
        "name": "evidence_classification_civic",
        "schema": {
            "type": "object",
            "properties": {
                "evidence_level": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D", "E", "unsure"],
                },
                "explanation": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["evidence_level", "explanation", "confidence"],
        },
    }

    return json_schema
