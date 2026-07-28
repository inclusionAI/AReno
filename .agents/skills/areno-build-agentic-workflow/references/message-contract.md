# Agentic Message Contract

- A tool call is an assistant message containing nonempty `tool_calls` with stable IDs, function names, and JSON arguments.
- Each call is followed by a `tool` message whose `tool_call_id` matches exactly.
- Tool results are data, not assistant-authored prose.
- Preserve complete prior messages for each subsequent turn.
- The trajectory records the exact model output and executed result; do not synthesize a successful call when parsing failed.
- Multimodal user content follows OpenAI content items and remains associated with the correct sample.
- Training masks include only intended assistant spans; tool-result spans are controlled by the explicit training option.
