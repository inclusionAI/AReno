# Request Contracts

- Discover the model with `GET /v1/models` where available.
- Text chat uses OpenAI-compatible `messages` and bounded `max_tokens`.
- Image content uses `type: image_url`; local files become data URLs without placing the payload in command arguments.
- Tools require a processor/chat template that supports tools. Surface its capability error instead of prompt hacks.
- Preserve raw model semantics; do not invent or silently strip reasoning fields.
- Cancellation or failure must release scheduler/cache ownership. Verify a subsequent independent request remains clean.
