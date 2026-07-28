:orphan:

Conversation normalization and tool-message pairing
====================================================

Multi-turn agentic conversations collected from different sources
(ShareGPT, OpenAI, Anthropic, HuggingFace) often use different role names
for the same concept.  AReno provides a normalizer that maps these aliases
to the standard roles and validates tool-call / tool-response pairing
before training begins.

When to use
-----------

Enable normalization when your training data:

* Comes from multiple sources with inconsistent role naming.
* Contains tool calls that may be missing responses.
* Needs to pass the agentic contract before expensive model loading.

Quick example
-------------

.. code-block:: python

   from areno.api.data import normalize_conversation, normalize_dataset

   # One conversation
   messages = [
       {"role": "human", "content": "What's the weather?"},
       {"role": "bot", "content": None, "tool_calls": [
           {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}
       ]},
       {"role": "function", "content": "25C sunny", "tool_call_id": "c1"},
       {"role": "bot", "content": "Beijing is 25C and sunny."},
   ]

   result = normalize_conversation(messages)
   assert result.ok
   # result.messages now uses standard roles: user, assistant, tool, assistant

   # Batch with structured report
   report = normalize_dataset([messages, bad_messages])
   print(report.to_human_string())
   # Total: 2  Passed: 1  Failed: 1  Skipped: 0
   # Errors:
   #   [missing_tool_response] sample #1, turn #2: ...

   # JSON output for programmatic use
   print(report.to_json())

Input contract
--------------

Each sample is a ``list[dict]`` of messages.  Every message must have:

* ``role`` (str): one of the recognized aliases or a standard role.
* ``content`` (str | None): message text.  ``None`` is replaced with ``""``.

Assistant messages with tool calls also carry:

* ``tool_calls`` (list[dict]): each call must have an ``id`` and either a
  ``function`` sub-dict with ``name``/``arguments`` or a flat
  ``name``/``arguments`` pair.

Tool messages also carry:

* ``tool_call_id`` (str): must match an ``id`` from a preceding
  assistant ``tool_calls`` entry.

Supported role aliases
----------------------

==============  ============  ==================
Alias           Standard role  Notes
==============  ============  ==================
human           user          ShareGPT
person          user
speaker         user
user            user          Already standard
bot             assistant     ShareGPT
gpt             assistant
model           assistant
chatbot         assistant
ai              assistant
assistant       assistant     Already standard
function        tool          OpenAI legacy
tool            tool          OpenAI current
tool_result     tool          Anthropic style
tool_response   tool
function_response  tool
system          system        Already standard
instruction     system
developer       system
==============  ============  ==================

Unknown roles raise ``ConversationValidationError`` and are never guessed.

Validation rules
----------------

1. **Tool-call pairing**: every ``tool_calls`` id must have exactly one
   matching ``tool`` response with the same ``tool_call_id``.

2. **No orphan responses**: a ``tool`` message whose ``tool_call_id`` does
   not match any preceding ``tool_calls`` id is rejected.

3. **No interruption**: a ``user`` message may not appear while tool calls
   are still pending.

4. **Role alternation**: consecutive ``user`` messages, consecutive
   ``assistant`` messages (without tool calls), ``system`` outside the first
   position, and ``tool`` not following ``assistant`` or ``tool`` are all
   rejected.

5. **Parallel tool calls**: multiple ``tool`` messages in a row are
   allowed when they respond to parallel ``tool_calls`` from one
   assistant message.

Defaults and backward compatibility
------------------------------------

Normalization is **opt-in**.  When not called, data flows through the
pipeline unchanged.  The normalizer introduces no new configuration keys
on existing trainers; it is a standalone utility that callers invoke
explicitly before training.

Error reporting
---------------

Errors are located to **sample index** and **turn index** without dumping
full message content:

* ``raise_on_error=True`` (default): raises on the first problem.
* ``raise_on_error=False``: collects all problems in
  ``NormalizeResult.errors``.

Batch reports provide both human-readable (``to_human_string``) and
structured JSON (``to_json`` / ``to_dict``) output.

Limitations
-----------

* The normalizer does not modify tool-call *arguments* beyond parsing
  JSON strings into dicts.  Schema validation is out of scope.
* The normalizer does not truncate or split overlength conversations.
  Use the length-handling utilities for that.
* The normalizer operates on in-memory lists; it does not read or write
  files directly.  Wrap it in your data loader as needed.