..
   AReno CLI docs — areno logs

   Documentation for the ``areno logs`` command: filtering and following
   run logs from the CLI.

``areno logs`` -- Filter and follow run logs
=============================================

Filter and follow AReno run logs directly from the CLI without writing
one-off scripts.  The command reads from AReno's existing local artifact
files (under the metrics log directory) and streams output incrementally
so large logs never need to be loaded into memory.

Synopsis
--------

::

    areno logs RUN_ID [OPTIONS]

Arguments
---------

``RUN_ID``
    A log file path, a metrics directory path, or a dashboard job ID.
    When a directory is given, AReno scans for text log files (``*.log``,
    ``*.txt``, ``*.out``, run-config files) and skips binary TensorBoard
    event files.

Options
-------

``--tail N``
    Show only the last *N* lines.  When omitted, all lines are read.
    Default: all lines.

``-f, --follow``
    Keep polling for new log output after reaching the end of the file,
    similar to ``tail -f``.  Use ``Ctrl-C`` to stop.  Default: off.

``--rank N``
    Filter by distributed rank (integer ≥ 0).  Lines without a rank
    indicator are excluded when this filter is active.  Default: no filter.

``--stage {train|eval|rollout|serve}``
    Filter by training stage.  The stage is inferred from the logger name.
    Default: no filter.

``--severity {debug|info|warn|error}``
    Filter by log severity.  ``WARNING`` lines are matched by ``warn``.
    Default: no filter.

``--grep PATTERN``
    Filter by regular expression matched against the log message
    (case-sensitive).  Default: no filter.

``--output {text|json}``
    Output format.  ``text`` produces human-readable lines with timestamp,
    stage, rank, severity, and source context.  ``json`` produces one JSON
    object per line (JSONL).  Default: ``text``.

``--poll-interval SECONDS``
    Seconds between poll cycles in follow mode.  Default: ``1.0``.

Defaults and backward compatibility
-----------------------------------

All filter options default to "no filtering".  Without any options,
``areno logs <path>`` reads and prints every line in the file — the same
behaviour as ``cat``.  No existing CLI behaviour is changed when the
command is not used.

Output fields
-------------

Text mode
~~~~~~~~~

Each line includes:

- ``[timestamp]`` -- original log timestamp
- ``[stage]`` -- inferred stage (train/eval/rollout/serve), if available
- ``[rank N]`` -- distributed rank, if available
- ``[SEVERITY]`` -- uppercase severity (coloured when stdout is a TTY)
- ``(source)`` -- logger name or file label
- ``message`` -- the log message body

JSON mode
~~~~~~~~~

Each line is a JSON object with these keys:

``timestamp``
    Log timestamp string.

``severity``
    Lowercase severity (``debug``, ``info``, ``warn``, ``error``).

``source``
    Logger name or file label.

``message``
    Log message body.

``rank``
    Distributed rank integer, or ``null`` if not present.

``stage``
    Inferred stage string, or ``null`` if not available.

Error output
------------

Validation errors are written to stderr and identify the affected stage
and input:

- Text: ``Error [stage=log_filter] Invalid severity 'trace'. ... (input: severity)``
- JSON: ``{"error": {"stage": "log_filter", "input": "severity", "message": "..."}}``

Limitations
-----------

- Follow mode uses polling (not WebSocket/SSE); there is a small delay
  controlled by ``--poll-interval``.
- Only local artifact files are supported; no remote or database log
  sources.
- Rank is extracted from the message text or logger name using a simple
  pattern match; lines without a ``rank=N`` token report ``rank=null``.

Examples
--------

Read all logs from a metrics directory::

    areno logs /tmp/areno/tfevent

Show the last 50 lines::

    areno logs /tmp/areno/tfevent --tail 50

Follow logs in real time::

    areno logs /tmp/areno/tfevent --follow

Follow and filter to errors on rank 0::

    areno logs /tmp/areno/tfevent --follow --severity error --rank 0

Combined filters with JSON output for scripting::

    areno logs /tmp/areno/tfevent --tail 100 --stage train --grep "loss" --output json