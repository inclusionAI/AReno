Documentation QA
================

Use this checklist before a public documentation release and for pull requests
that change user-facing commands, APIs, examples, or navigation. Run structural
checks without a GPU first, then record GPU-dependent verification separately.
The pull request should state which commands ran, their results, and anything
that was intentionally skipped.

Build and link checks
---------------------

#. Create an isolated documentation environment and install the documentation
   requirements:

   .. code-block:: bash

      python -m venv .venv-docs
      . .venv-docs/bin/activate
      python -m pip install -r docs/requirements.txt

#. Build every page from a clean Sphinx environment and treat warnings as
   errors:

   .. code-block:: bash

      python -m sphinx -E -a -W --keep-going \
        -b html docs dist/docs

   This is the required CPU-safe structure check. It catches malformed RST,
   duplicate labels, missing internal targets, and pages that are neither in a
   toctree nor explicitly marked ``:orphan:``.

#. Check external links in a network-enabled environment:

   .. code-block:: bash

      python -m sphinx -E -a -W --keep-going \
        -b linkcheck docs dist/docs-linkcheck

   Review each failure instead of blindly retrying. Record transient rate
   limits or unavailable third-party sites in the pull request; fix links that
   are stale, redirected permanently, or point to private locations.

Navigation and discoverability
------------------------------

* Confirm each new public page appears in a ``toctree`` that a reader can reach
  from ``docs/index.rst``.
* Search for ``:orphan:`` and review every match. Keep it only for pages that
  are intentionally outside the sidebar and are linked from a discoverable
  page.
* Follow links from the rendered page, not only from the source file. Confirm
  that headings, labels, and previous/next navigation land at the intended
  content.
* Remove superseded pages or add an explicit migration link; do not leave two
  pages presenting different current contracts.

API and CLI contract review
---------------------------

* Compare documented Python signatures, dataclass fields, return values, and
  exceptions with the current implementation under ``areno/api``.
* Compare CLI flags, defaults, and accepted values with current help output:

  .. code-block:: bash

     areno train --help
     areno serve --help
     areno check --help
     areno env --help

* Verify that copied identifiers resolve to real files and symbols. In
  particular, check dataset-loader, reward-function, agent-function, algorithm,
  and model-adapter examples against their registries or loading paths.
* Search all public docs for the old signature or flag when correcting contract
  drift. Updating one page is insufficient if a cookbook or troubleshooting
  page still teaches the previous behavior.

Example command freshness
-------------------------

* Run CPU-safe parsing and ``--help`` checks from a clean source checkout.
  Commands should not depend on an author's shell aliases, working directory,
  cache layout, or uncommitted files.
* Confirm every referenced path exists and every multiline shell command can be
  pasted without repairing quoting or continuation characters.
* Keep model IDs, dataset IDs, algorithm names, and option names consistent
  with current examples and registries.
* For commands that load a model or start workers, perform the GPU verification
  below rather than presenting successful argument parsing as an end-to-end
  run.

CPU and GPU verification boundary
---------------------------------

The following checks do not require a GPU:

* strict HTML and linkcheck builds;
* navigation, spelling, and public-content review;
* CLI help and argument-parsing checks that do not initialize the engine;
* CPU tests covering the documented data, loss, config, or utility behavior.

The following claims require a supported Linux, NVIDIA GPU, CUDA, and PyTorch
environment:

* installation of the ``areno_accel`` extension;
* successful model loading, rollout, optimization, checkpoint, or serving;
* memory use, throughput, numerical equivalence, and performance claims;
* completion of a training or serving quickstart.

For GPU verification, record the GPU model, OS, Python, CUDA, PyTorch, and AReno
commit. Start with ``areno check``, then run the smallest official command from
the page being reviewed. Include the success signal named by that page, not
only an exit code or a screenshot of startup logs.

Troubleshooting coverage
------------------------

* First-run installation, training, and serving pages must link to a relevant
  page under ``docs/troubleshooting`` at the point where a user can fail.
* Error guidance should name the first diagnostic command or metric to inspect
  and the evidence required for the next step.
* Confirm that unresolved failures can reach
  :doc:`/troubleshooting/report-issue`, which lists the environment, command,
  traceback, and reproduction details maintainers need.
* Avoid advice that silently changes model quality, sequence length, or
  algorithm semantics merely to hide an error.

Public-content and privacy review
---------------------------------

Review source and rendered output for material that must not enter public
documentation:

* private hosts, package indexes, object stores, dashboards, or repository
  URLs;
* real API keys, tokens, passwords, cookies, credentials, or signed URLs;
* personal names, email addresses, usernames, home directories, and local
  absolute paths;
* internal project names, incident details, customer data, or screenshots with
  identifying information.

Use searches as triage, then inspect every match manually. Placeholder secrets,
loopback serving URLs, and temporary paths may be legitimate examples, but
they must be clearly synthetic and portable.

.. code-block:: bash

   rg -n '(/home/|/Users/|[A-Za-z]:\\|api[_-]?key|token|password)' docs README.md
   rg -n 'https?://' docs README.md

Final sign-off
--------------

Before approving the documentation release, confirm that:

#. the strict HTML build completed with no warnings;
#. linkcheck failures were fixed or explicitly classified as transient;
#. new pages are discoverable and intentional orphans were reviewed;
#. API signatures, CLI flags, defaults, and examples match the current code;
#. CPU-safe checks passed and GPU-dependent claims have recorded GPU evidence;
#. first-run pages link to actionable troubleshooting and issue-report guidance;
#. the public-content and privacy review found no sensitive material; and
#. the pull request lists exact validation commands, results, environment, and
   skipped checks.
