:orphan:

Dashboard
=========

AReno ships a local, low-intrusion dashboard for inspecting training and
serving jobs without standing up a separate monitoring service. The dashboard
runs entirely on the host machine, reads from local artifact files, and
requires no external database.

Starting and stopping
---------------------

.. code-block:: bash

   areno dashboard --start --host 0.0.0.0 --port 8765
   areno dashboard --stop

The dashboard serves a single-page React app and a small JSON API from the
same HTTP server. Open ``http://<host>:<port>`` in a browser to view the
dashboard.

Run list search, filtering, and sorting
---------------------------------------

The Jobs page provides a toolbar above the run list for search, filtering,
sorting, and a reset action. All toolbar state is persisted in URL query
parameters, so refreshing the page or sharing the URL preserves the current
view.

Search
^^^^^^

The search box matches the following fields with a case-insensitive
substring query:

* Job ID
* Job name
* Model checkpoint (``ckpt`` or ``model_path``)
* Dataset path

Filters
^^^^^^^

Three dropdown filters narrow the run list:

* **State** — ``all`` (default), ``running``, ``succeeded``, ``failed``,
  ``stopped``, ``exited``, ``unknown``
* **Type** — ``all`` (default), ``train``, ``serve``
* **Algorithm** — ``all`` (default), then every algorithm present in the
  run list (``sft``, ``dpo``, ``gspo``, ``grpo``, ``ppo``, …). The
  options are derived dynamically from the data.

Sort
^^^^

The sort dropdown orders runs by:

* **Start time (newest)** — default, preserves original behavior
* **Start time (oldest)**
* **Duration (longest)**
* **Duration (shortest)**

Duration is computed from ``created_at`` to ``updated_at`` for each run.

URL parameters
^^^^^^^^^^^^^^

The toolbar state is encoded in the following query parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``q``
     - Search query string
   * - ``state``
     - State filter (omitted when ``all``)
   * - ``type``
     - Type filter (omitted when ``all``)
   * - ``algo``
     - Algorithm filter (omitted when ``all``)
   * - ``sort``
     - Sort key (omitted when ``newest``)
   * - ``page``
     - Current page number (omitted when page 1)

When all parameters are absent the dashboard behaves exactly as it did
before the feature was added.

Reset button
^^^^^^^^^^^^

The **Reset** button clears all search text, filters, and sort selection,
restoring the default view. It is disabled when no filters are active.

Testing with synthetic data
---------------------------

A seed script is provided for exercising the toolbar with many records
without running real training jobs:

.. code-block:: bash

   python scripts/seed_dashboard.py --count 50
   areno dashboard --start

This writes 50 synthetic jobs with mixed states, algorithms, types, and
timestamps to ``.areno-dashboard-state.json`` in the current directory.
The dashboard reads this file on startup and on each polling cycle.

Minimal example
---------------

1. Generate synthetic data and start the dashboard:

   .. code-block:: bash

      python scripts/seed_dashboard.py --count 30
      areno dashboard --start

2. Open the dashboard URL in a browser.

3. Type ``gsm8k`` in the search box — only runs with that dataset appear.

4. Select ``State: failed`` — the list narrows further to failed runs on
   gsm8k.

5. Change the sort to ``Duration (longest)`` — the remaining runs
   reorder by elapsed time.

6. Click **Reset** — all filters clear and the full list returns to the
   default newest-first ordering.

7. Refresh the browser — the URL contains the current filter parameters
   and the view is restored automatically.

Limitations
-----------

* Filtering and sorting run client-side on the polled run list (up to 200
  registered jobs). This is fast enough for typical single-node usage but
  is not designed for multi-node aggregations.
* The dashboard does not persist filter state across sessions beyond URL
  parameters. Closing the tab and reopening without the URL loses the
  active filters.
* Duration is only available for runs where both ``created_at`` and
  ``updated_at`` are present and parseable. Jobs with corrupted
  timestamps show ``duration_s: null``.