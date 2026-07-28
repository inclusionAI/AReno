Elevator-dispatch agentic RL
============================

Elevator dispatch is an episodic agentic AReno recipe with a time dimension.
Each prompt presents a building (floors, car capacity, waiting passengers, and a
pending arrival schedule) and the policy returns a full episode as an ordered
``move/open/close`` action string via a ``dispatch`` tool call. AReno replays
the episode deterministically, advancing the clock one tick per action and
landing arrivals on schedule, then scores delivered passengers, wait, and
invalid actions.

.. code-block:: bash

   python examples/agentic/elevator/dataset_generator.py \
     --output /tmp/areno-elevator.jsonl --count 2048 --seed 2026 --arrivals 6

   areno train \
     --agent-fn examples/agentic/elevator/run_agent.py \
     --dataset-loader-fn examples/agentic/elevator/dataset_loader.py \
     --reward-fn-path examples/agentic/elevator/reward.py \
     --algo gspo

A first-come-first-served (FCFS) baseline is available so trained-vs-baseline
improvement is reportable:

.. code-block:: bash

   python examples/agentic/elevator/baseline.py --count 256 --seed 2026 --arrivals 6

Key adaptation points:

* Replace the dispatch-contract action space with your own, but keep one
  ``AgentTrajectory`` turn so AReno tokenizes the episode as a single assistant
  output.
* Keep the event-queue advance inside the engine so the same building plus
  action string replays deterministically.
* Score with delivered passengers, mean wait, and an invalid-action penalty so
  a no-op policy cannot win, and board only up to capacity (overload
  prevention).
* Report ``delivered_passengers``, ``mean_wait``, and ``invalid_rate`` and
  compare against the FCFS baseline.

See :doc:`/reference/agentic-rollout-api` for the agentic rollout API contract.
