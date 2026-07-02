:orphan:

Reward function API
===================

Reward files should expose a callable named ``reward_fn``:

.. code-block:: python

   def reward_fn(example, completions) -> list[float]:
       ...

Parameters:

``example``
   The source record returned by the dataset loader.

``completions``
   The generated completions to score.

Return value:

``list[float]``
   One reward score per completion.

Keep this API stable for task code. If the reward needs extra metadata, add it
through the dataset loader record rather than through global state.
