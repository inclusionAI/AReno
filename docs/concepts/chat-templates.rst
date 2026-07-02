Chat templates
==============

Chat templates turn structured messages into the exact prompt text expected by
the selected tokenizer. AReno keeps this rendering close to rollout and serving
so a training run and an OpenAI-compatible endpoint can use the same model
conversation format.

Use chat templates when a model expects role-based messages such as ``system``,
``user``, ``assistant``, or tool-related turns. Dataset loaders should keep raw
task fields and normalized prompts separate; tokenizer-specific formatting
belongs in the training or serving path.

Thinking mode
-------------

Some reasoning or chat checkpoints expose an ``enable_thinking`` option through
their tokenizer chat template. AReno's CLI exposes ``--disable-thinking`` for
training and serving, which passes ``enable_thinking=False`` when supported and
falls back to the normal template call otherwise.

Where to go next
----------------

* :doc:`/cli/training` documents the training flag.
* :doc:`/cli/inference` documents the serving flag.
* :doc:`dataset-formats` explains how dataset rows provide prompt inputs.
