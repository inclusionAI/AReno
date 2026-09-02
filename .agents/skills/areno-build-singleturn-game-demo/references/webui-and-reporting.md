# WebUI, documentation, and delivery

Read this reference before implementing WebUI, finalizing README, starting
services, or opening a PR.

## Playable WebUI

After training and evaluation evidence is complete, implement a polished playable
WebUI using current Tic-Tac-Toe or terminal-hacking WebUI code as engineering
references, not as visual or gameplay templates.

- Render from public JSON view, never by parsing the training prompt.
- Use clear board, card, node, path, counter, or other components appropriate to
  the selected mechanic, with responsive layout, deliberate visual hierarchy,
  loading, error, win or loss, and disabled states.
- Map every UI interaction to the same structured action schema verified by the
  game and reward layer.
- Keep display or session progression outside training data. Human play may be
  sequential while model inference remains one independent state-to-output request.
- Handle model latency asynchronously without hiding the current game state.
- Add a representative public-view JSON example to README.

If authorized by the invoking request, start the trained checkpoint through
AReno's OpenAI-compatible server on port `8000` and WebUI on port `8001`. Use
the current serve command, keep CUDA graphs enabled, probe `/v1/models` and a
real completion, then probe UI and gameplay API. Report actual process IDs, log
paths, and whether services remain running.

## README contents

The demo README must contain:

1. rules and accessible Chinese source links;
2. comparison of at least five researched candidates;
3. differences from every current overlapping-risk demo;
4. single-turn state-to-output contract;
5. internal state, public view, action schema, oracle, and reward design;
6. generator CLI, split sizes, seeds, difficulty, leak checks, and data inspection;
7. smoke, baseline, training, checkpoint, post-eval, serving, and WebUI commands;
8. full experiment configuration and baseline or post-training table;
9. per-seed and per-difficulty results;
10. all failed formal experiments and resulting iterations;
11. tests actually run;
12. Future WebUI visualization layout, components, rendering, and action mapping;
13. known limitations and next steps.

Do not claim any command, result, checkpoint, or service that was not observed
in the current work.

## Final repository hygiene

Before committing:

- inspect `git diff` and status;
- remove caches, local environments, downloaded models, generated formal datasets,
  checkpoints, logs, credentials, endpoints, and absolute local paths;
- preserve only source, focused tests, tiny fixtures if justified, README, and
  compact result records;
- verify no unrelated user changes are included;
- run relevant formatting, linting, and focused tests;
- verify documentation commands against current help.

Commit on a dedicated branch, push, and open a focused PR only when authorized.
Link the motivating issue if one exists, state exact validation, and disclose
skipped GPU or platform checks. Do not auto-close an issue whose acceptance
evidence remains incomplete.

## Final response contract

Report all of the following with direct artifact or PR links where available:

1. selected game and rationale;
2. non-overlap evidence;
3. single-turn state or output design;
4. files added or changed;
5. dataset sizes, splits, seeds, and difficulty distribution;
6. reward formula and exploit resistance;
7. baseline versus post-training table;
8. results for at least two eval seeds;
9. exact training and evaluation commands;
10. focused test results;
11. failed experiments and iterations;
12. WebUI state schema, layout, and interaction mapping;
13. remaining limitations and recommended next step;
14. running service endpoints and probe status when requested.

Evidence takes priority over narrative. If blocked, identify the missing
permission, resource, credential, or external state and the last verified milestone.
