# Research and game selection

Read this reference before choosing the game or creating its directory.

## Existing-demo exclusion audit

Inspect the current branch, not a remembered list. At minimum inventory every
demo under `examples/agentic/`, `examples/sft/`, `examples/vl/`, multimodal,
math, and any newly added example roots. For each record:

- task/game name;
- core state representation;
- model output form;
- reward objective;
- training paradigm.

Reject a candidate whose core mechanic or task structure overlaps an existing
demo. A new story, board size, symbols, wording, or visual theme is not a new
mechanic. In particular, do not recreate:

- Tic-Tac-Toe, Gomoku, or other line-making move games;
- Codebreaker, Mastermind, terminal likeness deduction, or isomorphic code games;
- DuelGrid-style board combat or local tactical placement;
- shopping or product constraint selection;
- coding, math verification, or music-generation examples with a cosmetic theme;
- anything newly present in checked-out `examples/` even if absent above.

The final README must include a concrete “Differences from existing demos” table
based on mechanics, state, output, reward, and training shape.

## Chinese-first internet research

Compare at least five candidate mini-games before selecting one. Prefer public or
well-known classic logic games whose rules can be independently reimplemented.

The environment may lack overseas network access. Search in Chinese and prefer
mainland-accessible sources such as Baidu or 360 search and encyclopedias, Zhihu,
Bilibili, CSDN, CNBlogs, Juejin, Jianshu, Chinese university or education sites,
and Chinese game-rule sites. GitHub may be used for necessary API or open-source
implementation references, but not as the only rules source.

- Do not depend on an overseas search engine, community, video site, or rule site.
- If a page fails, times out, requires a proxy, or cannot be fetched reliably,
  switch sources immediately instead of repeatedly waiting.
- Search snippets may support initial screening, but before final selection open
  at least one full Chinese rules page.
- Cross-check selected rules using at least two independent accessible Chinese
  sources.
- Keep actual working URLs in README and summarize rules in original words; do
  not copy long passages.

## Candidate requirements

A candidate must satisfy every item:

1. State generation is deterministic, cheap, and scalable.
2. A reliable algorithm computes valid or optimal answers.
3. Reward is deterministic and needs no human or LLM judge.
4. One input contains everything needed for one output.
5. Difficulty is parameterized and supports diverse data.
6. Random guessing is materially worse than genuine reasoning.
7. A realistic training budget can expose a learning curve.
8. It does not overlap current AReno demos.
9. It uses no copied proprietary puzzle bank, prose, or art.
10. JSON state and finite structured actions naturally support an intuitive WebUI.

## Required comparison record

Put a table in README with at least these columns:

| Candidate | Rules/mechanic | Single-turn conversion | Oracle | Reward | Scale/difficulty | Existing-demo similarity risk | UI fit | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Explain the final selection using trainability and validation quality, not novelty
alone. Preserve sources actually used, including sources for rejected candidates
when they materially informed the decision.
