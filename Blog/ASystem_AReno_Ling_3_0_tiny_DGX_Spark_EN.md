# ASystem AReno: Post-Training Ling-3.0-tiny on a Single DGX Spark

> Ling-3.0-tiny is now officially open source.
>
> With 7.9B total parameters and only 1.3B activated parameters per token at inference time, what makes this lightweight model exciting is not only that it is small, but that it can move into local devices, real tasks, and developers' own workflows.
>
> So here comes the next question: if the model can run on DGX Spark, can it also be trained on DGX Spark?
>
> This is where AReno enters the story.

The article ["Ling-3.0-tiny officially open-sourced: bringing 1.3B activated parameters into real tasks"](https://mp.weixin.qq.com/s/YyrzNIZpnas4lerpC63DaQ) shows Ling-3.0-tiny running on local devices such as DGX Spark, MacBook, and Mac mini. For developers, the implication is straightforward: models no longer have to stay behind cloud APIs or large-scale clusters. They can move into local devices, local data environments, and concrete workflows.

But getting a model to run is only the first step.

Once a model enters a concrete task, developers will naturally ask a more practical question: can the model keep training according to my task rules, feedback signals, and tool environment? Can it move from "answering" to "doing"? Can it become more stable and controllable through repeated feedback?

This is exactly the question [AReno](https://github.com/inclusionAI/AReno) is designed to explore.

In this experiment, we used open-source AReno to post-train Ling-3.0-tiny on DGX Spark for an Agentic RL task. The model plays tic-tac-toe through tool calls, receives rule-based feedback from the environment, and improves its decision stability with GSPO training.

Tic-tac-toe may sound like a small game, but in this experiment it plays an important role: it is a **minimal verifiable task**.

It is small enough for developers to understand quickly, yet complete enough to cover state understanding, tool calling, environment feedback, reward design, and post-training evaluation. In other words, the point is not to make Tiny "play a game." The point is to see whether AReno can close the post-training loop once Tiny can already run efficiently on a local device.

## From Local Inference to Local Post-Training

As Ling-3.0-tiny starts running on local devices such as DGX Spark, MacBook, and Mac mini, lightweight models are moving into lower-cost environments that are closer to developers' daily workflows and real tasks.

The AReno Spark experiment pushes the question one step further: if a model can already run locally, can it also complete an observable, reproducible, and iterative post-training loop locally?

Our results show that DGX Spark is not only a lightweight model inference device. It can also become a starting point for developers to validate post-training ideas. In the same kind of lightweight environment, developers can define tasks, generate data, design rewards, sample rollouts, update policies, and evaluate the resulting behavior.

This extends the value of Tiny from "lower deployment cost" to "lower task adaptation cost."

## Why Ling-3.0-tiny Fits Task-Level Post-Training

In post-training, model size has a direct impact on whether developers can quickly complete a task adaptation loop.

When the model is too large, each experiment often requires more GPU memory, more complex parallelism, and longer training waits. Developers then struggle to answer practical questions quickly: did changing the reward actually change the model behavior? Did the model learn the new constraints after the environment changed? Did adjusting the tool-calling logic reduce errors?

Ling-3.0-tiny helps fill that gap:

- **Lower cost**: it is better suited for training, debugging, and reproducing experiments in lightweight environments such as DGX Spark.
- **Faster iteration**: reward, data, prompt, and agent logic changes can show effects more quickly.
- **More direct feedback**: developers can compare behavior changes against reward curves, response length, and evaluation demos.
- **Better fit for Agentic RL exploration**: tool calls, environment interaction, and rule-based feedback make it possible to observe how a model changes across multi-step decisions.

So instead of only showing that Tiny can run, we want to show that it can also train: in a clear, feedback-rich, reproducible task environment, AReno can connect behavior observation with policy updates.

## Starting From a Minimal Verifiable Task: Tic-Tac-Toe Agentic RL

We chose tic-tac-toe as the first validation task not because it is complex, but because it is clear.

The board has only nine cells. The rules are simple and the outcome is easy to judge. At the same time, it is not a pure single-step classification task. The model must understand the current board state, choose a legal position, avoid occupied cells, and take a winning move when one is available.

This makes tic-tac-toe a good minimal environment for Agentic RL:

- The model must call the `choose_square` tool to make a move.
- The environment checks whether the move is legal, optimal, or immediately winning.
- The reward distinguishes invalid actions, legal but non-optimal actions, optimal actions, and winning actions.
- After training, the model can be evaluated again in the same environment to see whether its behavior improved.

In other words, this is not an experiment where you can only look at metrics. Developers can return to the interactive environment and directly observe how the model calls tools, chooses actions, makes mistakes, and becomes more stable after training.

More importantly, this experiment is not an isolated demo. It represents a reusable path for lightweight model task adaptation:

- **Baseline evaluation**: first let Ling-3.0-tiny play in the same environment and observe whether it understands the state, calls the right tool, and selects legal actions.
- **Feedback definition**: convert rules, invalid moves, optimal moves, and winning moves into a clear reward.
- **Post-training adaptation**: use AReno to connect rollout, reward, training, and sampling into one loop so the model can update its policy from interaction feedback.
- **Re-evaluation**: return to the same environment and use demos, reward curves, and response length to judge whether the model actually improved.

This case is therefore not only about Tiny learning tic-tac-toe. It demonstrates a more general method: identify a model's behavior boundary in a concrete task, use AReno to post-train it for task adaptation, and evaluate the improvement in the same environment.

## Before Training: The Model Understands the Rules, but Its Actions Are Unstable

Before training, Ling-3.0-tiny already understands part of the task context. Once it enters an interactive environment, however, its actions are not stable enough.

For example, in the initial behavior, the model may make judgment errors or even place a move in an already occupied square. This is a typical issue in Agentic RL: the model appears to "know the rules," but tool-call parameters, state understanding, and action selection all become stress points in real interaction.

![Ling-3.0-tiny tic-tac-toe behavior before training](assets/ling-tiny-tictactoe-before.gif)

This is the value of an interactive task.

If we only test offline question answering, it is hard to tell whether the model failed because it did not know the rules, misunderstood the state, produced an invalid tool call, or knew a legal move but chose a poor strategy. An interactive environment separates these issues clearly: each step can be evaluated for legality, state constraint violations, and missed opportunities.

The same logic applies to more realistic business tasks. Whether a customer-service agent called the right tool, whether a workflow agent filled a form correctly, or whether a coding agent modified files as requested all require environment feedback. Tic-tac-toe simply compresses that idea into a small and easy-to-understand task.

## Dataset Design: Reproducible Task States

To make training reproducible, we first generate a set of intermediate tic-tac-toe board states. The model learns to choose the next action under different board conditions.

The dataset generation logic is intentionally simple:

1. Create an empty 3x3 board, using `.` for empty cells.
2. Start from X and randomly play 0 to 6 moves.
3. Each move is randomly selected from currently legal empty cells.
4. Stop once the game reaches a win/loss state or the board ends.
5. Keep only states where it is X's turn, the game is unfinished, and the state is not duplicated.
6. Use a fixed random seed so the same parameters generate the same data.
7. If the random process ends on O's turn, the generator adds one extra random O move so the final state is X's turn.

One sample looks like this:

```json
{"id":"generated-00000","board":[["X",".","O"],[".",".","."],["X","O","."]]}
```

The model needs to choose the next action based on the current board.

## Reward Design: Turning Task Rules Into Training Feedback

We designed a simple reward for the tic-tac-toe task. The goal is not complexity. The goal is to express the most important task constraints clearly:

- No valid `choose_square` tool call: `-1.0`
- Unparseable parameter or occupied square: `-1.0`
- A move that lets X win immediately: `1.0`
- An optimal move without an immediate win: `0.8`
- A legal but non-optimal move: `0.0`

This reward captures three key requirements.

First, the model must learn to act through the correct tool format. Second, it must respect the environment state and avoid occupied cells. Third, it must go beyond legality and pursue better strategies.

This is the kind of post-training experience AReno aims to provide: developers do not need to build a massive system before starting. They can begin with a clear task, define the environment, define the reward, run rollouts and training, and observe how the model behavior changes.

## Spark Results: The Model Becomes More Stable Through Feedback

This experiment was run on DGX Spark using AReno's GSPO algorithm for 400 steps.

During training, `rollout/rewards_mean` increased from around `-0.5` to about `0.4`. This indicates that the model gradually reduced invalid actions, increased the ratio of effective actions, and learned to choose more reasonable moves.

![rollout/rewards_mean curve](assets/ling-tiny-rewards-mean.png)

At the same time, `response_len` dropped to around `850 tokens`. This means the model became more concise when completing the task, reducing unnecessary long reasoning and ineffective output. For Agentic tasks, this also matters: the model should not only be correct, but also complete the task in a more stable and efficient way.

![response_len curve](assets/ling-tiny-response-len.png)

One metric reflects better task feedback, while the other reflects more converged model output. Taken together, they show that post-training is changing the model's behavior.

Beyond metrics, the more intuitive test is to return to the same interactive environment and evaluate the model again.

After training, Ling-3.0-tiny becomes much more stable in tool calling and action selection. It better respects the board state and chooses more reasonable moves at key moments.

![Ling-3.0-tiny tic-tac-toe behavior after training](assets/ling-tiny-tictactoe-after.gif)

For AReno, the significance is not just that the reward goes up.

More importantly, the result shows that in a lightweight environment such as DGX Spark, developers can complete a relatively full post-training loop: task definition, data generation, reward design, rollout sampling, policy update, and behavior evaluation in the same environment.

This extends Tiny from "a model that can run on a local device" to "a model that can continue adapting inside a local task." For developers, that means they can start from a small task, validate whether their reward and agent logic work, and then migrate the approach to more realistic workflows.

## How to Reproduce the Experiment on DGX Spark

We provide a complete Docker-based guide for running the training on DGX Spark. Developers can reproduce the tic-tac-toe experiment directly, and after it runs successfully, replace the dataset, reward, or agent function to adapt the flow to their own post-training tasks.

Guide: [tiny v3 on spark](https://github.com/inclusionAI/AReno/blob/main/examples/agentic/tictactoe/DGX_SPARK_GUIDE_CN.md)

## Back to AReno: Closing the Lightweight Model Adaptation Loop

[AReno](https://github.com/inclusionAI/AReno), short for **ASystem Reinforcement Learning Nano**, is a local LLM post-training toolkit initiated by the Ant Group ASystem team and maintained under the InclusionAI open-source ecosystem.

If Ling-3.0-tiny answers how a lightweight model can enter local devices and real task environments, AReno answers what happens next: after the model enters a task, how can it continue improving from feedback and become better adapted to that task?

The original goal of AReno is to lower the engineering barrier for LLM post-training and task adaptation.

In the past, reinforcement learning post-training often meant wiring together training frameworks, inference services, kernel dependencies, environment setup, cluster resources, reward design, and agent trajectory collection. If any part of the loop failed, a concrete task adaptation idea could easily get stuck before it even started running.

AReno takes the opposite direction: let more developers start from one machine, one small model, and one clear task, then run the full post-training loop and see how the model improves through feedback.

In this loop, AReno connects several key components. Today, AReno supports:

- SFT / DPO-style training
- RL post-training
- Agentic RL
- Local model serving
- Custom reward functions
- The training, inference, sampling, and optimization loop

For developers, AReno's strengths can be summarized in five points:

- **Affordable to train**: optimized for small models and local environments, lowering the resource barrier for post-training experiments.
- **Runnable end to end**: connects rollout, reward, train, and serve into one loop.
- **Observable**: uses reward curves, response length, and evaluation demos to observe model behavior changes.
- **Easy to modify**: developers can replace data, environments, rewards, and agent functions.
- **Portable to new tasks**: tic-tac-toe is only an example; the same method can move to tool calling, structured extraction, workflow agents, and more.

Tic-tac-toe is just the beginning.

AReno will continue adding examples for small models, Agentic RL, developer practice, and education. We hope developers can start from smaller and more intuitive tasks, then gradually migrate post-training methods into their own real scenarios.

## Join Us

We believe a truly vibrant open-source project is not just about releasing code. It is about helping more people use it, modify it, and contribute back.

If you are interested in any of the following areas, we welcome you to join us:

- Lightweight model post-training
- Agentic RL
- Reward function design
- Tool calling and workflow agents
- Small-model task adaptation
- Local AI toolchains
- AI education and developer communities

Ling-3.0-tiny gives lightweight models a foundation for entering local devices and real tasks. AReno aims to further lower the barrier for post-training and task adaptation.

Starting from this tic-tac-toe experiment on DGX Spark, we see more than a game demo. We see a general path: run the model locally, define a task with AReno, design feedback, train the model, and return to the same environment to verify whether it truly improved.

This path can continue into tool-call repair, structured extraction, domain-specific instruction following, personal writing style adaptation, workflow agents, and more.

From running, to training, to adapting to your own task.

That is the next step Ling-3.0-tiny x AReno hopes to bring to developers.

Visit the [AReno GitHub repository](https://github.com/inclusionAI/AReno) to reproduce the Spark post-training experiment, open issues, join discussions, or contribute your own Tiny task adaptation case.

> Ant Group ASystem is building next-generation AI infrastructure software and exploring new methods toward general intelligence on top of it. We have open-sourced several projects, including Awex, AMem, AState, and AReno.
>
> Contact us: pub_asystem@antgroup.com
