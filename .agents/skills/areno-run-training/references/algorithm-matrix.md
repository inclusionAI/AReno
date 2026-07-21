# Algorithm Matrix

Confirm the active contract in `areno/api/algorithms.py` and the selected trainer.

| Algorithm | Rollout | Required input | Main checks |
| --- | --- | --- | --- |
| SFT | No | supervised messages/text | response masks and labels |
| DPO | No | prompt plus chosen/rejected | pair order, reference logprobs, beta |
| GSPO | Yes | prompts and `reward_fn` | grouped samples and logprob agreement |
| GRPO | Yes | prompts and `reward_fn` | grouped advantages and token ratios |
| PPO | Yes | prompts and configured roles | ref/value/reward lifecycle and GAE |
| Agentic RL | Yes | `agent_fn`, reward, tool schemas | transcript order, timeout, loss mask |

Inspect `AlgorithmSpec.requires_rollout`; do not create a second implementation registry in a skill script.
