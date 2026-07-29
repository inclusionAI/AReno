# NaN Triage

Check the first non-finite boundary in order:

1. rollout logits/logprobs and sampled tokens;
2. reward and grouped advantages;
3. fixed-token train/ref logprobs;
4. logprob difference and policy ratio;
5. policy/value/KL losses;
6. gradients before clipping;
7. optimizer states and updated weights.

Compare finite counts, min/max, dtype, and masks. Do not apply broad `nan_to_num`, clamp ratios blindly, or lower LR before finding the source.
