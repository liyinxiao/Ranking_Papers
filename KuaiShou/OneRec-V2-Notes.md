# OneRec-V2: Key Learnings

## What Changed from V1

| Aspect | V1 | V2 |
|---|---|---|
| Architecture | Encoder-Decoder | Lazy Decoder-Only |
| FLOPs (1B model) | 296 GFLOPs | 19 GFLOPs (15x reduction) |
| Max scale | 1B | 8B |
| Prediction | Session-wise (6 videos) | Single item (3 tokens) |
| Post-training | DPO with Reward Model | GBPO with real user feedback + RM hybrid |
| Traffic | Small experiment | 25% of total traffic |

## 1. Lazy Decoder-Only Architecture

### The Problem with V1

In V1's encoder-decoder, 97.66% of compute goes to encoding user history (512+ tokens through N transformer layers), only 2.34% to actual item generation (3 semantic ID tokens). Massively wasteful.

### The Solution: Kill the Encoder

V2 replaces the transformer encoder with a **Context Processor** — just linear projections, no self-attention:

```
V1: User history → Transformer Encoder (N layers, expensive) → K, V
V2: User history → Linear + RMSNorm (single projection, cheap) → K, V
```

The decoder still cross-attends to these K, V pairs, but the "encoding" is a linear layer, not a transformer. They even share K=V (same tensor for both), and multiple decoder layers share the same K, V pairs.

### Q: Cross-attention without an encoder?

Yes. Cross-attention just needs K, V from somewhere. In V1 they come from a transformer encoder. In V2 they come from a linear projection. The decoder's query tokens still attend to user context — the mechanism is the same, but the context preparation is 15x cheaper.

### Q: Why is it called "lazy"?

The decoder **lazily** avoids doing any heavy computation on context. It only runs transformer layers on the 3 tokens it needs to predict: `[BOS, s1, s2] → [s1, s2, s3]`. Always exactly 3 tokens, regardless of user history length. ~100% of FLOPs go to target decoding.

### Q: V1 generated full sessions — what happened to that?

V2 dropped session-wise generation and went back to single-item prediction. This is what makes the "lazy" design possible — fixed 3-token decoder input.

To fill a page, they use **Pass@k**: run inference multiple times independently. V2's argument is that the 15x compute savings let you run it k times and still come out ahead, while scaling to 8B parameters more than compensates for losing session-wise inter-item context.

### Scaling Results

Loss follows a clean scaling law: `L = 3.13 + 3660/N^0.489`

| Model | Convergence Loss |
|---|---|
| 0.1B | 3.57 |
| 1B | 3.27 |
| 4B | 3.20 |
| 8B | 3.19 |
| 4B MoE (0.5B active) | 3.22 (beats 2B dense) |

Additional efficiency tricks:
- **KV-sharing across layers:** No loss degradation
- **GQA:** Reduce KV heads from 14→1 with negligible loss change
- **MoE:** 4B total (0.5B active) beats 2B dense

## 2. Preference Alignment with Real User Feedback

### The Problem with V1 (which we identified)

V1's reward model is the bottleneck — reward hacking, staleness, only 1% of users sampled for rollouts. The entire IPA loop depends on a proxy signal.

### V2's Answer: Use Actual User Feedback

Now that OneRec serves 25% of real traffic, they have direct user signals instead of a proxy reward model.

### Duration-Aware Reward Shaping

Raw watch time is biased by video length (60s video naturally gets more watch time than 10s). Solution:

1. Bucket videos by duration (log-scale)
2. For each user, compute percentile rank of play time within the matching bucket
3. Top 25% → positive (A=+1), explicit "dislike" → negative (A=-1), rest filtered out (A=0)

This normalizes for duration bias — a user watching 80% of a short video is more engaged than watching 30% of a long one.

### GBPO (Gradient-Bounded Policy Optimization)

Replaces DPO. Key insight: for negative samples, traditional clipping methods (PPO, GRPO, ECPO) can still cause gradient explosion when the policy ratio is 1.

```
Traditional RL gradient for negative sample: ∝ 1/π_θ
  → Small π_θ means huge gradient → explosion

BCE gradient for negative sample: ∝ 1/(1-π_θ)
  → Small π_θ means small gradient → stable
```

GBPO bounds RL gradients with the more stable BCE gradients. Two strengths:
- No samples discarded (unlike clipping)
- Gradients bounded by BCE loss (stable for negative samples)

### Reward Model vs User Feedback vs Hybrid

From A/B tests (Table 7):

| Signal | App Stay Time | Interaction (likes, follows, etc.) |
|---|---|---|
| Reward Model | Good | **Best** |
| User Feedback | **Best** | Moderate |
| Hybrid | Good | Best |

User feedback is better for the primary metric (stay time), reward model is better for interaction metrics. The hybrid gets best of both — confirming V1's reward model had blind spots that real signals don't.

### Self-Improvement Effect

Training on OneRec's own generated samples (on-policy) dramatically outperforms training only on traditional pipeline samples. With OneRec samples, all metrics improve; without them, interaction metrics (video view, comment, forward) actually degrade.

## 3. Online A/B Test Results

| Metric | V2 vs V1 (Kuaishou) | V2 vs V1 (Kuaishou Lite) |
|---|---|---|
| App Stay Time | **+0.467%** | **+0.741%** |
| Watch Time | +1.367% | +0.597% |
| Video View | +1.484% | +0.716% |
| Like | +8.286% | +7.605% |
| Follow | +8.910% | +9.445% |

At 400M DAU scale, these are massive improvements.

## Key Takeaway

V2 is a story of **trading model elegance for engineering pragmatism**. V1's session-wise generation was theoretically appealing (inter-item dependencies!), but V2 showed that a simpler single-item architecture, scaled 8x larger with 15x less compute, wins in practice. And replacing the reward model proxy with real user feedback addressed exactly the vulnerability we identified in V1.
