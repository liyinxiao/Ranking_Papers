# OneRec: Key Learnings

## What It Does

OneRec replaces the traditional multi-stage cascade pipeline (retrieval → pre-ranking → ranking) with a **single unified generative model** that directly generates the final recommended video list end-to-end. Deployed at KuaiShou with +1.6% watch time, +1.7% DAU.

## Three Core Components

### 1. Encoder-Decoder with MoE

- Encoder processes user behavior sequences
- Decoder autoregressively generates recommended videos using semantic IDs
- Sparse MoE scales to 1B params without proportional compute increase

### 2. Session-wise List Generation

**Q: The decoder still generates each item one token at a time, right?**

Yes. The decoder is still autoregressive, generating token by token. The difference is **what it can attend to**:

- **Point-wise:** When generating video 3, the model only sees user history. Each slot is independent. You need hand-crafted rules to ensure diversity/coherence.
- **Session-wise:** When generating video 3, the model sees user history **+ video 1 + video 2** via causal attention. It learns inter-item dependencies directly from data.

The output sequence looks like:
```
[BOS] a₁ b₁ c₁ [SEP] a₂ b₂ c₂ [SEP] a₃ b₃ c₃ ...
      video 1          video 2          video 3
```

Same mechanism as GPT generating a coherent paragraph vs generating each sentence independently — token-by-token either way, but the causal context window makes the difference.

### 3. Semantic IDs via Residual Quantization (RQ)

**Q: What are a₁, b₁, c₁?**

Each video is compressed into L=3 levels of discrete codes via Residual Quantization. Each level quantizes the **leftover error** from the previous level:

```
Level 1: Find nearest centroid c₁ → code a_9 (coarse: "sports")
         Residual: r₁ = v - c₁

Level 2: Quantize r₁ → code b_7 (medium: "basketball")
         Residual: r₂ = r₁ - c₂

Level 3: Quantize r₂ → code c_1 (fine: "NBA highlights")

Reconstruction: c₁ + c₂ + c₃ ≈ v
```

The decoder generates **coarse-to-fine, in order** (a → b → c). This enables beam search to narrow the search space at each level.

**Q: Wouldn't multiple videos have the same codes?**

Possible but rare. Balanced K-means forces equal-sized clusters. With K=16K per level and 3 levels: 16K³ = 4×10¹² unique codes >> 10¹⁰ videos. If collisions happen, break ties with original embeddings.

**Q: Why not just use video IDs directly?**

- A softmax over 10 billion video IDs is infeasible
- With 3 levels × 16K codes, you only need 3 softmax calls over 16K
- Bonus: similar videos share prefix codes → the model can generalize

## Iterative Preference Alignment (IPA)

**Q: Is this post-training with RL?**

Post-training yes, but **not RL**. It uses DPO (Direct Preference Optimization), which is a direct loss on preference pairs — no policy gradient, no critic, no environment interaction.

### How DPO Works

Given a (chosen, rejected) pair, the loss pushes the model to prefer chosen:

```
L = -log σ( β · [log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)] )
```

- `π` = current policy (being trained)
- `π_ref` = frozen reference (prevents diverging too far)
- `β` = temperature

It's just a classification loss. No RL machinery needed.

### How IPA Makes It Iterative

```
Round 1: policy₀ → generate N sessions → reward model scores → pick best/worst → DPO → policy₁
Round 2: policy₁ → generate N sessions → reward model scores → pick best/worst → DPO → policy₂
...
```

Each round, the reference model is updated to the previous round's output. As the model improves, even its worst outputs are decent → the margin between best/worst shrinks → **harder negatives → stronger learning signal**. Like self-play in AlphaGo.

### The Reward Model Dependency

**Q: The success of this approach entirely depends on the reward model?**

Yes. This is the critical vulnerability. Unlike LLM DPO where you have actual human preferences, here the reward model is the **only source of preference signal** — you never see what users would have done with an alternative session.

**Risks:**
- **Reward hacking:** Model finds sessions that score high on RM but aren't actually good (e.g., all clickbait)
- **Staleness:** RM is pretrained on historical data, but user preferences shift
- **Compounding errors:** If RM has systematic bias, it compounds across IPA rounds

**Mitigations in the paper:**
- **Multi-target RM:** Predicts watch time + view prob + follow prob + like prob (harder to hack all four)
- **DPO ratio (`r_DPO`):** Only a fraction of training steps use DPO; rest use NTP on real data to stay grounded
- **β penalty:** KL divergence against reference model prevents aggressive divergence

**Bottom line:** The ceiling of IPA is the quality of the reward model. If the RM can't distinguish genuinely good from superficially good, IPA optimizes for the superficial.
