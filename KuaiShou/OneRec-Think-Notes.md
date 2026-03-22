# OneRec-Think: Key Learnings

## What It Does

Adds **explicit Chain-of-Thought reasoning** to generative recommendation. Instead of directly predicting the next item, the model first generates a natural language reasoning trace, then recommends.

```
V1/V2:  User history → [item tokens]
Think:  User history → <think> reasoning </think> → [item tokens]
```

Deployed at KuaiShou with +0.159% App Stay Time.

## Three-Stage Training

### Stage 1: Itemic Alignment

Teach an LLM to understand semantic ID tokens via multi-task pre-training:
- **User Persona Grounding** — interleave items with text, generate user profile
- **Sequential Preference Modeling** — predict next item from history
- **Itemic Dense Captioning** — given item tokens, generate text description
- **General Language Modeling** — preserve LLM capabilities

### Stage 2: Reasoning Activation

CoT fails on raw noisy user histories. Solution — progressive distillation:
1. **Pruned contexts first:** Retrieve top-k most relevant history items to the target. Generate clean reasoning traces from these easy examples.
2. **Noisy sequences next:** Train on full messy histories using the clean rationales as supervision. Model learns to reason despite noise.

### Stage 3: Reasoning Enhancement (RL)

GRPO with **Rollout-Beam reward**: generate K candidates via beam search after each reasoning path. Reward if *any* candidate matches ground truth — captures multi-validity of preferences (many items could be correct). Denser signal than single-point evaluation.

## Q&A Deep Dives

### Q: What "fresh context" does V2 use in the online stage?

The actual input is the same — user history. The freshness is about **timeliness**:
- **Think (offline, hours ago):** Generates reasoning + first 2 tokens using a snapshot of user history from hours ago
- **V2 (online, right now):** Sees the user's latest interactions (videos watched in the last minutes/hours) and its own model weights are more recent from continuous streaming training

The first 2 tokens lock in the coarse category (e.g., "cooking videos"), V2 picks the specific item within that category using the most current data.

### Q: Think only generates the first 2 tokens of one video?

Yes. The split per video:
```
Think (offline):   token 1 (coarse) + token 2 (medium)  →  cached
V2 (online):       token 3 (fine)                        →  real-time
```

Think doesn't generate a session of multiple videos. The "multiple candidates" come from **beam search** — each beam is a different possible next video, not a sequence:

```
Reasoning path 1: "User likes cooking..."
  → Beam 1: (a_9, b_7)  ← different video options
  → Beam 2: (a_9, b_3)     not a sequence
  → Beam 3: (a_4, b_5)
```

They run T different reasoning paths × m beam candidates = T×m candidate prefixes total. To fill a page of 6 videos, pick the best prefixes and complete each with V2 independently.

### Q: So the real innovation is just CoT + doing it offline?

Yes. The two core contributions:

1. **CoT for recommendation works.** Explicit reasoning improves accuracy. The model connects interests ("likes Battlefield + GTA6" → "cares about GPU performance" → recommend GPU comparison video) instead of just pattern-matching co-occurrence.

2. **Think-Ahead makes it deployable.** CoT is expensive, but coarse item direction doesn't need real-time freshness. Reasoning + coarse selection offline, final selection online.

Everything else is supporting machinery:
- Itemic Alignment → LLM doesn't natively understand semantic IDs
- Progressive distillation → CoT fails on raw noisy histories
- Rollout-Beam reward → standard RL rewards too sparse for recommendation

## How Think Fits in the OneRec Family

| | V1 | V2 | Think |
|---|---|---|---|
| Base model | Custom enc-dec | Custom lazy decoder | LLM (e.g., Qwen) |
| Reasoning | Implicit | Implicit | Explicit (text CoT) |
| Prediction | Session-wise (6 videos) | Single item (3 tokens) | Single item (3 tokens) |
| Deployment | Standalone | Standalone | Think (offline coarse) + V2 (online fine) |

Think doesn't replace V2 — they're complementary. Think is essentially a **reasoning-powered retrieval stage** that generates coarse candidates, and V2 is the **real-time ranking/selection stage**. They've come full circle back to a two-stage system, just implemented with generation instead of traditional retrieval + ranking.
