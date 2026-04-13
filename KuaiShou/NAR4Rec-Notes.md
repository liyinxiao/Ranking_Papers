# NAR4Rec: Non-autoregressive Generative Models for Reranking Recommendation

## What It Does

Reranking stage in a traditional multi-stage pipeline. Takes ~60 candidates from ranking, selects and orders ~6 to show the user. Not part of the OneRec family — works within the existing pipeline rather than replacing it.

## Core Idea

Replace autoregressive (sequential, item-by-item) generation with non-autoregressive (all positions at once) generation. Outputs a probability matrix in a single forward pass: P(item i at position j) for all items and positions simultaneously. ~5x faster inference.

## Three Innovations

### 1. Matching Model
Candidates change per request (unlike fixed vocab in NLP), so they use a candidates encoder + position encoder with cross-attention, then dot product to get probabilities. Shared position embeddings help with sparse data.

### 2. Unlikelihood Training
Split sequences by user feedback quality:
- High utility (good feedback) -> maximize likelihood
- Low utility (bad feedback) -> maximize unlikelihood (learn to avoid)

### 3. Contrastive Decoding
Non-autoregressive models predict each position independently, risking duplicate/similar items. Fix at decode time:

```
y_i = argmax (1-α) × model_confidence - α × max_similarity_to_already_selected
```

**What "contrastive" means here:** Contrasting two competing signals — model quality (pull toward) vs. redundancy penalty (push away). The selection picks items that are both high-quality AND distinct from what's already chosen.

**Not the same "contrastive" as in contrastive learning** (SimCLR, etc.) which is about pulling/pushing in embedding space during training. This is a decode-time selection strategy.

**Essentially the same as MMR (Maximal Marginal Relevance, 1998):**
```
MMR = argmax λ × relevance(item) - (1-λ) × max_similarity(item, selected)
```
The name "contrastive decoding" is more modern branding for this classic idea.

## Relation to OneRec

Opposite philosophy:
- **OneRec**: replace entire multi-stage pipeline with one generative model
- **NAR4Rec**: improve the reranking stage within the existing pipeline

NAR4Rec's contrastive decoding is relevant to the diversity problem that OneRec V1/V2's point-wise generation doesn't address internally — OneRec generates items independently and has no built-in mechanism for diversity in the final list.

