"""
OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment
PyTorch reproduction of the model architecture described in the paper.

Key components:
1. Balanced K-means for semantic ID generation (residual quantization)
2. T5-style encoder-decoder with sparse MoE in decoder
3. Session-wise reward model
4. DPO / IPA training logic
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ============================================================
# Configuration
# ============================================================

@dataclass
class OneRecConfig:
    # Semantic ID
    num_codebook_levels: int = 3          # L = 3
    codebook_size: int = 8192             # K = 8192 per level
    embedding_dim: int = 128             # multi-modal embedding dim

    # Encoder
    enc_num_layers: int = 12
    enc_hidden_dim: int = 1024
    enc_num_heads: int = 16
    enc_ffn_dim: int = 4096

    # Decoder
    dec_num_layers: int = 12
    dec_hidden_dim: int = 1024
    dec_num_heads: int = 16
    dec_ffn_dim: int = 4096

    # MoE (decoder only)
    num_experts: int = 24                 # N_MoE = 24
    num_active_experts: int = 2           # K_MoE = 2

    # Sequence lengths
    max_history_len: int = 256            # n = 256
    session_size: int = 5                 # m = 5

    # Special tokens
    bos_token_id: int = 0
    sep_token_id: int = 1
    pad_token_id: int = 2

    # Training
    dropout: float = 0.1
    dpo_beta: float = 0.1
    dpo_sample_ratio: float = 0.01       # r_DPO = 1%
    num_beam_candidates: int = 128        # N = 128

    @property
    def vocab_size(self):
        """Total vocab = codebook_size * num_levels + special tokens."""
        return self.codebook_size * self.num_codebook_levels + 3  # BOS, SEP, PAD


# ============================================================
# 1. Balanced K-Means for Semantic ID Generation
# ============================================================

class BalancedKMeans:
    """
    Algorithm 1: Balanced K-means Clustering.
    Partitions items into K equal-sized clusters.
    """

    def __init__(self, num_clusters: int, max_iters: int = 100):
        self.K = num_clusters
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) item embeddings
        Returns:
            assignments: (N,) cluster assignments
        """
        N, D = embeddings.shape
        w = N // self.K  # items per cluster

        # Initialize centroids randomly
        indices = torch.randperm(N)[:self.K]
        self.centroids = embeddings[indices].clone()

        prev_assignments = None

        for _ in range(self.max_iters):
            assignments = torch.full((N,), -1, dtype=torch.long)
            unassigned = torch.ones(N, dtype=torch.bool)

            for k in range(self.K):
                # Compute distances from centroid k to all unassigned items
                unassigned_idx = torch.where(unassigned)[0]
                if len(unassigned_idx) == 0:
                    break

                dists = torch.cdist(
                    self.centroids[k:k+1],
                    embeddings[unassigned_idx]
                ).squeeze(0)

                # Assign w nearest unassigned items
                num_to_assign = min(w, len(unassigned_idx))
                _, nearest = torch.topk(dists, num_to_assign, largest=False)
                chosen = unassigned_idx[nearest]

                assignments[chosen] = k
                unassigned[chosen] = False

                # Update centroid
                self.centroids[k] = embeddings[chosen].mean(dim=0)

            # Check convergence
            if prev_assignments is not None and torch.equal(assignments, prev_assignments):
                break
            prev_assignments = assignments.clone()

        return assignments


class ResidualQuantizer(nn.Module):
    """
    Multi-level residual K-means quantization.
    Converts multi-modal embeddings into L-level semantic IDs.
    """

    def __init__(self, config: OneRecConfig):
        super().__init__()
        self.L = config.num_codebook_levels
        self.K = config.codebook_size
        self.D = config.embedding_dim

        # Codebooks: L levels, each with K centroids of dim D
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(self.K, self.D))
            for _ in range(self.L)
        ])

    def build_codebooks(self, embeddings: torch.Tensor):
        """Fit balanced K-means at each level with residual computation."""
        residuals = embeddings.clone()

        for l in range(self.L):
            kmeans = BalancedKMeans(self.K)
            kmeans.fit(residuals)
            self.codebooks[l].data = kmeans.centroids.clone()

            # Compute assignments and update residuals
            dists = torch.cdist(residuals, kmeans.centroids)
            assignments = dists.argmin(dim=1)
            residuals = residuals - kmeans.centroids[assignments]

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D)
        Returns:
            semantic_ids: (N, L) indices into each codebook level
        """
        residuals = embeddings.clone()
        semantic_ids = []

        for l in range(self.L):
            dists = torch.cdist(residuals, self.codebooks[l])
            ids = dists.argmin(dim=1)  # (N,)
            semantic_ids.append(ids)
            residuals = residuals - self.codebooks[l][ids]

        return torch.stack(semantic_ids, dim=1)  # (N, L)


# ============================================================
# 2. Core Transformer Building Blocks
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1,
                 is_causal: bool = False, is_cross: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.is_causal = is_causal
        self.is_cross = is_cross

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                kv: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        kv_input = kv if self.is_cross and kv is not None else x
        S = kv_input.shape[1]

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


# ============================================================
# 3. Sparse Mixture-of-Experts (Eq. 2)
# ============================================================

class MoELayer(nn.Module):
    """
    Sparse MoE layer for the decoder (Eq. 2).
    Only K_MoE out of N_MoE experts are activated per token.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int,
                 num_active: int, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active

        # Expert FFNs
        self.experts = nn.ModuleList([
            FeedForward(hidden_dim, ffn_dim, dropout)
            for _ in range(num_experts)
        ])

        # Router: projects hidden state to expert scores
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
            aux_loss: load balancing loss
        """
        B, T, D = x.shape

        # Compute gate scores: s_{i,t} = Softmax_i(H^l_t^T @ e^l_i)
        router_logits = self.router(x)                    # (B, T, N_MoE)
        router_probs = F.softmax(router_logits, dim=-1)   # (B, T, N_MoE)

        # Top-k selection
        topk_probs, topk_indices = torch.topk(
            router_probs, self.num_active, dim=-1
        )  # (B, T, K_MoE)

        # Normalize gate values among selected experts
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute expert outputs (only for selected experts)
        output = torch.zeros_like(x)
        for k in range(self.num_active):
            expert_idx = topk_indices[:, :, k]       # (B, T)
            gate_val = topk_probs[:, :, k:k+1]       # (B, T, 1)

            for e in range(self.num_experts):
                mask = (expert_idx == e)              # (B, T)
                if mask.any():
                    indices = mask.nonzero(as_tuple=True)
                    expert_input = x[indices[0], indices[1]]  # (num_tokens, D)
                    expert_out = self.experts[e](expert_input.unsqueeze(1)).squeeze(1)
                    output[indices[0], indices[1]] += gate_val[indices[0], indices[1]] * expert_out

        # Load balancing auxiliary loss
        # Fraction of tokens routed to each expert
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        flat_indices = topk_indices.reshape(-1, self.num_active)
        for e in range(self.num_experts):
            tokens_per_expert[e] = (flat_indices == e).float().sum()
        tokens_per_expert = tokens_per_expert / (B * T * self.num_active)

        # Mean router probability per expert
        mean_probs = router_probs.mean(dim=[0, 1])

        aux_loss = self.num_experts * (tokens_per_expert * mean_probs).sum()

        return x + output, aux_loss


# ============================================================
# 4. Encoder & Decoder Layers
# ============================================================

class EncoderLayer(nn.Module):
    """Encoder layer: fully visible self-attention + FFN + RMSNorm."""

    def __init__(self, config: OneRecConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.enc_hidden_dim)
        self.self_attn = MultiHeadAttention(
            config.enc_hidden_dim, config.enc_num_heads,
            config.dropout, is_causal=False
        )
        self.norm2 = RMSNorm(config.enc_hidden_dim)
        self.ffn = FeedForward(config.enc_hidden_dim, config.enc_ffn_dim, config.dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Decoder layer: causal self-attention + cross-attention + MoE FFN + RMSNorm."""

    def __init__(self, config: OneRecConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.dec_hidden_dim)
        self.self_attn = MultiHeadAttention(
            config.dec_hidden_dim, config.dec_num_heads,
            config.dropout, is_causal=True
        )
        self.norm2 = RMSNorm(config.dec_hidden_dim)
        self.cross_attn = MultiHeadAttention(
            config.dec_hidden_dim, config.dec_num_heads,
            config.dropout, is_causal=False, is_cross=True
        )
        self.norm3 = RMSNorm(config.dec_hidden_dim)
        self.moe = MoELayer(
            config.dec_hidden_dim, config.dec_ffn_dim,
            config.num_experts, config.num_active_experts, config.dropout
        )

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        x = x + self.cross_attn(self.norm2(x), kv=encoder_out, mask=memory_mask)
        x, aux_loss = self.moe(self.norm3(x))
        return x, aux_loss


# ============================================================
# 5. OneRec Model
# ============================================================

class OneRec(nn.Module):
    """
    Full OneRec encoder-decoder with MoE.
    Encoder: bidirectional self-attention over user history semantic IDs.
    Decoder: causal generation of session semantic IDs with MoE FFN.
    """

    def __init__(self, config: OneRecConfig):
        super().__init__()
        self.config = config

        # Shared token embedding for semantic IDs + special tokens
        self.token_embedding = nn.Embedding(config.vocab_size, config.enc_hidden_dim)
        self.position_embedding = nn.Embedding(
            config.max_history_len * (config.num_codebook_levels + 1) + 64,  # generous max
            config.enc_hidden_dim
        )

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.enc_num_layers)
        ])
        self.encoder_norm = RMSNorm(config.enc_hidden_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.dec_num_layers)
        ])
        self.decoder_norm = RMSNorm(config.dec_hidden_dim)

        # Output head: predicts codebook token at each position
        self.lm_head = nn.Linear(config.dec_hidden_dim, config.vocab_size, bias=False)

        # Tie embeddings with output head
        self.lm_head.weight = self.token_embedding.weight

    def _make_encoder_input(self, history_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert semantic IDs to flat token sequence with SEP tokens.

        Args:
            history_ids: (B, n, L) semantic IDs for n history items, L levels each.
                         Values should be in [0, K-1] per level.
        Returns:
            token_ids: (B, T_enc) flattened token sequence with level offsets and SEPs.
        """
        B, n, L = history_ids.shape
        config = self.config
        device = history_ids.device

        sequences = []
        for i in range(n):
            for l in range(L):
                # Offset each level's tokens to avoid collision:
                # level 0: [3, K+2], level 1: [K+3, 2K+2], level 2: [2K+3, 3K+2]
                offset_ids = history_ids[:, i, l] + 3 + l * config.codebook_size
                sequences.append(offset_ids.unsqueeze(1))
            # Add SEP after each item
            sep = torch.full((B, 1), config.sep_token_id, device=device, dtype=torch.long)
            sequences.append(sep)

        return torch.cat(sequences, dim=1)  # (B, n*(L+1))

    def _make_decoder_input(self, session_ids: torch.Tensor) -> torch.Tensor:
        """
        Construct decoder input with BOS before each item (Eq. 3).

        Args:
            session_ids: (B, m, L) semantic IDs for target session.
        Returns:
            token_ids: (B, T_dec) = (B, m*(L+1))
        """
        B, m, L = session_ids.shape
        config = self.config
        device = session_ids.device

        sequences = []
        for i in range(m):
            # BOS before each item
            bos = torch.full((B, 1), config.bos_token_id, device=device, dtype=torch.long)
            sequences.append(bos)
            for l in range(L):
                offset_ids = session_ids[:, i, l] + 3 + l * config.codebook_size
                sequences.append(offset_ids.unsqueeze(1))

        return torch.cat(sequences, dim=1)  # (B, m*(L+1))

    def encode(self, encoder_input_ids: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the encoder."""
        B, T = encoder_input_ids.shape
        positions = torch.arange(T, device=encoder_input_ids.device).unsqueeze(0)

        x = self.token_embedding(encoder_input_ids) + self.position_embedding(positions)

        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)

        return self.encoder_norm(x)

    def decode(self, decoder_input_ids: torch.Tensor,
               encoder_out: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the decoder. Returns logits and total MoE aux loss."""
        B, T = decoder_input_ids.shape
        positions = torch.arange(T, device=decoder_input_ids.device).unsqueeze(0)

        x = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)

        total_aux_loss = 0.0
        for layer in self.decoder_layers:
            x, aux_loss = layer(x, encoder_out, tgt_mask, memory_mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.decoder_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss

    def forward(self, history_ids: torch.Tensor, session_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training with NTP loss (Eq. 4).

        Args:
            history_ids: (B, n, L) user history semantic IDs
            session_ids: (B, m, L) target session semantic IDs
        Returns:
            ntp_loss: next-token prediction loss
            logits: (B, T_dec, vocab_size)
            aux_loss: MoE load balancing loss
        """
        # Build input sequences
        enc_input = self._make_encoder_input(history_ids)
        dec_input = self._make_decoder_input(session_ids)

        # Encode
        encoder_out = self.encode(enc_input)

        # Decode
        logits, aux_loss = self.decode(dec_input, encoder_out)

        # NTP loss (Eq. 4): predict next token from each position
        # Input:  [BOS, s1, s2, s3, BOS, s1, s2, s3, ...]
        # Target: [s1,  s2, s3, BOS, s1,  s2, s3, PAD, ...]  (shifted left by 1)
        target = torch.cat([
            dec_input[:, 1:],
            torch.full((dec_input.shape[0], 1), self.config.pad_token_id,
                       device=dec_input.device, dtype=torch.long)
        ], dim=1)

        ntp_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=self.config.pad_token_id
        )

        return ntp_loss, logits, aux_loss


    @torch.no_grad()
    def generate(self, history_ids: torch.Tensor,
                 beam_size: int = 128,
                 max_tokens: Optional[int] = None
                 ) -> torch.Tensor:
        """
        Beam search generation for inference / DPO candidate generation.

        Args:
            history_ids: (B, n, L) user history
            beam_size: number of candidate sessions (N=128 in paper)
        Returns:
            candidates: (B, beam_size, m, L) generated sessions
        """
        config = self.config
        B = history_ids.shape[0]
        if max_tokens is None:
            max_tokens = config.session_size * (config.num_codebook_levels + 1)

        enc_input = self._make_encoder_input(history_ids)
        encoder_out = self.encode(enc_input)

        device = history_ids.device
        # Simple greedy/beam search (simplified for clarity)
        # In practice, this would use KV-cache and proper beam management

        all_candidates = []
        for b in range(B):
            enc_b = encoder_out[b:b+1].expand(beam_size, -1, -1)

            # Start with BOS
            generated = torch.full(
                (beam_size, 1), config.bos_token_id,
                device=device, dtype=torch.long
            )
            beam_scores = torch.zeros(beam_size, device=device)

            for step in range(max_tokens - 1):
                logits, _ = self.decode(generated, enc_b)
                next_logits = logits[:, -1, :]  # (beam_size, vocab)
                log_probs = F.log_softmax(next_logits, dim=-1)

                if step == 0:
                    # First step: take top-k from single beam
                    topk_scores, topk_ids = log_probs[0].topk(beam_size)
                    beam_scores = topk_scores
                    generated = torch.cat([
                        generated, topk_ids.unsqueeze(1)
                    ], dim=1)
                else:
                    # Subsequent steps: keep top beam_size overall
                    vocab_size = log_probs.shape[-1]
                    candidate_scores = beam_scores.unsqueeze(1) + log_probs
                    candidate_scores_flat = candidate_scores.view(-1)

                    topk_scores, topk_flat_ids = candidate_scores_flat.topk(beam_size)
                    beam_idx = topk_flat_ids // vocab_size
                    token_idx = topk_flat_ids % vocab_size

                    beam_scores = topk_scores
                    generated = torch.cat([
                        generated[beam_idx], token_idx.unsqueeze(1)
                    ], dim=1)

            all_candidates.append(generated)

        return torch.stack(all_candidates, dim=0)  # (B, beam_size, T)


# ============================================================
# 6. Session-wise Reward Model (Eq. 5-7)
# ============================================================

class RewardModel(nn.Module):
    """
    Session-wise reward model R(u, S).
    Uses target-aware attention and multi-tower prediction.
    """

    def __init__(self, item_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.item_dim = item_dim

        # Target-aware attention: e_i = v_i ⊙ u
        self.target_attn = nn.MultiheadAttention(
            embed_dim=item_dim, num_heads=num_heads, batch_first=True
        )

        # Self-attention for intra-session interaction (Eq. 5)
        self.session_self_attn = nn.MultiheadAttention(
            embed_dim=item_dim, num_heads=num_heads, batch_first=True
        )

        # Multi-tower prediction heads (Eq. 6)
        self.tower_swt = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.tower_vtr = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.tower_wtr = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.tower_ltr = nn.Sequential(
            nn.Linear(item_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, user_repr: torch.Tensor, session_items: torch.Tensor
                ) -> dict:
        """
        Args:
            user_repr: (B, D) user behavior representation
            session_items: (B, m, D) item embeddings in the session
        Returns:
            rewards: dict with keys 'swt', 'vtr', 'wtr', 'ltr', each (B,)
            combined_reward: (B,) weighted sum
        """
        B, m, D = session_items.shape

        # Target-aware representation: e_i = v_i ⊙ u (via attention)
        user_expanded = user_repr.unsqueeze(1).expand_as(session_items)
        target_aware = session_items * user_expanded  # element-wise as ⊙

        # Self-attention among session items (Eq. 5)
        h_f, _ = self.session_self_attn(target_aware, target_aware, target_aware)

        # Sum pooling over session items
        h_pooled = h_f.sum(dim=1)  # (B, D)

        # Multi-tower predictions (Eq. 6)
        r_swt = self.tower_swt(h_pooled).squeeze(-1)
        r_vtr = self.tower_vtr(h_pooled).squeeze(-1)
        r_wtr = self.tower_wtr(h_pooled).squeeze(-1)
        r_ltr = self.tower_ltr(h_pooled).squeeze(-1)

        # Combined reward (simple weighted sum)
        combined = r_swt + r_vtr + r_wtr + r_ltr

        return {
            'swt': r_swt, 'vtr': r_vtr, 'wtr': r_wtr, 'ltr': r_ltr,
            'combined': combined
        }

    def compute_loss(self, rewards: dict, labels: dict) -> torch.Tensor:
        """Binary cross-entropy loss for RM training (Eq. 7)."""
        loss = 0.0
        for key in ['swt', 'vtr', 'wtr', 'ltr']:
            loss = loss + F.binary_cross_entropy(rewards[key], labels[key])
        return loss


# ============================================================
# 7. DPO / IPA Training (Eq. 10, Algorithm 2)
# ============================================================

class IPATrainer:
    """
    Iterative Preference Alignment trainer.
    Implements Algorithm 2 from the paper.
    """

    def __init__(self, model: OneRec, reward_model: RewardModel,
                 config: OneRecConfig, lr: float = 2e-4):
        self.model = model
        self.reward_model = reward_model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ref_model = None  # Snapshot of M_t for DPO

    def snapshot_reference(self):
        """Save current model as reference M_t for DPO."""
        import copy
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_dpo_loss(self, history_ids: torch.Tensor,
                         winner_ids: torch.Tensor,
                         loser_ids: torch.Tensor) -> torch.Tensor:
        """
        DPO loss (Eq. 10):
        L_DPO = -log σ(β * (log π(S_w|H) / π_ref(S_w|H) - log π(S_l|H) / π_ref(S_l|H)))
        """
        beta = self.config.dpo_beta

        # Log-probabilities under current model
        log_p_win = self._log_prob(self.model, history_ids, winner_ids)
        log_p_lose = self._log_prob(self.model, history_ids, loser_ids)

        # Log-probabilities under reference model
        with torch.no_grad():
            log_p_win_ref = self._log_prob(self.ref_model, history_ids, winner_ids)
            log_p_lose_ref = self._log_prob(self.ref_model, history_ids, loser_ids)

        # DPO objective
        log_ratio_win = log_p_win - log_p_win_ref
        log_ratio_lose = log_p_lose - log_p_lose_ref

        loss = -F.logsigmoid(beta * (log_ratio_win - log_ratio_lose)).mean()
        return loss

    def _log_prob(self, model: OneRec, history_ids: torch.Tensor,
                  session_ids: torch.Tensor) -> torch.Tensor:
        """Compute log P(session | history) for a model."""
        ntp_loss, logits, _ = model(history_ids, session_ids)
        # ntp_loss is the average NTP loss = -log P / num_tokens
        # Total log P = -ntp_loss * num_tokens
        m, L = session_ids.shape[1], session_ids.shape[2]
        num_tokens = m * (L + 1) - 1  # exclude last position (PAD target)
        return -ntp_loss * num_tokens

    def train_step(self, history_ids: torch.Tensor,
                   session_ids: torch.Tensor,
                   user_reprs: Optional[torch.Tensor] = None,
                   session_embeds: Optional[torch.Tensor] = None
                   ) -> dict:
        """
        Single training step implementing Algorithm 2.

        With probability r_DPO: generate candidates, select preference pairs, train with NTP + DPO.
        Otherwise: train with NTP only.
        """
        self.model.train()
        use_dpo = torch.rand(1).item() < self.config.dpo_sample_ratio

        if use_dpo and self.ref_model is not None and user_reprs is not None:
            # Generate N candidates via beam search
            self.model.eval()
            candidates = self.model.generate(
                history_ids, beam_size=self.config.num_beam_candidates
            )
            self.model.train()

            # Score with reward model (simplified — would need to decode
            # candidates back to embeddings in practice)
            # Here we assume session_embeds are available for the candidates
            B = history_ids.shape[0]
            rewards = []
            for i in range(self.config.num_beam_candidates):
                r = self.reward_model(user_reprs, session_embeds)['combined']
                rewards.append(r)
            rewards = torch.stack(rewards, dim=1)  # (B, N)

            # Select winner (max reward) and loser (min reward)
            winner_idx = rewards.argmax(dim=1)
            loser_idx = rewards.argmin(dim=1)

            # In practice, decode candidates back to (B, m, L) format
            # Here we use session_ids as a placeholder for winner
            winner_ids = session_ids  # placeholder
            loser_ids = session_ids   # placeholder

            # Combined loss: L = L_NTP + λ * L_DPO
            ntp_loss, logits, aux_loss = self.model(history_ids, session_ids)
            dpo_loss = self.compute_dpo_loss(history_ids, winner_ids, loser_ids)
            loss = ntp_loss + dpo_loss + 0.01 * aux_loss

            metrics = {'ntp_loss': ntp_loss.item(), 'dpo_loss': dpo_loss.item(),
                       'aux_loss': aux_loss.item(), 'total_loss': loss.item()}
        else:
            # NTP only
            ntp_loss, logits, aux_loss = self.model(history_ids, session_ids)
            loss = ntp_loss + 0.01 * aux_loss
            metrics = {'ntp_loss': ntp_loss.item(), 'aux_loss': aux_loss.item(),
                       'total_loss': loss.item()}

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return metrics


# ============================================================
# 8. Example Usage
# ============================================================

if __name__ == "__main__":
    # Use small config for demo
    config = OneRecConfig(
        codebook_size=256,       # small for demo (paper uses 8192)
        enc_num_layers=2,
        dec_num_layers=2,
        enc_hidden_dim=128,
        dec_hidden_dim=128,
        enc_num_heads=4,
        dec_num_heads=4,
        enc_ffn_dim=512,
        dec_ffn_dim=512,
        num_experts=8,           # small for demo (paper uses 24)
        num_active_experts=2,
        max_history_len=32,      # small for demo (paper uses 256)
        session_size=5,
    )

    model = OneRec(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"OneRec parameters: {param_count / 1e6:.1f}M")

    # Dummy data
    B = 4
    n = 32   # history length
    m = 5    # session size
    L = 3    # codebook levels

    history_ids = torch.randint(0, config.codebook_size, (B, n, L))
    session_ids = torch.randint(0, config.codebook_size, (B, m, L))

    # Forward pass
    ntp_loss, logits, aux_loss = model(history_ids, session_ids)
    print(f"NTP loss: {ntp_loss.item():.4f}")
    print(f"MoE aux loss: {aux_loss.item():.4f}")
    print(f"Decoder output shape: {logits.shape}")

    # Reward model
    reward_model = RewardModel(item_dim=128)
    user_repr = torch.randn(B, 128)
    session_embeds = torch.randn(B, m, 128)
    rewards = reward_model(user_repr, session_embeds)
    print(f"Reward scores: swt={rewards['swt'].mean():.4f}, "
          f"vtr={rewards['vtr'].mean():.4f}")

    # IPA training step
    trainer = IPATrainer(model, reward_model, config)
    trainer.snapshot_reference()
    metrics = trainer.train_step(history_ids, session_ids)
    print(f"Training metrics: {metrics}")
