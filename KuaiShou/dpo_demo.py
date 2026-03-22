"""
DPO (Direct Preference Optimization) Demo

Shows the core idea: given (chosen, rejected) pairs, train the model to
prefer chosen over rejected — no RL needed, just a binary cross-entropy-like loss.

Three parts:
  1) Vanilla DPO on a toy example
  2) DPO applied to session generation (OneRec style)
  3) Iterative Preference Alignment (IPA): self-improvement loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

torch.manual_seed(42)
random.seed(42)


# ======================================================================
# Part 1: Vanilla DPO — the core math
# ======================================================================
def part1_vanilla_dpo():
    """
    Simplest possible DPO example.

    Setup: A model outputs logits over 10 items for a given user.
    We have preference pairs: (chosen_item, rejected_item).
    DPO adjusts the model so P(chosen) increases relative to P(rejected).

    DPO loss:
        L = -log σ( β * (log π(chosen) - log π_ref(chosen)
                       - log π(rejected) + log π_ref(rejected)) )

    Where:
        π     = current policy (model being trained)
        π_ref = reference policy (frozen copy from before DPO)
        β     = temperature (controls how far π can deviate from π_ref)
    """
    print("=" * 60)
    print("Part 1: Vanilla DPO — The Core Math")
    print("=" * 60)

    NUM_ITEMS = 10
    BETA = 0.1

    # Simple model: linear layer maps user embedding to item logits
    policy = nn.Linear(8, NUM_ITEMS)
    ref_policy = copy.deepcopy(policy)  # frozen reference
    ref_policy.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    # Fake user embedding
    user = torch.randn(1, 8)

    # Preference pairs: (chosen_item, rejected_item)
    # "User prefers item 3 over item 7, item 2 over item 5, etc."
    preferences = [(3, 7), (3, 5), (2, 7), (2, 5), (3, 0), (2, 0)]

    print(f"\n  Preference pairs: chosen > rejected")
    for c, r in preferences:
        print(f"    Item {c} > Item {r}")

    # Show initial probabilities
    with torch.no_grad():
        probs = F.softmax(policy(user), dim=-1).squeeze()
        print(f"\n  Before DPO — P(item):")
        for i in range(NUM_ITEMS):
            bar = "█" * int(probs[i] * 100)
            marker = " ← preferred" if i in [2, 3] else ""
            print(f"    Item {i}: {probs[i]:.3f} {bar}{marker}")

    # DPO training
    for epoch in range(200):
        total_loss = 0
        for chosen_id, rejected_id in preferences:
            # Get log-probs from current policy
            logits = policy(user)
            log_probs = F.log_softmax(logits, dim=-1).squeeze()
            log_pi_chosen = log_probs[chosen_id]
            log_pi_rejected = log_probs[rejected_id]

            # Get log-probs from reference (frozen)
            ref_logits = ref_policy(user)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1).squeeze()
            log_ref_chosen = ref_log_probs[chosen_id]
            log_ref_rejected = ref_log_probs[rejected_id]

            # DPO loss: -log σ(β * (Δ_chosen - Δ_rejected))
            # where Δ = log π - log π_ref
            reward_chosen = log_pi_chosen - log_ref_chosen
            reward_rejected = log_pi_rejected - log_ref_rejected
            loss = -F.logsigmoid(BETA * (reward_chosen - reward_rejected))

            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Show final probabilities
    with torch.no_grad():
        probs = F.softmax(policy(user), dim=-1).squeeze()
        print(f"\n  After DPO — P(item):")
        for i in range(NUM_ITEMS):
            bar = "█" * int(probs[i] * 100)
            marker = " ← preferred" if i in [2, 3] else ""
            print(f"    Item {i}: {probs[i]:.3f} {bar}{marker}")

    print("\n  → DPO shifted probability mass toward preferred items")
    print("    No RL (no reward signal, no policy gradient, no critic)")
    print("    Just a loss on preference pairs.\n")


# ======================================================================
# Part 2: DPO for Session Generation (OneRec Style)
# ======================================================================

# --- Tiny autoregressive session model ---
class TinySessionModel(nn.Module):
    """Minimal autoregressive model: generates a sequence of token IDs."""

    def __init__(self, vocab_size=20, embed_dim=32, max_len=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)
        self.attn = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=64, batch_first=True
        )
        self.head = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x):
        """x: (B, T) → logits: (B, T, vocab)"""
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.attn(h, src_mask=mask, is_causal=True)
        return self.head(h)

    def log_prob_of_sequence(self, sequence):
        """
        Compute log P(sequence) = sum of log P(token_t | token_<t).
        This is what DPO needs.
        """
        input_ids = sequence[:, :-1]   # teacher-forced input
        target_ids = sequence[:, 1:]   # what we predict
        logits = self.forward(input_ids)  # (B, T-1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log-prob of each target token
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        return token_log_probs.sum(dim=-1)  # (B,) total log-prob

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=12):
        """Greedy autoregressive generation."""
        tokens = prompt.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(tokens)
            # Sample (not greedy) for diversity
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


class RewardModel(nn.Module):
    """
    Simulates OneRec's reward model.
    Scores a session (sequence of tokens) → scalar reward.
    In OneRec this predicts watch time, like probability, etc.
    """

    def __init__(self, vocab_size=20, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        # Secret preference: reward sequences with more tokens in [5-9]
        # (simulates "user likes certain content categories")

    def forward(self, sequence):
        h = self.embed(sequence).mean(dim=1)  # pool
        return self.mlp(h).squeeze(-1)


def dpo_loss(policy, ref_policy, chosen_seq, rejected_seq, beta=0.1):
    """
    The DPO loss for sequences.

    L = -log σ( β * [ log π(chosen)/π_ref(chosen)
                     - log π(rejected)/π_ref(rejected) ] )
    """
    # Log-probs under current policy
    log_pi_chosen = policy.log_prob_of_sequence(chosen_seq)
    log_pi_rejected = policy.log_prob_of_sequence(rejected_seq)

    # Log-probs under frozen reference
    with torch.no_grad():
        log_ref_chosen = ref_policy.log_prob_of_sequence(chosen_seq)
        log_ref_rejected = ref_policy.log_prob_of_sequence(rejected_seq)

    # DPO: implicit reward difference
    chosen_reward = log_pi_chosen - log_ref_chosen
    rejected_reward = log_pi_rejected - log_ref_rejected

    loss = -F.logsigmoid(beta * (chosen_reward - rejected_reward))
    return loss.mean()


def part2_session_dpo():
    """
    DPO applied to autoregressive session generation.
    1. Pretrain a session model (NTP)
    2. Generate candidate sessions
    3. Score with reward model → pick chosen/rejected
    4. Train with DPO loss
    """
    print("=" * 60)
    print("Part 2: DPO for Session Generation")
    print("=" * 60)

    VOCAB = 20
    BOS = 0

    # --- Step 1: Pretrained session model (simulate with random init) ---
    policy = TinySessionModel(vocab_size=VOCAB)
    ref_policy = copy.deepcopy(policy)
    ref_policy.requires_grad_(False)

    reward_model = RewardModel(vocab_size=VOCAB)
    # Pretrain reward model to prefer sequences with tokens 5-9
    rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.01)
    for _ in range(500):
        # Good sessions: tokens from 5-9
        good = torch.randint(5, 10, (8, 10))
        # Bad sessions: tokens from 10-19
        bad = torch.randint(10, 20, (8, 10))
        good_r = reward_model(good)
        bad_r = reward_model(bad)
        rm_loss = F.relu(1.0 - good_r + bad_r).mean()  # margin loss
        rm_optimizer.zero_grad()
        rm_loss.backward()
        rm_optimizer.step()

    print("\n  Reward model trained (prefers tokens 5-9 over 10-19)")

    # --- Step 2-4: DPO training loop ---
    dpo_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    prompt = torch.tensor([[BOS]])

    for step in range(100):
        # Step 2: Generate N candidate sessions
        N = 8
        candidates = []
        rewards = []
        for _ in range(N):
            session = policy.generate(prompt, max_new_tokens=10)
            r = reward_model(session).item()
            candidates.append(session)
            rewards.append(r)

        # Step 3: Pick best (chosen) and worst (rejected)
        best_idx = max(range(N), key=lambda i: rewards[i])
        worst_idx = min(range(N), key=lambda i: rewards[i])
        chosen = candidates[best_idx]
        rejected = candidates[worst_idx]

        # Step 4: DPO update
        loss = dpo_loss(policy, ref_policy, chosen, rejected, beta=0.1)
        dpo_optimizer.zero_grad()
        loss.backward()
        dpo_optimizer.step()

        if (step + 1) % 25 == 0:
            avg_r = sum(rewards) / N
            print(f"  Step {step+1:3d} | DPO loss: {loss.item():.4f} | "
                  f"Avg reward: {avg_r:.3f} | "
                  f"Best: {rewards[best_idx]:.3f} | Worst: {rewards[worst_idx]:.3f}")

    # Show before/after
    print(f"\n  Sample generated sessions after DPO:")
    for i in range(3):
        session = policy.generate(prompt, max_new_tokens=10)
        r = reward_model(session).item()
        tokens = session[0, 1:].tolist()  # skip BOS
        print(f"    Session {i+1}: {tokens}  reward={r:.3f}")

    print("\n  → DPO pushes the model toward sessions the reward model likes")
    print("    No policy gradient. No critic. Just preference pairs.\n")


# ======================================================================
# Part 3: Iterative Preference Alignment (IPA)
# ======================================================================
def part3_iterative():
    """
    The "iterative" part of IPA:
    - After each DPO round, the improved model becomes the new reference
    - Generate new (harder) preference pairs from the improved model
    - Repeat

    Key insight: as the model improves, its bad outputs get closer to
    its good outputs → harder negatives → stronger learning signal.
    """
    print("=" * 60)
    print("Part 3: Iterative Preference Alignment (Self-Improvement)")
    print("=" * 60)

    VOCAB = 20
    BOS = 0
    NUM_ROUNDS = 4
    STEPS_PER_ROUND = 50

    policy = TinySessionModel(vocab_size=VOCAB)

    # Same reward model as Part 2
    reward_model = RewardModel(vocab_size=VOCAB)
    rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.01)
    for _ in range(500):
        good = torch.randint(5, 10, (8, 10))
        bad = torch.randint(10, 20, (8, 10))
        rm_loss = F.relu(1.0 - reward_model(good) + reward_model(bad)).mean()
        rm_optimizer.zero_grad()
        rm_loss.backward()
        rm_optimizer.step()

    prompt = torch.tensor([[BOS]])

    for round_num in range(NUM_ROUNDS):
        # Freeze current model as reference for this round
        ref_policy = copy.deepcopy(policy)
        ref_policy.requires_grad_(False)

        dpo_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        round_rewards = []
        round_margins = []

        for step in range(STEPS_PER_ROUND):
            # Generate candidates from CURRENT (improving) model
            N = 8
            candidates = []
            rewards = []
            for _ in range(N):
                session = policy.generate(prompt, max_new_tokens=10)
                r = reward_model(session).item()
                candidates.append(session)
                rewards.append(r)

            best_idx = max(range(N), key=lambda i: rewards[i])
            worst_idx = min(range(N), key=lambda i: rewards[i])

            margin = rewards[best_idx] - rewards[worst_idx]
            round_rewards.append(sum(rewards) / N)
            round_margins.append(margin)

            loss = dpo_loss(policy, ref_policy, candidates[best_idx],
                            candidates[worst_idx], beta=0.1)
            dpo_optimizer.zero_grad()
            loss.backward()
            dpo_optimizer.step()

        avg_reward = sum(round_rewards) / len(round_rewards)
        avg_margin = sum(round_margins) / len(round_margins)

        # Show sample
        session = policy.generate(prompt, max_new_tokens=10)
        tokens = session[0, 1:].tolist()
        r = reward_model(session).item()

        print(f"\n  Round {round_num + 1}:")
        print(f"    Avg reward:  {avg_reward:.3f}")
        print(f"    Avg margin (best-worst): {avg_margin:.3f}")
        print(f"    Sample session: {tokens}  reward={r:.3f}")

    print(f"""
  How IPA works across rounds:

    Round 1: policy_0 generates → score → DPO → policy_1
                                                   ↓
    Round 2: policy_1 generates → score → DPO → policy_2
                                                   ↓
    Round 3: policy_2 generates → score → DPO → policy_3
                                                   ↓
    ...

  Each round:
    - The reference model is the PREVIOUS round's output
    - Candidates come from the CURRENT (improving) model
    - As the model gets better, the gap between best/worst shrinks
      → harder negatives → the model keeps improving
    - Like self-play: competing against a stronger version of yourself
""")


if __name__ == "__main__":
    print()
    part1_vanilla_dpo()
    part2_session_dpo()
    part3_iterative()
