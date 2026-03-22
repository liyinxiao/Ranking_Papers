"""
OneRec Session-wise List Generation Demo

Compares:
  1) Point-wise: generates one video at a time (independent predictions)
  2) Session-wise: generates a full session as one sequence, learning inter-video dependencies

Each video is represented by L=3 levels of semantic ID tokens from Residual Quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ─── Config ───
NUM_VIDEOS = 200        # total video corpus size
CODEBOOK_SIZE = 16      # K: codes per RQ level
NUM_RQ_LEVELS = 3       # L: semantic ID depth
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
SEQ_LEN = 64            # max encoder input length
SESSION_SIZE = 6        # videos per session
BATCH_SIZE = 8
NUM_EPOCHS = 30

# Special tokens
PAD_TOKEN = 0
BOS_TOKEN = 1
SEP_TOKEN = 2
SPECIAL_TOKENS = 3  # offset for codebook tokens

DECODER_VOCAB = SPECIAL_TOKENS + CODEBOOK_SIZE * NUM_RQ_LEVELS
# Each RQ level gets its own token range: level l -> [SPECIAL + l*K, SPECIAL + (l+1)*K)


# ─── Semantic ID Assignment (simulated RQ) ───
def assign_semantic_ids(num_videos, codebook_size, num_levels):
    """Simulate Residual Quantization: assign L-level codes to each video."""
    ids = {}
    for v in range(num_videos):
        codes = [random.randint(0, codebook_size - 1) for _ in range(num_levels)]
        ids[v] = codes
    return ids

VIDEO_SEMANTIC_IDS = assign_semantic_ids(NUM_VIDEOS, CODEBOOK_SIZE, NUM_RQ_LEVELS)


def codes_to_tokens(codes):
    """Convert RQ codes [c0, c1, c2] to token IDs with level offsets."""
    return [SPECIAL_TOKENS + level * CODEBOOK_SIZE + c for level, c in enumerate(codes)]


def tokens_to_video(tokens):
    """Decode token IDs back to RQ codes, find nearest video."""
    codes = [tokens[l] - SPECIAL_TOKENS - l * CODEBOOK_SIZE for l in range(NUM_RQ_LEVELS)]
    # Find exact match or closest
    for vid, vcodes in VIDEO_SEMANTIC_IDS.items():
        if vcodes == codes:
            return vid
    return -1  # no exact match


# ─── Synthetic Training Data ───
def generate_user_history(length=20):
    """Simulate a user's watch history as a sequence of video IDs."""
    return [random.randint(0, NUM_VIDEOS - 1) for _ in range(length)]


def generate_high_quality_session(history):
    """
    Simulate a high-quality session: videos that are coherent with history.
    In real OneRec, these are sessions where user watched >=5 videos,
    spent enough time, and showed engagement (likes, shares).
    """
    # Simple heuristic: mix some from history + some related new ones
    session = []
    for _ in range(SESSION_SIZE):
        if random.random() < 0.3 and history:
            session.append(random.choice(history))
        else:
            session.append(random.randint(0, NUM_VIDEOS - 1))
    return session


def encode_history(history, max_len=SEQ_LEN):
    """Encode user history as flattened semantic ID tokens for the encoder."""
    tokens = []
    for vid in history[-max_len // NUM_RQ_LEVELS:]:  # truncate to fit
        tokens.extend(codes_to_tokens(VIDEO_SEMANTIC_IDS[vid]))
    # Pad
    while len(tokens) < max_len:
        tokens.append(PAD_TOKEN)
    return tokens[:max_len]


# ─── Point-wise target: predict one video's codes ───
def make_pointwise_target(session):
    """Target = just the first video's semantic ID tokens (next-item prediction)."""
    return codes_to_tokens(VIDEO_SEMANTIC_IDS[session[0]])


# ─── Session-wise target: full session as one flat sequence ───
def make_sessionwise_target(session):
    """
    Target = [BOS] v1_codes [SEP] v2_codes [SEP] ... vN_codes
    The model learns to generate the entire session autoregressively,
    capturing inter-video coherence and diversity.
    """
    tokens = [BOS_TOKEN]
    for i, vid in enumerate(session):
        tokens.extend(codes_to_tokens(VIDEO_SEMANTIC_IDS[vid]))
        if i < len(session) - 1:
            tokens.append(SEP_TOKEN)
    return tokens


# ─── Model Components ───
class UserEncoder(nn.Module):
    """Encodes user behavior sequence with self-attention."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(DECODER_VOCAB, EMBED_DIM, padding_idx=PAD_TOKEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

    def forward(self, src):
        # src: (B, seq_len) token IDs
        pad_mask = src == PAD_TOKEN  # (B, seq_len)
        x = self.embed(src)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x  # (B, seq_len, dim)


class PointwiseDecoder(nn.Module):
    """Predicts one video's L semantic ID tokens independently."""

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(EMBED_DIM, CODEBOOK_SIZE) for _ in range(NUM_RQ_LEVELS)
        ])

    def forward(self, memory):
        # memory: (B, seq_len, dim) -> pool -> predict each level
        pooled = memory.mean(dim=1)  # (B, dim)
        logits = [head(pooled) for head in self.heads]  # L x (B, K)
        return logits


class SessionwiseDecoder(nn.Module):
    """
    Autoregressive decoder that generates an entire session.
    Uses causal self-attention + cross-attention to encoder memory.
    """

    def __init__(self, max_len=SESSION_SIZE * (NUM_RQ_LEVELS + 1) + 1):
        super().__init__()
        self.embed = nn.Embedding(DECODER_VOCAB, EMBED_DIM)
        self.pos_embed = nn.Embedding(max_len, EMBED_DIM)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.output_proj = nn.Linear(EMBED_DIM, DECODER_VOCAB)
        self.max_len = max_len

    def forward(self, tgt, memory):
        # tgt: (B, T) decoder input tokens
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.embed(tgt) + self.pos_embed(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt.device)
        x = self.decoder(x, memory, tgt_mask=causal_mask, tgt_is_causal=True)
        return self.output_proj(x)  # (B, T, vocab)

    @torch.no_grad()
    def generate(self, memory, max_tokens=None):
        """Autoregressive generation with greedy decoding."""
        if max_tokens is None:
            max_tokens = self.max_len
        B = memory.shape[0]
        tokens = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=memory.device)
        for _ in range(max_tokens - 1):
            logits = self.forward(tokens, memory)  # (B, T, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


# ─── Training Loop Comparison ───
def train_pointwise():
    """Train point-wise model: predict next single video."""
    print("=" * 60)
    print("POINT-WISE: Predict one video at a time")
    print("=" * 60)

    encoder = UserEncoder()
    decoder = PointwiseDecoder()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for _ in range(BATCH_SIZE):
            history = generate_user_history()
            session = generate_high_quality_session(history)

            src = torch.tensor([encode_history(history)])
            target_codes = VIDEO_SEMANTIC_IDS[session[0]]

            memory = encoder(src)
            logits = decoder(memory)  # L x (1, K)

            loss = sum(
                F.cross_entropy(logits[l], torch.tensor([target_codes[l]]))
                for l in range(NUM_RQ_LEVELS)
            )
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss / BATCH_SIZE:.4f}")

    # Demo inference
    history = generate_user_history()
    src = torch.tensor([encode_history(history)])
    memory = encoder(src)
    logits = decoder(memory)

    print("\n  Inference: must run model 6 times for a session of 6 videos")
    predicted_codes = [logits[l].argmax(dim=-1).item() for l in range(NUM_RQ_LEVELS)]
    print(f"  Single prediction codes: {predicted_codes}")
    print("  -> Need hand-crafted rules for diversity/coherence across 6 slots\n")
    return encoder, decoder


def train_sessionwise():
    """Train session-wise model: generate entire session at once."""
    print("=" * 60)
    print("SESSION-WISE: Generate full session as one sequence")
    print("=" * 60)

    encoder = UserEncoder()
    decoder = SessionwiseDecoder()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for _ in range(BATCH_SIZE):
            history = generate_user_history()
            session = generate_high_quality_session(history)

            src = torch.tensor([encode_history(history)])
            target_tokens = make_sessionwise_target(session)
            tgt = torch.tensor([target_tokens[:-1]])   # decoder input (shifted right)
            label = torch.tensor([target_tokens[1:]])   # prediction target

            memory = encoder(src)
            logits = decoder(tgt, memory)  # (1, T, vocab)

            loss = F.cross_entropy(logits.view(-1, DECODER_VOCAB), label.view(-1))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss / BATCH_SIZE:.4f}")

    # Demo inference: generate full session in one pass
    history = generate_user_history()
    src = torch.tensor([encode_history(history)])
    memory = encoder(src)

    max_tokens = 1 + SESSION_SIZE * NUM_RQ_LEVELS + (SESSION_SIZE - 1)  # BOS + codes + SEPs
    generated = decoder.generate(memory, max_tokens=max_tokens)
    gen_tokens = generated[0].tolist()

    print(f"\n  Inference: single autoregressive pass generates the whole session")
    print(f"  Raw tokens: {gen_tokens}")

    # Parse generated session
    videos = []
    buf = []
    for t in gen_tokens[1:]:  # skip BOS
        if t == SEP_TOKEN:
            if len(buf) == NUM_RQ_LEVELS:
                vid = tokens_to_video(buf)
                videos.append((buf, vid))
            buf = []
        else:
            buf.append(t)
    if len(buf) == NUM_RQ_LEVELS:
        vid = tokens_to_video(buf)
        videos.append((buf, vid))

    print(f"  Generated session ({len(videos)} videos):")
    for i, (toks, vid) in enumerate(videos):
        codes = [toks[l] - SPECIAL_TOKENS - l * CODEBOOK_SIZE for l in range(NUM_RQ_LEVELS)]
        match = f"video {vid}" if vid >= 0 else "no exact match"
        print(f"    Slot {i+1}: codes={codes} -> {match}")
    print("  -> Model learned inter-video structure (coherence, diversity) from data\n")
    return encoder, decoder


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    train_pointwise()
    train_sessionwise()

    print("=" * 60)
    print("KEY DIFFERENCE:")
    print("  Point-wise  = 6 independent calls + hand-crafted assembly rules")
    print("  Session-wise = 1 autoregressive pass, model learns session structure")
    print("=" * 60)
