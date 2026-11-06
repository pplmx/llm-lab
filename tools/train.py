#!/usr/bin/env python3
"""Train the toy GPT that LLM Lab visualizes and export it as weights.json.

The model here is byte-level and deliberately tiny (2 layers, 96 wide, 4 heads)
so that the Go implementation can redo the whole forward pass on every
keystroke. The point is not the text quality but the arithmetic: every tensor
the app draws comes from this exact computation.

    python3 tools/train.py            # writes internal/model/weights.json
    python3 tools/train.py --steps 4000 --out /tmp/w.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- hyperparams

CONFIG = {"vocab": 256, "d_model": 96, "n_layer": 2, "n_head": 4, "d_ff": 384, "ctx": 64}

CORPUS_SENTENCES = [
    "A transformer reads a sequence of tokens and predicts the next one.",
    "Each layer has two blocks: attention, and a small feed forward network.",
    "Attention lets every token look at the tokens before it.",
    "The model turns each token into a vector with the embedding table.",
    "Position matters, so a positional vector is added to the token vector.",
    "Queries ask a question, keys answer it, and values carry the content.",
    "The dot product of a query and a key measures how much they match.",
    "A causal mask hides the future so the model cannot cheat.",
    "Softmax turns the scores of a row into a probability distribution.",
    "The attention output is a weighted sum of the value vectors.",
    "Each head learns a different pattern of attention.",
    "Residual connections carry the input around each block.",
    "Layer normalization keeps the numbers in a comfortable range.",
    "The feed forward network expands the vector, applies GELU, and shrinks it back.",
    "After the last layer a final normalization cleans up the residual stream.",
    "The unembedding matrix turns the final vector into logits over the vocabulary.",
    "Temperature sharpens or flattens the distribution before sampling.",
    "A high temperature makes the model more random.",
    "A low temperature makes the model more predictable.",
    "Training compares the predicted distribution with the true next token.",
    "Cross entropy measures how surprised the model was.",
    "Gradients flow backwards through every matrix in the network.",
    "The learning rate decides how far each step moves the weights.",
    "A small model can still learn the shape of the language.",
    "The cat sat on the mat.",
    "The dog ran across the yard.",
    "The sun rises in the east and sets in the west.",
    "A language model is a very large function of its input.",
    "Tokens are the atoms that the model reads and writes.",
    "Byte level tokenization means a character may take several tokens.",
    "The residual stream is the highway that carries information forward.",
    "Attention mixes information between positions, not within one position.",
    "The feed forward network works on each position on its own.",
    "Weights are shared across positions, so the model can grow in length.",
    "The context window limits how far back the model can look.",
    "Every row of the attention matrix sums to one.",
    "The diagonal of the attention matrix is often large at the start.",
    "Some heads focus on the previous token, others on the first token.",
    "The output of a head is projected back into the model width.",
    "Layer normalization subtracts the mean and divides by the standard deviation.",
    "The scale factor keeps the dot products from growing with the head size.",
    "GELU is a smooth version of the rectified linear unit.",
    "The model repeats the same block a few times, and that is enough.",
    "Parameters are learned, activations are computed.",
    "The forward pass is a straight line of matrix multiplications.",
    "The backward pass walks the same line in reverse.",
    "A bigger model means more layers, more width, and more heads.",
    "The vocabulary of a byte level model has two hundred and fifty six entries.",
    "A prompt is just a sequence of bytes until the model reads it.",
    "The last position predicts what comes next.",
    "Sampling picks one token from the distribution.",
    "Greedy decoding always picks the most likely token.",
    "Top k sampling keeps only the most likely few tokens.",
    "The model does not know the future, only the past.",
    "Softmax is invariant to adding the same constant to every score.",
    "The attention matrix is triangular because of the causal mask.",
    "Each token keeps its own residual stream as it moves through the layers.",
    "Learning means moving the weights a little on every batch.",
    "The loss goes down, and the text starts to look like the corpus.",
    "This model is small, so its sentences are short and a little strange.",
    "Everything you see in the window is computed from these numbers.",
]


def build_corpus(repeats: int = 40, seed: int = 0) -> str:
    rng = random.Random(seed)
    blocks = []
    for _ in range(repeats):
        order = CORPUS_SENTENCES[:]
        rng.shuffle(order)
        blocks.append(" ".join(order))
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------- model


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, cfg["d_ff"], bias=False)
        self.fc2 = nn.Linear(cfg["d_ff"], d, bias=False)

    def forward(self, x, n_head):
        b, t, d = x.shape
        dh = d // n_head
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=2)
        q = q.view(b, t, n_head, dh).transpose(1, 2)
        k = k.view(b, t, n_head, dh).transpose(1, 2)
        v = v.view(b, t, n_head, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(out)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg["vocab"], cfg["d_model"])
        self.pos = nn.Embedding(cfg["ctx"], cfg["d_model"])
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layer"]))
        self.lnf = nn.LayerNorm(cfg["d_model"], eps=1e-5)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, self.cfg["n_head"])
        x = self.lnf(x)
        logits = x @ self.tok.weight.T
        return logits


# ------------------------------------------------------------------ training


def train(args) -> TinyGPT:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg = CONFIG
    text = build_corpus(args.repeats)
    data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    print(f"corpus: {len(data):,} bytes, vocab 256")

    model = TinyGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    batch, block = args.batch, min(48, cfg["ctx"])
    steps = args.steps
    warmup = min(100, steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    start = time.time()
    model.train()
    last = float("nan")
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ix = torch.randint(0, len(data) - block - 1, (batch,))
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg["vocab"]), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  {time.time() - start:5.1f}s")
    model.final_loss = last
    return model


# ------------------------------------------------------------------- export


def flat(t: torch.Tensor) -> list[float]:
    return [round(float(v), 6) for v in t.detach().flatten().tolist()]


def export(model: TinyGPT, cfg, path: str, meta: dict):
    model.eval()
    prompt = "The transformer reads the tokens and"
    ids = list(prompt.encode("utf-8"))[: cfg["ctx"]]
    with torch.no_grad():
        logits = model(torch.tensor([ids], dtype=torch.long))[0, -1]

    out = {
        "config": cfg,
        "tok_embed": flat(model.tok.weight),
        "pos_embed": flat(model.pos.weight),
        "lnf_g": flat(model.lnf.weight),
        "lnf_b": flat(model.lnf.bias),
        "layers": [],
        "corpus": meta["corpus"],
        "steps": meta["steps"],
        "loss": meta["loss"],
        "reference": {
            "prompt": prompt,
            "ids": ids,
            "logits": [round(float(v), 5) for v in logits.tolist()],
        },
    }
    for blk in model.blocks:
        out["layers"].append(
            {
                "ln1_g": flat(blk.ln1.weight),
                "ln1_b": flat(blk.ln1.bias),
                "wq": flat(blk.qkv.weight[0 : cfg["d_model"], :].T),
                "wk": flat(blk.qkv.weight[cfg["d_model"] : 2 * cfg["d_model"], :].T),
                "wv": flat(blk.qkv.weight[2 * cfg["d_model"] :, :].T),
                "wo": flat(blk.proj.weight.T),
                "ln2_g": flat(blk.ln2.weight),
                "ln2_b": flat(blk.ln2.bias),
                "w1": flat(blk.fc1.weight.T),
                "w2": flat(blk.fc2.weight.T),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--out", default="internal/model/weights.json")
    args = p.parse_args()

    model = train(args)
    export(
        model,
        CONFIG,
        args.out,
        {"corpus": "toy English sentences about transformers", "steps": args.steps, "loss": model.final_loss},
    )


if __name__ == "__main__":
    main()
