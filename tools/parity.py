#!/usr/bin/env python3
"""Dump every intermediate tensor of the toy GPT for cross-checking the Go port.

Both sides load the same weights.json, so any difference reported by
TestParityAllTensors is an implementation difference, not a rounding artifact.

    python3 tools/parity.py            # writes internal/model/testdata/parity.json
"""
from __future__ import annotations

import gzip
import json
import math
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "internal/model/weights.json")
OUT = os.path.join(ROOT, "internal/model/testdata/parity.json.gz")

PROMPT = "Attention mixes information between positions."


def rows(t):
    """[T, D] -> [[float, ...], ...] rounded like the exporter."""
    return [[round(float(v), 6) for v in r] for r in t.detach()]


def forward(w, ids):
    d = w["config"]["d_model"]
    n_head = w["config"]["n_head"]
    dh = d // n_head
    T = len(ids)

    tok = torch.tensor(w["tok_embed"], dtype=torch.float32).view(-1, d)
    pos = torch.tensor(w["pos_embed"], dtype=torch.float32).view(-1, d)
    idx = torch.tensor(ids, dtype=torch.long)

    out = {"prompt": PROMPT, "ids": ids, "heads_per_layer": []}
    emb = tok[idx]
    pe = pos[:T]
    x = emb + pe
    out["embed"] = rows(emb)
    out["pos"] = rows(pe)
    out["input"] = rows(x)

    for li, lw in enumerate(w["layers"]):
        def m(name, shape):
            return torch.tensor(lw[name], dtype=torch.float32).view(*shape)

        ln1g, ln1b = m("ln1_g", (d,)), m("ln1_b", (d,))
        wq, wk, wv = m("wq", (d, d)), m("wk", (d, d)), m("wv", (d, d))
        wo = m("wo", (d, d))
        ln2g, ln2b = m("ln2_g", (d,)), m("ln2_b", (d,))
        w1, w2 = m("w1", (d, w["config"]["d_ff"])), m("w2", (w["config"]["d_ff"], d))

        h = F.layer_norm(x, (d,), ln1g, ln1b, eps=1e-5)
        q, k, v = h @ wq, h @ wk, h @ wv
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        layer_heads = []
        concat = torch.zeros(T, d)
        for hi in range(n_head):
            sl = slice(hi * dh, (hi + 1) * dh)
            qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
            scores = (qh @ kh.T) / math.sqrt(dh)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            head_out = attn @ vh
            concat[:, sl] = head_out
            layer_heads.append({
                "q": rows(qh), "k": rows(kh), "v": rows(vh),
                "scores": [[None if v == float("-inf") else round(float(v), 6) for v in r] for r in scores],
                "attn": rows(attn), "out": rows(head_out),
            })
        attn_proj = concat @ wo
        x1 = x + attn_proj
        h2 = F.layer_norm(x1, (d,), ln2g, ln2b, eps=1e-5)
        pre = h2 @ w1
        hidden = F.gelu(pre)
        ffn = hidden @ w2
        x2 = x1 + ffn

        out["heads_per_layer"].append({
            "in": rows(x),
            "norm1": rows(h),
            "heads": layer_heads,
            "attn_proj": rows(attn_proj),
            "after_attn": rows(x1),
            "norm2": rows(h2),
            "hidden_pre": rows(pre),
            "hidden": rows(hidden),
            "ffn_out": rows(ffn),
            "after_ffn": rows(x2),
        })
        x = x2

    lnfg = torch.tensor(w["lnf_g"], dtype=torch.float32)
    lnfb = torch.tensor(w["lnf_b"], dtype=torch.float32)
    fin = F.layer_norm(x, (d,), lnfg, lnfb, eps=1e-5)
    logits = fin[-1] @ tok.T
    out["final_norm"] = rows(fin)
    out["logits"] = [round(float(v), 5) for v in logits]
    return out


def main():
    w = json.load(open(WEIGHTS, encoding="utf-8"))
    ids = list(PROMPT.encode("utf-8"))[: w["config"]["ctx"]]
    data = forward(w, ids)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Gzipped: the raw JSON is a couple of megabytes for a 46-token prompt.
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT}: {len(ids)} tokens, {len(data['heads_per_layer'])} layers")


if __name__ == "__main__":
    main()
