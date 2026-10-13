#!/usr/bin/env python3
"""Writes the files of chapter 21 from `karpathy/tinyllamas`, stories260K.

Run by authors only (the outputs are committed):

    uv run --with torch --with safetensors --with sentencepiece --with huggingface_hub --with numpy \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        exercises/21_the_graduation/make_fixtures.py

Outputs, next to this script:
  stories260K.safetensors  the checkpoint, converted from stories260K.pt
  config.json              the `model_args` found inside the .pt
  tok512.pb                tok512.model under the extension ZML wants
  activations.safetensors  what PyTorch computes for `PROMPT`, layer by layer

There is no `transformers` class for this model: the reference below is
`model.py` of https://github.com/karpathy/llama2.c, reduced to what
inference needs. It is checked against the text of the repo's readme.
"""
import json
import math
import os

import sentencepiece
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO = "karpathy/tinyllamas"
REVISION = "0bd21da7698eaf29a0d7de3992de8a46ef624add"
PROMPT = "Once upon a time"
README_STORY = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park."
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(name):
    return hf_hub_download(REPO, "stories260K/" + name, revision=REVISION)


def rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rope(x, positions):
    """x: (s, heads, hd). Pairs are NEIGHBOURS: (x0, x1), (x2, x3), ..."""
    hd = x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    angles = torch.outer(positions.float(), freqs)[:, None, :]
    re, im = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out_re = re * angles.cos() - im * angles.sin()
    out_im = re * angles.sin() + im * angles.cos()
    return torch.stack([out_re, out_im], dim=-1).flatten(-2)


def forward(w, args, ids, acts=None):
    """ids: (s,) -> logits (s, vocab). `acts`, when given, collects activations."""
    def keep(name, value):
        if acts is not None:
            acts[name] = value.clone()
        return value

    s = ids.shape[0]
    heads, kv_heads = args["n_heads"], args["n_kv_heads"]
    hd = args["dim"] // heads
    positions = torch.arange(s)
    mask = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)

    x = keep("tok_embeddings.out", w["tok_embeddings.weight"][ids])
    for i in range(args["n_layers"]):
        p = f"layers.{i}."
        keep(p + "in", x)
        h = keep(p + "attention_norm.out", rms_norm(x, w[p + "attention_norm.weight"]))
        q = F.linear(h, w[p + "attention.wq.weight"]).view(s, heads, hd)
        k = F.linear(h, w[p + "attention.wk.weight"]).view(s, kv_heads, hd)
        v = F.linear(h, w[p + "attention.wv.weight"]).view(s, kv_heads, hd)
        q, k = rope(q, positions), rope(k, positions)
        keep(p + "attention.q_rope", q)
        # Each key/value head serves `heads / kv_heads` query heads in a row.
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd) + mask
        attn = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(s, -1)
        x = x + keep(p + "attention.out", F.linear(attn, w[p + "attention.wo.weight"]))
        h = keep(p + "ffn_norm.out", rms_norm(x, w[p + "ffn_norm.weight"]))
        up = F.silu(F.linear(h, w[p + "feed_forward.w1.weight"])) * F.linear(h, w[p + "feed_forward.w3.weight"])
        x = x + keep(p + "feed_forward.out", F.linear(up, w[p + "feed_forward.w2.weight"]))
        keep(p + "out", x)
    x = keep("norm.out", rms_norm(x, w["norm.weight"]))
    return F.linear(x, w["output.weight"])


def main():
    checkpoint = torch.load(fetch("stories260K.pt"), map_location="cpu", weights_only=True)
    args = checkpoint["model_args"]
    state = checkpoint["model"]

    # `torch.compile` left its mark on every name: "_orig_mod.". It is kept, a
    # port has to deal with it. `output.weight` is `tok_embeddings.weight`
    # under a second name, and safetensors refuses shared storage.
    assert torch.equal(state["_orig_mod.output.weight"], state["_orig_mod.tok_embeddings.weight"])
    weights = {k: v.contiguous() for k, v in state.items() if k != "_orig_mod.output.weight"}
    save_file(weights, os.path.join(HERE, "stories260K.safetensors"))

    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(args, f, indent=2)
        f.write("\n")

    with open(fetch("tok512.model"), "rb") as src, open(os.path.join(HERE, "tok512.pb"), "wb") as dst:
        dst.write(src.read())

    # The reference run.
    w = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=fetch("tok512.model"))
    ids = torch.tensor([tokenizer.bos_id()] + tokenizer.encode(PROMPT))

    acts = {}
    with torch.no_grad():
        logits = forward(w, args, ids, acts)
        story = ids.tolist()
        for _ in range(120):
            story.append(int(forward(w, args, torch.tensor(story))[-1].argmax()))
    text = tokenizer.decode(story)
    assert text.startswith(README_STORY), text

    acts["input_ids"] = ids.to(torch.int32)
    acts["logits"] = logits
    save_file(acts, os.path.join(HERE, "activations.safetensors"))

    print("bos, eos:", tokenizer.bos_id(), tokenizer.eos_id())
    print("prompt ids:", ids.tolist())
    print("story ids:", story)
    print("story:", repr(text))
    for k, v in sorted(acts.items()):
        print(f"  {k:40s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
