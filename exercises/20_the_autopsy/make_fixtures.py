#!/usr/bin/env python3
"""Writes `probes.safetensors`: one extra reference tensor for chapter 20.

Run by authors only (the output is committed):

    uv run --with torch --with transformers --with safetensors --with numpy \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        exercises/20_the_autopsy/make_fixtures.py

Chapter 19's `activations.safetensors` has the input and the output of every
MODULE. The attention recipe itself (matmul, mask, softmax, matmul) is plain
function calls inside `GPTNeoSelfAttention._attn`: no module, no hook, no
tensor in the file. But one module sits in the middle of it:

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = self.attn_dropout(attn_weights)        # <- here
    attn_output = torch.matmul(attn_weights, value)

In eval mode a dropout does nothing, so what goes INTO it is the softmax
output, untouched: a free probe in the middle of the functional code.
"""
import os

import torch
from safetensors.torch import load_file, save_file
from transformers import GPTNeoConfig, GPTNeoForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
CH19 = os.path.join(HERE, "..", "19_once_upon_a_time")
PROBE = "transformer.h.0.attn.attention.attn_dropout"


def main():
    model = GPTNeoForCausalLM(GPTNeoConfig.from_pretrained(os.path.join(CH19, "config.json")))
    model.load_state_dict(load_file(os.path.join(CH19, "model.safetensors")), strict=False)
    model.tie_weights()
    model.eval()

    probes = {}

    def hook(_module, inputs, _output):
        # inputs[0] is {batch, heads, query, key}: drop the batch axis.
        probes[PROBE + ".in"] = inputs[0][0].detach().clone()

    model.get_submodule(PROBE).register_forward_hook(hook)

    acts = load_file(os.path.join(CH19, "activations.safetensors"))
    with torch.no_grad():
        logits = model(input_ids=acts["input_ids"].long()[None]).logits[0]
    # Same weights, same prompt, same answer as chapter 19's reference run.
    assert torch.allclose(logits[-1], acts["last_logits"], atol=1e-4)

    save_file(probes, os.path.join(HERE, "probes.safetensors"))
    for k, v in probes.items():
        print(f"  {k:50s} {tuple(v.shape)}")
        print("  rows sum to 1:", bool(torch.allclose(v.sum(-1), torch.ones(v.shape[:-1]))))
        print("  nothing above the diagonal:", bool((v.triu(1) == 0).all()))


if __name__ == "__main__":
    main()
