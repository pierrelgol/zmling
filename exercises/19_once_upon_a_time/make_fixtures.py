#!/usr/bin/env python3
"""Writes the files of chapters 19 and 20 from `roneneldan/TinyStories-1M`.

Run by authors only (the outputs are committed):

    uv run --with torch --with transformers --with safetensors --with numpy \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        exercises/19_once_upon_a_time/make_fixtures.py

Outputs, next to this script:
  model.safetensors        the checkpoint, converted from pytorch_model.bin
  config.json              untouched
  tokenizer.json           untouched
  activations.safetensors  what PyTorch computes for `PROMPT`, layer by layer
"""
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import GPTNeoConfig, GPTNeoForCausalLM

REPO = "roneneldan/TinyStories-1M"
REVISION = "77f1b168e219585646439073245fe87e56b3023e"
PROMPT = "Once upon a time, there was a little dragon"
NEW_TOKENS = 40
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(name):
    return hf_hub_download(REPO, name, revision=REVISION)


def main():
    state = torch.load(fetch("pytorch_model.bin"), map_location="cpu", weights_only=True)

    # 34 of the 48 MB are not weights: every layer saved its causal mask, a
    # 2048 x 2048 table of booleans (`attn.attention.bias`), and the value it
    # writes where the mask says no (`masked_bias`).
    weights = {k: v.contiguous() for k, v in state.items() if not k.endswith(("attention.bias", "masked_bias"))}
    save_file(weights, os.path.join(HERE, "model.safetensors"))

    for name in ("config.json", "tokenizer.json"):
        with open(fetch(name), "rb") as src, open(os.path.join(HERE, name), "wb") as dst:
            dst.write(src.read())

    # The reference run. There is no `lm_head.weight` in the file: the model
    # ties it to `transformer.wte.weight` when it is built.
    model = GPTNeoForCausalLM(GPTNeoConfig.from_pretrained(fetch("config.json")))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert [k for k in missing if not k.endswith(("attention.bias", "masked_bias"))] == ["lm_head.weight"], missing
    assert not unexpected, unexpected
    model.tie_weights()
    model.eval()

    acts = {}

    def hook(name):
        def fn(_module, inputs, output):
            if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].is_floating_point():
                acts[name + ".in"] = inputs[0][0].detach().clone()
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                acts[name + ".out"] = out[0].detach().clone()
        return fn

    for name, module in model.named_modules():
        if name and "dropout" not in name:
            module.register_forward_hook(hook(name))

    tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
    ids = torch.tensor(tokenizer.encode(PROMPT).ids)
    with torch.no_grad():
        logits = model(input_ids=ids[None]).logits[0]

    # The 50257-wide logits are kept for the last position only.
    keep = {k: v for k, v in acts.items() if v.shape[-1] != 50257}
    keep["input_ids"] = ids.to(torch.int32)
    keep["last_logits"] = logits[-1]
    save_file(keep, os.path.join(HERE, "activations.safetensors"))

    # Greedy generation, the slow way: the whole sequence again for each token.
    story = ids.tolist()
    with torch.no_grad():
        for _ in range(NEW_TOKENS):
            story.append(int(model(input_ids=torch.tensor([story])).logits[0, -1].argmax()))

    print("prompt ids:", ids.tolist())
    print("next token:", int(logits[-1].argmax()), repr(tokenizer.decode([int(logits[-1].argmax())])))
    print("story ids:", story)
    print("story:", repr(tokenizer.decode(story)))
    for k, v in sorted(keep.items()):
        print(f"  {k:50s} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
