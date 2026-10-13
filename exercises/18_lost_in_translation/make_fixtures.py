#!/usr/bin/env python3
"""Writes the files of chapter 18 from the real `prajjwal1/bert-tiny` checkpoint.

Run by authors only (the outputs are committed):

    uv run --with torch --with transformers --with safetensors \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        exercises/18_lost_in_translation/make_fixtures.py

Outputs, next to this script:
  model.safetensors        the checkpoint, converted from pytorch_model.bin
  config.json              untouched
  tokenizer.json           built from vocab.txt (the repo has no tokenizer.json)
  activations.safetensors  what PyTorch computes for `SENTENCE`, layer by layer
"""
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForPreTraining

REPO = "prajjwal1/bert-tiny"
REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
SENTENCE = "the capital of france is [MASK]."
# Exercise 127: the reference answers are typed in the exercise itself.
QUIZ = [
    "He was born in New [MASK] City.",
    "He opened the [MASK] and went inside.",
    "The prime [MASK] of Japan.",
    "The interpreter speaks English and [MASK].",
]
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(name):
    return hf_hub_download(REPO, name, revision=REVISION)


def main():
    state = torch.load(fetch("pytorch_model.bin"), map_location="cpu", weights_only=True)

    # Not weights: a buffer (0, 1, 2, ...) and two tensors that are other
    # tensors under a second name. safetensors refuses shared storage, and the
    # port has to know about the tying anyway.
    assert torch.equal(state["cls.predictions.decoder.weight"], state["bert.embeddings.word_embeddings.weight"])
    assert torch.equal(state["cls.predictions.decoder.bias"], state["cls.predictions.bias"])
    dropped = ["bert.embeddings.position_ids", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    weights = {k: v.contiguous() for k, v in state.items() if k not in dropped}
    save_file(weights, os.path.join(HERE, "model.safetensors"))

    with open(fetch("config.json")) as src, open(os.path.join(HERE, "config.json"), "w") as dst:
        dst.write(src.read())

    tokenizer = BertWordPieceTokenizer(fetch("vocab.txt"), lowercase=True)
    tokenizer.save(os.path.join(HERE, "tokenizer.json"))

    # The reference run.
    model = BertForPreTraining(BertConfig.from_pretrained(fetch("config.json")))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, missing
    model.eval()

    acts = {}

    def hook(name):
        def fn(_module, inputs, output):
            if inputs and isinstance(inputs[0], torch.Tensor):
                acts[name + ".in"] = inputs[0][0].detach().clone()
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                acts[name + ".out"] = out[0].detach().clone()
        return fn

    for name, module in model.named_modules():
        if name and "dropout" not in name:
            module.register_forward_hook(hook(name))

    ids = torch.tensor(tokenizer.encode(SENTENCE).ids)
    with torch.no_grad():
        out = model(input_ids=ids[None])

    mask_pos = ids.tolist().index(tokenizer.token_to_id("[MASK]"))
    logits = out.prediction_logits[0, mask_pos]
    top = torch.topk(logits, 5)
    print("tokens:", [tokenizer.id_to_token(i) for i in ids.tolist()])
    print("ids:", ids.tolist())
    print("top 5 at [MASK]:", [tokenizer.id_to_token(i) for i in top.indices.tolist()], top.values.tolist())

    # The 30522-wide logits are kept for the [MASK] position only.
    keep = {k: v.to(torch.float32) for k, v in acts.items() if v.dtype == torch.float32 and v.shape[-1] != 30522}
    keep["input_ids"] = ids.to(torch.int32)
    keep["mask_logits"] = logits
    save_file(keep, os.path.join(HERE, "activations.safetensors"))
    for k, v in sorted(keep.items()):
        print(f"  {k:60s} {tuple(v.shape)}")

    # The quiz, AFTER the file is written: the hooks are still recording.
    # Nothing is stored, the top word of each sentence is printed.
    # The margin to the runner-up must be wide, so that no rounding on any
    # machine can change the winner.
    for sentence in QUIZ:
        quiz_ids = torch.tensor(tokenizer.encode(sentence).ids)
        with torch.no_grad():
            quiz_logits = model(input_ids=quiz_ids[None]).prediction_logits[0]
        best = torch.topk(quiz_logits[quiz_ids.tolist().index(tokenizer.token_to_id("[MASK]"))], 2)
        margin = float(best.values[0] - best.values[1])
        assert margin > 1.0, (sentence, margin)
        print(f"quiz: {sentence!r} -> {tokenizer.id_to_token(int(best.indices[0]))!r} (margin {margin:.2f})")


if __name__ == "__main__":
    main()
