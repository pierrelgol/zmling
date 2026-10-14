#!/usr/bin/env python3
"""Writes parcels/, the model repositories of chapter 26.

Usage:  python3 make_fixtures.py     (from any directory)

Only the standard library is used. Each directory under parcels/ looks like
a HuggingFace repository of a (very) small model:

    floor14/    config.json, model.safetensors                    F32
    overseas/   config.json, generation_config.json,
                model.safetensors.index.json, 3 shards            F16
    annex/      config.json, model.safetensors.index.json,
                2 shards                                          BF16
    tube/       config.json, model.safetensors (not a router)     F32

The models are "routers": given the station a parcel is at, they answer the
next station. Every weight is a small integer, so that the routes are exact
on every machine. Names follow the `transformers` conventions:
`model.embed_tokens.weight`, `model.layers.N.fc.weight`, `lm_head.weight`.

The files are committed, so you only need to run this if you change them.
"""

import json
import os
import struct

STATIONS = ["dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement"]
N = len(STATIONS)


def bf16(value):
    # A bf16 is the upper half of an f32. Small integers lose nothing.
    raw = struct.pack("<f", value)
    assert raw[:2] == b"\0\0", value
    return raw[2:]


PACK = {
    "F32": lambda v: struct.pack("<f", v),
    "F16": lambda v: struct.pack("<e", v),
    "BF16": bf16,
}


def safetensors(dtype, tensors):
    """tensors: list of (name, matrix or vector)."""
    entries = ['"__metadata__":{"format":"pt"}']
    data = b""
    for name, values in tensors:
        if isinstance(values[0], list):
            shape = [len(values), len(values[0])]
            flat = [v for row in values for v in row]
        else:
            shape = [len(values)]
            flat = values
        raw = b"".join(PACK[dtype](v) for v in flat)
        entry = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        entries.append("%s:%s" % (json.dumps(name), json.dumps(entry, separators=(",", ":"))))
        data += raw
    header = ("{\n" + ",\n".join(entries) + "\n}").encode()
    header += b" " * (-len(header) % 8)
    json.loads(header)
    return struct.pack("<Q", len(header)) + header + data


def zeros():
    return [[0] * N for _ in range(N)]


def identity():
    m = zeros()
    for i in range(N):
        m[i][i] = 1
    return m


def transition(table):
    """weight[next][current] = 1: a Linear that moves a one-hot vector."""
    m = zeros()
    for current, nxt in enumerate(table):
        m[nxt][current] = 1
    return m


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(N)) for j in range(N)] for i in range(N)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def route(step, eos, max_length):
    tok, out = 0, [0]
    while tok not in eos and len(out) < max_length:
        tok = step(tok)
        out.append(tok)
    return " > ".join(STATIONS[t] for t in out)


def relu(v):
    return [max(0, x) for x in v]


def apply(m, v):
    return [sum(m[i][k] * v[k] for k in range(N)) for i in range(N)]


def argmax(v):
    return max(range(N), key=lambda i: (v[i], -i))


def onehot(i):
    return [1 if k == i else 0 for k in range(N)]


def write(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes)" % (os.path.relpath(path, HERE), len(blob)))


def write_json(path, value):
    write(path, (json.dumps(value, indent=2) + "\n").encode())


def write_repo(name, config, dtype, shards, extra=None):
    repo = os.path.join(HERE, "parcels", name)
    write_json(os.path.join(repo, "config.json"), config)
    for filename, value in (extra or {}).items():
        write_json(os.path.join(repo, filename), value)
    if len(shards) == 1:
        write(os.path.join(repo, "model.safetensors"), safetensors(dtype, shards[0]))
        return
    weight_map, total = {}, 0
    for i, tensors in enumerate(shards):
        filename = "model-%05d-of-%05d.safetensors" % (i + 1, len(shards))
        for tensor_name, values in tensors:
            weight_map[tensor_name] = filename
            flat = [v for row in values for v in row] if isinstance(values[0], list) else values
            total += len(flat) * len(PACK[dtype](0))
        write(os.path.join(repo, filename), safetensors(dtype, tensors))
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    write_json(os.path.join(repo, "model.safetensors.index.json"), index)


HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # --- floor14: a "chute". One table lookup, one Linear. ---------------
    table = [1, 2, 3, 5, 5, 5, 6, 7]
    head = transition(table)
    write_repo(
        "floor14",
        {
            "architectures": ["ChuteForRouting"],
            "model_type": "chute",
            "vocab_size": N,
            "hidden_size": N,
            "bos_token_id": 0,
            "eos_token_id": 5,
            "torch_dtype": "float32",
            "transformers_version": "4.44.0",
        },
        "F32",
        [[("model.embed_tokens.weight", identity()), ("lm_head.weight", head)]],
    )
    print("  " + route(lambda t: argmax(apply(head, onehot(t))), [5], 20))

    # --- overseas: a "conveyor", two layers, three shards, half floats. --
    # The last hop is decided by 2049 against 2048: two numbers an f32
    # tells apart and an f16 does not. The 1 of 2049 comes from the bias.
    table = [1, 2, 4, 7, 7, 5, 6, 7]
    cycle = transition([0] + [i % 7 + 1 for i in range(1, N)])
    w0 = matmul(transpose(cycle), transition(table))
    b0 = onehot(0)
    w1 = cycle
    head = zeros()
    for j in range(1, N):
        head[j][j] = 2048
    head[6][7] = 2048
    head[7][0] = 1
    write_repo(
        "overseas",
        {
            "_name_or_path": "/mnt/scratch/kenji/run_0412/final",
            "architectures": ["ConveyorForRouting"],
            "model_type": "conveyor",
            "vocab_size": N,
            "hidden_size": N,
            "num_hidden_layers": 2,
            "hidden_act": "relu",
            "initializer_range": 0.02,
            "bos_token_id": 0,
            "eos_token_id": [6, 7],
            "pad_token_id": None,
            "max_length": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.51.3",
            "use_cache": True,
        },
        "F16",
        [
            [("model.embed_tokens.weight", identity())],
            [
                ("model.layers.0.fc.weight", w0),
                ("model.layers.0.fc.bias", b0),
                ("model.layers.1.fc.weight", w1),
            ],
            [("lm_head.weight", head)],
        ],
        extra={
            "generation_config.json": {
                "_from_model_config": True,
                "bos_token_id": 0,
                "eos_token_id": [6, 7],
                "max_new_tokens": None,
                "top_k": None,
                "temperature": 0.7,
                "transformers_version": "4.51.3",
            }
        },
    )

    def overseas(t, with_bias=True, half=False):
        h = relu([a + (b if with_bias else 0) for a, b in zip(apply(w0, onehot(t)), b0)])
        logits = apply(head, relu(apply(w1, h)))
        if half:
            logits = [struct.unpack("<e", struct.pack("<e", x))[0] for x in logits]
        return argmax(logits)

    print("  " + route(overseas, [6, 7], 20))
    print("  without the bias:  " + route(lambda t: overseas(t, with_bias=False), [6, 7], 20))
    print("  computed in f16:   " + route(lambda t: overseas(t, half=True), [6, 7], 20))

    # --- annex: a conveyor again, bf16, no bias, no lm_head (tied). ------
    table = [1, 3, 3, 4, 3, 5, 6, 7]
    mirror = transition([0] + [N - i for i in range(1, N)])
    a0 = matmul(transpose(mirror), transition(table))
    a1 = mirror
    write_repo(
        "annex",
        {
            "architectures": ["ConveyorForRouting"],
            "model_type": "conveyor",
            "vocab_size": N,
            "hidden_size": N,
            "num_hidden_layers": 2,
            "bos_token_id": 0,
            "eos_token_id": 5,
            "max_length": 6,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.38.2",
        },
        "BF16",
        [
            [("model.embed_tokens.weight", identity()), ("model.layers.0.fc.weight", a0)],
            [("model.layers.1.fc.weight", a1)],
        ],
    )
    print("  " + route(lambda t: argmax(relu(apply(a1, relu(apply(a0, onehot(t)))))), [5], 6))

    # --- tube: an architecture the mail room has never heard of. ---------
    write_repo(
        "tube",
        {
            "architectures": ["PneumaticTubeForRouting"],
            "model_type": "pneumatic_tube",
            "vocab_size": N,
            "hidden_size": N,
            "bos_token_id": 0,
            "eos_token_id": 7,
            "torch_dtype": "float32",
            "transformers_version": "5.0.0",
        },
        "F32",
        [[("tube.pressure", [3, 1, 4, 1, 5, 9, 2, 6])]],
    )
