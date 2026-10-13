#!/usr/bin/env python3
"""Writes house.safetensors, the weights of the finale of chapter 16 (exercise 114).

Usage:  python3 make_fixtures.py     (from any directory)

Standard library only. The format is explained in
../14_heavy_lifting/make_fixtures.py and in exercise 097.

The model is a "bigram" language model: it looks at ONE word and scores
every word of its vocabulary as the possible next one. Nobody trained
it: the weights below are computed by hand, so that it recites the
motto of the casino.

    token embedding     8 words x 3 numbers
    linear head         3 numbers -> 8 scores (no bias)

The trick: 8 words fit exactly in 3 bits. The embedding of word number
i is its binary code, written with -1 for 0 and +1 for 1:

    5 = 0b101  ->  [ +1, -1, +1 ]

Row n of the head is the code of the word that comes BEFORE word n in
the motto. The score of n is the dot product of the two codes: 3 when
all three bits agree (n is the right next word), at most 1 otherwise.
"""

import json
import os
import struct

VOCABULARY = [".", "again", "always", "house", "play", "please", "the", "wins"]
MOTTO = "the house always wins please play again ."


def code(i):
    return [1.0 if i & bit else -1.0 for bit in (4, 2, 1)]


def weights():
    ids = [VOCABULARY.index(word) for word in MOTTO.split()]
    assert sorted(ids) == list(range(8)), "every word exactly once"
    embed, head = [], [None] * 8
    for i in range(8):
        embed += code(i)
    # After the final "." the motto starts over: the house never stops.
    for previous, following in zip(ids, ids[1:] + ids[:1]):
        head[following] = code(previous)
    return embed, [x for row in head for x in row]


def safetensors(tensors):
    entries = ['"__metadata__":{"casino":"Royale"}']
    data = b""
    for name, shape, values in tensors:
        raw = struct.pack("<%df" % len(values), *values)
        entry = {"dtype": "F32", "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        entries.append("%s:%s" % (json.dumps(name), json.dumps(entry, separators=(",", ":"))))
        data += raw
    header = ("{\n" + ",\n".join(entries) + "\n}").encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + data


def recite(embed, head):
    """Self-check: run the model in plain Python, greedily."""
    token, words = VOCABULARY.index("the"), ["the"]
    while VOCABULARY[token] != ".":
        e = embed[3 * token : 3 * token + 3]
        scores = [sum(a * b for a, b in zip(e, head[3 * n : 3 * n + 3])) for n in range(8)]
        token = scores.index(max(scores))
        words.append(VOCABULARY[token])
    return " ".join(words)


if __name__ == "__main__":
    embed, head = weights()
    assert recite(embed, head) == MOTTO, recite(embed, head)
    blob = safetensors([("embed.weight", [8, 3], embed), ("head.weight", [8, 3], head)])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "house.safetensors")
    with open(path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes)" % (path, len(blob)))
