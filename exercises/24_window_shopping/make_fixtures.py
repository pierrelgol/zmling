#!/usr/bin/env python3
"""Writes inspector.safetensors, the weights of the quiz of chapter 24.

Usage:  python3 make_fixtures.py     (from any directory)

Only the standard library is used (the file format is explained in
exercises/14_heavy_lifting/make_fixtures.py). Nothing here was trained:
every weight is chosen by hand, so that every number the network prints
can be explained. The script also runs the network in plain Python on
the drawings of exercise 168 and prints the scores, as a reference.

The network, PyTorch names and layouts:

    conv1   Conv2d(1, 2, 3)    weight [2, 1, 3, 3]   bias [2]
    conv2   Conv2d(2, 2, 3)    weight [2, 2, 3, 3]   bias [2]
    fc      Linear(8, 5)       weight [5, 8]         bias [5]

    x [1, 14, 14] -> conv1, relu, max pool 2 -> [2, 6, 6]
                  -> conv2, relu, max pool 2 -> [2, 2, 2]
                  -> flatten (channel, row, column) -> [8] -> fc -> [5]

The file is committed, so you only need to run this if you change it.
"""

import json
import os
import struct

# conv1: two stroke detectors. On a stroke one pixel wide the matching
# detector answers 6, the other one 0, and next to the stroke it is negative.
VERTICAL = [[-1, 2, -1]] * 3
HORIZONTAL = [[-1] * 3, [2] * 3, [-1] * 3]
# conv2: "how much of that stroke is there in this neighbourhood", for
# each of the two kinds separately.
ONES = [[1] * 3] * 3
ZEROS = [[0] * 3] * 3

CONV1_W = [[VERTICAL], [HORIZONTAL]]
CONV1_B = [-2, -2]
CONV2_W = [[ONES, ZEROS], [ZEROS, ONES]]
CONV2_B = [-8, -8]

# fc: one template per letter. The 8 features are, in PyTorch's flatten
# order (channel, row, column):
#   vertical strokes in the top-left, top-right, bottom-left, bottom-right
#   quarter of the drawing, then the same four for horizontal strokes.
# A letter gets +1 for a stroke it expects, -1 for one it does not, and
# pays 2 per expected stroke through its bias (a stroke is worth about 4:
# without the bias, "O" would win every time, since it expects everything).
LETTERS = ["L", "7", "U", "C", "O"]
TEMPLATES = [
    [1, 0, 1, 0, 0, 0, 1, 1],  # L: left side, bottom
    [0, 1, 0, 1, 1, 1, 0, 0],  # 7: right side, top
    [1, 1, 1, 1, 0, 0, 1, 1],  # U: both sides, bottom
    [1, 0, 1, 0, 1, 1, 1, 1],  # C: left side, top and bottom
    [1, 1, 1, 1, 1, 1, 1, 1],  # O: everything
]
FC_W = [[1 if t else -1 for t in template] for template in TEMPLATES]
FC_B = [-2 * sum(template) for template in TEMPLATES]


def flatten(x):
    return [v for row in x for v in (flatten(row) if isinstance(row, list) else [row])]


TENSORS = [
    ("conv1.weight", [2, 1, 3, 3], flatten(CONV1_W)),
    ("conv1.bias", [2], CONV1_B),
    ("conv2.weight", [2, 2, 3, 3], flatten(CONV2_W)),
    ("conv2.bias", [2], CONV2_B),
    ("fc.weight", [5, 8], flatten(FC_W)),
    ("fc.bias", [5], FC_B),
]


def safetensors(tensors):
    entries = ['"__metadata__":{"street":"rue des Vitrines"}']
    data = b""
    for name, shape, values in tensors:
        count = 1
        for dim in shape:
            count *= dim
        assert count == len(values), name
        raw = struct.pack("<%df" % count, *values)
        entry = {"dtype": "F32", "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        entries.append("%s:%s" % (json.dumps(name), json.dumps(entry, separators=(",", ":"))))
        data += raw
    header = ("{\n" + ",\n".join(entries) + "\n}").encode()
    header += b" " * (-len(header) % 8)
    json.loads(header)  # self-check: still valid JSON
    return struct.pack("<Q", len(header)) + header + data


# The reference network, in plain Python.


def conv(x, w, b):
    rows, cols = len(x[0]) - 2, len(x[0][0]) - 2
    return [
        [
            [
                b[o] + sum(w[o][c][i][j] * x[c][r + i][q + j] for c in range(len(x)) for i in range(3) for j in range(3))
                for q in range(cols)
            ]
            for r in range(rows)
        ]
        for o in range(len(w))
    ]


def relu(x):
    return [[[max(0, v) for v in row] for row in m] for m in x]


def pool(x):
    return [
        [[max(m[2 * r][2 * c], m[2 * r][2 * c + 1], m[2 * r + 1][2 * c], m[2 * r + 1][2 * c + 1]) for c in range(len(m[0]) // 2)] for r in range(len(m) // 2)]
        for m in x
    ]


def features(drawing):
    x = [[[1 if ch == "#" else 0 for ch in row] for row in drawing]]
    x = pool(relu(conv(x, CONV1_W, CONV1_B)))
    return pool(relu(conv(x, CONV2_W, CONV2_B)))


def scores(flat):
    return [sum(w * v for w, v in zip(row, flat)) + b for row, b in zip(FC_W, FC_B)]


def best(values):
    return LETTERS[values.index(max(values))]


def drawings():
    """The `signs` of exercise 168, read from the exercise itself."""
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    with open(os.path.join(here, "168_quiz_the_sign_inspector.zig")) as f:
        for line in f:
            line = line.strip()
            if line.startswith('"') and line.rstrip(",").endswith('"') and len(line.rstrip(",")) == 16:
                rows.append(line.rstrip(",").strip('"'))
    assert len(rows) % 14 == 0, len(rows)
    return [rows[i : i + 14] for i in range(0, len(rows), 14)]


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    blob = safetensors(TENSORS)
    path = os.path.join(here, "inspector.safetensors")
    with open(path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes)" % (path, len(blob)))

    try:
        signs = drawings()
    except OSError:
        signs = []
    for i, sign in enumerate(signs):
        f = features(sign)
        chw = [f[c][r][q] for c in range(2) for r in range(2) for q in range(2)]
        hwc = [f[c][r][q] for r in range(2) for q in range(2) for c in range(2)]
        print("sign %d: features %s -> %s %s   (flattened row, column, channel: %s)" % (i, chw, scores(chw), best(scores(chw)), best(scores(hwc))))
