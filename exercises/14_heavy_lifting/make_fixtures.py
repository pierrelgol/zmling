#!/usr/bin/env python3
"""Writes gym.safetensors, the weights file of chapter 14.

Usage:  python3 make_fixtures.py     (from any directory)

Only the standard library is used, to show that a .safetensors file is
nothing more than:

    8 bytes     N, the size of the header: unsigned 64-bit, little-endian
    N bytes     the header: JSON, name -> dtype, shape, data_offsets
    the rest    the numbers of every tensor, raw, back to back

`data_offsets` are [begin, end) byte positions counted from the first byte
after the header. Exercise 097 reads this file back by hand.

The file is committed, so you only need to run this if you change it.
"""

import json
import os
import struct

# safetensors dtype name -> struct format character
FORMATS = {"F32": "f", "I32": "i", "U8": "B"}

# name, dtype, shape, values in row-major order (last axis varies fastest).
TENSORS = [
    # The plates on the rack, in kg. Not part of any model.
    ("plates", "F32", [4], [20, 10, 5, 2.5]),
    # Combination of locker 3. It is always the same one.
    ("locker_combination", "I32", [3], [36, 24, 36]),
    # The Coach: a two-layer network that judges your lifts.
    #   inputs: squat, bench, deadlift (in multiples of your body weight)
    #   hidden: [squat + bench + deadlift,  bench - squat]
    #   output: one score per verdict
    ("coach.0.weight", "F32", [2, 3], [1, 1, 1, -1, 1, 0]),
    ("coach.1.weight", "F32", [3, 2], [0, 0, 1, 0, 0, 8]),
    ("coach.1.bias", "F32", [3], [4, 0, 0]),
    # The smoothie bar (exercise 102): three blenders in a row.
    ("bar.0.weight", "F32", [2, 2], [1, 1, 1, -1]),
    ("bar.0.bias", "F32", [2], [0, 0]),
    ("bar.1.weight", "F32", [2, 2], [2, 0, 0, 3]),
    ("bar.1.bias", "F32", [2], [-1, 0]),
    ("bar.2.weight", "F32", [2, 2], [1, 1, 0, 1]),
    ("bar.2.bias", "F32", [2], [0.5, 0]),
]


def safetensors(tensors):
    entries = ['"__metadata__":{"gym":"Iron Tensor"}']
    data = b""
    for name, dtype, shape, values in tensors:
        count = 1
        for dim in shape:
            count *= dim
        assert count == len(values), name
        raw = struct.pack("<%d%s" % (count, FORMATS[dtype]), *values)
        entry = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        entries.append("%s:%s" % (json.dumps(name), json.dumps(entry, separators=(",", ":"))))
        data += raw

    # One entry per line: JSON does not mind, and humans can read it.
    header = ("{\n" + ",\n".join(entries) + "\n}").encode()
    # Like the official library, pad the header with spaces so that the
    # data starts at a multiple of 8 bytes.
    header += b" " * (-len(header) % 8)
    json.loads(header)  # self-check: still valid JSON
    return struct.pack("<Q", len(header)) + header + data


if __name__ == "__main__":
    blob = safetensors(TENSORS)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym.safetensors")
    with open(path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes)" % (path, len(blob)))
