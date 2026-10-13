"""Reference values for chapter 17 (author-only: learners never run this).

Recomputes the forward pass of the MNIST checkpoint in PyTorch and writes
`activations.safetensors` next to this file. The exercises compare their ZML
port against it.

    uv run --with torch --with safetensors --with numpy \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        python make_fixtures.py <mnist_dir> <t10k-labels-idx1-ubyte.gz>

<mnist_dir> holds `mnist.safetensors` and `t10k-images.idx3-ubyte`: it is the
Bazel repository `@mnist` of the ZML workspace (`bazel info output_base`, then
`external/+non_module_deps+mnist`). The labels are not in `@mnist`: get
`t10k-labels-idx1-ubyte.gz` from a public mirror, for instance
https://storage.googleapis.com/cvdf-datasets/mnist/ (the md5 is checked below).

What was found out while writing this (see `Net` and `preprocess`):

  - nobody ships the training script with these weights. `Net` is rebuilt from
    the four names of the checkpoint, and `load_state_dict(strict=True)` proves
    that names and shapes fit;
  - the pixels must be divided by 255 (torchvision's `ToTensor()`), with no
    mean/std normalization: cross-entropy on the 10000 test images is 0.077
    that way, 17.8 on raw 0..255 pixels, and accuracy drops from 98.31% to
    90.6% with the usual `Normalize((0.1307,), (0.3081,))`;
  - there is no relu after `fc2`. `zml/examples/mnist/mnist.zig` has one, and
    feeds raw pixels: it still finds the right digit almost every time
    (98.26%), which is exactly why exercise 118 exists.
"""

import gzip
import hashlib
import pathlib
import sys

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

N_LOGITS = 200  # images whose logits are kept
N_HIDDEN = 8  # images whose hidden layer is kept (500 floats each)
LABELS_MD5 = "ec29112dd5afa0611ce80d1b7f02629c"


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def hidden(self, x):
        return torch.relu(self.fc1(x.flatten(1)))

    def forward(self, x):
        return self.fc2(self.hidden(x))


def preprocess(images_u8):
    """What torchvision's `ToTensor()` does: u8 0..255 -> f32 0..1."""
    return torch.from_numpy(images_u8.astype(np.float32)) / 255


def main():
    mnist_dir, labels_gz = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    assert hashlib.md5(labels_gz.read_bytes()).hexdigest() == LABELS_MD5, "not the MNIST test labels"

    raw = (mnist_dir / "t10k-images.idx3-ubyte").read_bytes()
    magic, count, rows, cols = np.frombuffer(raw, dtype=">u4", count=4)
    assert (magic, count, rows, cols) == (2051, 10000, 28, 28)
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, rows, cols)
    labels = np.frombuffer(gzip.decompress(labels_gz.read_bytes()), dtype=np.uint8, offset=8)
    assert labels.shape == (count,)

    net = Net()
    net.load_state_dict(load_file(mnist_dir / "mnist.safetensors"), strict=True)
    net.eval()

    with torch.no_grad():
        x = preprocess(images)
        hidden = net.hidden(x)
        logits = net(x)
        raw_logits = net(x * 255)  # the port that forgot the division
        relu_logits = torch.relu(logits)  # the port with one relu too many
    digits = logits.argmax(1)

    # The exercises print digits and counts: none of them may depend on the
    # last bits of a float, so the winner must win by a clear margin.
    top2 = logits.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    assert margin[:1000].min() > 0.05, margin[:1000].min()

    labels_t = torch.from_numpy(labels.astype(np.int64))
    report = lambda name, d: print(f"{name:28} right {int((d == labels_t).sum())}/10000, "
                                   f"first 200: {int((d[:200] == labels_t[:200]).sum())}, first 1000: {int((d[:1000] == labels_t[:1000]).sum())}")
    report("reference (x/255, no relu)", digits)
    report("raw pixels", raw_logits.argmax(1))
    report("relu after fc2", relu_logits.argmax(1))
    print("cross-entropy", float(torch.nn.functional.cross_entropy(logits, labels_t)),
          "raw:", float(torch.nn.functional.cross_entropy(raw_logits, labels_t)))
    print("raw differs at", (raw_logits.argmax(1) != digits).nonzero().flatten()[:8].tolist())
    print("relu differs at", (relu_logits.argmax(1) != digits).nonzero().flatten()[:8].tolist())
    print("wrong in first 1000:", (digits[:1000] != labels_t[:1000]).nonzero().flatten().tolist())
    print("min margin first 1000:", float(margin[:1000].min()))
    # Exercise 116 runs an all-zero image, exercise 117 a handful of parcels.
    with torch.no_grad():
        print("blank image reads", int(net(torch.zeros(1, 28, 28)).argmax()))
    pulled = [0, 1, 2, 3, 4, 115, 674, 1522]
    print("parcels", pulled, "read", digits[pulled].tolist(), "labels", labels[pulled].tolist())

    out = pathlib.Path(__file__).parent / "activations.safetensors"
    save_file(
        {
            "hidden": hidden[:N_HIDDEN].numpy(),  # {8, 500} f32, after the relu
            "logits": logits[:N_LOGITS].numpy(),  # {200, 10} f32
            "digits": digits.numpy().astype(np.uint8),  # {10000} u8, PyTorch's answers
            "labels": labels.copy(),  # {10000} u8, the truth
        },
        out,
    )
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
