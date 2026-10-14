# zmling

Learn [ZML](https://github.com/zml/zml) by fixing tiny broken programs.

zmling is to ZML what [Ziglings](https://codeberg.org/ziglings/exercises) is to Zig, and
[Rustlings](https://github.com/rust-lang/rustlings) to Rust: about a hundred and fifty small programs, each
one broken on purpose, each one teaching one thing. You read the comments, fix the program, and
move on to the next. The first ones need a word or two. Then the gaps grow: a function body, the
code that compiles and runs it, a model struct, and in the last chapters most of the program,
each exercise leaning on what you wrote in the earlier ones. By the end you can write, compile, load and run a neural network with ZML
without looking anything up.

You should already know Zig. You don't need to know machine learning: the exercises explain the
ideas they use.

## Getting started

You need:

- [Zig](https://ziglang.org/download/) **0.17.0**
- [Bazelisk](https://github.com/bazelbuild/bazelisk) installed as `bazel` (ZML is built with Bazel;
  Bazelisk downloads the right version)
- about 30 GB of free disk space (Bazel's build of ZML and its dependencies)
- Linux x86_64 is what zmling was tested on. ZML also supports macOS, so it should work there

```sh
git clone --recursive https://github.com/pierrelgol/zmling
cd zmling
zig build
```

(Forgot `--recursive`? `git submodule update --init`.)

The **first** `zig build` downloads and compiles ZML's dependencies: count ten minutes or more,
depending on your machine and connection. After that, each exercise takes around 15 seconds to
build. Every exercise binary links all of ZML (about 1 GB!), so the runner deletes it once it has
run.

`zig build` runs the exercises in order and stops at the first one that fails:

```
Compiling 01_shape_shifters/001_hello_shape.zig...
zmling/01_shape_shifters/001_hello_shape.zig:59:47: error: expected type expression, found ')'
    const sofa = zml.Shape.init(.{ 2, 3 }, ???);
                                              ^
Edit exercises/01_shape_shifters/001_hello_shape.zig and run 'zig build' again.
```

Open that file, read it from the top, fix it, run `zig build` again. That's the whole loop.

| Command | |
|---|---|
| `zig build` | continue from the first unsolved exercise |
| `zig build -Dn=42` | run only exercise 42 |
| `zig build -Ds=42` | start at exercise 42 |
| `zig build -Dreset` | forget your progress |
| `zig build -Dbazel=bazelisk` | use another Bazel executable |

An exercise passes when it compiles, runs, and prints exactly the expected output on stdout.
Everything ZML logs goes to stderr and is ignored, so feel free to add `std.debug.print` and
`std.log` calls while you investigate.

## The chapters

| | Chapter | You learn |
|---|---|---|
| 01 | Shape Shifters | `zml.Shape`, data types, axes, tags, reshaping |
| 02 | Ghost Tensors | tensors vs buffers, platforms, compiling and running a function |
| 03 | The Arithmetic Diner | element-wise math, activations, constants, dtype conversion |
| 04 | Luggage Tags | tagged axes, in depth |
| 05 | Broadcast News | broadcasting |
| 06 | The Reduction Kitchen | sum, mean, max, argmax, softmax |
| 07 | Origami Club | reshape, transpose, split and merge axes, padding |
| 08 | Slice of Life | slicing, concatenating, stacking, splitting |
| 09 | Dot Com Bubble | `dot`: matrix products by axis name |
| 10 | Treasure Maps | comparisons, select, gather, sort, top-k |
| 11 | Time Loops | control flow inside a compiled graph |
| 12 | The Exe Files | executables, multiple outputs, model structs and their buffers |
| 13 | Layer Cake | `zml.nn`: linear layers, normalization, embeddings, an MLP |
| 14 | Heavy Lifting | loading weights from safetensors files |
| 15 | Pay Attention! | attention, causal masks, rotary embeddings, KV caches |
| 16 | Casino Royale | random numbers, sampling, and a tiny language model |
| 17 | The Customs Office | the porting workflow on a real checkpoint (MNIST): inspect, declare, preprocess, verify against PyTorch |
| 18 | Lost in Translation | porting a HuggingFace encoder (bert-tiny): `config.json`, tokenizer, tied weights |
| 19 | Once Upon a Time | porting a decoder (TinyStories-1M, a GPT-Neo): read the author's code, then write a story |
| 20 | The Autopsy | debugging a port that runs and writes garbage: reference activations, layer by layer |
| 21 | The Graduation | a whole Llama 2 (stories260K) from its author's `model.py`: RoPE, grouped-query attention, KV cache |
| 22 | The Watchmaker | numerics: f32 / f16 / bf16, `bitCast`, stagnating sums, NaN and overflow, integer traps, `reducePrecision` |
| 23 | Hot Off the Press | the production generation loop: padded prefill and decode Exes, buffer donation, `FnExe` with baked weights, stop tokens, streaming decode, run-time sampling settings |
| 24 | Window Shopping | convolutions by hand, then `conv1d` / `conv2d` (stride, padding, dilation, groups, NCHW vs NHWC), pooling, resizing, a LeNet-style CNN |
| 25 | The Petting Zoo | blocks real ports need: pooling heads and cosine ranking, padding masks, cross-attention, positional encodings, mixture of experts, recurrence in the graph |
| 26 | The Mail Room | the plumbing before the first tensor: command-line flags, HuggingFace config quirks, sharded checkpoints, the virtual file system, Loader options |
| 27 | The Hall of Mirrors | under the Tensor API: `zml.meta.visit`, `mapAlloc`, `Bufferized`, `Tensor.id`, and how a struct of Tensors becomes an ordered list of Exe inputs |
| 28 | Footprints in the Snow | what a compile call does: the three times, the thread-local Compiler, Tensors as handles, constants vs arguments, blocks and captures, source locations, reading a failed compile |

## How it works

ZML can only be built with Bazel, so `zig build` is a thin front end. It builds
`runner/runner.zig`, which makes the `exercises/` directory visible to the ZML workspace (through
the symlink `zml/zmling`), builds one exercise with `bazel build //zmling/<chapter>:<number>`,
runs it, and compares its output with the expected one.

You can run an exercise by hand to see everything it prints, logs included:

```sh
cd zml && bazel run //zmling/02_ghost_tensors:010
```

The `zml` directory is the real ZML repository (a git submodule): when an exercise mentions a
function, go read it in `zml/zml/tensor.zig`. Reading ZML's source is a skill this course wants
you to pick up.

## Troubleshooting

- **`AccessDenied` on `/tmp/zig-cache`**: Bazel shares one Zig cache between users.
  `sudo chmod -R 777 /tmp/zig-cache /var/tmp/zig-cache`.
- **`unable to create the symlink zml/zmling`**: the submodule is missing, see above.
- **Walls of `warning: unknown argument` during the first build**: harmless, they come from
  ZML's C++ dependencies.
- **A ZML panic with a long stack trace**: that's often the exercise. The first line is the message;
  then look for the first frame that is in your file.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every exercise has a patch in `patches/patches/`;
`./patches/heal.sh` applies them all and checks that every exercise can be solved, and
`./patches/verify_broken.sh` checks that none of them passes unsolved.

## Credits

The runner is adapted from Ziglings (MIT). ZML is developed by [ZML](https://zml.ai).
