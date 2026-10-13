# zmling

Learn [ZML](https://github.com/zml/zml) by fixing tiny broken programs.

zmling is to ZML what [Ziglings](https://codeberg.org/ziglings/exercises) is to Zig, and
[Rustlings](https://github.com/rust-lang/rustlings) to Rust: a hundred-odd small programs, each
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
