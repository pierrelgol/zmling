# Writing zmling exercises

This is the authoring guide. Read `exercises/01_shape_shifters/` and
`exercises/02_ghost_tensors/` first: they are the reference for tone, length and layout.

## How it fits together

- `zml/` is the ZML repository as a git submodule. It is a Bazel workspace and is **never modified**.
- The runner keeps a symlink `zml/zmling -> ../exercises`, so each chapter directory is a Bazel
  package of the ZML workspace. `exercises/harness/defs.bzl` declares one binary per exercise:
  `exercises/03_foo/017_bar.zig` is the target `//zmling/03_foo:017` (number only, see the
  comment in `defs.bzl` for why).
- `runner/runner.zig` holds the ordered exercise table: file, expected **stdout**, optional hint.
  `zig build` builds it and runs exercises in order until one fails.
- `answers/` (git-ignored) mirrors `exercises/` with the solved files. `zml/zmling_answers ->
  ../answers` makes them buildable as `//zmling_answers/<chapter>:NNN`.
- `patches/patches/NNN_name.patch` turns the exercise into the answer.

## Workflow for one exercise

1. Write the **solved** program in `answers/<chapter>/NNN_name.zig`, prose included.
   Each chapter dir (in `answers/` and in `exercises/`) needs this `BUILD.bazel`:

   ```starlark
   load("//zmling/harness:defs.bzl", "zmling_exercises")

   zmling_exercises()
   ```

2. Run it until the output is right (from `zml/`):

   ```sh
   bazel run --disk_cache= --noshow_progress --ui_event_filters=-info,-stdout,-stderr //zmling_answers/<chapter>:NNN 2>&1 | grep -v '^warning'
   ```

   Each linked binary weighs about 1 GB: `--disk_cache=` keeps it out of your Bazel disk cache,
   and you should delete `zml/bazel-bin/zmling_answers/<chapter>/NNN` when you are done with it.
   A build takes ~15 s. Bazel runs one command at a time per workspace, so build several targets
   in one command when you can (`bazel build //zmling_answers/<chapter>:all`).

3. Copy it to `exercises/<chapter>/NNN_name.zig` and break it. Check that it now fails
   (`bazel run //zmling/<chapter>:NNN`): compile error, panic, or different stdout.
4. `./patches/mkpatch.sh NNN` (from the project root) writes the patch.
5. Add the entry to the table in `runner/runner.zig`, with the exact stdout of the answer.
6. `zig fmt` both files. `./patches/heal.sh` and `./patches/verify_broken.sh` must pass.

## Rules for exercises

- One self-contained file. Imports: `std` and `zml` only. `pub fn main(init: std.process.Init) !void`.
- Results go to **stdout** through the three-line `stdout_writer` boilerplate, ending with
  `try stdout.flush();`. Logs go to stderr and are ignored.
- Output must be deterministic and stable: no pointers, no timings, floats printed with an explicit
  precision (`{d:.3}`) unless they are exact small integers/halves. No trailing whitespace.
- Use `.cpu`: `const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});`
- Free what you allocate (`defer x.deinit()`, `defer slice.free(gpa)`), like the reference chapters.
- Every exercise starts with a `//` comment block (the lesson), opening and closing with a bare
  `//` line, code samples indented 5 spaces after `//`, lines wrapped at ~66 columns. It explains
  the concept and *why* it exists, lists the relevant API with real signatures/semantics, and then
  sets the task inside the chapter's story. 40 to 120 comment lines is normal. Depth over brevity.
- Progressive disclosure: only use what earlier exercises taught, plus this exercise's one new
  idea. When you need something not yet taught, explain it in a sentence where it is used.
- Ways to break an exercise, from gentle to hard: a `???` placeholder; a wrong argument or value;
  a missing line; code that panics inside ZML (the learner must read ZML's message); a function
  body to write; a whole program to write. Vary them, and follow the difficulty curve below.
- Difficulty curve. The amount the learner writes grows with the chapter number, and inside a
  chapter with the exercise number. An exercise is never "one fix in one place" after chapter 02:
  - Ch 01-02: two or three broken spots. Quizzes: a few lines written by hand.
  - Ch 03-08: two to four broken spots, of different kinds. At least two exercises per chapter ask
    for a whole function body, and the quiz asks for one or more whole helpers.
  - Ch 09-12: writing function bodies is the normal task. The plumbing taught in chapter 02
    (placeholders, compile, Buffers, eval, read back) is progressively left to the learner:
    by chapter 12 they write it from memory. Quizzes are mostly blank below the lesson.
  - Ch 13-16: the learner writes struct fields, `init` and `forward` methods and most of `main`.
    The last quizzes are close to a whole program written by hand.
  - Ch 17-21 (porting real models): the learner ports a published checkpoint. They write the
    `Config`, the structs, `init`, `forward` and most of `main`; the broken spot is often a
    *silent* one (the program runs and prints the wrong thing). Every answer is checked against
    the PyTorch reference, and the last quiz is a whole port.
- Build on earlier work. From chapter 03 on, at least a third of the exercises must need
  something from an *earlier chapter* that this exercise does not re-explain (say where it was
  taught: "you wrote this in exercise 44, bring it back"). Longer exercises have several parts,
  each using the previous one.
- What stays given: the lesson prose, the data (Buffer contents, expected numbers), and the
  `stdout.print` format strings, so that the expected output is unambiguous.
- A blanked function body is `return ???;` (a parse error), never an empty body or `_ = x;`
  stubs: Zig's "unused parameter" error must not get there before the lesson does.
- Comments next to a broken spot say what is wanted, not how to write it.
- The last exercise of a chapter is a quiz named `NNN_quiz_*.zig`: it combines the chapter with
  much less hand-holding.
- A broken exercise must fail **for the reason the prose says**. If the text promises a ZML panic,
  check that the learner really gets that panic (Zig's "unused constant" error likes to get there
  first).
- Hints (in the runner table) are optional, short, and never give the answer.
- Humor: each chapter has its own little world and recurring characters. Keep it light, keep it
  kind, and never let a joke get in the way of the explanation.

## ZML facts that were verified by running code (ZML at the pinned submodule commit, Zig 0.17)

The docs under `zml/docs` are out of date. **The source in `zml/zml/*.zig` is the reference**:
read the function you are about to teach, and its `test` blocks, before writing about it.
Never write an API name from memory.

- These do **not** exist: `Tensor.matmul`, `Tensor.ones`, `Tensor.unsqueeze`, `zml.Context`,
  `zml.HostBuffer`, `zml.aio`, `nn.Embedding`, `testing.compileAndCall`. Do not use
  `testing.testLayer`, `testing.compareFloats`, `Platform.format`, `Tensor.linspace`,
  or `zml.testing.env()` (test-only).
- Canonical pipeline (see `exercises/02_ghost_tensors/`):
  `platform.compileFn(gpa, io, func, .{placeholders...}, .{})` -> `zml.Exe`;
  `zml.Buffer.fromBytes(io, platform, shape, bytes)` / `.fromSlice(io, platform, slice)` /
  `.scalar(io, platform, value, dtype)`; `exe.eval(gpa, io, .{buffers...})` (single output only);
  `buffer.getValue(T, io)` / `buffer.toSliceAlloc(gpa, io)`.
  Models with a `forward` method: `platform.compile(gpa, io, model, .forward, .{args...}, .{})`,
  see `zml/examples/mnist/mnist.zig`. Several outputs: `exe.runner(gpa)` + `runner.run(...)`, or
  `exe.args/results/call`, see `zml/zml/ops.zig` (`test @"while"`) and `zml/zml/exe.zig`.
- A Tensor operation outside a compile call panics with `attempt to use null value` in
  `Compiler.current`. This includes `Tensor.scalar`, `Tensor.constant`, `Tensor.arange`, ...
- Passing the same placeholder twice to a compile call panics:
  `Tensor with id ... has already been used once as an argument`.
- A Buffer of the wrong shape panics at run time:
  `Expected argument 1 to have shape {hours=4,i32}, got {hours=5,i32}`.
- Shape/dtype mismatches in ops are panics during tracing, not Zig errors:
  `mul expects tensor shapes to match, got {knocks=3,f32} and {1,f32}`.
- No dtype promotion. Rank-0 tensors broadcast automatically; otherwise only size-1 axes of the
  *right* operand are stretched, and only when both sides have the same tags in the same order
  (see `binaryOpHelper`). Element-wise ops do not compare tags when the sizes match.
- `Tensor.any` and `Tensor.all` do not compile at the pinned commit.
- Shapes print as `{2,3,f32}` / `{h=2,w=3,i32}`, tensors as `Tensor({2,3,f32})`.
  `{any}` on an array prints `{ 1, 2, 3 }`. Don't print a `zml.Slice` with `{f}` (it has a pointer).
- `Slice.items(T)` does not check `T` against the dtype.
- `init.gpa` is a leak-checking allocator in debug builds: deinit everything.

## Porting chapters (17-21): assets and verified facts

- Only tiny, real, ungated checkpoints, committed in the chapter directory (MNIST comes from
  ZML's own `@mnist` Bazel repo). No downloads at run time, no Python for learners.
- Each chapter has a `make_fixtures.py`, run by authors with `uv run --with torch ...` (see its
  docstring). It pins the upstream revision, converts the checkpoint to safetensors, and dumps
  the reference activations as `{module}.in` / `{module}.out`. Keep fixtures small (KB).
- Print only what cannot drift between machines: text, token ids, top-1 words, booleans such as
  `matches PyTorch: true`. Never raw activations or logits.
- `zml.testing.expectClose(io, want, got, .{})` works from `main`. On mismatch it prints a report
  on stderr and returns `error.TestUnexpectedResult`.
- The runner hides stderr when a program exits 0, so a learner never sees that report from a
  program that "passes". The debugging exercises (136-140) therefore print each verdict on stdout
  *and* stderr, end with `std.process.exit(1)` when a layer diverges, and declare
  `pub const std_options = .{ .log_level = .err }` to keep `warning(pjrt)` lines out of the way.
  The healthy answer still exits 0 and is checked on stdout like every other exercise.
- The default tolerances of `expectClose` are loose for one layer: exact vs tanh GELU, or a wrong
  LayerNorm eps, pass them. Exercises that must catch those use
  `.{ .absolute_tolerance = 1e-5, .relative_tolerance = 1e-4 }`.
- `Decoder.reset()` does not compile at the pinned commit, and decoders keep state between
  texts: make a new decoder for each text.
- `zml.Buffer.deinitAll(T, &buffers)` wants the *model* type and a flat struct. Passing
  `zml.Bufferized(T)` segfaults the Zig compiler without a message. For a model holding slices
  of layers: `zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(T, &buffers)`, then free the
  slices `bufferize` allocated.
- `zml.tokenizer.Tokenizer.fromBytes` reads a HuggingFace `tokenizer.json` (BPE and WordPiece).
  The encoder does not add special tokens ([CLS], [SEP], BOS): the port adds them.
- `Tensor.gelu` is the tanh approximation (HF `gelu_new`). The exact one is written with `erf`.
- `zml.nn.sdpa` divides by `sqrt(head_dim)` unless `.scale` is given, and accepts fewer
  key/value heads than query heads. `zml.nn.rope` defaults to the `.real_im_pass` layout (HF
  `rotate_half`); llama2.c and Meta's code need `.interleaved`.
