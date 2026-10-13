//
// Chapter 21: The Graduation
//
// Last chapter. There is a gown with your name on it, a stage,
// folding chairs, and in the chairs everybody you met on the
// way: Madame Zelda and her ghosts, Earl from the diner, the
// Coach, Monte in his dealer's vest. In the front row, the
// whole herd of chapter 15: Tina, Kuzco, Paca and Dolly.
//
// They came for you, and for the other graduate: Tiny, Tina's
// daughter, the smallest llama the Academy ever admitted.
// 260 000 parameters. She majored in storytelling, and
// tradition wants the smallest graduate to give the speech.
//
// Tiny is `stories260K`, from the `karpathy/tinyllamas`
// repository: a real Llama 2, trained on children's stories.
// She was not made with the `transformers` library. No
// `LlamaForCausalLM`, no model card, no `config.json`. What
// exists is ONE Python file, `model.py`, in
// https://github.com/karpathy/llama2.c, and a `stories260K.pt`
// that the training script saved. That's the normal situation
// of a port: a paper, a repository, a checkpoint, and you.
//
// So the four previous chapters end here: nobody prepared
// anything. Your first stop is Brenda, at the registrar's
// desk, and Brenda wants a TRANSCRIPT: what is in that file?
//
// What `torch.save` wrote, in the author's `train.py`:
//
//     checkpoint = {
//         "model": raw_model.state_dict(),
//         "optimizer": optimizer.state_dict(),
//         "model_args": model_args,
//         "iter_num": iter_num,
//         "best_val_loss": best_val_loss,
//         "config": config,
//     }
//
// A training checkpoint is the author's desk drawer, and a
// port needs two things out of it:
//
//  - "model", the weights. `make_fixtures.py`, next to this
//    file, copied them to `stories260K.safetensors` WITHOUT
//    cleaning up, on purpose.
//  - "model_args", the numbers the model was built with. The
//    script wrote them to `config.json`, as they were:
//
//     { "dim": 64, "n_layers": 5, "n_heads": 8,
//       "n_kv_heads": 4, "vocab_size": 512, "multiple_of": 4,
//       "max_seq_len": 512, "dropout": 0.05 }
//
// Three things in there are not what `model.py` promises, and
// they are what this exercise is about.
//
// 1. THE NAMES. `model.py` says `self.tok_embeddings`. The
//    file says `_orig_mod.tok_embeddings.weight`. The author
//    trained with `torch.compile`, which wraps the model in
//    another module and keeps the real one in a field called
//    `_orig_mod`. Every name got the prefix. (Other classics:
//    `module.` from `DataParallel`, `model.` from a training
//    framework.) You know the cure since exercise 99: a
//    narrower view, `store.view().withPrefix("_orig_mod")`.
//
// 2. WHAT IS MISSING. The source has
//
//     self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
//     # share the unembedding parameters with the embedding parameters
//     self.tok_embeddings.weight = self.output.weight
//
//    ONE matrix under two names: the "tied" head. The .pt had
//    both names, and safetensors refuses to store the same
//    memory twice, so the script kept `tok_embeddings.weight`
//    only. A converted checkpoint often lacks a tensor the
//    source has: before you call it a bug, look for the line
//    that ties it to another one.
//
// 3. WHAT THE CONFIG DOES NOT SAY. The feed-forward has a
//    hidden size, and `model_args` has no such number
//    (`hidden_dim` is None). It is computed, in the middle of
//    a constructor:
//
//     hidden_dim = 4 * dim
//     hidden_dim = int(2 * hidden_dim / 3)
//     hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
//
//    You can port the formula. But the checkpoint already
//    holds the ANSWER: `w1` is an `nn.Linear(dim, hidden_dim)`,
//    so its weight is {hidden_dim, dim}. The same goes for the
//    number of layers (count them), the vocabulary, and the
//    number of key/value heads (`wk` is narrower than `wq`).
//
// The general rule: A CONFIG IS A CLAIM, A SHAPE IS A FACT.
// When the two can be compared, compare them, once, at the
// start, in a program like this one. When the config is
// silent, read the shapes. And some numbers are in neither:
// `norm_eps: float = 1e-5` is a default in the `ModelArgs`
// class and was never saved anywhere. The author's source is
// part of the config. You will need that one tomorrow.
//
// The tools, all from chapter 14:
//
//     registry.tensors.keys()       every name in the file
//     view.getShape("a.b.weight")   ?zml.Shape, null if absent
//     shape.dim(0)                  one size of it, an i64
//     view.hasKey("a.b")            is there a name STARTING
//                                   with this? (so it works
//                                   for a whole layer)
//     view.withPrefix("layers").withLayer(3)    "layers.3."
//
// To do:
//
//  1. `main` asks for shapes through `view`, and dies of
//     `error.NoSuchTensor`. The view is on the wrong drawer.
//  2. `countLayers`: how many `layers.N.` does the file have?
//     Do not trust the config: that's the point.
//  3. `hiddenFromFormula`: the three lines of Python above.
//     Mind the `int(...)`: it rounds DOWN.
//  4. `hiddenFromShapes` and `kvHeadsFromShapes`: the same
//     facts, from the checkpoint. A head is `dim / n_heads`
//     numbers wide, for the queries and for the keys.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// What `model_args` held. The JSON has more keys than we read.
const Config = struct {
    dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    multiple_of: u32,
};

const View = zml.io.TensorStore.View;

/// The shape of a tensor that must be there.
fn shapeOf(view: View, key: []const u8) !zml.Shape {
    return view.getShape(key) orelse error.NoSuchTensor;
}

/// One size of a shape, as an unsigned number.
fn size(shape: zml.Shape, axis: u8) u32 {
    return @intCast(shape.dim(axis));
}

/// How many blocks the checkpoint really has.
fn countLayers(view: View) u32 {
    return ???;
}

/// The hidden size of the feed-forward, the way model.py computes it.
fn hiddenFromFormula(dim: u32, multiple_of: u32) u32 {
    return ???;
}

/// The hidden size of the feed-forward, the way the checkpoint states it.
fn hiddenFromShapes(view: View) !u32 {
    return ???;
}

/// The number of key/value heads, from the width of `wk`.
fn kvHeadsFromShapes(view: View, config: Config) !u32 {
    return ???;
}

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The claims.
    const config_bytes = try readFile(gpa, io, dir ++ "config.json");
    defer gpa.free(config_bytes);
    const parsed = try std.json.parseFromSlice(Config, gpa, config_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const config = parsed.value;

    // The facts.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "stories260K.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    var prefixed: usize = 0;
    var biases: usize = 0;
    for (registry.tensors.keys()) |name| {
        if (std.mem.startsWith(u8, name, "_orig_mod.")) prefixed += 1;
        if (std.mem.endsWith(u8, name, ".bias")) biases += 1;
    }
    try stdout.print("tensors in the file: {d}\n", .{registry.tensors.count()});
    try stdout.print("  named _orig_mod.something: {d}\n", .{prefixed});
    try stdout.print("  biases: {d}\n", .{biases});

    // The names of model.py start here.
    const view = store.view();

    const table = try shapeOf(view, "tok_embeddings.weight");
    try stdout.print("the token table: {f}\n", .{table});
    try stdout.print("wq: {f}  wk: {f}\n", .{
        try shapeOf(view, "layers.0.attention.wq.weight"),
        try shapeOf(view, "layers.0.attention.wk.weight"),
    });

    try stdout.print("            config  checkpoint\n", .{});
    try stdout.print("vocabulary  {d:6}  {d:10}\n", .{ config.vocab_size, size(table, 0) });
    try stdout.print("dim         {d:6}  {d:10}\n", .{ config.dim, size(table, 1) });
    try stdout.print("layers      {d:6}  {d:10}\n", .{ config.n_layers, countLayers(view) });
    try stdout.print("kv heads    {d:6}  {d:10}\n", .{ config.n_kv_heads, try kvHeadsFromShapes(view, config) });
    try stdout.print("hidden size, by the formula of model.py: {d}\n", .{hiddenFromFormula(config.dim, config.multiple_of)});
    try stdout.print("hidden size, by the shape of w1: {d}\n", .{try hiddenFromShapes(view)});
    try stdout.print("a separate output matrix: {}\n", .{view.hasKey("output.")});
    try stdout.flush();
}
