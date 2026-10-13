//
// The Little Dragon Library closes at six, and at seven it opens
// again, for story hour. Madame Plume has read every book on
// the shelves aloud, twice, and the children have started to
// correct her.
//
// This morning a box was left at the door. Inside: a
// storyteller. His name is Neo, he is very small (a million
// weights, give or take the vocabulary), and he knows one
// thing: how a story for a four-year-old goes on. The label
// says `roneneldan/TinyStories-1M`, a real model, trained by
// real people, and the box holds what a model repository on
// the Hugging Face hub usually holds:
//
//     pytorch_model.bin   48 MB   the weights
//     config.json          1 KB   the sizes and the options
//     tokenizer.json       2 MB   text <-> token ids
//
// Pip, the apprentice, had one look at the list of tensors.
// "wte, wpe, h.0.ln_1, attn, mlp, c_fc, c_proj, ln_f. That's
// a GPT-2. I know GPT-2 by heart. Done by lunch."
//
// It is not a GPT-2. It is a GPT-Neo, which was written to
// LOOK like one, and this chapter is about every place where
// it differs. None of them makes a port crash. All of them
// make it tell worse stories. The rule of the chapter, which
// Madame Plume has embroidered on a cushion:
//
//     READ THE AUTHOR'S CODE. DO NOT ASSUME.
//
// Before any code: what is in the box? Three surprises.
//
// 1. THE FILE IS A PROGRAM. `pytorch_model.bin` is a Python
//    pickle: instructions for rebuilding Python objects, which
//    the reader EXECUTES. ZML does not read pickles, and you do
//    not want it to. A port starts in Python, once, with a
//    conversion to safetensors, the header-plus-raw-bytes
//    format you opened by hand in exercise 097:
//
//     state = torch.load("pytorch_model.bin", weights_only=True)
//     save_file(state, "model.safetensors")
//
//    (`weights_only=True` refuses to run anything else than
//    tensor rebuilding. Newer repositories ship the safetensors
//    file directly. Old ones need you.) The script is next to
//    this file, `make_fixtures.py`. You never have to run it.
//
// 2. NOT EVERYTHING IN A FILE IS A WEIGHT. The conversion made
//    the file shrink from 48 MB to 15. Every block had saved
//    two "buffers", tensors that PyTorch keeps next to the
//    weights and that nobody trained:
//
//     transformer.h.0.attn.attention.bias          {1,1,2048,2048} bool
//     transformer.h.0.attn.attention.masked_bias   {} f32
//
//    A "bias" with four axes of booleans. It is the causal
//    mask of exercise 104, the triangle that hides the future,
//    stored for the longest possible text, 4 MB per block.
//    The model's source says where it comes from:
//
//     bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool))
//
//    A port computes that, it does not load it. The script
//    dropped both, and your file holds weights only.
//
// 3. NOT EVERY WEIGHT IS IN THE FILE. The last layer of the
//    model is `lm_head`, a Linear that turns 64 numbers into
//    50257 scores. Look for `lm_head.weight`. And then read
//    this line of the class `GPTNeoForCausalLM`:
//
//     _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
//
//    "Tied" means: the same tensor, used twice. One table,
//    read forwards to turn a token into numbers, and backwards
//    to turn numbers into token scores. The file stores it
//    once. (Exercise 132 uses it.)
//
// And a fourth, smaller, which Pip is about to meet. A GPT-Neo
// block owns six Linears. PyTorch gives a Linear a bias unless
// told otherwise, and the author told otherwise for some:
//
//     self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
//     self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
//     self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
//     self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
//
// In exercise 099 a missing name was a panic:
//
//     panic: Checkpoint has no tensor named coach.0.bias
//
// and in exercise 101 you looked before you leapt, with
// `hasKey`. The View has a method that does both at once:
//
//     view.maybeCreateTensor(subkey, tags, .replicated, .{})  -> ?zml.Tensor
//
// `null` when the file has no such entry, the placeholder
// otherwise. The third argument says where the weight lives
// (`.replicated`: whole, on every device, what
// `createReplicatedTensor` does without asking) and the fourth
// is for splitting a weight across devices: `.{}`, not today.
// `zml.nn.Linear.init` takes a `?Tensor` for the bias, so the
// two fit together with nothing in between.
//
// The names, finally. This is what the file calls the parts of
// block 0, and nothing but the file can tell you:
//
//     transformer.h.0.ln_1.weight
//     transformer.h.0.attn.attention.q_proj.weight
//     transformer.h.0.mlp.c_fc.weight
//
// `attn.attention`: two levels, because the author wrapped one
// class in another. The names of YOUR struct fields are free.
// The strings given to the View are not.
//
// One more tool, new here: `view.withLayer(i)` is
// `withPrefix` for a number. `h.withLayer(3)` stands on
// "transformer.h.3.".
//
// Madame Plume wants an inventory before anybody plugs
// anything in. To do:
//
//  1. Write `census`. `registry.tensors` is the map of exercise
//     098: `.values()` gives every entry, with its `name` and
//     its `shape`, and a Shape can `count()` its numbers.
//
//  2. `Voice.init` panics. Pip typed the names he knew by
//     heart. Compare the message with the names of the file.
//
//  3. Then `linear` panics, on the first Linear it meets. Read
//     the message, then the four lines of PyTorch above.
//
//  4. In `main`: is `lm_head.weight` in the box, does each
//     Linear of block 0 have a bias, and how many blocks are
//     there? All three are questions for a View. (Mind what
//     exercise 099 said about `hasKey`: it matches the
//     BEGINNING of a name. In a model of twelve blocks,
//     `hasKey("h.1")` would be true three times over. Ask for
//     something only one block can answer.)
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

const Census = struct {
    /// How many tensors the file holds.
    tensors: usize,
    /// How many numbers, all tensors together.
    numbers: usize,
    /// The tensor with the most numbers.
    biggest: zml.safetensors.Tensor,
};

fn census(registry: *const zml.safetensors.TensorRegistry) Census {
    return ???;
}

/// A Linear from the "weight" and (if there is one) the "bias"
/// found under `view`.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        view.createReplicatedTensor("bias", .{.dout}),
        .d,
    );
}

/// The four Linears of the attention of one block.
const Voice = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,

    /// `view` must stand on a block: "transformer.h.N.".
    pub fn init(view: zml.io.TensorStore.View) Voice {
        const attention = view.withPrefix("attn");
        return .{
            .q_proj = linear(attention.withPrefix("q_proj")),
            .k_proj = linear(attention.withPrefix("k_proj")),
            .v_proj = linear(attention.withPrefix("v_proj")),
            .out_proj = linear(attention.withPrefix("out_proj")),
        };
    }

    pub fn biases(self: Voice) usize {
        var count: usize = 0;
        for ([_]zml.nn.Linear{ self.q_proj, self.k_proj, self.v_proj, self.out_proj }) |layer| {
            if (layer.bias != null) count += 1;
        }
        return count;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const box = census(&registry);
    try stdout.print("The box: {d} tensors, {d} numbers.\n", .{ box.tensors, box.numbers });
    try stdout.print("The biggest: {s} {f}, {d}% of the numbers.\n", .{
        box.biggest.name,
        box.biggest.shape,
        100 * box.biggest.shape.count() / box.numbers,
    });

    // Is the head of the model in the file?
    const has_head = ???;
    try stdout.print("lm_head.weight is in the box: {}\n", .{has_head});

    const blocks = store.view().withPrefix("transformer").withPrefix("h");

    // How many blocks? Count for as long as there is one.
    var block_count: usize = 0;
    ???;
    try stdout.print("Blocks: {d}\n", .{block_count});

    try stdout.print("The Linears of block 0:\n", .{});
    const block = blocks.withLayer(0);
    for ([_][]const u8{
        "attn.attention.q_proj",
        "attn.attention.k_proj",
        "attn.attention.v_proj",
        "attn.attention.out_proj",
        "mlp.c_fc",
        "mlp.c_proj",
    }) |name| {
        const layer = block.withPrefix(name);
        // Does this Linear have a bias?
        const has_bias = ???;
        // (A Shape does not let `print` pad it: it goes through a
        // small buffer first, as in exercise 098.)
        var shape_buffer: [32]u8 = undefined;
        const shape = try std.fmt.bufPrint(&shape_buffer, "{f}", .{layer.getShape("weight").?});
        try stdout.print("  {s:<24} {s:<13} bias: {s}\n", .{ name, shape, if (has_bias) "yes" else "no" });
    }

    const voice: Voice = .init(block);
    try stdout.print("The voice of block 0 is wired: 4 Linears, {d} bias.\n", .{voice.biases()});
    try stdout.flush();
}
