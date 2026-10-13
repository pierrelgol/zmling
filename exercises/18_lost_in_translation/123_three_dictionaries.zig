//
// Nine ids go in. The first thing Bertie does with them is
// look them up, and he owns not one dictionary but three. This
// is the first module of the Python model, and the first you
// port:
//
//     class BertEmbeddings(nn.Module):
//         def __init__(self, config):
//             self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, ...)
//             self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
//             self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
//             self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
//             self.dropout = nn.Dropout(config.hidden_dropout_prob)
//
//         def forward(self, input_ids=None, token_type_ids=None, position_ids=None, ...):
//             ...
//             inputs_embeds = self.word_embeddings(input_ids)
//             token_type_embeddings = self.token_type_embeddings(token_type_ids)
//             embeddings = inputs_embeds + token_type_embeddings
//
//             position_embeddings = self.position_embeddings(position_ids)
//             embeddings = embeddings + position_embeddings
//
//             embeddings = self.LayerNorm(embeddings)
//             embeddings = self.dropout(embeddings)
//             return embeddings
//
// How to read a PyTorch module when you port it:
//
// `__init__` is the list of weights. Every `self.x = nn.Thing`
// is a child, and the name of the attribute IS the name in the
// checkpoint: `self.LayerNorm`, inside `self.embeddings`,
// inside `self.bert`, holds a tensor called
//
//     bert.embeddings.LayerNorm.weight
//
// capital letters included. Your `init` must ask for exactly
// that string (exercise 101: a view, a prefix per level). Your
// Zig FIELDS, on the other hand, are yours: call them what you
// like, nothing ever compares them with the file. The program
// prints the names found under `bert.embeddings`, so that you
// don't have to guess.
//
// `forward` is the computation. Port it line by line, and
// watch for three things that are not written in it:
//
//  - `nn.Dropout` does nothing outside of training. A port
//    drops the dropouts.
//
//  - `position_ids` is None when nobody passes it, and then it
//    is 0, 1, 2... So "the position embedding" of a sentence
//    of 9 tokens is the first 9 rows of the table, added row
//    by row. No gather needed: a `slice` (exercise 54), and a
//    `rename` to make `.pos` line up with `.s`.
//
//  - `token_type_ids` is None too, and then it is all zeros.
//    (BERT can read TWO sentences at once, and marks the
//    tokens of the second one with a 1. We give him one.) So
//    ROW 0 of the smallest table is added to every token. It
//    is easy to believe that an argument nobody passes does
//    nothing. This one adds 128 trained numbers to every token.
//
// PORTING RULE 4: the default arguments of the reference
// `forward` are part of the model. For each one, find out
// what None turns into.
//
// The LayerNorm you know from exercise 91:
//
//     zml.nn.LayerNorm{ .weight = ..., .bias = ..., .eps = ... }
//
// with the eps of the resume, not ZML's default. (An honest
// warning: with 1e-5 instead of 1e-12, the check below still
// says `true`. The difference is far below what any comparison
// tolerates. A test will not find that mistake for you, only
// reading the config class will.)
//
// And now the fourth file pays for itself. Before Bertie left,
// his old agency recorded what every module received and
// produced for our test sentence. Each one is stored under
// the name of the module, plus `.in` or `.out`:
//
//     input_ids                 {9} i32
//     bert.embeddings.out       {9,128} f32
//
// so you can run YOUR embeddings on the same ids and compare:
//
//     zml.testing.expectClose(io, want, got, .{}) !void
//
// It takes two Buffers, and returns an error when they differ
// by more than a rounding error (it also prints, on stderr,
// how far apart they are). Tags are not compared: the
// reference has none.
//
// PORTING RULE 5: never port a whole model and then look at
// what comes out. Port one module, compare it with the
// reference, and only then write the next one.
//
// The trainee's version is below.
//
//  1. `init`: two tables are missing. Once that compiles, it
//     panics: he asked the file for a name he made up.
//
//  2. `forward` runs, and does not match. Compare it with the
//     Python, line by line.
//
//  3. In `main`: compile, and run on the reference ids.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/18_lost_in_translation/";

const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    // Not in this config.json: the default of `BertConfig`.
    layer_norm_eps: f32 = 1e-12,
};

const Embeddings = struct {
    word: zml.nn.TokenEmbedding,
    position: zml.Tensor,
    token_type: zml.Tensor,
    norm: zml.nn.LayerNorm,

    /// `view` must stand on "bert.embeddings".
    pub fn init(view: zml.io.TensorStore.View, config: Config) Embeddings {
        return .{
            .word = .{ .weight = view.withPrefix("word_embeddings").createReplicatedTensor("weight", .{ .voc, .d }) },
            // Tag them {.pos, .d} and {.typ, .d}.
            .position = ???,
            .token_type = ???,
            .norm = .{
                .weight = view.withPrefix("norm").createReplicatedTensor("weight", .{.d}),
                .bias = view.withPrefix("norm").createReplicatedTensor("bias", .{.d}),
                .eps = config.layer_norm_eps,
            },
        };
    }

    /// tokens: {.s} i32 -> {.s, .d}
    pub fn forward(self: Embeddings, tokens: zml.Tensor) zml.Tensor {
        var x = self.word.forward(tokens);
        // Positions 0, 1, 2...: the first rows of the table.
        x = x.add(self.position.slice(.pos, .{ .end = tokens.dim(.s) }).rename(.{ .pos = .s }));
        return self.norm.forward(x);
    }
};

/// The whole file, as bytes. The caller frees them.
fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// One tensor of a file of weights (or of activations), as a Buffer.
fn bufferFromStore(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);

    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);

    return .fromBytes(io, platform, shape, bytes);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const config_bytes = try readFile(gpa, io, dir ++ "config.json");
    defer gpa.free(config_bytes);
    const parsed = try std.json.parseFromSlice(Config, gpa, config_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const config = parsed.value;

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const prefix = "bert.embeddings.";
    try stdout.print("in the file, under {s}\n", .{prefix});
    var it = registry.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        try stdout.print("  {s} {f}\n", .{ name[prefix.len..], entry.value_ptr.shape });
    }

    // The recording of Bertie at work, in PyTorch.
    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    var ids = try bufferFromStore(gpa, io, platform, &acts, "input_ids");
    defer ids.deinit();
    var want = try bufferFromStore(gpa, io, platform, &acts, "bert.embeddings.out");
    defer want.deinit();

    const model: Embeddings = .init(store.view().withPrefix("bert").withPrefix("embeddings"), config);

    // The weights, as in exercise 100.
    var weights = try zml.mem.bufferize(gpa, Embeddings, &model);
    defer zml.Buffer.deinitAll(Embeddings, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Embeddings, &model, &weights, &store, .{});
    try loader.await(io);

    // `got`: what `model` makes of `ids`.
    ???;

    const same = if (zml.testing.expectClose(io, want, got, .{})) |_| true else |_| false;
    try stdout.print("embeddings: {f}\n", .{got.shape()});
    try stdout.print("same as PyTorch: {}\n", .{same});
    try stdout.flush();
}
