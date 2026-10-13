//
// Bertie thinks: nine tokens in, nine vectors of 128 numbers
// out. Madame Vogel is not paid for vectors. She wants a WORD
// where the delegate mumbled, and the part of a model that
// turns thoughts back into words is called a HEAD. Bertie's,
// in the file, is everything under `cls.predictions`:
//
//     class BertPredictionHeadTransform(nn.Module):
//         def forward(self, hidden_states):
//             hidden_states = self.dense(hidden_states)
//             hidden_states = self.transform_act_fn(hidden_states)
//             hidden_states = self.LayerNorm(hidden_states)
//             return hidden_states
//
//     class BertLMPredictionHead(nn.Module):
//         def __init__(self, config):
//             self.transform = BertPredictionHeadTransform(config)
//             self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
//             self.bias = nn.Parameter(torch.zeros(config.vocab_size))
//
//         def forward(self, hidden_states):
//             hidden_states = self.transform(hidden_states)
//             hidden_states = self.decoder(hidden_states)
//             return hidden_states
//
// A Linear, the same GELU as before, a LayerNorm, and then
// `decoder`: a big Linear from 128 numbers to 30522 scores,
// one for each word of the glossary. The highest score is
// Bertie's guess.
//
// The trainee ported it as written: `decoder` is an
// `nn.Linear`, so he asked the file for
// `cls.predictions.decoder.weight` and
// `cls.predictions.decoder.bias`. You know what ZML says when
// a name is not in the file. Neither of them is.
//
// The file is not broken, and nothing was lost. Two lines
// were left out of the excerpt above. In `__init__`, a
// comment:
//
//     # The output weights are the same as the input embeddings, but there is
//     # an output-only bias for each token.
//
// and, in the class of the whole model:
//
//     _tied_weights_keys = {
//         "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
//         "cls.predictions.decoder.bias": "cls.predictions.bias",
//     }
//
// TIED WEIGHTS. The decoder has no matrix of its own: it uses
// the word table, the very tensor the embeddings look ids up
// in. Read it one way, {.voc, .d}, it turns an id into a
// vector. Read it the other way, it scores a vector against
// every word: a dot product over `.d` with each row. One
// tensor, two jobs, and nearly four million numbers saved (which,
// for Bertie, is nearly all of him).
//
// And look at the shapes: an `nn.Linear(128, 30522)` stores
// its weight as {30522, 128}, out first. That IS the layout
// of the table. No transpose anywhere: `dot` contracts `.d`
// and leaves `.voc`.
//
// The bias is tied too, the other way around: `decoder.bias`
// is `cls.predictions.bias` under a second name, and that one
// is in the file.
//
// In PyTorch the two names point at the same memory, and a
// checkpoint may contain the tensor twice, once, or (in
// safetensors, which refuses to store the same memory under
// two names) only under one of them. So:
//
// PORTING RULE 8: when the model needs a weight that the file
// does not have, don't go looking for a better file. Look for
// "tie" in the reference (`_tied_weights_keys`,
// `tie_weights`, `tie_word_embeddings` in the config, or a
// plain `self.a.weight = self.b.weight`). The weight is there,
// under another name. In your port, tying is not a copy: it is
// using the SAME Tensor in two places of `forward`.
//
// (The opposite also happens. This file has `bert.pooler.*`
// and `cls.seq_relationship.*`, which you never asked for:
// they belong to another task Bertie was trained on, "does
// sentence B follow sentence A?". A port may leave weights
// unused. It should know which ones, and why.)
//
// The model is now complete: `Bert` holds the `Encoder` of the
// last exercise and the `Head`. One sentence has nine
// positions and you want the scores of one: `forward` takes
// the position as a second input, a scalar, and picks that
// row with a `gather` (exercise 70) before the head. No need
// to score 30522 words nine times.
//
// To do:
//
//  1. `Head`: fix what it asks the file for, and write its
//     `forward`. It needs a tensor it does not own, so its
//     arguments are yours to change, and `Bert.forward` with
//     them.
//
//  2. In `main`: the two input Buffers and the run. `where` is
//     a scalar: `zml.Buffer.scalar` (exercise 14).
//
//  3. `best`: the index of the highest score. Once the scores
//     are on your side, this is plain Zig:
//     `std.mem.indexOfMax(f32, slice)` returns a `usize`.
//
// Then run it, and meet the real Bertie. He is, how to put
// it, a very small model.
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

/// A torch `nn.Linear`, with its bias: "weight" is {out, in}.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        view.createReplicatedTensor("bias", .{.dout}),
        .d,
    );
}

fn layerNorm(view: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = view.createReplicatedTensor("weight", .{.d}),
        .bias = view.createReplicatedTensor("bias", .{.d}),
        .eps = eps,
    };
}

/// The GELU BERT was trained with: x/2 * (1 + erf(x / sqrt(2))).
fn geluExact(x: zml.Tensor) zml.Tensor {
    return x.scale(0.5).mul(x.scale(1.0 / @sqrt(2.0)).erf().addConstant(1));
}

const Embeddings = struct {
    word: zml.nn.TokenEmbedding,
    position: zml.Tensor,
    token_type: zml.Tensor,
    norm: zml.nn.LayerNorm,

    /// `view` must stand on "bert.embeddings".
    pub fn init(view: zml.io.TensorStore.View, config: Config) Embeddings {
        return .{
            .word = .{ .weight = view.withPrefix("word_embeddings").createReplicatedTensor("weight", .{ .voc, .d }) },
            .position = view.withPrefix("position_embeddings").createReplicatedTensor("weight", .{ .pos, .d }),
            .token_type = view.withPrefix("token_type_embeddings").createReplicatedTensor("weight", .{ .typ, .d }),
            .norm = layerNorm(view.withPrefix("LayerNorm"), config.layer_norm_eps),
        };
    }

    /// tokens: {.s} i32 -> {.s, .d}
    pub fn forward(self: Embeddings, tokens: zml.Tensor) zml.Tensor {
        var x = self.word.forward(tokens);
        x = x.add(self.position.slice(.pos, .{ .end = tokens.dim(.s) }).rename(.{ .pos = .s }));
        x = x.add(self.token_type.slice(.typ, .{ .end = 1 }).squeeze(.typ).broad(x.shape()));
        return self.norm.forward(x);
    }
};

const Layer = struct {
    query: zml.nn.Linear,
    key: zml.nn.Linear,
    value: zml.nn.Linear,
    attn_out: zml.nn.Linear,
    attn_norm: zml.nn.LayerNorm,
    up: zml.nn.Linear,
    down: zml.nn.Linear,
    out_norm: zml.nn.LayerNorm,
    heads: u32,

    /// `view` must stand on one layer: "bert.encoder.layer.0".
    pub fn init(view: zml.io.TensorStore.View, config: Config) Layer {
        const self = view.withPrefix("attention").withPrefix("self");
        const attn_output = view.withPrefix("attention").withPrefix("output");
        return .{
            .query = linear(self.withPrefix("query")),
            .key = linear(self.withPrefix("key")),
            .value = linear(self.withPrefix("value")),
            .attn_out = linear(attn_output.withPrefix("dense")),
            .attn_norm = layerNorm(attn_output.withPrefix("LayerNorm"), config.layer_norm_eps),
            .up = linear(view.withPrefix("intermediate").withPrefix("dense")),
            .down = linear(view.withPrefix("output").withPrefix("dense")),
            .out_norm = layerNorm(view.withPrefix("output").withPrefix("LayerNorm"), config.layer_norm_eps),
            .heads = config.num_attention_heads,
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        const split = .{ .h = self.heads, .hd = .auto };
        const q = self.query.forward(x, .f32).splitAxis(.dout, split).rename(.{ .s = .q });
        const k = self.key.forward(x, .f32).splitAxis(.dout, split).rename(.{ .s = .k });
        const v = self.value.forward(x, .f32).splitAxis(.dout, split).rename(.{ .s = .k });
        const heard = zml.nn.sdpa(q, k, v, .{}).merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const y = self.attn_norm.forward(self.attn_out.forward(heard, .f32).rename(.{ .dout = .d }).add(x));

        const up = geluExact(self.up.forward(y, .f32)).rename(.{ .dout = .d });
        return self.out_norm.forward(self.down.forward(up, .f32).rename(.{ .dout = .d }).add(y));
    }
};

const Encoder = struct {
    embeddings: Embeddings,
    layers: []Layer,

    /// `view` must stand on "bert". The caller frees `layers`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Encoder {
        const layers = try gpa.alloc(Layer, config.num_hidden_layers);
        for (layers, 0..) |*layer, i| {
            layer.* = .init(view.withPrefix("encoder").withPrefix("layer").withLayer(i), config);
        }
        return .{
            .embeddings = .init(view.withPrefix("embeddings"), config),
            .layers = layers,
        };
    }

    /// tokens: {.s} i32 -> what the last layer thinks, {.s, .d}
    pub fn forward(self: Encoder, tokens: zml.Tensor) zml.Tensor {
        var x = self.embeddings.forward(tokens);
        for (self.layers) |layer| x = layer.forward(x);
        return x;
    }
};

/// BertLMPredictionHead: from what Bertie thinks of one token to
/// a score for every word of the glossary.
const Head = struct {
    dense: zml.nn.Linear,
    norm: zml.nn.LayerNorm,
    decoder: zml.nn.Linear,

    /// `view` must stand on "cls.predictions".
    pub fn init(view: zml.io.TensorStore.View, config: Config) Head {
        return .{
            .dense = linear(view.withPrefix("transform").withPrefix("dense")),
            .norm = layerNorm(view.withPrefix("transform").withPrefix("LayerNorm"), config.layer_norm_eps),
            .decoder = linear(view.withPrefix("decoder")),
        };
    }

    /// x: {.d} -> scores {.voc}
    pub fn forward(self: Head, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const Bert = struct {
    encoder: Encoder,
    head: Head,

    /// `view` must stand on the root of the file. The caller
    /// frees `encoder.layers`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Bert {
        return .{
            .encoder = try .init(gpa, view.withPrefix("bert"), config),
            .head = .init(view.withPrefix("cls").withPrefix("predictions"), config),
        };
    }

    /// tokens: {.s} i32, where: {} i32, the position to guess
    /// -> a score for every word of the glossary, {.voc}
    pub fn forward(self: Bert, tokens: zml.Tensor, where: zml.Tensor) zml.Tensor {
        const x = self.encoder.forward(tokens).gather(.{ .s = where }, .{});
        return self.head.forward(x);
    }
};

/// The ids of `text` the way Bertie was trained to read them:
/// [CLS], the tokens of the text, [SEP]. The caller frees them.
fn encodeForBert(
    gpa: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    encoder: *zml.tokenizer.Tokenizer.Encoder,
    text: []const u8,
) ![]u32 {
    const words = try encoder.encodeAlloc(gpa, text);
    defer gpa.free(words);

    const ids = try gpa.alloc(u32, words.len + 2);
    ids[0] = tokenizer.tokenId("[CLS]").?;
    @memcpy(ids[1 .. ids.len - 1], words);
    ids[ids.len - 1] = tokenizer.tokenId("[SEP]").?;
    return ids;
}

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

    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    const model: Bert = try .init(gpa, store.view(), config);
    defer gpa.free(model.encoder.layers);

    var weights = try zml.mem.bufferize(gpa, Bert, &model);
    defer gpa.free(weights.encoder.layers);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Bert, &weights);

    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Bert, &model, &weights, &store, .{});
    try loader.await(io);

    try stdout.print("cls.predictions.decoder.weight in the file: {}\n", .{store.view().hasKey("cls.predictions.decoder.weight")});
    try stdout.print("cls.predictions.bias in the file: {}\n", .{store.view().hasKey("cls.predictions.bias")});

    const tokenizer_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "tokenizer.json", gpa);
    defer gpa.free(tokenizer_path);
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(gpa, io, tokenizer_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const sentence = "The capital of France is [MASK].";
    const ids = try encodeForBert(gpa, &tokenizer, &encoder, sentence);
    defer gpa.free(ids);
    const where: i32 = @intCast(std.mem.indexOfScalar(u32, ids, tokenizer.tokenId("[MASK]").?).?);

    // What PyTorch scored at that position: 30522 numbers.
    var want = try bufferFromStore(gpa, io, platform, &acts, "mask_logits");
    defer want.deinit();

    const tokens: zml.Tensor = .init(.{ .s = ids.len }, .i32);
    var exe = try platform.compile(gpa, io, model, .forward, .{ tokens, zml.Tensor.init(.{}, .i32) }, .{});
    defer exe.deinit();

    // `got`: the scores for `ids`, at the position `where`.
    ???;

    // The word with the highest score.
    const scores = try got.toSliceAlloc(gpa, io);
    defer scores.free(gpa);
    const best: u32 = ???;

    const same = if (zml.testing.expectClose(io, want, got, .{})) |_| true else |_| false;
    try stdout.print("scores: {f}, same as PyTorch: {}\n", .{ got.shape(), same });
    try stdout.print("{s}\n", .{sentence});
    try stdout.print("[MASK] = {s}\n", .{tokenizer.decode(best).?});
    try stdout.flush();
}
