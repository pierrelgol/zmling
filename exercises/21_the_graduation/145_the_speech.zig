//
// The gown fits, the tassel is on the right side, the notes
// are shared. One day left, and Tiny has still not said a
// word. Today she gets assembled, and she speaks.
//
// You have every part, each one checked against PyTorch:
//
//     rmsNorm        exercise 142
//     FeedForward    exercise 142
//     Attention      exercises 143 and 144
//
// What is left of `model.py` is the two classes that only
// plug parts together. The block:
//
//     class TransformerBlock(nn.Module):
//         def __init__(self, layer_id: int, args: ModelArgs):
//             ...
//             self.attention = Attention(args)
//             self.feed_forward = FeedForward(...)
//             self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
//             self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
//
//         def forward(self, x, freqs_cos, freqs_sin):
//             h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
//             out = h + self.feed_forward.forward(self.ffn_norm(h))
//             return out
//
// "Pre-norm": the norm is applied to what goes INTO each part,
// and the part's result is added to the untouched `x`. And the
// model:
//
//     class Transformer(nn.Module):
//         def __init__(self, params: ModelArgs):
//             ...
//             self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
//             self.layers = torch.nn.ModuleList()
//             for layer_id in range(params.n_layers):
//                 self.layers.append(TransformerBlock(layer_id, params))
//             self.norm = RMSNorm(params.dim, eps=params.norm_eps)
//             self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
//
//             # share the unembedding parameters with the embedding parameters
//             self.tok_embeddings.weight = self.output.weight
//
//         def forward(self, tokens, targets=None):
//             h = self.tok_embeddings(tokens)
//             h = self.dropout(h)
//             for layer in self.layers:
//                 h = layer(h, self.freqs_cos, self.freqs_sin)
//             h = self.norm(h)
//             ...
//             logits = self.output(h)
//
// No position embedding to add: RoPE lives inside the
// attention. Three things from the transcript come due:
//
//  - THE PREFIX. Every name of the file starts with
//    `_orig_mod.`. Handle it ONCE, at the top, where the view
//    is made. No `init` below should ever know about it.
//
//  - THE LAYERS. `n_layers` is a number in a file: the struct
//    holds a slice, `layers: []Block`, allocated in `init`
//    (hence the allocator and the `!`), one
//    `view.withPrefix("layers").withLayer(i)` each.
//
//  - THE TIED HEAD. There is no `output.weight` to load. The
//    logits are the final vectors against the token table
//    itself, {.voc, .d}: `x.dot(table, .d)` leaves `.voc`.
//
// Then, the words. Tiny's vocabulary is 512 pieces, in
// `tok512.pb`. It is a SentencePiece model, not a HuggingFace
// `tokenizer.json`, and `zml.tokenizer` reads both: it picks
// by the extension, and `.pb` is the one it wants here (the
// author's file is `tok512.model`: same bytes, renamed).
//
//     var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(gpa, io, abs_path);
//     var encoder = try tokenizer.encoder();     // text -> ids
//     var decoder = try tokenizer.decoder();     // ids -> text
//     const ids: []u32 = try encoder.encodeAlloc(gpa, "some text");
//     var text: std.ArrayList(u8) = try decoder.decodeAlloc(gpa, ids);
//
// All four have a `deinit` (`text.deinit(gpa)`), `ids` is
// yours to free. And one trap, the oldest in the trade. The
// author's `sample.py`:
//
//     start_ids = enc.encode(start, bos=True, eos=False)
//
// `bos=True`. Every story Tiny ever read started with token 1,
// "beginning of sequence". ZML's encoder gives you the text
// and nothing else: 4 tokens for "Once upon a time". Without
// the BOS in front, the program runs, and Tiny mumbles
// something that looks like a story and is not the reference.
// The BOS is yours to add, by hand, in `main`.
//
// The rule: THE TOKENIZER IS PART OF THE MODEL, AND SO IS THE
// SCRIPT THAT CALLS IT. Special tokens are added by the
// author's glue code, not by the weights. Read `sample.py`
// like you read `model.py`, and check the ids against the
// reference before you check a single logit.
//
// Writing, without a cache (that's tomorrow). Shapes are
// static, so the Exe is compiled once for `story_len` tokens.
// The story sits in an array of that size, padded with zeros,
// and `Llama.next` gathers the logits at position `last`, the
// last real token. The padding after it is harmless: the
// causal mask keeps every token from looking to its right.
// You wrote this loop in exercise 133.
//
// To do:
//
//  1. `Block.forward`: two lines of Python.
//  2. `Llama`: fields, `init`, `forward`. `next` is given.
//  3. `main` panics: `Checkpoint has no tensor named ...`,
//     with a name that IS in model.py. You know whose fault
//     that is.
//  4. `main`: the prompt as tokens, BOS first, in `story`.
//  5. `main`: the loop. One run of `next_exe` per new token,
//     until the story is `story_len` tokens long.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

const prompt = "Once upon a time";

/// Token 1: "beginning of sequence".
const bos: u32 = 1;

/// The whole speech, prompt included, in tokens.
const story_len = 106;

/// `model_args`, as far as inference needs them.
const Config = struct {
    dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
};

/// `ModelArgs.norm_eps`: in the source, and nowhere else.
const norm_eps: f32 = 1e-5;

/// model.py, RMSNorm.forward. x: {.s, .d}, weight: {.d}
fn rmsNorm(x: zml.Tensor, weight: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .d, norm_eps).mul(weight.broad(x.shape()));
}

/// A torch `nn.Linear(bias=False)`: its weight is {out, in}.
fn linear(view: zml.io.TensorStore.View, name: []const u8) zml.nn.Linear {
    return .init(view.withPrefix(name).createReplicatedTensor("weight", .{ .dout, .d }), null, .d);
}

const FeedForward = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(view: zml.io.TensorStore.View) FeedForward {
        return .{ .w1 = linear(view, "w1"), .w2 = linear(view, "w2"), .w3 = linear(view, "w3") };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: FeedForward, x: zml.Tensor) zml.Tensor {
        const gate = self.w1.forward(x, .f32).silu();
        const up = self.w3.forward(x, .f32);
        return self.w2.forward(gate.mul(up).rename(.{ .dout = .d }), .f32).rename(.{ .dout = .d });
    }
};

const Attention = struct {
    wq: zml.nn.Linear,
    wk: zml.nn.Linear,
    wv: zml.nn.Linear,
    wo: zml.nn.Linear,
    heads: u32,
    kv_heads: u32,

    pub fn init(view: zml.io.TensorStore.View, config: Config) Attention {
        return .{
            .wq = linear(view, "wq"),
            .wk = linear(view, "wk"),
            .wv = linear(view, "wv"),
            .wo = linear(view, "wo"),
            .heads = config.n_heads,
            .kv_heads = config.n_kv_heads,
        };
    }

    /// h: {.s, .d} -> {.s, .d}
    pub fn forward(self: Attention, h: zml.Tensor) zml.Tensor {
        const q = self.wq.forward(h, .f32).splitAxis(.dout, .{ .h = self.heads, .hd = .auto });
        const k = self.wk.forward(h, .f32).splitAxis(.dout, .{ .h = self.kv_heads, .hd = .auto });
        const v = self.wv.forward(h, .f32).splitAxis(.dout, .{ .h = self.kv_heads, .hd = .auto });

        const layout: zml.nn.RopeOpts = .{ .layout = .interleaved };
        const heard = zml.nn.sdpa(
            zml.nn.rope(q, null, layout).rename(.{ .s = .q }),
            zml.nn.rope(k, null, layout).rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{ .attn_mask = zml.nn.causalAttnMask(.{ .q = h.dim(.s), .k = h.dim(.s) }, .f32, null) },
        );

        const merged = heard.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.wo.forward(merged, .f32).rename(.{ .dout = .d });
    }
};

const Block = struct {
    attention_norm: zml.Tensor,
    attention: Attention,
    ffn_norm: zml.Tensor,
    feed_forward: FeedForward,

    pub fn init(view: zml.io.TensorStore.View, config: Config) Block {
        return .{
            .attention_norm = view.withPrefix("attention_norm").createReplicatedTensor("weight", .{.d}),
            .attention = .init(view.withPrefix("attention"), config),
            .ffn_norm = view.withPrefix("ffn_norm").createReplicatedTensor("weight", .{.d}),
            .feed_forward = .init(view.withPrefix("feed_forward")),
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Block, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const Llama = struct {
    // The token table, the blocks, the last norm.
    ???,

    /// The model, with placeholders made from the file. The caller
    /// frees `layers`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Llama {
        return ???;
    }

    /// tokens: {.s} -> logits: {.s, .voc}
    pub fn forward(self: Llama, tokens: zml.Tensor) zml.Tensor {
        return ???;
    }

    /// tokens: {.s}, last: {} -> the most likely token after
    /// position `last`: {} i32
    pub fn next(self: Llama, tokens: zml.Tensor, last: zml.Tensor) zml.Tensor {
        const logits = self.forward(tokens).gather(.{ .s = last }, .{});
        return logits.argMax(.voc).indices.squeeze(.voc);
    }
};

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// One tensor of a file, as a Buffer with the given tags.
fn reference(gpa: std.mem.Allocator, io: std.Io, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8, tags: anytype) !zml.Buffer {
    const shape = (store.view().getShape(key) orelse return error.NoSuchTensor).withTags(tags);
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);
    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);
    return .fromBytes(io, platform, shape, bytes);
}

/// Chapter 20's verdict: are the two close enough? (On a `false`,
/// `expectClose` prints its report on stderr: read it.)
fn matches(io: std.Io, want: zml.Buffer, got: zml.Buffer) bool {
    zml.testing.expectClose(io, want, got, .{}) catch return false;
    return true;
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

    // Tiny's weights.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "stories260K.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    // What PyTorch computed.
    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    // Tiny's words.
    const tokenizer_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "tok512.pb", gpa);
    defer gpa.free(tokenizer_path);
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(gpa, io, tokenizer_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    const model: Llama = try .init(gpa, store.view(), config);
    defer gpa.free(model.layers);

    // A struct with a slice in it: `bufferize` allocates the same
    // slice on the Buffer side. The device memory goes first, the
    // slice after (defers run bottom to top).
    var weights = try zml.mem.bufferize(gpa, Llama, &model);
    defer gpa.free(weights.layers);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Llama, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Llama, &model, &weights, &store, .{});
    try loader.await(io);

    // The story: the BOS, the prompt, and zeros to be written over.
    var story: [story_len]u32 = @splat(0);
    var len: usize = 0;
    // `story` starts with the BOS and the tokens of `prompt`, and
    // `len` says how many tokens that is.
    ???;
    try stdout.print("the prompt, in tokens: {any}\n", .{story[0..len]});

    // First the proof: the logits of the prompt against PyTorch's.
    {
        var want = try reference(gpa, io, platform, &acts, "logits", .{ .s, .voc });
        defer want.deinit();

        const tokens: zml.Tensor = .init(.{ .s = len }, .u32);
        var exe = try platform.compile(gpa, io, model, .forward, .{tokens}, .{});
        defer exe.deinit();

        var tokens_buffer: zml.Buffer = try .fromBytes(io, platform, tokens.shape(), std.mem.sliceAsBytes(story[0..len]));
        defer tokens_buffer.deinit();
        var got = try exe.eval(gpa, io, .{ weights, tokens_buffer });
        defer got.deinit();

        try stdout.print("the logits {f} match PyTorch: {}\n", .{ got.shape(), matches(io, want, got) });
    }

    // Then the speech.
    {
        const tokens: zml.Tensor = .init(.{ .s = story_len }, .u32);
        const last: zml.Tensor = .init(.{}, .i32);
        var next_exe = try platform.compile(gpa, io, model, .next, .{ tokens, last }, .{});
        defer next_exe.deinit();

        // One token per run, until `story` is full.
        ???;
    }

    var text = try decoder.decodeAlloc(gpa, &story);
    defer text.deinit(gpa);
    try stdout.print("{s}\n", .{text.items});
    try stdout.flush();
}
