//
// The dress rehearsal went badly. Tiny gave her speech, word
// for word, and it took forever, because of HOW she gives it:
// to find her next word, she starts again from "Once", reads
// everything she has said so far, and adds one word. Then she
// starts again from "Once". The Dean looked at his watch 101
// times.
//
// The author knows. From the docstring of `generate`, in
// `model.py`:
//
//     Also note this is a super inefficient version of
//     sampling with no key/value cache.
//
// Yesterday's loop ran 106 positions through 5 layers for
// EACH new token. And the causal mask guarantees that almost
// all of it was computed before: a token never looks to its
// right, so nothing about token 3 changes when token 40 shows
// up. Its keys and its values, in every layer, are what they
// were. Tiny must learn her speech BY HEART: write the keys
// and values down once, and from then on run ONE token per
// step, the newest.
//
// You built this on a toy in exercise 108, the gossip archive:
// a cache of fixed size, mostly empty, `dynamicUpdateSlice` to
// file a row at a position known at run time only, and a mask
// so that nobody listens to the empty rows. Today, on a real
// model, three things grow.
//
// 1. ONE ARCHIVE PER LAYER. Each of the 5 attentions has its
//    own keys and values. One tensor holds them all:
//
//     k, v:  {.layer = 5, .k = story_len, .h = 4, .hd = 8}
//
//    (4 heads, not 8: what is remembered is the SHARED notes
//    of exercise 144. That is what grouped-query attention is
//    for.) Layer number `i` is a plain Zig number while the
//    graph is built, so reading its part is an ordinary slice,
//    `cache.k.slice(.layer, .single(i))`, and writing it is
//    the op you know with one more offset:
//
//     cache.k.dynamicUpdateSlice(.{ .layer = i_as_tensor, .k = pos }, new_k)
//
//    `new_k` is {.k = n, .h, .hd}: n rows, filed from row
//    `pos` on. It has no `.layer` axis, and gets one of size
//    1 for free. Its other axes must come in the order of the
//    cache's: hence `.k` before `.h` up there, which is how
//    the keys leave the Linear. Offsets are rank 0 tensors of
//    the same integer type: `zml.Tensor.scalar(i, .i32)` for
//    the layer, and `pos`, an input of the Exe.
//
// 2. THE POSITIONS ARE NO LONGER 0, 1, 2... So far the spin
//    was `rope(x, null, opts)`: positions counted from 0
//    along `.s`. The newest token, alone in its Exe, would be
//    spun as position 0 at every step: a stammer. It must be
//    spun at ITS position, and the second argument of `rope`
//    exists for that:
//
//     zml.nn.rope(x, positions, opts)    // positions: {.s}
//
//    The keys are spun BEFORE they are filed. What sits in
//    the cache is ready to use and is never touched again.
//
// 3. TWO EXES, ONE FUNCTION. The prompt is 5 tokens that
//    nobody has to guess: run them in one go. Then one token
//    at a time. Shapes are static, so that makes two Exes:
//
//     prefill    tokens {.s = 5}, pos = 0
//     decode     tokens {.s = 1}, pos = 5, 6, 7...
//
//    but `forward` is written once: "n tokens, the first of
//    them at position `pos`". Every real LLM runtime has this
//    pair. See `prefill` and `decode` in
//    zml/examples/llm/models/llama/.
//
// The mask is the one of exercise 108, with n rows: the token
// at position p may look at the rows 0 to p of the cache.
// Rows after p are empty (or hold a later token of the
// prompt, during the prefill, which is just as forbidden).
//
// And since only the next word matters, only the LAST position
// goes through the final norm and the head. `model.py` again:
//
//     # inference-time mini-optimization: only forward the output on the very last position
//     logits = self.output(h[:, [-1], :])
//
// The rule, and it is the one of the whole chapter: AN
// OPTIMIZATION IS A SECOND PORT OF THE SAME MODEL, AND THE
// FIRST PORT IS ITS REFERENCE. Never write the cache first.
// Yesterday's slow model was proven against PyTorch;
// today's must tell the same story, token for token, or it
// is wrong, no matter how fast.
//
// (The empty rows are zeros, not garbage: `main` fills the
// cache with zeros. A masked row gets a weight of exactly 0,
// and 0 times whatever was lying in memory is not always 0.
// 0 times NaN is NaN.)
//
// To do, in `Attention.forward`:
//
//  1. `positions`: {.s} i32, counting from `pos`.
//  2. `filed`: the cache, with the new keys and values of
//     this layer written in.
//  3. `visible`: the mask, {.q, .k}.
//
// And in `main`:
//
//  4. The two Exes and their runners.
//  5. The loop: which runner, which tokens, at which position?
//     `feed` does one run and swaps the cache.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

const prompt = "Once upon a time";

/// Token 1: "beginning of sequence".
const bos: u32 = 1;

/// The whole speech, prompt included, in tokens. Also the number of
/// rows of the cache.
const story_len = 106;

/// The speech as PyTorch gives it: `make_fixtures.py` prints these.
const reference_story = [story_len]u32{
    1,   403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421, 395, 317, 426,
    338, 401, 396, 267, 337, 410, 408, 419, 292, 411, 322, 265, 282, 295, 433, 426,
    385, 328, 432, 358, 394, 261, 370, 432, 352, 266, 268, 388, 426, 338, 391, 266,
    267, 337, 335, 312, 432, 398, 312, 286, 267, 414, 270, 333, 415, 426, 13,  438,
    310, 439, 419, 357, 336, 432, 313, 438, 310, 432, 278, 316, 439, 419, 298, 414,
    267, 265, 282, 295, 433, 426, 436, 317, 286, 296, 418, 269, 279, 292, 416, 439,
    413, 409, 416, 327, 263, 415, 294, 267, 400, 426,
};

/// `model_args`, as far as inference needs them.
const Config = struct {
    dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
};

/// Every key and every value computed so far, for every layer:
/// {.layer, .k = story_len, .h, .hd}
const Cache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
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

    /// A mask {.q, .k}: 0 where the token at `positions[q]` may look at
    /// row k of the cache (rows 0 to its own position), -inf elsewhere.
    fn visible(positions: zml.Tensor, rows: i64) zml.Tensor {
        return ???;
    }

    /// h: {.s, .d}, the tokens at positions pos, pos + 1...
    /// Returns what they heard, {.s, .d}, and the cache with the keys
    /// and values of layer number `layer` filed.
    pub fn forward(self: Attention, h: zml.Tensor, pos: zml.Tensor, cache: Cache, layer: usize) struct { zml.Tensor, Cache } {
        const q = self.wq.forward(h, .f32).splitAxis(.dout, .{ .h = self.heads, .hd = .auto });
        const k = self.wk.forward(h, .f32).splitAxis(.dout, .{ .h = self.kv_heads, .hd = .auto });
        const v = self.wv.forward(h, .f32).splitAxis(.dout, .{ .h = self.kv_heads, .hd = .auto });

        const positions = ???;
        const layout: zml.nn.RopeOpts = .{ .layout = .interleaved };

        const filed: Cache = ???;

        const heard = zml.nn.sdpa(
            zml.nn.rope(q, positions, layout).rename(.{ .s = .q }),
            filed.k.slice(.layer, .single(@intCast(layer))),
            filed.v.slice(.layer, .single(@intCast(layer))),
            .{ .attn_mask = visible(positions, cache.k.dim(.k)) },
        );

        const merged = heard.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.wo.forward(merged, .f32).rename(.{ .dout = .d }), filed };
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

    pub fn forward(self: Block, x: zml.Tensor, pos: zml.Tensor, cache: Cache, layer: usize) struct { zml.Tensor, Cache } {
        const heard, const filed = self.attention.forward(rmsNorm(x, self.attention_norm), pos, cache, layer);
        const h = x.add(heard);
        return .{ h.add(self.feed_forward.forward(rmsNorm(h, self.ffn_norm))), filed };
    }
};

const Llama = struct {
    tok_embeddings: zml.nn.TokenEmbedding,
    layers: []Block,
    norm: zml.Tensor,

    /// The model, with placeholders made from the file. The caller
    /// frees `layers`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Llama {
        const layers = try gpa.alloc(Block, config.n_layers);
        for (layers, 0..) |*layer, i| layer.* = .init(view.withPrefix("layers").withLayer(i), config);
        return .{
            .tok_embeddings = .{ .weight = view.withPrefix("tok_embeddings").createReplicatedTensor("weight", .{ .voc, .d }) },
            .layers = layers,
            .norm = view.withPrefix("norm").createReplicatedTensor("weight", .{.d}),
        };
    }

    /// tokens: {.s}, sitting at positions pos, pos + 1...
    /// Returns the most likely token after the last one, {} i32, and
    /// the cache with all of them filed.
    pub fn forward(self: Llama, tokens: zml.Tensor, pos: zml.Tensor, cache: Cache) struct { zml.Tensor, Cache } {
        var x = self.tok_embeddings.forward(tokens);
        var filed = cache;
        for (self.layers, 0..) |layer, i| x, filed = layer.forward(x, pos, filed, i);

        // Only the last position has a next word to offer.
        const last = rmsNorm(x.slice(.s, .single(-1)), self.norm);
        const logits = last.dot(self.tok_embeddings.weight, .d);
        return .{ logits.argMax(.voc).indices.squeeze(.voc), filed };
    }
};

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// One run: `tokens` enter at position `pos`, the next token comes
/// out, and the cache that comes out replaces the one that went in.
fn feed(
    io: std.Io,
    platform: *zml.Platform,
    runner: *zml.Exe.Runner,
    weights: zml.Bufferized(Llama),
    cache: *zml.Bufferized(Cache),
    tokens: []const u32,
    pos: usize,
) !u32 {
    var tokens_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = tokens.len }, .u32), std.mem.sliceAsBytes(tokens));
    defer tokens_buffer.deinit();
    var pos_buffer: zml.Buffer = try .scalar(io, platform, @as(i32, @intCast(pos)), .i32);
    defer pos_buffer.deinit();

    var next_buffer: zml.Buffer = undefined;
    var filed: zml.Bufferized(Cache) = undefined;
    runner.run(io, .{ weights, tokens_buffer, pos_buffer, cache.* }, .{ &next_buffer, &filed }, .{ .wait = true });
    defer next_buffer.deinit();

    // Nothing was donated: the old cache is ours to free.
    cache.k.deinit();
    cache.v.deinit();
    cache.* = filed;

    return @intCast(try next_buffer.getValue(i32, io));
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

    // Tiny's words.
    const tokenizer_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "tok512.pb", gpa);
    defer gpa.free(tokenizer_path);
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(gpa, io, tokenizer_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    const model: Llama = try .init(gpa, store.view().withPrefix("_orig_mod"), config);
    defer gpa.free(model.layers);

    var weights = try zml.mem.bufferize(gpa, Llama, &model);
    defer gpa.free(weights.layers);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Llama, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Llama, &model, &weights, &store, .{});
    try loader.await(io);

    // The story: the BOS, the prompt, and zeros to be written over.
    var story: [story_len]u32 = @splat(0);
    const prompt_len = len: {
        const ids = try encoder.encodeAlloc(gpa, prompt);
        defer gpa.free(ids);
        story[0] = bos;
        @memcpy(story[1..][0..ids.len], ids);
        break :len 1 + ids.len;
    };

    // The cache: a ghost for the compiler, and zeros for the device.
    const cache_shape: zml.Shape = .init(.{
        .layer = config.n_layers,
        .k = story_len,
        .h = config.n_kv_heads,
        .hd = config.dim / config.n_heads,
    }, .f32);
    const cache: Cache = .{ .k = .fromShape(cache_shape), .v = .fromShape(cache_shape) };

    const zeros = try gpa.alloc(u8, cache_shape.byteSize());
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var cache_buffers: zml.Bufferized(Cache) = .{
        .k = try .fromBytes(io, platform, cache_shape, zeros),
        .v = try .fromBytes(io, platform, cache_shape, zeros),
    };
    defer cache_buffers.k.deinit();
    defer cache_buffers.v.deinit();

    // One function, two Exes.
    const pos: zml.Tensor = .init(.{}, .i32);

    // `prefill`, for the prompt, and `decode`, for one token. Each
    // with its runner.
    ???;

    // The prompt in one go, then one token at a time: the one that
    // just came out. Count in `positions_run` every position that
    // you feed to the model.
    var positions_run: usize = 0;
    ???;

    var text = try decoder.decodeAlloc(gpa, &story);
    defer text.deinit(gpa);
    try stdout.print("{s}\n", .{text.items});

    const new_tokens = story_len - prompt_len;
    try stdout.print("the same speech as PyTorch, token for token: {}\n", .{std.mem.eql(u32, &story, &reference_story)});
    try stdout.print("positions run through the model, by heart: {d}\n", .{positions_run});
    try stdout.print("positions run through the model, yesterday: {d}\n", .{new_tokens * story_len});
    try stdout.flush();
}
