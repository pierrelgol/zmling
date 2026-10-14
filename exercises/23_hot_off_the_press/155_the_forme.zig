//
// Chapter 23: Hot Off the Press
//
// Tiny graduated, and Tiny got a job. The Morning Tensor, a daily
// of the old school, has given her the children's page: one story
// a night, in the paper by five, when the trucks leave.
//
// It is a night job, and you got it too. Downstairs, in the press
// room, you report to Gus, the pressman, forty years in the trade
// and ink up to the elbows. Upstairs is Dolores, the night
// editor, who has a red pencil and a voice that goes through two
// floors.
//
// What you built in chapter 21 was a model that is RIGHT: the
// same story as PyTorch, token for token. This chapter is about
// the distance between that and a model that is IN PRODUCTION.
// Nothing in it changes a single number. Everything in it is what
// zml/examples/llm/models/llama/ does and your graduation piece
// does not. Keep `inference.zig` and `session.zig` of that
// directory open: you are going to rewrite them, small.
//
// First night, first problem. Exercise 146 compiled its prefill
// like this:
//
//     zml.Tensor.init(.{ .s = prompt_len }, .u32)
//
// Five tokens, because "Once upon a time" behind a BOS is five
// tokens. Tonight Dolores sends down three openings, of 5, 8 and
// 12 tokens. A `zml.Shape` has no "any length": every size is a
// number, known when the graph is built (exercise 087 said it of
// the batch). Three lengths, three Exes, three compilations. And
// a reader who types a prompt nobody has seen before waits for a
// compiler.
//
// Gus has had this problem since 1962. A page is set in a FORME:
// a steel frame of one size, the size the press takes. The text
// is never exactly that long, so the holes are filled with blank
// slugs, which print nothing. One forme, any text that fits.
//
//     prefill    tokens {.s = forme}    the prompt, then blanks
//     decode     tokens {.s = 1}        as in exercise 146
//
// ONE prefill Exe, for every prompt of at most `forme` tokens. It
// is what `runPrefill` does in the llama example's `session.zig`:
// a buffer of `seqlen` tokens, set to 0, the prompt copied at the
// front. (There, the forme is as long as the whole cache. A
// server that minds its latency compiles a few sizes, say 32, 128
// and 512, and takes the smallest that fits: the prefill costs
// what the forme holds, not what the prompt holds. These sizes
// are called buckets.)
//
// You have padded a page before: exercise 133, for Neo, who had
// no cache. The causal mask made the filler harmless, and a
// `gather` picked position `len - 1`. With a cache and two Exes,
// the same idea has two new corners.
//
// 1. THE LAST POSITION IS A BLANK. In exercise 146, `x.slice(.s,
//    .single(-1))` was "the last token of the prompt". It is now
//    a blank slug, and what comes after a blank is nobody's
//    business. The token you want is the one predicted after
//    position `len - 1`. Exercise 133 gave `len - 1` to the graph
//    as one more input. The llama example goes the other way, and
//    so do you tonight: the press prints a prediction for EVERY
//    position, {.s}, and the host picks its own. No extra input,
//    and an output that has exactly the shape of the tokens that
//    came in. Remember that last detail tomorrow night.
//
// 2. THE BLANKS ARE FILED TOO. The prefill writes `forme` rows of
//    keys and values in the cache: the prompt's, then rows
//    computed from blanks. Garbage, in rows `len` to `forme - 1`.
//    Two things you already have make it harmless, and you should
//    be able to say why before you go on:
//
//  - the mask of exercise 146: a token at position p looks at
//    rows 0 to p, so no token of the prompt ever saw a blank;
//
//  - the decode: the first new token is filed at row `len`, over
//    the garbage, before anybody looks at that row. The next one
//    at `len + 1`. And so on.
//
// So the rule for the host is: the decode starts at position
// `len`, the true length. Not at `forme`. Start at `forme` and
// every token is spun (exercise 143) for a seat it does not sit
// in, with garbage rows visible on its left. The program runs.
// The story is another one.
//
// And the cache needs at least `forme` rows, or the prefill does
// not fit in it.
//
// To do:
//
//  1. `visible`: the mask. You wrote it in exercise 146.
//
//  2. `Llama.forward`: the model, cache included, with ONE
//     prediction per position: {.s} i32. No slice.
//
//  3. In `main`: `locked`, the forme. The prompt, then blanks.
//     Token 0 is as good a blank as any.
//
//  4. The prefill line is Gus's, from memory. It runs. Read what
//     he picks.
//
//  5. The decode loop: one token per run, the one that just came
//     out, at ITS position. `feed` returns the prediction made
//     after `tokens[pick]`.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// Token 1: "beginning of sequence".
const bos: u32 = 1;

/// The forme: how many tokens the prefill press takes. Always.
const forme = 16;

/// The rows of the cache: the longest text it can hold.
const rows = 64;

/// The length of a story, opening included, in tokens.
const story_len = 48;

/// Tonight's openings.
const openings = [_][]const u8{
    "Once upon a time",
    "One day, a little dog",
    "Lily and her mom went to the park",
};

/// The story as PyTorch gives it: chapter 21's `make_fixtures.py`
/// prints these.
const reference_story = [106]u32{
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
/// {.layer, .k = rows, .h, .hd}
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
    fn visible(positions: zml.Tensor, cache_rows: i64) zml.Tensor {
        return ???;
    }

    /// h: {.s, .d}, the tokens at positions pos, pos + 1...
    /// Returns what they heard, {.s, .d}, and the cache with the keys
    /// and values of layer number `layer` filed.
    pub fn forward(self: Attention, h: zml.Tensor, pos: zml.Tensor, cache: Cache, layer: usize) struct { zml.Tensor, Cache } {
        const q = self.wq.forward(h, .f32).splitAxis(.dout, .{ .h = self.heads, .hd = .auto });
        const k = self.wk.forward(h, .f32).splitAxis(.dout, .{ .h = self.kv_heads, .hd = .auto });
        const v = self.wv.forward(h, .f32).splitAxis(.dout, .{ .h = self.kv_heads, .hd = .auto });

        const positions = zml.Tensor.arange(.{ .end = h.dim(.s) }, .i32).withTags(.{.s}).add(pos);
        const layout: zml.nn.RopeOpts = .{ .layout = .interleaved };

        const at = .{ .layer = zml.Tensor.scalar(layer, .i32), .k = pos };
        const filed: Cache = .{
            .k = cache.k.dynamicUpdateSlice(at, zml.nn.rope(k, positions, layout).rename(.{ .s = .k })),
            .v = cache.v.dynamicUpdateSlice(at, v.rename(.{ .s = .k })),
        };

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
    /// Returns, for EACH of them, the most likely token after it,
    /// {.s} i32, and the cache with all of them filed.
    pub fn forward(self: Llama, tokens: zml.Tensor, pos: zml.Tensor, cache: Cache) struct { zml.Tensor, Cache } {
        return ???;
    }
};

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// A cache full of zeros, on the device. (Zeros, not garbage:
/// exercise 146 says why.)
fn blankCache(gpa: std.mem.Allocator, io: std.Io, platform: *zml.Platform, shape: zml.Shape) !zml.Bufferized(Cache) {
    const zeros = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(zeros);
    @memset(zeros, 0);
    return .{
        .k = try .fromBytes(io, platform, shape, zeros),
        .v = try .fromBytes(io, platform, shape, zeros),
    };
}

/// One run: `tokens` (as many as the Exe was compiled for) enter at
/// position `pos`, and the cache that comes out replaces the one
/// that went in. Returns the token predicted after `tokens[pick]`.
fn feed(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    runner: *zml.Exe.Runner,
    weights: zml.Bufferized(Llama),
    cache: *zml.Bufferized(Cache),
    tokens: []const u32,
    pos: usize,
    pick: usize,
) !u32 {
    var tokens_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = tokens.len }, .u32), std.mem.sliceAsBytes(tokens));
    defer tokens_buffer.deinit();
    var pos_buffer: zml.Buffer = try .scalar(io, platform, @as(i32, @intCast(pos)), .i32);
    defer pos_buffer.deinit();

    var next_buffer: zml.Buffer = undefined;
    var filed: zml.Bufferized(Cache) = undefined;
    runner.run(io, .{ weights, tokens_buffer, pos_buffer, cache.* }, .{ &next_buffer, &filed }, .{ .wait = true });
    defer next_buffer.deinit();

    cache.k.deinit();
    cache.v.deinit();
    cache.* = filed;

    const next = try next_buffer.toSliceAlloc(gpa, io);
    defer next.free(gpa);
    return @intCast(next.items(i32)[pick]);
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

    const model: Llama = try .init(gpa, store.view().withPrefix("_orig_mod"), config);
    defer gpa.free(model.layers);

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var weights = try zml.mem.bufferize(gpa, Llama, &model);
    defer gpa.free(weights.layers);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Llama, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Llama, &model, &weights, &store, .{});
    try loader.await(io);

    // The cache: a ghost for the compiler.
    const cache_shape: zml.Shape = .init(.{
        .layer = config.n_layers,
        .k = rows,
        .h = config.n_kv_heads,
        .hd = config.dim / config.n_heads,
    }, .f32);
    const cache: Cache = .{ .k = .fromShape(cache_shape), .v = .fromShape(cache_shape) };
    const pos: zml.Tensor = .init(.{}, .i32);

    // One function, two Exes. And no third one, whatever the opening.
    var prefill = try platform.compile(gpa, io, model, .forward, .{ zml.Tensor.init(.{ .s = forme }, .u32), pos, cache }, .{});
    defer prefill.deinit();
    var prefill_runner = try prefill.runner(gpa);
    defer prefill_runner.deinit(gpa);

    var decode = try platform.compile(gpa, io, model, .forward, .{ zml.Tensor.init(.{ .s = 1 }, .u32), pos, cache }, .{});
    defer decode.deinit();
    var decode_runner = try decode.runner(gpa);
    defer decode_runner.deinit(gpa);

    try stdout.print("the forme holds {d} tokens; presses set up for {d} openings: 2\n", .{ forme, openings.len });

    var first_story: [story_len]u32 = undefined;
    for (openings, 0..) |opening, n| {
        var cache_buffers = try blankCache(gpa, io, platform, cache_shape);
        defer cache_buffers.k.deinit();
        defer cache_buffers.v.deinit();

        // The BOS, then the opening.
        var story: [story_len]u32 = @splat(0);
        const ids = try encoder.encodeAlloc(gpa, opening);
        defer gpa.free(ids);
        if (1 + ids.len > forme) return error.OpeningTooLong;
        story[0] = bos;
        @memcpy(story[1..][0..ids.len], ids);
        var len: usize = 1 + ids.len;
        try stdout.print("\"{s}\": {d} tokens and {d} blanks\n", .{ opening, len, forme - len });

        // `locked`: the forme, [forme]u32. The opening, then blanks.
        ???;

        // The whole forme in one go, at position 0.
        story[len] = try feed(gpa, io, platform, &prefill_runner, weights, &cache_buffers, &locked, 0, forme - 1);
        len += 1;

        // Then one token per run, until the story has `story_len` tokens.
        ???;

        // A decoder remembers the text before: a new one per story.
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var text = try decoder.decodeAlloc(gpa, &story);
        defer text.deinit(gpa);
        try stdout.print("{s}\n", .{text.items});

        if (n == 0) first_story = story;
    }

    try stdout.print("the first story is PyTorch's, token for token: {}\n", .{std.mem.eql(u32, &first_story, reference_story[0..story_len])});
    try stdout.flush();
}
