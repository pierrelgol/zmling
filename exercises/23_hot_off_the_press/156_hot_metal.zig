//
// Second night. Gus watches your press run and says nothing for a
// while, which is how Gus says a lot.
//
// "Every page, you cast a whole new set of plates and throw the
// old set in the bin. You know what lead costs?"
//
// He means `feed`. Read what exercise 146 did after each run:
//
//     runner.run(io, .{ weights, tokens_buffer, pos_buffer, cache.* },
//         .{ &next_buffer, &filed }, .{ .wait = true });
//     cache.k.deinit();
//     cache.v.deinit();
//     cache.* = filed;
//
// The Exe takes a cache and RETURNS a cache: a new Buffer, new
// memory on the device, with every row of the old one copied into
// it and one row changed. Then the host frees the old one. For
// Tiny that is 160 kB allocated, copied and freed per token. For
// a model with 32 layers and a few thousand rows it is gigabytes,
// per token, and for a moment both copies exist: the cache costs
// twice its size.
//
// In a hot-metal shop nothing is thrown away. Yesterday's type
// goes back in the pot and tonight's is cast from the same lead.
// ZML has the pot:
//
//     pub fn reuseBuffer(self: Tensor, origin: Tensor) Tensor
//
// `y.reuseBuffer(x)` is `y`, same values, with one promise
// attached for the compiler: WRITE THIS OUTPUT IN THE MEMORY OF
// THAT INPUT. `x` is said to be DONATED. The update of one row no
// longer has to be a copy of all the others. The conditions:
//
//  - `origin` is an input of the Exe (or something that already
//    reuses one);
//
//  - both have the same size in bytes. The shapes may differ, and
//    even the dtype: ZML panics with `.reuseBuffer expects two
//    tensors with same byteSize` otherwise;
//
//  - the tensor that carries the promise must be what the
//    function RETURNS. A `reuseBuffer` in the middle of a graph,
//    lost before the output, promises nothing.
//
// The llama example does it for the cache (`KvCache.reuseBuffer`,
// in `model.zig`. Right above it, `updateAt` files the rows with
// `scatterSlices` where you wrote `dynamicUpdateSlice`: with
// offsets of rank 0, `zml.ops.scatter` rewrites the one into the
// other, so it is the op you know) and for the tokens: the
// prediction for each position goes out in the memory of the
// token that came in.
//
//     next_tokens.convert(tokens.dtype()).reuseBuffer(tokens)
//
// That is why the output had to have the shape of the input, last
// night. And it gives the decode loop a nice property: the Buffer
// that comes out, one token, IS the Buffer that goes in at the
// next run. No upload from the host at all.
//
// Now the part that bites. A donation is VISIBLE FROM THE
// OUTSIDE: after the call, the input Buffer you passed is dead.
// Its memory belongs to the output. Read it and PJRT answers
// `error.InvalidArgument` ("Buffer has been deleted or donated").
// What is left of it is a HANDLE, a small object on the PJRT
// side, and somebody has to destroy that. Who? It depends on how
// you called, and `zml/zml/exe.zig` says it in so many words:
//
//     exe.call(io, args, &results, opts)
//         The raw way of exercise 082. "The caller owns all input
//         handles, including donated inputs." You `deinit()` the
//         dead Buffer yourself, and you `deinit()` the output,
//         which has a handle of its own.
//
//     runner.run(io, inputs, outputs, opts)
//         The runner of exercise 083. "Destroys donated input
//         handles and fills the output destinations." Nothing for
//         you to free. And since `outputs` are pointers, you may
//         give the ADDRESS OF THE DONATED BUFFER as destination:
//
//     runner.run(io, .{ weights, plate.*, pos_buffer, cache.* },
//         .{ plate, cache }, .{ .wait = true });
//
// `plate` and `cache` are `*zml.Buffer` and
// `*zml.Bufferized(Cache)`: read as inputs, destroyed by the
// runner, then overwritten with the outputs. The variable is
// recast in place, and the single `defer x.deinit()` next to its
// declaration frees whatever it holds at the end.
//
// So the three lines of exercise 146 are now a bug.
// `cache.k.deinit()` after a runner destroys a handle a second
// time, and what you get is not a ZML panic: it is, typically, a
// `Segmentation fault` with `PJRT_Buffer_Destroy` at the top of
// the trace and YOUR `deinit` a few lines below. Learn to
// recognize that trace. Forgetting a `deinit` after a raw call is
// worse, because it is silent: `init.gpa` counts the host's
// memory, not the device's.
//
// An Exe tells you what it melts down:
// `exe.donated_input_indices` is a `[]const usize`, the indices
// of the donated inputs among `exe.input_shapes`.
//
// To do:
//
//  1. `Block.forward`: norm, attention, add; norm, feed-forward,
//     add. Exercise 145.
//
//  2. `Llama.forward` returns what last night's did. It runs, and
//     `main` reports that nothing is recast. Make the three
//     outputs reuse the three inputs: the tokens (they come in as
//     u32) and both halves of the cache.
//
//  3. `feed`: one run through the runner, `plate` and `cache`
//     recast in place.
//
//  4. In `main`, Gus does one page BY HAND, the raw way:
//     `args.set`, `decode.call`, and `results.get` into `page`, a
//     `Page`. The frees that follow are written: read them, they
//     are the lesson.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// Token 1: "beginning of sequence".
const bos: u32 = 1;

/// The forme: how many tokens the prefill press takes.
const forme = 16;

/// The rows of the cache.
const rows = 128;

/// The length of the story, opening included, in tokens.
const story_len = 106;

const opening = "Once upon a time";

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
        const shape: zml.Shape = .init(.{ .q = positions.dim(.s), .k = cache_rows }, .i32);
        const row = zml.Tensor.arange(.{ .end = cache_rows }, .i32).withTags(.{.k}).broad(shape);
        const me = positions.rename(.{ .s = .q }).broad(shape);
        return row.cmp(.LE, me).select(.scalar(0, .f32), .scalar(-std.math.inf(f32), .f32));
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
        return ???;
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

    /// tokens: {.s} u32, sitting at positions pos, pos + 1...
    /// Returns, for each of them, the most likely token after it, in
    /// the memory of `tokens`, and the cache, in its own memory.
    pub fn forward(self: Llama, tokens: zml.Tensor, pos: zml.Tensor, cache: Cache) struct { zml.Tensor, Cache } {
        var x = self.tok_embeddings.forward(tokens);
        var filed = cache;
        for (self.layers, 0..) |layer, i| x, filed = layer.forward(x, pos, filed, i);

        const logits = rmsNorm(x, self.norm).dot(self.tok_embeddings.weight, .d);
        const next = logits.argMax(.voc).indices.squeeze(.voc);
        return .{ next.convert(.u32), filed };
    }
};

/// What a raw call hands back.
const Page = struct {
    tokens: zml.Buffer,
    cache: zml.Bufferized(Cache),
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

/// One run: the tokens of `plate` enter at position `pos`. When it
/// returns, `plate` holds the predictions and `cache` the new cache:
/// both recast in place, nothing to free.
fn feed(
    io: std.Io,
    platform: *zml.Platform,
    runner: *zml.Exe.Runner,
    weights: zml.Bufferized(Llama),
    cache: *zml.Bufferized(Cache),
    plate: *zml.Buffer,
    pos: usize,
) !void {
    return ???;
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
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

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

    var prefill = try platform.compile(gpa, io, model, .forward, .{ zml.Tensor.init(.{ .s = forme }, .u32), pos, cache }, .{});
    defer prefill.deinit();
    var prefill_runner = try prefill.runner(gpa);
    defer prefill_runner.deinit(gpa);

    var decode = try platform.compile(gpa, io, model, .forward, .{ zml.Tensor.init(.{ .s = 1 }, .u32), pos, cache }, .{});
    defer decode.deinit();
    var decode_runner = try decode.runner(gpa);
    defer decode_runner.deinit(gpa);

    try stdout.print("tensors the press takes: {d}\n", .{decode.input_shapes.len});
    try stdout.print("tensors it melts down and recasts: {d}\n", .{decode.donated_input_indices.len});
    try stdout.print("bytes of cache that are no longer copied, per token: {d}\n", .{2 * cache_shape.byteSize()});

    var cache_buffers = try blankCache(gpa, io, platform, cache_shape);
    defer cache_buffers.k.deinit();
    defer cache_buffers.v.deinit();

    // The BOS, then the opening.
    var story: [story_len]u32 = @splat(0);
    const ids = try encoder.encodeAlloc(gpa, opening);
    defer gpa.free(ids);
    story[0] = bos;
    @memcpy(story[1..][0..ids.len], ids);
    var len: usize = 1 + ids.len;

    // The forme goes in, a prediction per position comes out IN IT.
    var locked: [forme]u32 = @splat(0);
    @memcpy(locked[0..len], story[0..len]);
    var forme_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = forme }, .u32), std.mem.sliceAsBytes(&locked));
    defer forme_buffer.deinit();
    try feed(io, platform, &prefill_runner, weights, &cache_buffers, &forme_buffer, 0);
    story[len] = (try forme_buffer.getValue([forme]u32, io))[len - 1];

    // One plate of one token. Each run recasts it into the next token,
    // which is the very thing the next run wants to be fed.
    var plate: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&story[len]));
    defer plate.deinit();
    len += 1;
    while (len < story_len) : (len += 1) {
        try feed(io, platform, &decode_runner, weights, &cache_buffers, &plate, len - 1);
        story[len] = (try plate.getValue([1]u32, io))[0];
    }

    var text = try decoder.decodeAlloc(gpa, &story);
    defer text.deinit(gpa);
    try stdout.print("{s}\n", .{text.items});
    try stdout.print("the same story as PyTorch, token for token: {}\n", .{std.mem.eql(u32, &story, &reference_story)});

    // One page by hand: the BOS alone, at position 0, on an empty cache.
    var args = try decode.args(gpa);
    defer args.deinit(gpa);
    var results = try decode.results(gpa);
    defer results.deinit(gpa);

    var hand_plate: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&bos));
    var hand_cache = try blankCache(gpa, io, platform, cache_shape);
    var hand_pos: zml.Buffer = try .scalar(io, platform, @as(i32, 0), .i32);
    defer hand_pos.deinit();

    // `page`: a `Page`, what the raw call gives back for those inputs.
    ???;

    const after_bos = (try page.tokens.getValue([1]u32, io))[0];
    const melted = if (hand_plate.getValue([1]u32, io)) |_| false else |_| true;
    try stdout.print("by hand: after the BOS comes token {d}\n", .{after_bos});
    try stdout.print("by hand: the plate that went in can no longer be read: {}\n", .{melted});

    // A raw call destroys nothing. Three dead handles, three live
    // outputs: six frees, and nobody but you to count them.
    hand_plate.deinit();
    hand_cache.k.deinit();
    hand_cache.v.deinit();
    page.tokens.deinit();
    page.cache.k.deinit();
    page.cache.v.deinit();

    try stdout.flush();
}
