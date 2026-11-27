//
// Sixth night. The owner has been reading about engagement. The
// Morning Tensor will now have SIX editions a day, and he wants
// the children's page of the evening one "less predictable".
//
// You know how to do unpredictable: `zml.nn.sampleTokens(logits,
// strategy, rng)`, exercises 112, 113 and 147. And you know what
// it cost at the graduation: `strategy` is a
// `zml.nn.SamplingStrategy`, a plain Zig value, so it is BAKED
// into the graph while it is traced (exercise 085). Three
// strategies were six Exes. Six editions would be twelve, and
// that is still a newspaper. An API takes a `temperature` and a
// `top_p` with EVERY request, as floats, from strangers. Nobody
// compiles a model per request.
//
// What changes between two calls of one Exe? Exercise 087: the
// contents of its input Buffers, and nothing else. So the dials
// must be TENSORS:
//
//     pub const DynamicSamplingStrategy = struct {
//         max_top_k: u32,          // still a Zig value: see below
//         top_k: Tensor,           // {} i32
//         temperature: Tensor,     // {} float
//         top_p: Tensor,           // {} float
//         min_p: Tensor,           // {} float
//     };
//
//     const Dials = zml.nn.DynamicSamplingStrategy;
//
//     Dials.init(dtype, max_top_k)
//         the four placeholders, for the compile call
//     Dials.makeBuffers(io, platform, dtype, opts)
//         four scalar Buffers, from a plain Zig value:
//         Dials.Opts{ .top_k, .temperature = 1.0,
//                     .top_p = 1.0, .min_p = 0.0 }
//     Dials.deinitBuffers(&buffers)
//
//     zml.nn.sampleTokensDynamic(logits, dials, rng)
//         -> .{ next_tokens, new_rng }
//
// Same contract as `sampleTokens`: `logits` has a `.voc` axis,
// the Rng goes in and a new one comes out. What happens in
// between is `fixupLogits`, in zml/nn/nn.zig, and it is short enough
// to read:
//
//  1. the candidates are sorted, best first, and only the best
//     `max_top_k` are kept. This one is a Zig number because it
//     decides a SHAPE: the `.topk` axis of everything that
//     follows. (0 means "sort the whole vocabulary": more work at
//     every token, and no ceiling.)
//
//  2. `top_k`: the candidates of rank `top_k` and beyond get the
//     lowest score there is. A dial can lower the number of
//     candidates. It cannot raise it above `max_top_k`: ask for
//     50 in a press built for 8 and you get 8, without a word.
//
//  3. `temperature`: the scores are DIVIDED by it. So not 0,
//     ever.
//
//  4. `top_p`: softmax, then a running total of the chances, best
//     first. A candidate stays while the total is at most
//     `top_p`. ("The smallest set of words that covers 90% of the
//     chances.")
//
//  5. `min_p`: a candidate stays if its chance is at least
//     `min_p` times the chance of the best one.
//
//  6. the first candidate always stays, whatever 4 and 5 say.
//
//  7. Gumbel noise and an argmax, as in `sampleTokens`.
//
// Which gives three ways to ask for the greedy choice with the
// dice still in the graph: `top_k = 1`, `top_p = 0`, `min_p = 1`.
// Tonight's first three editions. They are how you TEST a
// sampler: a draw cannot be compared with PyTorch, but a draw
// that has only one candidate can.
//
// The price: a sort, a softmax and a running total at every
// token, for every caller, including the one who wanted greedy.
// The llama example still uses the baked `sampleTokens`; the
// dynamic one is for when the settings belong to the request.
//
// THE DICE. A `zml.Tensor.Rng` is a state that must be threaded
// (exercise 111): in as an input, out as an output, back in at
// the next run. In a typed Exe it is one more field on each side.
// Two details:
//
//  - you do not write `reuseBuffer` for it. `Rng` does it itself
//    each time it draws (`Rng.update`, in zml/core/tensor.zig): the
//    state that comes out is cast in the memory of the state that
//    went in. Count the donations in tonight's first lines;
//
//  - so the rule of exercise 156 applies: give the runner the
//    address of the dice as the destination of the dice.
//
// And one old truth, because the owner will ask why Tuesday's
// evening edition cannot be printed again on Wednesday. It can:
// same seed, same dials, same press, same text. Dice that were
// seeded ONCE and then left on the table between two editions are
// not the same seed.
//
// To do:
//
//  1. `step`: the end of it. One draw per position, the dials and
//     the dice from `input`, and three outputs.
//
//  2. `feed`: the call. Five inputs that are not baked, three
//     outputs.
//
//  3. In `roll`: `dials`, the Buffers of `setting`, and their
//     defer.
//
//  4. In `main`, the placeholder `dials` was written by someone
//     who read "0 means no ceiling" and liked the sound of it. It
//     runs. Read the line of the night edition, then point 2 of
//     the list.
//
//  5. The reprint of the evening edition is not a reprint. Where
//     do its dice come from?
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// The forme: how many tokens the prefill press takes.
const forme = 16;

/// The rows of the cache.
const rows = 128;

/// The paper for one edition, in tokens.
const paper = 106;

/// The seed of every edition.
const seed = 1911;

/// How many candidates the press can hold, at most.
const type_cases = 8;

const opening = "Once upon a time";

const Edition = struct {
    name: []const u8,
    dials: zml.nn.DynamicSamplingStrategy.Opts,
};

/// The six editions. The last one asks for more than the press
/// holds.
const editions = [_]Edition{
    .{ .name = "morning", .dials = .{ .top_k = 1 } },
    .{ .name = "noon", .dials = .{ .top_k = 8, .top_p = 0.0 } },
    .{ .name = "afternoon", .dials = .{ .top_k = 8, .min_p = 1.0 } },
    .{ .name = "evening", .dials = .{ .top_k = 8 } },
    .{ .name = "evening, reprinted", .dials = .{ .top_k = 8 } },
    .{ .name = "night", .dials = .{ .top_k = 50 } },
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
};

/// What one run of the press takes...
const Input = struct {
    weights: Llama,
    tokens: zml.Tensor,
    pos: zml.Tensor,
    cache: Cache,
    dials: zml.nn.DynamicSamplingStrategy,
    rng: zml.Tensor.Rng,
};

/// ... and what comes out of it.
const Output = struct {
    tokens: zml.Tensor,
    cache: Cache,
    rng: zml.Tensor.Rng,
};

/// tokens: {.s} u32, at positions pos, pos + 1...
/// Returns, for each position, a token DRAWN after it, in the memory
/// of `tokens`, the cache, in its own memory, and the dice.
fn step(input: Input) Output {
    const model = input.weights;
    var x = model.tok_embeddings.forward(input.tokens);
    var filed = input.cache;
    for (model.layers, 0..) |layer, i| x, filed = layer.forward(x, input.pos, filed, i);

    const logits = rmsNorm(x, model.norm).dot(model.tok_embeddings.weight, .d);
    return ???;
}

/// The press: an Exe that knows the names of what it takes.
const Press = zml.FnExe(step);

/// The pressman: a runner with the weights locked in.
const Pressman = Press.Runner(.{.weights});

const Dials = zml.nn.DynamicSamplingStrategy;

/// What one printing keeps on the device from one run to the next.
const Desk = struct {
    cache: zml.Bufferized(Cache),
    dials: zml.Bufferized(Dials),
    rng: *zml.Bufferized(zml.Tensor.Rng),
};

/// One run: the tokens of `plate` enter at position `pos`. `plate`,
/// the cache and the dice are recast in place.
fn feed(io: std.Io, platform: *zml.Platform, pressman: *Pressman, desk: *Desk, plate: *zml.Buffer, pos: usize) !void {
    var pos_buffer: zml.Buffer = try .scalar(io, platform, @as(i32, @intCast(pos)), .i32);
    defer pos_buffer.deinit();
    ???;
}

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

/// Why the presses stopped.
const Stop = enum { the_end, no_paper };

fn isStop(stops: []const u32, token: u32) bool {
    return std.mem.indexOfScalar(u32, stops, token) != null;
}

/// The page as Tiny expects it: the start token, then the opening.
/// The caller frees it.
fn openPage(gpa: std.mem.Allocator, tokenizer: *const zml.tokenizer.Tokenizer, text: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const start = tokenizer.tokenId("<s>") orelse return error.NoSuchToken;
    var tokens: std.Io.Writer.Allocating = .initAligned(gpa, .of(u32));
    errdefer tokens.deinit();
    try encoder.appendTokens(&tokens.writer, &.{start});
    try encoder.encode(&tokens.writer, text);
    return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
}

/// Rolls the presses, the dials on `setting`, with the dice `rng`.
/// `story[0..start]` is the page as it opens; the rest of `story`
/// is the paper. Returns the length of the story, in tokens.
fn roll(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    prefill_man: *Pressman,
    decode_man: *Pressman,
    cache_shape: zml.Shape,
    story: []u32,
    start: usize,
    stops: []const u32,
    setting: Dials.Opts,
    rng: *zml.Bufferized(zml.Tensor.Rng),
) !usize {
    std.debug.assert(start <= forme and start < story.len and story.len <= rows);

    // `dials`: the Buffers of `setting`.
    ???;

    var desk: Desk = .{ .cache = try blankCache(gpa, io, platform, cache_shape), .dials = dials, .rng = rng };
    defer desk.cache.k.deinit();
    defer desk.cache.v.deinit();

    var locked: [forme]u32 = @splat(0);
    @memcpy(locked[0..start], story[0..start]);
    var forme_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = forme }, .u32), std.mem.sliceAsBytes(&locked));
    defer forme_buffer.deinit();
    try feed(io, platform, prefill_man, &desk, &forme_buffer, 0);

    var next: u32 = (try forme_buffer.getValue([forme]u32, io))[start - 1];
    var plate: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&next));
    defer plate.deinit();

    var len = start;
    while (!isStop(stops, next)) {
        story[len] = next;
        len += 1;
        if (len == story.len) break;
        try feed(io, platform, decode_man, &desk, &plate, len - 1);
        next = (try plate.getValue([1]u32, io))[0];
    }
    return len;
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

    // The dials and the dice: ghosts, like the rest.
    const dials: Dials = .init(.f32, 0);
    const rng: zml.Tensor.Rng = .init();

    // One function, two presses, and a pressman for each. Whatever
    // the number of editions.
    var prefill: Press = try .compile(gpa, io, platform, .{}, .{.{
        .weights = model,
        .tokens = zml.Tensor.init(.{ .s = forme }, .u32),
        .pos = pos,
        .cache = cache,
        .dials = dials,
        .rng = rng,
    }});
    defer prefill.deinit();
    var prefill_man: Pressman = try .init(&prefill, gpa, .{ .weights = weights });
    defer prefill_man.deinit(gpa);

    var decode: Press = try .compile(gpa, io, platform, .{}, .{.{
        .weights = model,
        .tokens = zml.Tensor.init(.{ .s = 1 }, .u32),
        .pos = pos,
        .cache = cache,
        .dials = dials,
        .rng = rng,
    }});
    defer decode.deinit();
    var decode_man: Pressman = try .init(&decode, gpa, .{ .weights = weights });
    defer decode_man.deinit(gpa);

    try stdout.print("presses set up: 2, for {d} editions\n", .{editions.len});
    try stdout.print("plates melted down and recast at each run: {d}\n", .{decode.raw.donated_input_indices.len});

    const stops = [_]u32{
        tokenizer.tokenId("<s>") orelse return error.NoSuchToken,
        tokenizer.tokenId("</s>") orelse return error.NoSuchToken,
    };
    const page = try openPage(gpa, &tokenizer, opening);
    defer gpa.free(page);

    var stories: [editions.len][paper]u32 = undefined;
    var lens: [editions.len]usize = undefined;
    // The dice.
    var rng_buffers = try rng.initBuffer(io, platform, seed);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    for (editions, &stories, &lens) |edition, *story, *len| {
        @memcpy(story[0..page.len], page);
        len.* = try roll(gpa, io, platform, &prefill_man, &decode_man, cache_shape, story, page.len, &stops, edition.dials, &rng_buffers);
    }

    const morning = stories[0][0..lens[0]];
    const noon = stories[1][0..lens[1]];
    const afternoon = stories[2][0..lens[2]];
    const evening = stories[3][0..lens[3]];
    const reprint = stories[4][0..lens[4]];
    const night = stories[5][0..lens[5]];

    var in_vocabulary = true;
    for (evening) |token| {
        if (token >= config.vocab_size) in_vocabulary = false;
    }

    try stdout.print("morning (top_k 1): the story of PyTorch: {}\n", .{std.mem.eql(u32, morning, &reference_story)});
    try stdout.print("noon (top_p 0): the morning edition again: {}\n", .{std.mem.eql(u32, noon, morning)});
    try stdout.print("afternoon (min_p 1): the morning edition again: {}\n", .{std.mem.eql(u32, afternoon, morning)});
    try stdout.print("evening (top_k 8, the dice roll): another story: {}\n", .{!std.mem.eql(u32, evening, morning)});
    try stdout.print("  it opens like the others: {}\n", .{std.mem.startsWith(u32, evening, page)});
    try stdout.print("  every word of it is one of Tiny's {d}: {}\n", .{ config.vocab_size, in_vocabulary });
    try stdout.print("evening, reprinted from the same seed: the evening edition again: {}\n", .{std.mem.eql(u32, reprint, evening)});
    try stdout.print("night (top_k 50, in a press that holds {d}): the evening edition again: {}\n", .{ type_cases, std.mem.eql(u32, night, evening) });
    try stdout.flush();
}
