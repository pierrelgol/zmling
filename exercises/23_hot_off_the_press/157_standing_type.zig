//
// Third night. The top of page one of the Morning Tensor has not
// changed since 1911: the name of the paper, the eagle, "All the
// Gradients Fit to Print". Gus does not set it every evening. It
// is STANDING TYPE: set once, locked in the press, left there.
// Only the news is new.
//
// Now look at how you call your press:
//
//     runner.run(io, .{ weights, plate.*, pos_buffer, cache.* },
//         .{ plate, cache }, .{ .wait = true });
//
// Two things are wrong with that line, and neither is a bug yet.
//
// IT IS POSITIONAL. The inputs are a tuple, and the only thing
// that says which Buffer is the position and which is a token is
// the order. Add an input to `forward` and every call site must
// be found and fixed by hand. Swap two Buffers of the same shape
// and nothing complains, ever. `forward` returns a tuple too:
// "the second output is the cache" lives in your head.
//
// IT CARRIES THE WEIGHTS EVERY TIME. Last night's Exe has 51
// inputs. 47 of them are Tiny's weights, the same 47 at every
// call, and at every call `run` walks through the 47 Buffers,
// checks their shapes and copies their handles into the argument
// table. For 4 that change.
//
// `zml/zml/core/exe.zig` has an answer to both, and `inference.zig` of
// the llama example is built on it:
//
//     pub fn FnExe(comptime function_: anytype) type
//
// A TYPED Exe, made from a function that takes exactly ONE struct
// with named fields, and returns one:
//
//     const Input = struct { weights: Llama, tokens: Tensor, ... };
//     const Output = struct { tokens: Tensor, cache: Cache };
//     fn step(input: Input) Output { ... }
//
//     const Press = zml.FnExe(step);
//
// (A free function, not a method: the model is not `self` any
// more, it is the field `input.weights`. Tuples are refused:
// `FnExe input must use named fields`.) From the two structs the
// type derives everything a call needs:
//
//     Press.Input     zml.Bufferized(Input): its fields, as Buffers
//     Press.Output    the fields of Output, as POINTERS to Buffers
//     Press.compile(gpa, io, platform, .{}, .{input})
//                     `input` is an `Input` made of placeholders.
//                     Mind the two pairs of braces: a tuple of
//                     arguments, holding one struct.
//     press.raw       the plain `zml.Exe` inside: `input_shapes`,
//                     `donated_input_indices` and the rest
//     press.deinit()
//
// And the standing type:
//
//     Press.Runner(.{.weights})
//
// (an enum literal per baked field, in a tuple) is a runner TYPE
// in which the field `weights` is BAKED: given once, at `init`,
// written once in the argument table, never touched again.
//
//     const Pressman = Press.Runner(.{.weights});
//     var pressman: Pressman = try .init(&press, gpa, .{
//         .weights = weights_buffers,
//     });
//     defer pressman.deinit(gpa);
//
//     pressman.run(io, .{
//         .inputs = .{ .tokens = ..., .pos = ..., .cache = ... },
//         .outputs = .{ .tokens = &..., .cache = &... },
//     }, .{ .wait = true });
//
// `.inputs` has every field of `Input` EXCEPT the baked ones;
// forget one, misspell one, and it is a compile error with the
// name in it. `.outputs` has the fields of `Output`, as pointers,
// and the rule of last night still holds: give the address of a
// donated Buffer and it is recast in place.
//
// One rule, and the reason for the order of the fields in the
// llama example: the baked fields must be THE FIRST FIELDS of
// `Input`, in order. The argument table is flat; what is baked
// sits at the front and each call writes after it. Bake a field
// that is not first and the compiler stops you:
//
//     FnExe baked fields must be an ordered prefix of its
//     bufferized input fields
//
// A runner does not own what is baked into it. It holds the
// handles, not the Buffers: the weights must outlive it, and two
// runners (prefill, decode) can share one set of weights, as they
// share one cache. That is the whole memory of an LLM server: one
// copy of the weights, a few Exes, one cache per conversation.
//
// To do:
//
//  1. The weights. Gus dropped the case: load them, the way of
//     exercises 100 and 145.
//
//  2. `Input` does not compile as it is. The message is above.
//
//  3. `step`: last night's `forward`, donations included, as a
//     function of one `Input`.
//
//  4. `feed`: the call, with names.
//
//  5. In `main`: `prefill` and `decode`, two `Press`, and
//     `prefill_man` and `decode_man`, their `Pressman`, defers
//     included.
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
    tokens: zml.Tensor,
    pos: zml.Tensor,
    cache: Cache,
    weights: Llama,
};

/// ... and what comes out of it.
const Output = struct {
    tokens: zml.Tensor,
    cache: Cache,
};

/// tokens: {.s} u32, at positions pos, pos + 1...
/// Returns, for each position, the most likely token after it, in
/// the memory of `tokens`, and the cache, in its own memory.
fn step(input: Input) Output {
    return ???;
}

/// The press: an Exe that knows the names of what it takes.
const Press = zml.FnExe(step);

/// The pressman: a runner with the weights locked in.
const Pressman = Press.Runner(.{.weights});

/// One run: the tokens of `plate` enter at position `pos`, and
/// `plate` and `cache` are recast in place.
fn feed(io: std.Io, platform: *zml.Platform, pressman: *Pressman, cache: *zml.Bufferized(Cache), plate: *zml.Buffer, pos: usize) !void {
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

    // `weights`: Tiny's weights, on the device.
    ???;

    // The cache: a ghost for the compiler.
    const cache_shape: zml.Shape = .init(.{
        .layer = config.n_layers,
        .k = rows,
        .h = config.n_kv_heads,
        .hd = config.dim / config.n_heads,
    }, .f32);
    const cache: Cache = .{ .k = .fromShape(cache_shape), .v = .fromShape(cache_shape) };
    const pos: zml.Tensor = .init(.{}, .i32);

    // One function, two presses, and a pressman for each.
    ???;

    const standing = decode_man.args.baked_count;
    try stdout.print("plates the press takes: {d}\n", .{decode.raw.input_shapes.len});
    try stdout.print("standing type, locked in once: {d}\n", .{standing});
    try stdout.print("plates set for each page: {d}\n", .{decode.raw.input_shapes.len - standing});
    try stdout.print("plates melted down and recast: {d}\n", .{decode.raw.donated_input_indices.len});

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

    var locked: [forme]u32 = @splat(0);
    @memcpy(locked[0..len], story[0..len]);
    var forme_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = forme }, .u32), std.mem.sliceAsBytes(&locked));
    defer forme_buffer.deinit();
    try feed(io, platform, &prefill_man, &cache_buffers, &forme_buffer, 0);
    story[len] = (try forme_buffer.getValue([forme]u32, io))[len - 1];

    var plate: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&story[len]));
    defer plate.deinit();
    len += 1;
    while (len < story_len) : (len += 1) {
        try feed(io, platform, &decode_man, &cache_buffers, &plate, len - 1);
        story[len] = (try plate.getValue([1]u32, io))[0];
    }

    var text = try decoder.decodeAlloc(gpa, &story);
    defer text.deinit(gpa);
    try stdout.print("{s}\n", .{text.items});
    try stdout.print("the same story as PyTorch, token for token: {}\n", .{std.mem.eql(u32, &story, &reference_story)});
    try stdout.flush();
}
