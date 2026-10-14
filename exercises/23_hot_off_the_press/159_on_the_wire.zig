//
// Fifth night. Dolores does not come down any more. She has had a
// wire installed: a ticker on her desk, a roll of paper tape, and
// whatever the press prints is supposed to come out of it,
// upstairs, as it is printed.
//
// It does. All at once, at 4:58, after ten minutes of nothing.
//
//     var text = try decoder.decodeAlloc(gpa, story[0..len]);
//
// `decodeAlloc` wants ALL the tokens, so it is called when the
// loop is over, so the first word reaches the reader when the
// last one has been computed. Tiny writes her 346 tokens in a
// blink. A big model writes twenty a second, and nobody watches
// an empty box for a quarter of a minute: every chat you have
// used prints as it goes. That is called STREAMING, and it is
// done on the host, one token at a time.
//
// Two ways that do not work, so you know them when you meet them.
//
// DECODE EVERYTHING AGAIN AT EACH TOKEN, print the end. 346
// calls, of 1, 2, 3... tokens: the work grows with the square of
// the length, and you must work out which part is new.
//
// DECODE EACH TOKEN ALONE. `tokenizer.decode(id)` is one piece of
// the vocabulary, as it is stored. The first line this program
// prints is made that way: `<s>` is there, the spaces are `▁`,
// and "girl" is `▁g`, `ir`, `l`. Turning pieces into text is not
// a lookup. It depends on what came BEFORE: no space in front of
// the first word of a text, a space in front of the others; a
// letter with an accent, or half of an emoji, may be split over
// several tokens, and its first bytes alone are not valid text.
// Decoding has a STATE.
//
// So the `Decoder` you have been making since exercise 129 is a
// small machine with a memory, and it can be fed one token at a
// time:
//
//     decoder.feedOne(token_id: u32, buffer: []u8) ![]const u8
//     decoder.finalize(buffer: []u8) ![]const u8
//
// `feedOne` returns the text that this token makes CERTAIN. Often
// a word or a piece of one. Sometimes nothing at all: a special
// token, or bytes the decoder holds back until the rest arrives.
// `finalize`, after the last token, returns whatever is still
// held back. With Tiny's SentencePiece file it happens to have
// nothing to return; with a `tokenizer.json` it may. Always call
// it, the loop is the same for every model.
//
// Three rules. The source states the first, in a comment above
// each function:
//
//  - "Output is only valid until the next feed/finalize call and
//    while the decoder AND buffer is alive." The slice you get
//    points into memory that the next call writes over. PRINT IT
//    (or copy it) BEFORE YOU FEED AGAIN. Keep the slices in an
//    array for later and you print what the memory holds later.
//    Gus's proof, in `main`, does that: when the program runs,
//    look at its second line. No, it does not always look the
//    same;
//
//  - `buffer` is scratch space that you own, where some decoders
//    write the text they return. (Tiny's SentencePiece one
//    ignores it and returns a slice of its own memory. The rule
//    is the same.) 64 bytes are plenty for one token;
//
//  - the state is the point, and the trap. A decoder that has
//    seen a text thinks the next token CONTINUES it. A new
//    decoder for each text (exercise 158; `Decoder.reset()` still
//    does not compile).
//
// And the opening? It goes through the decoder like the rest,
// before the first new token, or the decoder would take Tiny's
// first word for the start of the text. (A chat does not print
// its prompt back. Then the template's tokens are simply not fed,
// and the answer starts a text of its own.)
//
// Last thing. `stdout` is buffered: 4096 bytes, since exercise
// 001. Text that sits in that buffer is not on Dolores's tape.
// `try stdout.flush()` after each piece. It costs a system call
// per token, which is nothing next to one run of a model, and it
// is the difference between a stream and a delivery.
//
// To do:
//
//  1. `filed`, in `Attention.forward`: the cache with this
//     layer's new keys and values written in. Exercise 146.
//
//  2. In `roll`: the opening goes through `decoder` and out, then
//     the loop of last night, with each token on the tape as soon
//     as it is known, then `finalize`. The stop token still
//     prints nothing.
//
//  3. In `main`, the proof. The second line is wrong, and not in
//     the same way twice. Rule one.
//
//  4. Still in `main`: read the first word of the story on the
//     tape, and the character before it. Which decoder printed
//     it?
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// The forme: how many tokens the prefill press takes.
const forme = 16;

/// The rows of the cache.
const rows = 512;

const opening = "Once upon a time";

/// For the proof: "<s> Once upon a time, there was a little girl".
const proof_ids = [_]u32{ 1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421 };

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
};

/// What one run of the press takes...
const Input = struct {
    weights: Llama,
    tokens: zml.Tensor,
    pos: zml.Tensor,
    cache: Cache,
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
    const model = input.weights;
    var x = model.tok_embeddings.forward(input.tokens);
    var filed = input.cache;
    for (model.layers, 0..) |layer, i| x, filed = layer.forward(x, input.pos, filed, i);

    const logits = rmsNorm(x, model.norm).dot(model.tok_embeddings.weight, .d);
    const next = logits.argMax(.voc).indices.squeeze(.voc);
    return .{
        .tokens = next.convert(.u32).reuseBuffer(input.tokens),
        .cache = .{
            .k = filed.k.reuseBuffer(input.cache.k),
            .v = filed.v.reuseBuffer(input.cache.v),
        },
    };
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
    pressman.run(io, .{
        .inputs = .{ .tokens = plate.*, .pos = pos_buffer, .cache = cache.* },
        .outputs = .{ .tokens = plate, .cache = cache },
    }, .{ .wait = true });
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

/// Rolls the presses. `story[0..start]` is the page as it opens;
/// the rest of `story` is the paper. The text goes on `tape`,
/// through `decoder`, AS IT IS WRITTEN. Returns the length of the
/// story, in tokens, and why it is not longer.
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
    decoder: *zml.tokenizer.Tokenizer.Decoder,
    tape: *std.Io.Writer,
) !struct { usize, Stop } {
    std.debug.assert(start <= forme and start < story.len and story.len <= rows);

    var cache = try blankCache(gpa, io, platform, cache_shape);
    defer cache.k.deinit();
    defer cache.v.deinit();

    var locked: [forme]u32 = @splat(0);
    @memcpy(locked[0..start], story[0..start]);
    var forme_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = forme }, .u32), std.mem.sliceAsBytes(&locked));
    defer forme_buffer.deinit();
    try feed(io, platform, prefill_man, &cache, &forme_buffer, 0);

    var next: u32 = (try forme_buffer.getValue([forme]u32, io))[start - 1];
    var plate: zml.Buffer = try .fromBytes(io, platform, .init(.{ .s = 1 }, .u32), std.mem.asBytes(&next));
    defer plate.deinit();

    // `len`, `why`, and every piece of text on the tape on its way.
    var scratch: [64]u8 = undefined;
    ???;
    return .{ len, why };
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

    // One function, two presses, and a pressman for each.
    var prefill: Press = try .compile(gpa, io, platform, .{}, .{.{
        .weights = model,
        .tokens = zml.Tensor.init(.{ .s = forme }, .u32),
        .pos = pos,
        .cache = cache,
    }});
    defer prefill.deinit();
    var prefill_man: Pressman = try .init(&prefill, gpa, .{ .weights = weights });
    defer prefill_man.deinit(gpa);

    var decode: Press = try .compile(gpa, io, platform, .{}, .{.{
        .weights = model,
        .tokens = zml.Tensor.init(.{ .s = 1 }, .u32),
        .pos = pos,
        .cache = cache,
    }});
    defer decode.deinit();
    var decode_man: Pressman = try .init(&decode, gpa, .{ .weights = weights });
    defer decode_man.deinit(gpa);

    const stops = [_]u32{
        tokenizer.tokenId("<s>") orelse return error.NoSuchToken,
        tokenizer.tokenId("</s>") orelse return error.NoSuchToken,
    };

    // The proof: the same tokens, as the type case holds them...
    try stdout.print("in the type case: ", .{});
    for (proof_ids) |id| try stdout.print("[{s}]", .{tokenizer.decode(id) orelse "?"});

    // ... and as a decoder gives them back, one at a time.
    var proof = try tokenizer.decoder();
    defer proof.deinit();
    var scratch: [64]u8 = undefined;
    try stdout.print("\noff the decoder:  ", .{});
    var pieces: [proof_ids.len][]const u8 = undefined;
    for (proof_ids, &pieces) |id, *piece| piece.* = try proof.feedOne(id, &scratch);
    for (pieces) |piece| try stdout.print("[{s}]", .{piece});
    try stdout.print("\n--- on the tape\n", .{});

    const page = try openPage(gpa, &tokenizer, opening);
    defer gpa.free(page);
    var story: [rows]u32 = undefined;
    @memcpy(story[0..page.len], page);

    const len, const why = try roll(gpa, io, platform, &prefill_man, &decode_man, cache_shape, &story, page.len, &stops, &proof, stdout);

    try stdout.print("\n--- {d} tokens printed, and the presses stopped: {s}\n", .{ len, switch (why) {
        .the_end => "the story was over",
        .no_paper => "no more paper",
    } });
    try stdout.flush();
}
