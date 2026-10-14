//
// Fourth night, 4:40. The trucks are in the yard and Dolores is
// on the stairs.
//
// "STOP THE PRESSES! I asked for A story. She has finished it,
// and she has started ANOTHER one. On the same page!"
//
// Until tonight a `while (len < story_len)` decided how long a
// story is, and Tiny was cut in the middle of a sentence. A real
// loop has three reasons to STOP.
//
// 1. THE MODEL SAYS SO. A language model does not stop: it gives
//    a next token, always. What it can do is give a token that
//    MEANS "I am done", because the texts it was trained on ended
//    with one. Which token is a fact about the model, to be
//    looked up, like the layout of `rope`:
//
//  - most models have an EOS: `eos_token_id` in `config.json`. It
//    is sometimes a LIST (Llama 3 has two), which is why
//    `Config.EosTokens`, in the llama example, is a union of
//    `int` and `ints`, and `isEosToken` a loop;
//
//  - Tiny has an `</s>` in her vocabulary and never learned to
//    use it. Her training text was stories glued one after the
//    other, each one behind a BOS. So when a story is over she
//    writes `<s>`, the start of the next one. `run.c`, in
//    llama2.c, stops there too: "the BOS (=1) token delimits
//    sequences".
//
// A stop token is not text. It is not appended, not decoded, not
// fed back.
//
// 2. THE PAGE IS FULL. The editor's limit: `max_tokens`, in every
//    API you have used.
//
// 3. THE CACHE IS FULL. Row number `rows` does not exist. Nothing
//    in the graph will tell you: it has no errors to give, only
//    numbers. The host counts.
//
// Tonight 2 and 3 are one number: the length of the slice `story`
// that `roll` is given, never more than `rows`.
//
// Why is this loop on the HOST, when chapter 11 put loops inside
// a graph? Every token must come back to the host anyway, to be
// printed and compared with the stop tokens. And a graph that
// stops "when it wants" has an output of unknown length: there is
// no such shape. One run of the Exe is one token; the story is a
// Zig `while`.
//
// The order inside the loop matters, and it is the one of
// `runDecode` in the llama example's `session.zig`:
//
//     is `next` a stop token?       leave: the story is over
//     append `next` to the story
//     is the story full?            leave: no paper
//     run the decode on `next`, at its position: a new `next`
//
// Check the stop token after the append and it ends up in the
// text. Check the paper after the run and you have paid for a
// token you throw away.
//
// THE OTHER END OF THE PAGE. The BOS has been a `const bos: u32 =
// 1` since exercise 145, with the rule that came with it: THE
// ENCODER DOES NOT ADD SPECIAL TOKENS. Here is the other half of
// that rule: it does not READ them either. Gus was told the page
// starts with `<s>`, so he typed `<s>`:
//
//     encoder.encodeAlloc(gpa, "<s>Once upon a time")
//
// and got ten tokens where five were expected: a `<`, an `s`, a
// `>` and what follows, in pieces. Not one of them is token 1. To
// the encoder all text is text. A special token goes in BY ID,
// and the id is looked up by name, once:
//
//     tokenizer.tokenId(token: []const u8) ?u32
//
// An optional: `null` when the vocabulary has no such piece.
//
// What turns "a text" into "the tokens the model expects" is
// called the TEMPLATE, and somebody writes it by hand for each
// family of models. Tiny's is one token and the text. A chat
// model's is longer: read `tokenizePrompt` in `session.zig`,
// Llama 3's, with its `<|start_header_id|>`, `<|end_header_id|>`
// and `<|eot_id|>` by id around "user", the prompt and
// "assistant" as text. It is built with two methods of the
// encoder that write tokens to a `*std.Io.Writer`:
//
//     encoder.appendTokens(w, &.{ id, id, ... })   ids, as they are
//     encoder.encode(w, text)                      text, encoded
//
// and the writer is an allocating one, aligned for u32:
//
//     const Allocating = std.Io.Writer.Allocating;
//     var tokens: Allocating = .initAligned(gpa, .of(u32));
//     ... appendTokens and encode, on `&tokens.writer` ...
//     const bytes = try tokens.toOwnedSlice();
//     const ids: []u32 = @ptrCast(@alignCast(bytes));
//
// Get a template wrong and nothing crashes. The model answers, a
// little worse, forever.
//
// To do:
//
//  1. `rmsNorm`: Tiny's norm, exercise 142. The function of
//     `zml.nn` has no weight.
//
//  2. `openPage`: Gus's version is there. It runs.
//
//  3. `isStop`.
//
//  4. In `roll`: `why`, the loop, in the order above.
//     `story[0..start]` is the page as it opens, `next` the first
//     token Tiny wants to add, `plate` holds it on the device.
//
//  5. In `main`, two orders go through the press: a full page of
//     `rows` tokens, then a column of 40. Read the second text.
//     Exercise 147 warned you: a decoder REMEMBERS. (Its advice
//     was `decoder.reset()`, which does not compile at this
//     commit. A new decoder per text does.)
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// The forme: how many tokens the prefill press takes.
const forme = 16;

/// The rows of the cache.
const rows = 512;

const opening = "Once upon a time";

/// Tonight's orders: how many tokens of paper for each printing.
const orders = [_]usize{ rows, 40 };

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
    return ???;
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
    return ???;
}

/// The page as Tiny expects it: the start token, then the opening.
/// The caller frees it.
fn openPage(gpa: std.mem.Allocator, tokenizer: *const zml.tokenizer.Tokenizer, text: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    // Gus: "The page starts with <s>. So I typed <s>."
    const typed = try std.mem.concat(gpa, u8, &.{ "<s>", text });
    defer gpa.free(typed);
    return encoder.encodeAlloc(gpa, typed);
}

/// Rolls the presses. `story[0..start]` is the page as it opens;
/// the rest of `story` is the paper. Returns the length of the
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

    var len = start;
    const why: Stop = ???;
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

    // Both of Tiny's special tokens. She only ever writes the first.
    const stops = [_]u32{
        tokenizer.tokenId("<s>") orelse return error.NoSuchToken,
        tokenizer.tokenId("</s>") orelse return error.NoSuchToken,
    };

    const page = try openPage(gpa, &tokenizer, opening);
    defer gpa.free(page);
    try stdout.print("the page opens with: {any}\n", .{page});
    try stdout.print("the presses stop on: {any}\n", .{stops});

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    for (orders) |paper| {
        var story: [rows]u32 = undefined;
        @memcpy(story[0..page.len], page);
        const len, const why = try roll(gpa, io, platform, &prefill_man, &decode_man, cache_shape, story[0..paper], page.len, &stops);

        var text = try decoder.decodeAlloc(gpa, story[0..len]);
        defer text.deinit(gpa);

        try stdout.print("--- {d} tokens of paper\n", .{paper});
        try stdout.print("{s}\n", .{text.items});
        try stdout.print("--- {d} tokens printed, and the presses stopped: {s}\n", .{ len, switch (why) {
            .the_end => "the story was over",
            .no_paper => "no more paper",
        } });
    }
    try stdout.flush();
}
