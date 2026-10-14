//
// Seventh night, and the quiz. Gus has retired. He left at
// midnight, with his apron folded on the stool and a note on the
// press: "She's yours. Don't throw away the lead."
//
// Dolores wants tomorrow's children's page, and she wants it the
// way the shop does things now:
//
//  - the MORNING edition: the dials on greedy, the whole page of
//    `rows` tokens for paper, and every word on her tape as Tiny
//    writes it;
//
//  - the LATE edition: `late`, below. Sixty tokens of paper, and
//    NOT on the tape: nobody can promise what the dice will say
//    on your machine (exercise 110), so only facts that hold for
//    any dice are printed;
//
//  - the late edition AGAIN, from the same seed, to prove the
//    shop can reprint.
//
// No new idea tonight. This is the chapter in one file, and the
// file is nearly empty below the model. What you have: the
// constants, Tiny (`Llama` has its fields and its `init`, and no
// `forward`), `readFile`, `blankCache`, the beginning of `main`
// and its last lines. What the shop does, in the order you
// learned it:
//
//     the forme      155   one prefill of `forme` tokens, blanks,
//                          a prediction per position, the host picks
//     the lead       156   tokens and cache: recast, never copied
//     standing type  157   `zml.FnExe`, named fields, baked weights
//     the stops      158   `<s>` and `</s>` by id, the paper, and
//                          the template: special tokens by id
//     the wire       159   `feedOne`, `finalize`, `flush`
//     the dials      160   `sampleTokensDynamic`, the dice threaded
//
// TWO Exes, for the three printings. `type_cases` candidates at
// most.
//
// The last lines of `main` are written and say what they expect:
//
//     morning_len      usize        tokens of the morning edition
//     morning_stop     Stop         why it stopped
//     morning          []const u32  its tokens, opening included
//     late_story       []const u32  the late edition
//     reprint          []const u32  the late edition, again
//     page             []const u32  the opening, as Tiny expects it
//     decode           the Exe of one token (a `zml.FnExe(...)`)
//     decode_man       its runner
//
// The morning edition is PRINTED BY YOU, on the tape, before
// those lines: the text, as it comes, then one newline.
//
// Things that bite, in the order they will:
//
//  - the baked field is the first field;
//
//  - a prediction per position, and the host reads the one at
//    `len - 1`, not the last;
//
//  - the decode starts at the true length;
//
//  - after a runner, a donated Buffer is not yours to free. If
//    the program dies in `PJRT_Buffer_Destroy`, you know what you
//    did;
//
//  - `<s>` is an id, not three characters;
//
//  - the stop token is not text, and the opening is;
//
//  - a slice from `feedOne` dies at the next call;
//
//  - one decoder per text; the late editions need none;
//
//  - fresh dice for each printing, a cache of zeros for each
//    printing.
//
// When the story on the tape is not the one of the last three
// nights: turn the dials off, drop the stops, compare the first
// 106 tokens with `reference_story`, and walk back through the
// chapter one night at a time. Each of them still compiles on its
// own.
//
// The trucks leave at five.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// The forme: how many tokens the prefill press takes.
const forme = 16;

/// The rows of the cache.
const rows = 512;

/// The seed of every printing.
const seed = 1911;

/// How many candidates the press can hold, at most.
const type_cases = 8;

/// The dials of the late edition, and its paper, in tokens.
const late: zml.nn.DynamicSamplingStrategy.Opts = .{ .top_k = 8, .temperature = 0.9, .top_p = 0.95 };
const late_len = 60;

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

// The press, the pressman, the stops, the page, the printing: yours.
???

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

    // The weights, the two presses, the three printings.
    ???;

    try stdout.print("--- morning edition: {d} tokens, and the presses stopped: {s}\n", .{ morning_len, switch (morning_stop) {
        .the_end => "the story was over",
        .no_paper => "no more paper",
    } });
    try stdout.print("its first {d} tokens are PyTorch's: {}\n", .{ reference_story.len, std.mem.startsWith(u32, morning, &reference_story) });
    try stdout.print("--- late edition: the dice rolled, and it is another story: {}\n", .{!std.mem.startsWith(u32, morning, late_story)});
    try stdout.print("it opens like the morning one: {}\n", .{std.mem.startsWith(u32, late_story, page)});
    try stdout.print("reprinted from the same seed, it is the same: {}\n", .{std.mem.eql(u32, reprint, late_story)});
    try stdout.print("--- the shop: standing type {d}, set for each run {d}, recast at each run {d}\n", .{
        decode_man.args.baked_count,
        decode.raw.input_shapes.len - decode_man.args.baked_count,
        decode.raw.donated_input_indices.len,
    });
    try stdout.print("Put the paper to bed.\n", .{});
    try stdout.flush();
}
