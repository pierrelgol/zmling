//
// Seven o'clock. The children are on the carpet. Neo has
// been asked for a story and has answered: ".".
//
// One token. That is all a language model ever gives: the
// scores of the NEXT token. A story is a loop, the loop of
// exercise 111, where what comes out goes back in:
//
//     story = the prompt
//     repeat:
//         run the model on the story so far
//         pick a token from the scores of the last position
//         write it at the end of the story
//
// In Python this is five lines, because a PyTorch model takes
// a tensor of any length: 10 tokens, then 11, then 12. A ZML
// Exe does not. Shapes are STATIC (you have known it since
// exercise 015): an Exe compiled for `.{ .s = 10 }` panics on
// 11 tokens. Compiling one Exe per length would work, and
// would take the whole evening.
//
// The trick is as old as paper: write on a page of fixed
// size.
//
//     compile ONCE for `max_len` tokens
//     story:  [ Once | upon | a | time | , | ... | dragon | 0 | 0 | 0 | ... ]
//                                                 ^
//                                           position len - 1
//
// The page always holds `max_len` tokens. The first `len` are
// the story, the rest is filler. Two questions, and the whole
// exercise is in their answers.
//
// DOES THE FILLER CHANGE THE RESULT? No, and you already know
// why: the causal mask (exercise 104). A token only listens to
// the tokens BEFORE it. Whatever sits after position len - 1
// cannot be heard by positions 0 to len - 1. The position
// vectors do not depend on the length either: row p of the
// table for position p. So the first `len` rows of the result
// are exactly what they would be without the filler. Do not
// take my word for it: the program compares the scores it gets
// from a page of 43 with the ones PyTorch got from 10 tokens.
// (This works for a causal model. BERT, in chapter 18, listens
// in both directions, and needs a real padding mask.)
//
// WHERE ARE THE SCORES OF THE NEXT TOKEN? Not at the end of
// the page any more: at position len - 1, the last token that
// was really written. And `len` changes at every turn of the
// loop, so it cannot be baked into the Exe: it must come in as
// an INPUT, a scalar tensor, and the model must pick a row at
// a position that is only known when it runs. That is a
// `gather` (exercise 070):
//
//     x.gather(.{ .s = last }, .{})     x: {.s, .d}, last: {} i32  ->  {.d}
//
// The gathered axis is replaced by the axes of the indices,
// and a scalar has none: `.s` disappears, like the scalar
// token of the last quiz of chapter 16.
//
// So `forward` gets a second argument, the compile call a
// second placeholder, `eval` a third Buffer:
//
//     const last: zml.Tensor = .init(.{}, .i32);
//     ... platform.compile(gpa, io, model, .forward, .{ tokens, last }, .{});
//     ... zml.Buffer.scalar(io, platform, value, .i32)
//     ... exe.eval(gpa, io, .{ weights, story_buffer, last_buffer });
//
// Picking the token is the greedy choice of exercise 112, done
// on the host this time: read the scores back (`toSliceAlloc`,
// then `.items(f32)`) and take the index of the largest one.
// `std.mem.indexOfMax(f32, values)` does that.
//
// What this costs. Every turn of the loop reads the whole page
// again, 43 tokens to produce one. For Neo it is a blink. For
// a model a thousand times bigger it is the reason exercise 108
// exists: the KV cache, the next thing a serious port adds.
// And a real story hour would stop when Neo writes his
// end-of-text token (50256), or sample instead of always
// taking the favourite (chapter 16). Tonight: greedy, and a
// page of 43 tokens, which happens to end on a full stop.
//
// The porting rule: dynamic lengths in Python become a fixed
// size plus a "how much of it is real" input in ZML. Then ask
// the two questions above. What keeps the filler from leaking
// into the real part, and where do I read the answer?
//
// To do:
//
//  1. `GptNeo.forward` is the one of exercise 132: it reads
//     the end of the page. Give it the second argument
//     described above.
//
//  2. In `main`: the Exe, the page, and the loop. `story` must
//     be a `[max_len]u32`, the prompt first and `blank`
//     everywhere else. At the first turn, when the page holds
//     the prompt and nothing more, the scores are compared
//     with PyTorch's: the print for that is given, it wants
//     the Buffer of scores to be called `logits`.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

const prompt = "Once upon a time, there was a little dragon";

/// The length of the page: the story stops when it is full.
const max_len = 43;

/// What an empty place of the page holds. Any token will do.
const blank: u32 = 0;

/// The keys of `config.json` this program needs.
const Config = struct {
    vocab_size: u32,
    max_position_embeddings: u32,
    hidden_size: u32,
    num_layers: u32,
    num_heads: u32,
    window_size: u32,
    layer_norm_epsilon: f32,
    attention_layers: []const []const u8,
};

/// A Linear from the "weight" and (if there is one) the "bias"
/// found under `view`.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        view.maybeCreateTensor("bias", .{.dout}, .replicated, .{}),
        .d,
    );
}

/// A LayerNorm from the "weight" and the "bias" found under `view`.
fn layerNorm(view: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = view.createReplicatedTensor("weight", .{.d}),
        .bias = view.createReplicatedTensor("bias", .{.d}),
        .eps = eps,
    };
}

const Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    heads: u32,
    window: ?u32,

    /// `view` must stand on a block: "transformer.h.N.".
    pub fn init(view: zml.io.TensorStore.View, config: Config, kind: []const u8) Attention {
        const attention = view.withPrefix("attn").withPrefix("attention");
        return .{
            .q_proj = linear(attention.withPrefix("q_proj")),
            .k_proj = linear(attention.withPrefix("k_proj")),
            .v_proj = linear(attention.withPrefix("v_proj")),
            .out_proj = linear(attention.withPrefix("out_proj")),
            .heads = config.num_heads,
            .window = if (std.mem.eql(u8, kind, "local")) config.window_size else null,
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Attention, x: zml.Tensor) zml.Tensor {
        const split = .{ .h = self.heads, .hd = .auto };
        const q = self.q_proj.forward(x, .f32).splitAxis(.dout, split);
        const k = self.k_proj.forward(x, .f32).splitAxis(.dout, split);
        const v = self.v_proj.forward(x, .f32).splitAxis(.dout, split);

        const heard = zml.nn.sdpa(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{
                .attn_mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, .f32, self.window),
                .scale = .scalar(1, .f32),
            },
        );

        const merged = heard.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.out_proj.forward(merged, .f32).rename(.{ .dout = .d });
    }
};

const Mlp = struct {
    c_fc: zml.nn.Linear,
    c_proj: zml.nn.Linear,

    /// `view` must stand on a block: "transformer.h.N.".
    pub fn init(view: zml.io.TensorStore.View) Mlp {
        return .{
            .c_fc = linear(view.withPrefix("mlp").withPrefix("c_fc")),
            .c_proj = linear(view.withPrefix("mlp").withPrefix("c_proj")),
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const wide = self.c_fc.forward(x, .f32).gelu().rename(.{ .dout = .d });
        return self.c_proj.forward(wide, .f32).rename(.{ .dout = .d });
    }
};

const Block = struct {
    ln_1: zml.nn.LayerNorm,
    attn: Attention,
    ln_2: zml.nn.LayerNorm,
    mlp: Mlp,

    /// `view` must stand on a block: "transformer.h.N.".
    pub fn init(view: zml.io.TensorStore.View, config: Config, kind: []const u8) Block {
        return .{
            .ln_1 = layerNorm(view.withPrefix("ln_1"), config.layer_norm_epsilon),
            .attn = .init(view, config, kind),
            .ln_2 = layerNorm(view.withPrefix("ln_2"), config.layer_norm_epsilon),
            .mlp = .init(view),
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Block, x: zml.Tensor) zml.Tensor {
        const y = x.add(self.attn.forward(self.ln_1.forward(x)));
        return y.add(self.mlp.forward(self.ln_2.forward(y)));
    }
};

const GptNeo = struct {
    wte: zml.nn.TokenEmbedding,
    wpe: zml.Tensor,
    h: []Block,
    ln_f: zml.nn.LayerNorm,

    /// `view` must stand on the root of the file. The blocks are
    /// allocated: the caller frees `h`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !GptNeo {
        const transformer = view.withPrefix("transformer");
        const h = try gpa.alloc(Block, config.num_layers);
        for (h, 0..) |*block, i| {
            block.* = .init(transformer.withPrefix("h").withLayer(i), config, config.attention_layers[i]);
        }
        return .{
            .wte = .{ .weight = transformer.withPrefix("wte").createReplicatedTensor("weight", .{ .voc, .d }) },
            .wpe = transformer.withPrefix("wpe").createReplicatedTensor("weight", .{ .pos, .d }),
            .h = h,
            .ln_f = layerNorm(transformer.withPrefix("ln_f"), config.layer_norm_epsilon),
        };
    }

    /// tokens: {.s} i32 -> what the head receives: {.s, .d}
    pub fn hidden(self: GptNeo, tokens: zml.Tensor) zml.Tensor {
        const positions = self.wpe.slice(.pos, .{ .end = tokens.dim(.s) }).rename(.{ .pos = .s });
        var x = self.wte.forward(tokens).add(positions);
        for (self.h) |block| x = block.forward(x);
        return self.ln_f.forward(x);
    }

    /// tokens: {.s} i32 -> the scores of the token that comes
    /// after the last one: {.voc}
    pub fn forward(self: GptNeo, tokens: zml.Tensor) zml.Tensor {
        const x = self.hidden(tokens);
        const last = x.slice(.s, .{ .start = x.dim(.s) - 1 }).squeeze(.s);
        return last.dot(self.wte.weight, .d);
    }
};

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// The tensor `key` of a safetensors file, as a Buffer.
fn bufferFromStore(gpa: std.mem.Allocator, io: std.Io, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);
    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);
    return .fromBytes(io, platform, shape, bytes);
}

/// Are the two Buffers the same, up to rounding? When they are
/// not, `expectClose` says how far apart on stderr.
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

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const reference_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(reference_path);
    var reference_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, reference_path);
    defer reference_registry.deinit();
    var reference: zml.io.TensorStore = .fromRegistry(gpa, &reference_registry, &.{platform.replicated_mesh});
    defer reference.deinit();

    const tokenizer_bytes = try readFile(gpa, io, dir ++ "tokenizer.json");
    defer gpa.free(tokenizer_bytes);
    var tokenizer: zml.tokenizer.Tokenizer = try .fromBytes(gpa, tokenizer_bytes);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    const model: GptNeo = try .init(gpa, store.view(), config);
    defer gpa.free(model.h);

    var weights = try zml.mem.bufferize(gpa, GptNeo, &model);
    defer gpa.free(weights.h);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(GptNeo, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, GptNeo, &model, &weights, &store, .{});
    try loader.await(io);

    const prompt_ids = try encoder.encodeAlloc(gpa, prompt);
    defer gpa.free(prompt_ids);

    // One Exe for `max_len` tokens. The page, `story`. The loop,
    // until the page is full. At the first turn, and only then,
    // these three lines, right after the run:
    //
    //     var want = try bufferFromStore(gpa, io, platform, &reference, "last_logits");
    //     defer want.deinit();
    //     try stdout.print("{d} tokens on a page of {d}, the scores match PyTorch: {}\n", .{ len, max_len, matches(io, want, logits) });
    ???;

    var text = try decoder.decodeAlloc(gpa, &story);
    defer text.deinit(gpa);
    try stdout.print("{s}\n", .{text.items});
    try stdout.flush();
}
