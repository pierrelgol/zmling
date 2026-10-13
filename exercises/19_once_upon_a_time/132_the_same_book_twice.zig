//
// All the parts are on the table, and each of them has been
// checked against the original. What is left is the binding:
// the model from the first token id to the scores of the next
// token. Here is the whole `forward` of GPT-Neo, with the
// options a port does not need taken out:
//
//     inputs_embeds = self.wte(input_ids)
//     position_embeds = self.wpe(position_ids)
//     hidden_states = inputs_embeds + position_embeds
//     for i, block in enumerate(self.h):
//         hidden_states = block(hidden_states, ...)
//     hidden_states = self.ln_f(hidden_states)
//
// and, one class above, in `GPTNeoForCausalLM`:
//
//     logits = self.lm_head(hidden_states)
//
// Three things to get right. None is hard. Each is forgotten
// in one port out of two.
//
// 1. THE STACK. `self.h` is a list of 8 blocks, and 8 comes
//    from the config (`num_layers`). In Zig a list whose
//    length is known at run time is a slice, and a model
//    struct may hold one: `h: []Block`, allocated in `init`,
//    one `Block.init` per layer with `withLayer(i)`, and block
//    `i` is global or local according to
//    `config.attention_layers[i]`.
//
//    A slice in the model means a slice in its weights:
//    `zml.mem.bufferize` allocates a `[]Bufferized(Block)` of
//    the same length. Two things to free, then, in the right
//    order: the Buffers (device memory), and then the slice
//    that held them (host memory). `Buffer.deinitAll` only
//    handles structs without slices. The one that walks
//    through slices has a name that says exactly what it does
//    and what it leaves to you:
//
//     var weights = try zml.mem.bufferize(gpa, GptNeo, &model);
//     defer gpa.free(weights.h);
//     defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(GptNeo, &weights);
//
//    (Defers run last to first: the Buffers go, then the
//    slice. And `model.h` is yours to free as well.)
//
// 2. THE LAST NORM. In a pre-norm model the main line is never
//    normalized (exercise 131): every block washes a COPY.
//    So there is one more LayerNorm after the last block,
//    `ln_f`, "f" for final. Forget it and the head receives
//    numbers of the wrong size. It is one line in PyTorch, far
//    from the blocks, and it has no counterpart in a post-norm
//    model like BERT. Look for it in every decoder you port.
//
// 3. THE HEAD. `lm_head` is a Linear without bias from 64
//    numbers to 50257 scores, and exercise 128 found that the
//    file has no `lm_head.weight`. The class says why:
//
//     _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
//
//    The head IS the token table. A Linear without bias is a
//    dot product with each row of its weight; the rows of
//    `wte` are the vectors of the tokens, {.voc, .d}. So the
//    score of a token is the dot product of the final vector
//    with that token's own embedding:
//
//     x.dot(self.wte.weight, .d)        {.d} with {.voc, .d} -> {.voc}
//
//    No second tensor, no second copy in memory, and the same
//    `self.wte.weight` used at both ends of the model. Pip
//    declared an `lm_head` field, because the PyTorch class
//    has one. Fields follow the FILE, not the class.
//
// One decision is ours. PyTorch returns scores for every
// position: 10 tokens in, 10 rows of 50257 scores out, row i
// being the guess for token i+1. To continue a story only the
// last row matters. So `forward` keeps the last position
// (`slice` along `.s`, then `squeeze`, exercises 054 and 040)
// BEFORE the head: there is no point in computing half a
// million scores to throw nine tenths of them away.
//
// Two checks, inside out as always: `hidden`, the model up to
// and including `ln_f`, against what the reference calls
// `lm_head.in`; then the scores, against `last_logits`.
//
// And then, at last, something to read. The three tokens Neo
// likes best after "Once upon a time, there was a little
// dragon", decoded with the other half of the tokenizer:
//
//     var decoder = try tokenizer.decoder();
//     defer decoder.deinit();
//     var text = try decoder.decodeAlloc(gpa, ids);   // std.ArrayList(u8)
//     defer text.deinit(gpa);
//
// (`tokenizer.decode(id)` of exercise 129 gives the raw piece,
// with its `Ġ`. The decoder gives text, with a real space.)
//
// To do:
//
//  1. `GptNeo` panics as soon as it is built. Read the
//     message, and exercise 128. The fix removes more than it
//     adds, and `forward` must do without what you removed.
//
//  2. `GptNeo.init` has no blocks.
//
//  3. `hidden`: Pip stopped one line early. The first check
//     will tell you, once the program runs that far.
//
//  4. In `main`: the model, its weights, and the freeing of
//     both.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

const prompt = "Once upon a time, there was a little dragon";

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
    lm_head: zml.nn.Linear,

    /// `view` must stand on the root of the file. The blocks are
    /// allocated: the caller frees `h`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !GptNeo {
        const transformer = view.withPrefix("transformer");
        const h = try gpa.alloc(Block, config.num_layers);
        // Every block of the stack, each of its own kind.
        ???;
        return .{
            .wte = .{ .weight = transformer.withPrefix("wte").createReplicatedTensor("weight", .{ .voc, .d }) },
            .wpe = transformer.withPrefix("wpe").createReplicatedTensor("weight", .{ .pos, .d }),
            .h = h,
            .ln_f = layerNorm(transformer.withPrefix("ln_f"), config.layer_norm_epsilon),
            .lm_head = linear(view.withPrefix("lm_head")),
        };
    }

    /// tokens: {.s} i32 -> what the head receives: {.s, .d}
    pub fn hidden(self: GptNeo, tokens: zml.Tensor) zml.Tensor {
        const positions = self.wpe.slice(.pos, .{ .end = tokens.dim(.s) }).rename(.{ .pos = .s });
        var x = self.wte.forward(tokens).add(positions);
        for (self.h) |block| x = block.forward(x);
        return x;
    }

    /// tokens: {.s} i32 -> the scores of the token that comes
    /// after the last one: {.voc}
    pub fn forward(self: GptNeo, tokens: zml.Tensor) zml.Tensor {
        const x = self.hidden(tokens);
        const last = x.slice(.s, .{ .start = x.dim(.s) - 1 }).squeeze(.s);
        return self.lm_head.forward(last, .f32).rename(.{ .dout = .voc });
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

    // `model`, and `weights`: its Buffers, loaded. Everything that
    // was allocated on the way gets freed, in the right order.
    ???;

    const ids = try encoder.encodeAlloc(gpa, prompt);
    defer gpa.free(ids);
    const tokens: zml.Tensor = .init(.{ .s = ids.len }, .i32);
    var tokens_buffer: zml.Buffer = try .fromBytes(io, platform, tokens.shape(), std.mem.sliceAsBytes(ids));
    defer tokens_buffer.deinit();

    try stdout.print("{d} blocks between the two tables and the head\n", .{model.h.len});

    {
        var exe = try platform.compile(gpa, io, model, .hidden, .{tokens}, .{});
        defer exe.deinit();
        var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, tokens_buffer });
        defer got.deinit();
        var want = try bufferFromStore(gpa, io, platform, &reference, "lm_head.in");
        defer want.deinit();
        try stdout.print("what the head receives matches PyTorch: {}\n", .{matches(io, want, got)});
    }

    var exe = try platform.compile(gpa, io, model, .forward, .{tokens}, .{});
    defer exe.deinit();
    var logits: zml.Buffer = try exe.eval(gpa, io, .{ weights, tokens_buffer });
    defer logits.deinit();
    {
        var want = try bufferFromStore(gpa, io, platform, &reference, "last_logits");
        defer want.deinit();
        try stdout.print("the scores match PyTorch: {}\n", .{matches(io, want, logits)});
    }

    // The three favourites, best first.
    const scores = try logits.toSliceAlloc(gpa, io);
    defer scores.free(gpa);
    const values = scores.items(f32);

    try stdout.print("\"{s}\" and then:\n", .{prompt});
    for (0..3) |rank| {
        const best: u32 = @intCast(std.mem.indexOfMax(f32, values));
        var text = try decoder.decodeAlloc(gpa, &.{best});
        defer text.deinit(gpa);
        try stdout.print("  {d}. \"{s}\"\n", .{ rank + 1, text.items });
        values[best] = -std.math.inf(f32);
    }
    try stdout.flush();
}
