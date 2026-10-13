//
// Fourth patient, and Rigor is uneasy. "Doctor, this one is
// not dead. Listen." And the patient says, word for word, what
// PyTorch says: the dragon, the big castle, the family, too
// hungry to eat. The form explains why it is here anyway:
//
//     last_logits: DIVERGES        max_absolute_error: 0.27
//
// on scores that go from -20 to 20. "It tells the right
// story" is the weakest test there is. Greedy decoding keeps
// the largest of 50257 scores: they can all move a little and
// the same token still wins. Until two candidates are close.
// Then the other one wins, and from that token on it is
// another story. A longer text, another prompt, a sampler
// instead of an argmax, thirty blocks instead of eight: it
// will come out, in production, far from this morgue.
//
//     A PORT IS RIGHT WHEN ITS NUMBERS ARE RIGHT.
//
// The default walk is in `main`: it will be the first thing
// you read once the program compiles. The first blocks: ok.
// The last one and `ln_f`: DIVERGES. Yet the last block has
// the code of the first. And inside it one leaf is called
// sick, `ln_2`, while `ln_1`, the same library code from the
// same helper, is ok.
//
// The walk did not find where the bug is. It found where the
// bug gets LOUD enough for the default tolerance:
//
//     |want - got| <= 1e-3 + 1e-2 * max(|want|, |got|)
//
// One percent. That is made for the far end of a big model in
// half precision. Here, in f32 and one layer at a time, a
// healthy layer of this model is 1e-7 to 1e-6 away from
// PyTorch (we measured). Anything between 1e-6 and 1e-2 is
// called "ok", and is not.
//
//     KNOW THE NOISE OF A HEALTHY LAYER. SET THE TOLERANCE
//     JUST ABOVE IT. AN ERROR A THOUSAND TIMES THE NOISE IS
//     NOT NOISE, WHATEVER THE VERDICT SAYS.
//
// `zml.testing.CompareOpts` has four fields, all with defaults:
//
//     absolute_tolerance: f32 = 1e-3,
//     relative_tolerance: f32 = 1e-2,
//     minimum_close_fraction: f32 = 0.999,
//     epsilon_relative: f32 = 1e-6,       (for the report only)
//
// For this chapter, strict means 1e-5 absolute and 1e-4
// relative. It is for ONE layer in isolation. Noise adds up
// along the pipe: after eight blocks, the logits of a healthy
// port are 5e-5 off, a hundred times more than one layer.
// A tolerance is chosen for a depth.
//
// Small errors, and where they come from:
//
//  - the `eps` of a norm, 1 / sqrt(variance + eps). Harmless
//    while the variance is large, louder as it shrinks.
//  - an approximated function. `Tensor.gelu()` is the tanh
//    formula (Hugging Face: "gelu_new", "gelu_pytorch_tanh").
//    "gelu" alone is the exact one, with `erf` (chapter 18).
//    Swap them in this model: 2.5e-5 on one MLP, 4.6e-3 on
//    the logits, the same story. It passes EVERY default
//    check of this chapter. Only strict sees it, on the MLP.
//  - precision: weights stored in f16, a softmax done in f16.
//
// And where does a wrong constant come from? Rarely from a
// `forward`. Look at how this port reads `config.json`, and
// at the options of `parseFromSlice` in `main`. A key that is
// not in the struct is ignored. A field that is not in the
// file takes its default. Put the two together:
//
//     A FIELD WITH A DEFAULT CANNOT FAIL TO LOAD. KEEP
//     DEFAULTS FOR KEYS THAT REAL CONFIG FILES LEAVE OUT.
//
// To do:
//
//  1. `strict`, in `main`.
//
//  2. The strict walk: the eight blocks and `ln_f`, as the
//     default one above it. Then `openBlock` on block 0,
//     strict too. The first block, where the default
//     tolerance saw nothing at all.
//
//  3. Read. Which leaves are sick, in block 0 already? What
//     do they share? Fix it where it comes from, and in such
//     a way that the same mistake would now be an error
//     (`error.MissingField`) instead of a quiet number.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

const Config = struct {
    num_layers: u32,
    num_heads: u32,
    window_size: u32,
    layer_norm_eps: f32 = 1e-12,
    attention_layers: []const []const u8,
};

fn layerNorm(view: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = view.createReplicatedTensor("weight", .{.d}),
        .bias = view.createReplicatedTensor("bias", .{.d}),
        .eps = eps,
    };
}

/// PyTorch's `nn.Linear`, as a layer that can be examined alone.
const Linear = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,

    pub fn init(view: zml.io.TensorStore.View) Linear {
        return .{
            .weight = view.createReplicatedTensor("weight", .{ .dout, .d }),
            // q_proj, k_proj and v_proj have no bias in this checkpoint.
            .bias = view.maybeCreateTensor("bias", .{.dout}, .replicated, .{}),
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Linear, x: zml.Tensor) zml.Tensor {
        return zml.nn.Linear.init(self.weight, self.bias, .d).forward(x, .f32).rename(.{ .dout = .d });
    }
};

/// `transformer.h.N.attn.attention`
const Attention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: u32,
    /// "local" layers only listen to the last `window` tokens.
    window: ?u32,

    pub fn init(view: zml.io.TensorStore.View, config: Config, kind: []const u8) Attention {
        return .{
            .q_proj = .init(view.withPrefix("q_proj")),
            .k_proj = .init(view.withPrefix("k_proj")),
            .v_proj = .init(view.withPrefix("v_proj")),
            .out_proj = .init(view.withPrefix("out_proj")),
            .heads = config.num_heads,
            .window = if (std.mem.eql(u8, kind, "local")) config.window_size else null,
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Attention, x: zml.Tensor) zml.Tensor {
        const split = .{ .h = self.heads, .hd = .auto };
        const q = self.q_proj.forward(x).splitAxis(.d, split).rename(.{ .s = .q });
        const k = self.k_proj.forward(x).splitAxis(.d, split).rename(.{ .s = .k });
        const v = self.v_proj.forward(x).splitAxis(.d, split).rename(.{ .s = .k });
        const heard = zml.nn.sdpa(q, k, v, .{
            .attn_mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, .f32, self.window),
            // GPT-Neo does not divide the scores by sqrt(head_dim).
            .scale = .scalar(1, .f32),
        });
        return self.out_proj.forward(heard.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s }));
    }
};

/// `transformer.h.N.mlp`
const Mlp = struct {
    c_fc: Linear,
    c_proj: Linear,

    pub fn init(view: zml.io.TensorStore.View) Mlp {
        return .{ .c_fc = .init(view.withPrefix("c_fc")), .c_proj = .init(view.withPrefix("c_proj")) };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.c_proj.forward(self.c_fc.forward(x).gelu());
    }
};

/// `transformer.h.N`
const Block = struct {
    ln_1: zml.nn.LayerNorm,
    attn: Attention,
    ln_2: zml.nn.LayerNorm,
    mlp: Mlp,

    pub fn init(view: zml.io.TensorStore.View, config: Config, kind: []const u8) Block {
        return .{
            .ln_1 = layerNorm(view.withPrefix("ln_1"), config.layer_norm_eps),
            .attn = .init(view.withPrefix("attn").withPrefix("attention"), config, kind),
            .ln_2 = layerNorm(view.withPrefix("ln_2"), config.layer_norm_eps),
            .mlp = .init(view.withPrefix("mlp")),
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Block, x: zml.Tensor) zml.Tensor {
        // Pre-norm: normalize, THEN the sublayer, then add.
        const y = x.add(self.attn.forward(self.ln_1.forward(x)));
        return y.add(self.mlp.forward(self.ln_2.forward(y)));
    }
};

/// The sum PyTorch feeds to `transformer.drop`.
const Embeddings = struct {
    wte: zml.nn.TokenEmbedding,
    wpe: zml.Tensor,

    /// tokens: {.s} i32 -> {.s, .d}
    pub fn forward(self: Embeddings, tokens: zml.Tensor) zml.Tensor {
        const positions = self.wpe.slice(.pos, .{ .end = tokens.dim(.s) }).rename(.{ .pos = .s });
        return self.wte.forward(tokens).add(positions);
    }
};

const GptNeo = struct {
    embeddings: Embeddings,
    h: []Block,
    ln_f: zml.nn.LayerNorm,

    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !GptNeo {
        const transformer = view.withPrefix("transformer");
        const h = try gpa.alloc(Block, config.num_layers);
        for (h, 0..) |*block, i| {
            block.* = .init(transformer.withPrefix("h").withLayer(i), config, config.attention_layers[i]);
        }
        return .{
            .embeddings = .{
                .wte = .{ .weight = transformer.withPrefix("wte").createReplicatedTensor("weight", .{ .voc, .d }) },
                .wpe = transformer.withPrefix("wpe").createReplicatedTensor("weight", .{ .pos, .d }),
            },
            .h = h,
            .ln_f = layerNorm(transformer.withPrefix("ln_f"), config.layer_norm_eps),
        };
    }

    /// tokens: {.s} i32, last: {} i32 -> scores of the token after position `last`: {.voc}
    pub fn forward(self: GptNeo, tokens: zml.Tensor, last: zml.Tensor) zml.Tensor {
        var x = self.embeddings.forward(tokens);
        for (self.h) |block| x = block.forward(x);
        x = self.ln_f.forward(x).gather(.{ .s = last }, .{});
        // Tied: there is no `lm_head.weight` in the file, it is `wte`.
        return x.dot(self.embeddings.wte.weight, .d);
    }
};

/// ZML logs a warning on stderr at each compile. Errors only:
/// the reports of `expectClose` are easier to find.
pub const std_options: std.Options = .{ .log_level = .err };

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// Dr. Mortis's table: everything an examination needs.
const Morgue = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    /// The healthy body: `activations.safetensors`.
    acts: *zml.io.TensorStore,
    stdout: *std.Io.Writer,
    /// How many verdicts were bad.
    sick: u32 = 0,

    /// One line of the report: on stdout for the runner, and on
    /// stderr for you, in between the reports of `expectClose`.
    fn note(m: *Morgue, comptime fmt: []const u8, args: anytype) !void {
        try m.stdout.print(fmt, args);
        std.debug.print(fmt, args);
    }

    fn verdict(m: *Morgue, name: []const u8, ok: bool) !void {
        if (!ok) m.sick += 1;
        try m.note("{s}: {s}\n", .{ name, if (ok) "ok" else "DIVERGES" });
    }

    /// One tensor of the healthy body, as a Buffer. Yours to deinit.
    fn reference(m: *Morgue, key: []const u8) !zml.Buffer {
        const shape = m.acts.view().getShape(key) orelse {
            std.debug.print("The reference has no tensor named {s}\n", .{key});
            return error.NotFound;
        };
        const bytes = try m.gpa.alloc(u8, shape.byteSize());
        defer m.gpa.free(bytes);
        var io_buffer: [4096]u8 = undefined;
        var reader = try m.acts.view().getReader(key, m.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return .fromBytes(m.io, m.platform, shape, bytes);
    }

    /// Compiles `layer.forward` on its own, and runs it once on
    /// `in`, whose axes get `tags`. The result is yours to deinit.
    fn runAlone(
        m: *Morgue,
        layer: anytype,
        weights: zml.Bufferized(@TypeOf(layer)),
        in: zml.Buffer,
        tags: anytype,
    ) !zml.Buffer {
        const x = zml.Tensor.fromShape(in.shape()).withTags(tags);
        var exe = try m.platform.compile(m.gpa, m.io, layer, .forward, .{x}, .{});
        defer exe.deinit();
        return exe.eval(m.gpa, m.io, .{ weights, in });
    }

    /// Is `got` the tensor `key` of the healthy body? Prints the
    /// verdict under the name `key`, and returns it.
    fn compare(m: *Morgue, key: []const u8, got: zml.Buffer, opts: zml.testing.CompareOpts) !bool {
        var want = try m.reference(key);
        defer want.deinit();
        const ok = if (zml.testing.expectClose(m.io, want, got, opts)) |_| true else |_| false;
        try m.verdict(key, ok);
        return ok;
    }

    /// Feeds `layer` what PyTorch fed the module called `name`,
    /// compares the two answers, prints the verdict, returns it.
    fn examine(
        m: *Morgue,
        comptime fmt: []const u8,
        args: anytype,
        layer: anytype,
        weights: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
    ) !bool {
        var name_buffer: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, fmt, args);
        var key_buffer: [128]u8 = undefined;

        var in = try m.reference(try std.fmt.bufPrint(&key_buffer, "{s}.in", .{name}));
        defer in.deinit();
        var want = try m.reference(try std.fmt.bufPrint(&key_buffer, "{s}.out", .{name}));
        defer want.deinit();
        var got = try m.runAlone(layer, weights, in, .{ .s, .d });
        defer got.deinit();

        const ok = if (zml.testing.expectClose(m.io, want, got, opts)) |_| true else |_| false;
        try m.verdict(name, ok);
        return ok;
    }

    /// The complete examination of block `i`: its four parts and
    /// their six leaves, ten verdicts (you wrote it in exercise 138).
    fn openBlock(m: *Morgue, i: usize, block: Block, weights: zml.Bufferized(Block), opts: zml.testing.CompareOpts) !void {
        _ = try m.examine("transformer.h.{d}.ln_1", .{i}, block.ln_1, weights.ln_1, opts);
        _ = try m.examine("transformer.h.{d}.attn.attention", .{i}, block.attn, weights.attn, opts);
        _ = try m.examine("transformer.h.{d}.attn.attention.q_proj", .{i}, block.attn.q_proj, weights.attn.q_proj, opts);
        _ = try m.examine("transformer.h.{d}.attn.attention.k_proj", .{i}, block.attn.k_proj, weights.attn.k_proj, opts);
        _ = try m.examine("transformer.h.{d}.attn.attention.v_proj", .{i}, block.attn.v_proj, weights.attn.v_proj, opts);
        _ = try m.examine("transformer.h.{d}.attn.attention.out_proj", .{i}, block.attn.out_proj, weights.attn.out_proj, opts);
        _ = try m.examine("transformer.h.{d}.ln_2", .{i}, block.ln_2, weights.ln_2, opts);
        _ = try m.examine("transformer.h.{d}.mlp", .{i}, block.mlp, weights.mlp, opts);
        _ = try m.examine("transformer.h.{d}.mlp.c_fc", .{i}, block.mlp.c_fc, weights.mlp.c_fc, opts);
        _ = try m.examine("transformer.h.{d}.mlp.c_proj", .{i}, block.mlp.c_proj, weights.mlp.c_proj, opts);
    }

    /// The port as its users meet it: are its scores for the
    /// next token PyTorch's, and what story does it tell?
    fn wholeBody(m: *Morgue, model: GptNeo, weights: zml.Bufferized(GptNeo)) !void {
        const tok_bytes = try readFile(m.gpa, m.io, dir ++ "tokenizer.json");
        defer m.gpa.free(tok_bytes);
        var tokenizer: zml.tokenizer.Tokenizer = try .fromBytes(m.gpa, tok_bytes);
        defer tokenizer.deinit();
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();
        const prompt = try encoder.encodeAlloc(m.gpa, "Once upon a time, there was a little dragon");
        defer m.gpa.free(prompt);

        // One Exe for `max_len` tokens: the story grows inside it,
        // and the causal mask hides the unwritten end (chapter 19).
        const max_len = 50;
        var story: [max_len]u32 = @splat(0);
        @memcpy(story[0..prompt.len], prompt);
        const tokens: zml.Tensor = .init(.{ .s = max_len }, .i32);
        var exe = try m.platform.compile(m.gpa, m.io, model, .forward, .{ tokens, zml.Tensor.init(.{}, .i32) }, .{});
        defer exe.deinit();

        var len: usize = prompt.len;
        while (len < max_len) : (len += 1) {
            var story_buffer: zml.Buffer = try .fromBytes(m.io, m.platform, tokens.shape(), std.mem.sliceAsBytes(&story));
            defer story_buffer.deinit();
            var last_buffer: zml.Buffer = try .scalar(m.io, m.platform, @as(i32, @intCast(len - 1)), .i32);
            defer last_buffer.deinit();
            var logits = try exe.eval(m.gpa, m.io, .{ weights, story_buffer, last_buffer });
            defer logits.deinit();

            if (len == prompt.len) _ = try m.compare("last_logits", logits, .{});

            const scores = try logits.toSliceAlloc(m.gpa, m.io);
            defer scores.free(m.gpa);
            story[len] = @intCast(std.mem.indexOfMax(f32, scores.items(f32)));
        }

        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var text = try decoder.decodeAlloc(m.gpa, &story);
        defer text.deinit(m.gpa);
        try m.note("{s}\n", .{text.items});
    }
};

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

    // The weights, and the healthy body: two files, two stores.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    const model: GptNeo = try .init(gpa, store.view(), config);
    defer gpa.free(model.h);
    var weights = try zml.mem.bufferize(gpa, GptNeo, &model);
    defer gpa.free(weights.h);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(GptNeo, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, GptNeo, &model, &weights, &store, .{});
    try loader.await(io);

    var morgue: Morgue = .{ .gpa = gpa, .io = io, .platform = platform, .acts = &acts, .stdout = stdout };

    try morgue.note("The isolated walk, default tolerance:\n", .{});
    for (model.h, weights.h, 0..) |block, block_weights, i| {
        _ = try morgue.examine("transformer.h.{d}", .{i}, block, block_weights, .{});
    }
    _ = try morgue.examine("transformer.ln_f", .{}, model.ln_f, weights.ln_f, .{});

    try morgue.note("Inside block 7, default tolerance:\n", .{});
    try morgue.openBlock(7, model.h[7], weights.h[7], .{});

    const strict: zml.testing.CompareOpts = ???;
    try morgue.note("The isolated walk, strict:\n", .{});
    // The eight blocks, then `ln_f`.
    ???

    try morgue.note("Inside block 0, strict:\n", .{});
    // One call.
    ???

    try morgue.wholeBody(model, weights);
    try stdout.flush();
    // A sick port leaves like a failed test, with an error code:
    // that is when the runner shows you stderr.
    if (morgue.sick > 0) std.process.exit(1);
}
