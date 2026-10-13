//
// Second patient. Rigor reads the form: "It tells a story!
// The dragon is excited to explore the world. Twice. Then
// the sun is shining and the sun, and it was a little girl."
// Almost alive. Dr. Mortis has already walked the pipe: every
// block is sick in itself. Block 0 is open on the table, and
// `main` examines its parts, then the leaves of the sick one:
//
//     transformer.h.0.attn.attention            DIVERGES
//     transformer.h.0.attn.attention.q_proj     ok
//     transformer.h.0.attn.attention.k_proj     ok
//     transformer.h.0.attn.attention.v_proj     ok
//     transformer.h.0.attn.attention.out_proj   ok
//
// Four healthy organs in a sick body. The rule:
//
//     ALL THE CHILDREN ARE OK AND THE PARENT DIVERGES:
//     THE BUG IS IN THE GLUE BETWEEN THE CHILDREN.
//
// The glue is everything in a `forward` that is not a module:
// a `+`, a reshape, a `torch.matmul`, a `softmax`. Hooks only
// see modules, so the glue left nothing in the file. Here it
// is, in `GPTNeoSelfAttention` (names shortened):
//
//     q = self._split_heads(self.q_proj(hidden_states), ...)
//     k, v = ...                         # the same
//     w = torch.matmul(q, k.transpose(-1, -2))
//     w = torch.where(causal_mask, w, mask_value)
//     w = nn.functional.softmax(w, dim=-1)
//     w = self.attn_dropout(w)
//     out = torch.matmul(w, v)
//     out = self.out_proj(self._merge_heads(out, ...))
//
// Short glue is read line by line against the port, with the
// list of what goes wrong in an attention: a scale that
// should not be there (or is missing), no mask, key and
// value swapped, heads split as {hd, h}, heads merged in
// another order than they were split.
//
// Longer glue is cut in two with a PROBE: one more reference
// tensor, taken in the middle. Three ways to get one:
//
//  - a module that sits there and does nothing. Outside of
//    training a dropout returns its input. `attn_dropout`
//    stands between the softmax and the second matmul: what
//    goes INTO it is the attention weights, {heads, q, k};
//  - redo the lines yourself in the script, from tensors the
//    hooks did catch (`q_proj.out`, `k_proj.out`), and save;
//  - ask the model: many return their internals on demand
//    (`output_attentions=True`, `output_hidden_states=True`).
//
// `make_fixtures.py`, next to this file, uses the first one:
//
//     probe = "transformer.h.0.attn.attention.attn_dropout"
//     def hook(_module, inputs, _output):
//         probes[probe + ".in"] = inputs[0][0].detach().clone()
//     model.get_submodule(probe).register_forward_hook(hook)
//
// On the ZML side, a probe needs a function that STOPS there.
// That is why this port has its attention in two methods:
// `probs` goes from x to the weights, `forward` does the rest.
// `zml.nn.sdpa` is a single call. Nothing can be compared in
// the middle of it. To probe it, you unroll it, as here.
//
// A file knows no tags, only an order of axes: PyTorch's. So
// `probs` ends with a `transpose` to {h, q, k}. Inside the
// model the order never mattered, the tags did the work.
//
// For a quick look, there is also `Tensor.print`:
//
//     pub fn print(input: Tensor, name: []const u8) void
//
// Call it on any tensor inside a `forward`. Each time the Exe
// runs, it writes the name, the shape and the (abridged)
// values on stderr, once per device of the mesh. It shows
// that row 0 of the weights is `1, 0, 0, ...` or that there
// are NaNs. It cannot tell if 1600 floats are the right ones.
//
// To do:
//
//  1. In `main`, the examination of the probe. `examine` only
//     knows `.forward` and the main file: this one is by hand,
//     as in exercise 135. Compile `attn.probs`, feed it the
//     tensor `transformer.h.0.attn.attention.in`, compare with
//     `morgue.probe(probe_key)`, give the verdict.
//
//  2. Run it. The report of the probe says `close_fraction:
//     0.50`. Half of the weights are right! Which half? (What
//     is above the diagonal of a causal attention?) So the
//     mask is innocent, and so is everything after the probe.
//
//  3. Fix `probs`. Then remember this patient each time you
//     call `zml.nn.sdpa`: its default does the same thing,
//     unless `.scale` says otherwise (exercises 105 and
//     130). A default, or "as in the paper", is a guess
//     about your model. The author's code decides.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";
const probes_path = "zmling/20_the_autopsy/probes.safetensors";

const Config = struct {
    num_layers: u32,
    num_heads: u32,
    window_size: u32,
    layer_norm_epsilon: f32,
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

    fn splitHeads(self: Attention, x: zml.Tensor) zml.Tensor {
        return x.splitAxis(.d, .{ .h = self.heads, .hd = .auto });
    }

    /// How much each token listens to each other one.
    /// x: {.s, .d} -> {.h, .q, .k}
    pub fn probs(self: Attention, x: zml.Tensor) zml.Tensor {
        const q = self.splitHeads(self.q_proj.forward(x)).rename(.{ .s = .q });
        const k = self.splitHeads(self.k_proj.forward(x)).rename(.{ .s = .k });
        // "Scaled dot-product attention", as in the paper.
        const head_dim: f32 = @floatFromInt(q.dim(.hd));
        var scores = q.dot(k, .hd).scale(1.0 / @sqrt(head_dim));
        const mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, .f32, self.window);
        scores = scores.add(mask.broad(scores.shape()));
        // The order of the axes only matters to a file: PyTorch's.
        return scores.softmax(.k).transpose(.{ .h, .q, .k });
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Attention, x: zml.Tensor) zml.Tensor {
        const v = self.splitHeads(self.v_proj.forward(x)).rename(.{ .s = .k });
        const heard = self.probs(x).dot(v, .k).transpose(.{ .q, .h, .hd });
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
            .ln_1 = layerNorm(view.withPrefix("ln_1"), config.layer_norm_epsilon),
            .attn = .init(view.withPrefix("attn").withPrefix("attention"), config, kind),
            .ln_2 = layerNorm(view.withPrefix("ln_2"), config.layer_norm_epsilon),
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
            .ln_f = layerNorm(transformer.withPrefix("ln_f"), config.layer_norm_epsilon),
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

const probe_key = "transformer.h.0.attn.attention.attn_dropout.in";

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
    /// A few more organs: `probes.safetensors`.
    probes: *zml.io.TensorStore,
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
        return m.fromStore(m.acts, key);
    }

    /// The same, from the file of probes.
    fn probe(m: *Morgue, key: []const u8) !zml.Buffer {
        return m.fromStore(m.probes, key);
    }

    fn fromStore(m: *Morgue, store: *zml.io.TensorStore, key: []const u8) !zml.Buffer {
        const shape = store.view().getShape(key) orelse {
            std.debug.print("The reference has no tensor named {s}\n", .{key});
            return error.NotFound;
        };
        const bytes = try m.gpa.alloc(u8, shape.byteSize());
        defer m.gpa.free(bytes);
        var io_buffer: [4096]u8 = undefined;
        var reader = try store.view().getReader(key, m.io, &io_buffer);
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

    const probes_abs = try std.Io.Dir.cwd().realPathFileAlloc(io, probes_path, gpa);
    defer gpa.free(probes_abs);
    var probes_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, probes_abs);
    defer probes_registry.deinit();
    var probes: zml.io.TensorStore = .fromRegistry(gpa, &probes_registry, &.{platform.replicated_mesh});
    defer probes.deinit();

    var morgue: Morgue = .{ .gpa = gpa, .io = io, .platform = platform, .acts = &acts, .probes = &probes, .stdout = stdout };

    const attn = model.h[0].attn;
    const attn_weights = weights.h[0].attn;

    try morgue.note("Block 0, part by part:\n", .{});
    _ = try morgue.examine("transformer.h.{d}.ln_1", .{0}, model.h[0].ln_1, weights.h[0].ln_1, .{});
    _ = try morgue.examine("transformer.h.{d}.attn.attention", .{0}, attn, attn_weights, .{});
    _ = try morgue.examine("transformer.h.{d}.ln_2", .{0}, model.h[0].ln_2, weights.h[0].ln_2, .{});
    _ = try morgue.examine("transformer.h.{d}.mlp", .{0}, model.h[0].mlp, weights.h[0].mlp, .{});

    try morgue.note("The attention of block 0, leaf by leaf:\n", .{});
    _ = try morgue.examine("transformer.h.{d}.attn.attention.q_proj", .{0}, attn.q_proj, attn_weights.q_proj, .{});
    _ = try morgue.examine("transformer.h.{d}.attn.attention.k_proj", .{0}, attn.k_proj, attn_weights.k_proj, .{});
    _ = try morgue.examine("transformer.h.{d}.attn.attention.v_proj", .{0}, attn.v_proj, attn_weights.v_proj, .{});
    _ = try morgue.examine("transformer.h.{d}.attn.attention.out_proj", .{0}, attn.out_proj, attn_weights.out_proj, .{});

    try morgue.note("The probe, in the middle of the glue:\n", .{});
    // `attn.probs`, alone, on the input of the attention of block
    // 0, against the probe. One verdict, under the name `probe_key`.
    ???

    try morgue.wholeBody(model, weights);
    try stdout.flush();
    // A sick port leaves like a failed test, with an error code:
    // that is when the runner shows you stderr.
    if (morgue.sick > 0) std.process.exit(1);
}
