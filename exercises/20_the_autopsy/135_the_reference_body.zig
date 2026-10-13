//
// Welcome to the County Morgue for Models. Dr. Mortis, the
// coroner, examines ports that died. Not the loud deaths: a
// port that panics has left a note. Hers are the quiet ones.
// They compile, they run, and they say "the dragon dragon
// dragon dragon". Nobody is blamed here. Every port dies of
// something small, and her job (yours, from today) is to
// find WHERE.
//
// Reading a dead port from top to bottom, hoping to see the
// mistake, does not work: a wrong line looks exactly like a
// right one. What works is a healthy body to compare with,
// ORGAN BY ORGAN. PyTorch has it. Chapter 19's
// `activations.safetensors` is that body, and since a port of
// your own will need one, here is how it is made:
//
//     acts = {}
//     def hook(name):
//         def fn(_module, inputs, output):
//             acts[name + ".in"] = inputs[0][0].detach().clone()
//             if isinstance(output, tuple): output = output[0]
//             acts[name + ".out"] = output[0].detach().clone()
//         return fn
//
//     for name, module in model.named_modules():
//         module.register_forward_hook(hook(name))
//     model(input_ids=ids[None])        # ONE real prompt
//     save_file(acts, "activations.safetensors")
//
// (The real script, with its guards, is
// `19_once_upon_a_time/make_fixtures.py`.) A forward hook is a
// function PyTorch calls after a module has run, with what
// went in and what came out. `named_modules()` lists every
// module of the tree with its dotted name, the same names as
// in the weights file. So each module leaves two tensors:
//
//     transformer.h.0.ln_1.in     {10, 64}   what it was given
//     transformer.h.0.ln_1.out    {10, 64}   what it answered
//
// The `[0]` after `inputs[0]` and `output` drops the batch axis:
// PyTorch ran a batch of one prompt, your port has no batch.
// 10 is the prompt, "Once upon a time, there was a little
// dragon". The file also has `input_ids` and `last_logits`.
//
// With that file, a layer can be tested ALONE: give it the
// `.in` PyTorch gave it, compare what it answers with `.out`.
// No other layer is involved, so no other layer can be blamed.
// ZML's own model tests do exactly this, by hand (see
// `zml/examples/llm/models/lfm2_tests.zig`):
//
//  1. read `name.in` and `name.out` from the file (the given
//     `reference`);
//  2. make a placeholder with the shape of `.in`. A file knows
//     no tags, your layer wants some:
//
//     zml.Tensor.fromShape(in.shape()).withTags(.{ .s, .d })
//
//  3. compile the layer alone, `platform.compile(gpa, io,
//     layer, .forward, .{x}, .{})`, and run it with ITS
//     weights and the `.in` Buffer (exercise 101);
//  4. compare:
//
//     zml.testing.expectClose(io, want, got, opts) !void
//
//    Buffers or Slices. It returns an error when fewer than
//    99.9% of the numbers are close, "close" meaning
//    |want - got| <= 1e-3 + 1e-2 * the larger of the two
//    (those are the defaults of `zml.testing.CompareOpts`,
//    `.{}`). Before the error, it prints a report on stderr.
//    You will read one in the next exercise. Here a verdict
//    is enough:
//
//     const ok =
//         if (zml.testing.expectClose(...)) |_| true else |_| false;
//
// One condition, and it is the rule of this exercise:
//
//     A LAYER CAN BE TESTED ALONE ONLY IF IT IS A STRUCT WITH
//     ITS OWN `forward`. SHAPE YOUR PORT LIKE THE MODULE TREE.
//
// Your block of chapter 19 was one flat `forward`. It was
// right, and it could only be tested whole. The same model is
// below, cut along PyTorch's modules: `Block` holds an
// `Attention` and an `Mlp`, as `GPTNeoBlock` holds
// `attn.attention` and `mlp`. Every hook name now has a
// struct to answer for it. (`Linear` wraps `zml.nn.Linear` so
// that its `forward` takes one tensor and gives `.d` back.)
//
// To do:
//
//  1. Write `runAlone`: steps 2 and 3. Its `anytype` is plain
//     Zig: one function, compiled once per type of layer.
//
//  2. Write `examine`: steps 1 and 4 around `runAlone`, then
//     the given `verdict`. The name arrives as a format and
//     its arguments, as for `print`; `std.fmt.bufPrint(&buf,
//     fmt, args)` writes it in a buffer and returns the slice.
//     Float layers of this model all take `.{ .s, .d }`.
//
//  3. In `main`, Dr. Mortis starts with a healthy patient,
//     your port: the first examination is there, add the
//     others. The last line is her control: block 0 passed
//     off as block 1. A test that cannot say no is not a test.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

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

const GptNeo = struct {
    wte: zml.nn.TokenEmbedding,
    wpe: zml.Tensor,
    h: []Block,
    ln_f: zml.nn.LayerNorm,

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
};

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

    /// One tensor of the healthy body, as a Buffer. Yours to deinit.
    fn reference(m: Morgue, key: []const u8) !zml.Buffer {
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

    fn verdict(m: Morgue, name: []const u8, ok: bool) !void {
        try m.stdout.print("{s}: {s}\n", .{ name, if (ok) "ok" else "DIVERGES" });
    }

    /// Compiles `layer.forward` on its own, and runs it once on
    /// `in`, whose axes get `tags`. The result is yours to deinit.
    fn runAlone(
        m: Morgue,
        layer: anytype,
        weights: zml.Bufferized(@TypeOf(layer)),
        in: zml.Buffer,
        tags: anytype,
    ) !zml.Buffer {
        return ???;
    }

    /// Feeds `layer` what PyTorch fed the module called `name`,
    /// compares the two answers, prints the verdict, returns it.
    fn examine(
        m: Morgue,
        comptime fmt: []const u8,
        args: anytype,
        layer: anytype,
        weights: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
    ) !bool {
        return ???;
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

    const morgue: Morgue = .{ .gpa = gpa, .io = io, .platform = platform, .acts = &acts, .stdout = stdout };

    // The weights of a part are the same part of the weights.
    const block = model.h[0];
    const block_weights = weights.h[0];
    _ = try morgue.examine("transformer.h.{d}.ln_1", .{0}, block.ln_1, block_weights.ln_1, .{});
    // The same for, in this order (they are all in the file):
    //   transformer.h.0.attn.attention
    //   transformer.h.0.attn.attention.q_proj
    //   transformer.h.0.mlp
    //   transformer.h.0            the whole block
    //   transformer.ln_f

    try stdout.print("The control, block 0 on the table with the papers of block 1:\n", .{});
    const fooled = try morgue.examine("transformer.h.{d}", .{1}, block, block_weights, .{});
    try stdout.print("Dr. Mortis was fooled: {}\n", .{fooled});
    try stdout.flush();
}
