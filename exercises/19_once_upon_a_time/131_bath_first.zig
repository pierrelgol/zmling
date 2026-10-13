//
// The rule of the house, for the children as for the vectors:
// bath first, story after.
//
// A transformer block has two things to do to the sequence:
// let the tokens listen to each other (the attention, exercise
// 130), then let each token think alone (a small MLP). Around
// them, two LayerNorms (exercise 091) and two additions. Every
// transformer since 2017 has these six parts. What changes
// from one model to the next is the ORDER, and the order is
// not in the file: a file is a bag of tensors. It is in
// `forward`. Here is the one of `GPTNeoBlock`:
//
//     residual = hidden_states
//     hidden_states = self.ln_1(hidden_states)
//     attn_output, attn_weights = self.attn(hidden_states, ...)
//     hidden_states = attn_output + residual
//
//     residual = hidden_states
//     hidden_states = self.ln_2(hidden_states)
//     feed_forward_hidden_states = self.mlp(hidden_states)
//     hidden_states = residual + feed_forward_hidden_states
//
// Read it slowly. The vector is put aside (`residual`). A
// WASHED copy goes to the attention. What the attention says
// is added to the vector that was put aside, the unwashed
// one. Then the same again with the MLP:
//
//     x = x + attn(ln_1(x))
//     x = x + mlp(ln_2(x))
//
// That is called PRE-NORM: the norm comes before the sublayer,
// and the main line, the "residual stream", is only ever
// added to. The older arrangement, the one of the first
// transformer and of BERT, is POST-NORM:
//
//     x = ln_1(x + attn(x))
//     x = ln_2(x + mlp(x))
//
// Same six parts, same tensors in the file, same shapes. Pip
// ported a BERT last week (chapter 18), and wrote this block
// in the evening.
//
// The MLP is two Linears with an activation in between, and
// it is wider in the middle: 64 numbers, 256, 64 again.
//
//     hidden_states = self.c_fc(hidden_states)
//     hidden_states = self.act(hidden_states)
//     hidden_states = self.c_proj(hidden_states)
//
// And `self.act`? It is not written in the class at all:
//
//     self.act = ACT2FN[config.activation_function]
//
// A lookup in a table, with a key from `config.json`, where
// we find `"activation_function": "gelu_new"`. GELU is a
// smooth cousin of relu (exercise 090), and there are two of
// it in the wild. The exact one, which needs the error
// function `erf`:
//
//     gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// and an approximation with a tanh, which is what "gelu_new"
// means in this library:
//
//     0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi)
//         * (input + 0.044715 * torch.pow(input, 3.0))))
//
// ZML has the second one, `x.gelu()`: its doc comment shows
// that very formula. BERT's "gelu" is the first one, and Pip
// has kept his from last week:
//
//     x.scale(0.5).mul(x.scale(1.0 / @sqrt(2.0)).erf().addConstant(1))
//
// The two curves never differ by more than 0.0005. Now look
// at what `matches` tolerates: 0.001. With everything else
// right, a wrong GELU passes the check of the MLP, and the
// check of the block. (Try it at the end: put Pip's line
// back.) It is still wrong, in every block, at every token,
// and small errors that all lean the same way do not stay
// small.
//
// Two porting rules come out of this exercise.
//
// CHECK FROM THE INSIDE OUT. The reference has every module:
// `transformer.h.0.ln_1`, `.attn`, `.mlp`, and `transformer.h.0`
// itself. When the parts are right and the whole is wrong, the
// bug is in the wiring, and there are six lines of wiring.
//
// A TOLERANCE HIDES SMALL SYSTEMATIC ERRORS. When an op has
// two versions that are "almost the same", test that op alone,
// on what PyTorch fed it (`transformer.h.0.mlp.act.in`), and
// tighten the comparison:
//
//     const strict: zml.testing.CompareOpts = .{
//         .absolute_tolerance = 1e-5,
//         .relative_tolerance = 1e-4,
//     };
//
// The right formula in f32 passes that easily. The cousin
// does not.
//
// `check` is the one you completed in exercise 130, made
// generic: any model type `T` whose `forward` takes one
// {.s, .d} tensor, and the name of a module of the reference.
//
// To do:
//
//  1. `layerNorm`: a `zml.nn.LayerNorm` from a view, like
//     `linear` above it. Its two tensors are called "weight"
//     and "bias" in the file, both with one axis, and the eps
//     is in the config.
//
//  2. `activation`: the strict line says false.
//
//  3. `Mlp.forward`. Mind the `.dout` of each Linear.
//
//  4. `Block.init`: four parts, all found from the view of
//     the block. Look at what `Attention.init` and `Mlp.init`
//     expect.
//
//  5. `Block.forward` is Pip's. `ln_1`, `attn` and `mlp` say
//     true, the whole block does not.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

/// The keys of `config.json` this program needs.
const Config = struct {
    hidden_size: u32,
    num_heads: u32,
    window_size: u32,
    layer_norm_epsilon: f32,
    activation_function: []const u8,
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
    return ???;
}

/// Yours, from exercise 130.
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

/// The activation function of the MLP: `gelu_new`.
fn activation(x: zml.Tensor) zml.Tensor {
    return x.scale(0.5).mul(x.scale(1.0 / @sqrt(2.0)).erf().addConstant(1));
}

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
        return ???;
    }
};

const Block = struct {
    ln_1: zml.nn.LayerNorm,
    attn: Attention,
    ln_2: zml.nn.LayerNorm,
    mlp: Mlp,

    /// `view` must stand on a block: "transformer.h.N.".
    pub fn init(view: zml.io.TensorStore.View, config: Config, kind: []const u8) Block {
        return ???;
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Block, x: zml.Tensor) zml.Tensor {
        const y = self.ln_1.forward(x.add(self.attn.forward(x)));
        return self.ln_2.forward(y.add(self.mlp.forward(y)));
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

/// Runs `model.forward` on the reference tensor `module ++ ".in"`:
/// does it return the reference tensor `module ++ ".out"`?
/// The `check` of exercise 130, for any model made of a `forward`
/// that takes one {.s, .d} tensor.
fn check(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    reference: *zml.io.TensorStore,
    comptime T: type,
    model: T,
    comptime module: []const u8,
) !bool {
    var x_buffer = try bufferFromStore(gpa, io, platform, reference, module ++ ".in");
    defer x_buffer.deinit();
    var want = try bufferFromStore(gpa, io, platform, reference, module ++ ".out");
    defer want.deinit();

    const x: zml.Tensor = .init(.{ .s = x_buffer.shape().dim(0), .d = x_buffer.shape().dim(1) }, .f32);

    var weights = try zml.mem.bufferize(gpa, T, &model);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(T, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, T, &model, &weights, store, .{});
    try loader.await(io);

    var exe = try platform.compile(gpa, io, model, .forward, .{x}, .{});
    defer exe.deinit();
    var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, x_buffer });
    defer got.deinit();

    return matches(io, want, got);
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

    try stdout.print("activation_function: {s}\n", .{config.activation_function});

    // The activation alone, on what PyTorch fed it, and compared a
    // hundred times more strictly than `matches` does.
    {
        var x_buffer = try bufferFromStore(gpa, io, platform, &reference, "transformer.h.0.mlp.act.in");
        defer x_buffer.deinit();
        var want = try bufferFromStore(gpa, io, platform, &reference, "transformer.h.0.mlp.act.out");
        defer want.deinit();

        const x: zml.Tensor = .init(.{ .s = x_buffer.shape().dim(0), .d = x_buffer.shape().dim(1) }, .f32);
        var exe = try platform.compileFn(gpa, io, activation, .{x}, .{});
        defer exe.deinit();
        var got: zml.Buffer = try exe.eval(gpa, io, .{x_buffer});
        defer got.deinit();

        const strict: zml.testing.CompareOpts = .{ .absolute_tolerance = 1e-5, .relative_tolerance = 1e-4 };
        const same = if (zml.testing.expectClose(io, want, got, strict)) |_| true else |_| false;
        try stdout.print("the activation matches PyTorch, strictly: {}\n", .{same});
    }

    // Block 0, from the smallest part to the whole.
    const view = store.view().withPrefix("transformer").withPrefix("h").withLayer(0);
    const block: Block = .init(view, config, config.attention_layers[0]);

    try stdout.print("ln_1 matches PyTorch: {}\n", .{
        try check(gpa, io, platform, &store, &reference, zml.nn.LayerNorm, block.ln_1, "transformer.h.0.ln_1"),
    });
    try stdout.print("attn matches PyTorch: {}\n", .{
        try check(gpa, io, platform, &store, &reference, Attention, block.attn, "transformer.h.0.attn"),
    });
    try stdout.print("mlp matches PyTorch: {}\n", .{
        try check(gpa, io, platform, &store, &reference, Mlp, block.mlp, "transformer.h.0.mlp"),
    });
    try stdout.print("the whole block matches PyTorch: {}\n", .{
        try check(gpa, io, platform, &store, &reference, Block, block, "transformer.h.0"),
    });
    try stdout.flush();
}
