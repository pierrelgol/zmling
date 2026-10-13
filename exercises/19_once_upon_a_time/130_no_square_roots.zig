//
// Pip is typing the attention. From memory.
//
// "Split into heads, queries times keys, divide by the square
// root of the head size, mask, softmax, times values, glue the
// heads back, one last Linear. It's what you do with the
// llamas. It's what every transformer does. `zml.nn.sdpa`
// even does the division for you."
//
// Madame Plume, who has never written a line of code, puts
// the model's source on the desk, open at `_attn`, the method
// of `GPTNeoSelfAttention` where the scores are made:
//
//     attn_weights = torch.matmul(query, key.transpose(-1, -2))
//
//     causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
//     mask_value = torch.finfo(attn_weights.dtype).min
//     attn_weights = torch.where(causal_mask, attn_weights, mask_value)
//
//     attn_weights = nn.functional.softmax(attn_weights, dim=-1)
//     attn_output = torch.matmul(attn_weights, value)
//
// "Show me the square root," says Madame Plume.
//
// There is none. For comparison, the same place in GPT-2, by
// the same library:
//
//     scaling = query.size(-1) ** -0.5
//     attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
//
// GPT-Neo was trained WITHOUT the division, so its weights
// expect scores that are not divided. A head of Neo has 4
// numbers: divide by sqrt(4) and every score is halved, every
// softmax is flatter, every token listens to everybody a bit
// more than it should. Nothing crashes. The shapes are right.
// The model still writes English. It is simply a slightly
// deaf storyteller, and without a reference you would blame
// the weights.
//
// Exercise 105 said of `sdpa`: "the default `scale` is the
// 1/sqrt(hd) you want". For this model you don't:
//
//     pub const SdpaOpts = struct {
//         attn_mask: ?Tensor = null,
//         scale: ?Tensor = null,
//         softmax_bias: ?Tensor = null,
//     };
//
// `scale` is what the keys get multiplied by before the dot
// product. `null` means 1/sqrt(hd). A scalar Tensor of 1 means
// "leave my scores alone":
//
//     .scale = .scalar(1, .f32)
//
// The porting rule: in an attention, three things are never
// "the usual". WHAT the scores are divided by (sqrt(hd), or
// nothing, or a learned temperature), WHAT the mask hides, and
// WHICH Linears have a bias (exercise 128). Find each of them
// in the source, with your finger on the line.
//
// The mask, then. `self.bias` is the triangle of booleans the
// file was so full of. Where does it come from?
//
//     bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool))
//
//     if attention_type == "local":
//         bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))
//
// The first line is the causal mask of exercise 104. The
// second one removes a smaller triangle from it, for SOME
// layers: a "local" layer only hears the last `window_size`
// tokens, itself included. Which layers? `config.json`:
//
//     "attention_layers": ["global", "local", "global", "local", ...],
//     "window_size": 256,
//
// You have had the tool for this since exercise 104, and never
// used its last argument:
//
//     zml.nn.causalAttnMask(.{ .q = n, .k = n }, .f32, window)
//
// `window` is a `?u32`: `null` for the plain triangle, a number
// for "and no further back than this".
//
// Now the nasty part. The reference text has 10 tokens. With
// 10 tokens, nobody is further than 256 tokens away from
// anybody: a local layer and a global one compute EXACTLY the
// same thing, and a port without the window passes every
// check. It would pass them all the way to the 257th token of
// a story, in front of the children. So the script recorded
// one more thing: block 1 (the first local one) reading a
// story of 300 tokens, under the keys `long.transformer.h.1.attn.in`
// and `.out`. A reference must exercise the code path, or it
// proves nothing about it.
//
// `config.json` is read with Zig's JSON parser straight into
// the struct `Config` below: one field per key you care about,
// same names, and `.ignore_unknown_fields = true` for the
// rest. It is given.
//
// What a module of the reference is called tells you what it
// received. `transformer.h.0.attn` is the attention of block
// 0: its `.in` is what comes out of the LayerNorm before it,
// and its `.out` is the result of the last Linear, `out_proj`,
// BEFORE anything is added to anything. That is what `forward`
// must return here.
//
// To do:
//
//  1. `Attention.init`. The view stands on a block
//     ("transformer.h.N."), `kind` is "global" or "local".
//
//  2. `Attention.forward` is Pip's, from memory. It runs.
//
//  3. `check` has lost its middle: a model for the block, its
//     weights (the way `main` did it in exercise 129), an Exe
//     and a run.
//
// Fix one thing at a time and watch the three lines: which of
// them turn true tells you which bug you just fixed.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

/// The keys of `config.json` this program needs.
const Config = struct {
    hidden_size: u32,
    num_heads: u32,
    window_size: u32,
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

const Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    heads: u32,
    /// How far back a token may listen. `null`: all the way.
    window: ?u32,

    /// `view` must stand on a block: "transformer.h.N.".
    pub fn init(view: zml.io.TensorStore.View, config: Config, kind: []const u8) Attention {
        return ???;
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
            .{ .attn_mask = zml.nn.causalAttnMask(.{ .q = x.dim(.s), .k = x.dim(.s) }, .f32, null) },
        );

        const merged = heard.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.out_proj.forward(merged, .f32).rename(.{ .dout = .d });
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

/// Runs the attention of block `layer` on the reference input
/// `in_key`: does it return the reference output `out_key`?
fn check(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    reference: *zml.io.TensorStore,
    config: Config,
    layer: usize,
    in_key: []const u8,
    out_key: []const u8,
) !bool {
    var x_buffer = try bufferFromStore(gpa, io, platform, reference, in_key);
    defer x_buffer.deinit();
    var want = try bufferFromStore(gpa, io, platform, reference, out_key);
    defer want.deinit();

    // The file has no tags. The placeholder gets the dims of the
    // reference input, and our names.
    const x: zml.Tensor = .init(.{ .s = x_buffer.shape().dim(0), .d = x_buffer.shape().dim(1) }, .f32);

    // The attention of block `layer`, its weights on the device, an
    // Exe, and `got`: what it makes of `x_buffer`.
    ???;

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

    try stdout.print("{d} heads of {d} numbers each\n", .{ config.num_heads, config.hidden_size / config.num_heads });
    try stdout.print("blocks:", .{});
    for (config.attention_layers) |kind| try stdout.print(" {s}", .{kind});
    try stdout.print("\na local block hears the last {d} tokens\n", .{config.window_size});

    try stdout.print("block 0 ({s}),  10 tokens, matches PyTorch: {}\n", .{
        config.attention_layers[0],
        try check(gpa, io, platform, &store, &reference, config, 0, "transformer.h.0.attn.in", "transformer.h.0.attn.out"),
    });
    try stdout.print("block 1 ({s}),   10 tokens, matches PyTorch: {}\n", .{
        config.attention_layers[1],
        try check(gpa, io, platform, &store, &reference, config, 1, "transformer.h.1.attn.in", "transformer.h.1.attn.out"),
    });
    try stdout.print("block 1 ({s}),  300 tokens, matches PyTorch: {}\n", .{
        config.attention_layers[1],
        try check(gpa, io, platform, &store, &reference, config, 1, "long.transformer.h.1.attn.in", "long.transformer.h.1.attn.out"),
    });
    try stdout.flush();
}
