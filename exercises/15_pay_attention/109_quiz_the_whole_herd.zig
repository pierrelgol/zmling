//
// Quiz! Time to assemble the real thing: one complete
// self-attention layer, as found (a few dozen times over) in
// every Llama, and every llama.
//
// You have all the parts. What a real layer adds is four Linear
// layers, because so far queries, keys and values fell from the
// sky, and in a model they are COMPUTED from the tokens:
//
//     q_proj   "given what I am, what do I want to hear?"
//     k_proj   "given what I am, what do I know about?"
//     v_proj   "given what I am, what do I have to say?"
//     o_proj   mixes what the heads heard into one vector
//
// All four take `.d` numbers and return `.d` numbers, without
// bias. The whole layer is one model struct, and its `forward`
// turns `x: {.s, .d}` into a tensor of the same shape:
//
//  1. Project x three times: q, k and v. Split each of them
//     into `self.heads` heads. (What is the name of the axis a
//     Linear hands you back?)
//
//  2. Spin the queries and the keys: RoPE, default options,
//     positions 0, 1, 2... Not the values.
//
//  3. Scaled dot product attention, with a causal mask: no
//     llama hears the ones behind her. The sequence plays
//     both roles, so it must be given both names.
//
//  4. Glue the heads back together into one `.d` axis, and
//     call the sequence `.s` again.
//
//  5. Through `o_proj`. And then, the last ingredient, new but
//     tiny: the RESIDUAL. A layer does not replace a token by
//     what it heard, it ADDS what it heard to what the token
//     already was:
//
//         return x.add(heard)
//
//     That's why the output must have exactly the shape, and
//     the tags, of x. It is also what lets forty of these
//     layers be stacked: each one only nudges the tokens.
//
// Every one of these steps is a line or two that you have
// already written in this chapter or in chapter 13. Mind the
// tags at each step: `.dout`, `.d`, `.h`, `.hd`, `.s`, `.q`,
// `.k` all show up, each in its place. When ZML panics, the
// message names the shapes it got. Read them, the answer is
// usually right there.
//
// The weights are tiny and hand-made, so that you can follow
// a llama through the layer with a pencil if you feel brave.
// Then compile the model and let the herd talk.
//
const std = @import("std");
const zml = @import("zml");

const SelfAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    heads: u32,

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: SelfAttention, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const names = [_][]const u8{ "Tina ", "Kuzco", "Paca ", "Dolly" };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const weight_shape: zml.Shape = .init(.{ .dout = 4, .d = 4 }, .f32);
    const layer: SelfAttention = .{
        .q_proj = .init(.fromShape(weight_shape), null, .d),
        .k_proj = .init(.fromShape(weight_shape), null, .d),
        .v_proj = .init(.fromShape(weight_shape), null, .d),
        .o_proj = .init(.fromShape(weight_shape), null, .d),
        .heads = 2,
    };

    // Four llamas, four numbers each.
    const x: zml.Tensor = .init(.{ .s = 4, .d = 4 }, .f32);

    var exe = ???;
    defer exe.deinit();

    // "I want to hear about what I am": the identity.
    var q_weight: zml.Buffer = try .fromBytes(io, platform, weight_shape, std.mem.sliceAsBytes(&[4][4]f32{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    }));
    defer q_weight.deinit();

    // "I know about the opposite of what I am": the two halves, swapped.
    var k_weight: zml.Buffer = try .fromBytes(io, platform, weight_shape, std.mem.sliceAsBytes(&[4][4]f32{
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
    }));
    defer k_weight.deinit();

    // "What I say is what I am, only louder at the end."
    var v_weight: zml.Buffer = try .fromBytes(io, platform, weight_shape, std.mem.sliceAsBytes(&[4][4]f32{
        .{ 1, 0, 0, 0 },
        .{ 0, 2, 0, 0 },
        .{ 0, 0, 3, 0 },
        .{ 0, 0, 0, 4 },
    }));
    defer v_weight.deinit();

    // What was heard counts for half.
    var o_weight: zml.Buffer = try .fromBytes(io, platform, weight_shape, std.mem.sliceAsBytes(&[4][4]f32{
        .{ 0.5, 0, 0, 0 },
        .{ 0, 0.5, 0, 0 },
        .{ 0, 0, 0.5, 0 },
        .{ 0, 0, 0, 0.5 },
    }));
    defer o_weight.deinit();

    // Only the Tensors: `heads` and the tags stayed in the Exe.
    const layer_buffers: zml.Bufferized(SelfAttention) = .{
        .q_proj = .{ .weight = q_weight, .quantization = null },
        .k_proj = .{ .weight = k_weight, .quantization = null },
        .v_proj = .{ .weight = v_weight, .quantization = null },
        .o_proj = .{ .weight = o_weight, .quantization = null },
    };

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[4][4]f32{
        .{ 0, 0, 1, 0 }, // Tina
        .{ 1, 0, 0, 0 }, // Kuzco
        .{ 1, 0, 1, 0 }, // Paca
        .{ 0, 1, 0, 1 }, // Dolly
    }));
    defer x_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ layer_buffers, x_buffer });
    defer result.deinit();

    try stdout.print("after one round of gossip {f}:\n", .{result.shape()});
    for (names, try result.getValue([4][4]f32, io)) |name, row| {
        try stdout.print("  {s}", .{name});
        for (row) |r| try stdout.print(" {d:6.3}", .{r});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
