//
// Sixth enclosure. Sheldon the tortoise is 94, and he has never
// paid attention to anything.
//
// Attention looks at the WHOLE sequence at every step: to find
// the next word, token 1000 is compared with the 999 before
// it, and the KV cache of exercise 108 grows by one row per
// token, forever. Sheldon does it the old way. He walks along
// the path, one sign after the other, and all he carries is
// his MOOD: one vector, {.d}, always the same size. At each
// sign, the mood and the sign make the new mood:
//
//     h[t] = cell(x[t], h[t-1])
//
// That is RECURRENCE. It was the whole field before 2017 (RNN,
// LSTM, GRU), and it is back inside recent language models,
// mixed with attention layers, because a state of fixed size
// costs the same at token 10 and at token 100000: Mamba, RWKV,
// and the "linear attention" layers of Qwen 3.5, whose ZML port
// is `zml.nn.GatedDeltaNet`.
//
// The cell is GATED: it does not swallow every sign whole. A
// gate z, between 0 and 1 for each feature, decides how much
// of the mood is replaced by the candidate c:
//
//     z = sigmoid(gate_x(x) + gate_h(h))      how much to let in
//     c = tanh(cand_x(x) + cand_h(h))         what would come in
//     h' = h + z * (c - h)                    z = 0: keep h,
//                                             z = 1: take c
//
// (A GRU has one more gate, an LSTM two more and a second
// state. Same idea.) `Cell.step` is written for you: four
// Linears and three lines.
//
// Now, the loop over the signs. You have known two ways since
// chapter 11.
//
// UNROLLED: a Zig `for`. It runs while the graph is built, so
// the graph gets one copy of the cell PER STEP, and the number
// of steps is frozen in the Exe. `unrolled` below is that.
//
// IN THE GRAPH: `zml.ops.@"while"` (exercise 078). One copy of
// the cell, and the number of steps is a Tensor, decided at
// each call: the REAL length of the sentence, when the batch
// is padded. The contract, from `zml/core/ops.zig`:
//
//     pub fn @"while"(
//         While: type,
//         context: While,
//         initial_state: While.State,
//     ) While.State
//
//     While        a struct holding everything that does NOT
//                  change during the loop
//     While.State  a struct of Tensors: everything that does
//     cond(self, state) Tensor    one bool: go on?
//     body(self, state) State     one step. Every Tensor must
//                                 keep its shape and dtype.
//
// What is new here is what goes in the two structs when the
// loop is a model. Read `GatedDeltaNet` in `zml/nn/nn.zig`: it is
// this exercise with more algebra in `body`.
//
//  - The CONTEXT holds the whole sequence `xs` {.s, .d}, the
//    weights (our `Cell`: a struct of Linears is fine), and
//    the length. A loop cannot receive "the next x": it
//    receives everything, and picks.
//
//  - The STATE holds a step counter (i32, rank 0), the mood
//    `h`, and the TRAIL: every mood so far, {.s, .d}, because
//    the next layer wants one vector per position, not only
//    the last one. Shapes may not change inside a loop, so the
//    trail cannot grow: it starts as `zml.Tensor.zeroes(xs.shape())`
//    and is filled row by row. The archive of exercise 108.
//
//  - `body` picks the sign of the current step with a slice
//    whose index is a Tensor, and files the new mood at the
//    same index (exercise 059 for both):
//
//        self.xs.slice(.s, .dynSingle(state.step))                  {.d}
//        state.trail.dynamicUpdateSlice(.{ .s = state.step }, h)
//
//  - `cond`: `step < length`. And if somebody asks for more
//    steps than there are signs? There is no sign 5 to read,
//    and the loop does not know. GatedDeltaNet clamps the
//    length to the size of `.s` first (`.minimum(...)`, with
//    a scalar Tensor of the same dtype). Do the same.
//
// The path has 4 signs. Sheldon walks 2, 3, 4 of them, then is
// asked for 9: ONE Exe for all of that. Then the unrolled Exe
// walks its 4, and must end in the same mood.
//
// To do:
//
//  1. The fields of `Walk` and of `Walk.State`.
//  2. `cond` and `body`.
//  3. `Tortoise.looped`: the initial state, the loop, and what
//     comes back.
//  4. In `main`: the Exe of `looped`. Then, in the `for`, the
//     length as a Buffer and the run: two outputs, in `memory`.
//
const std = @import("std");
const zml = @import("zml");

const Cell = struct {
    gate_x: zml.nn.Linear,
    gate_h: zml.nn.Linear,
    cand_x: zml.nn.Linear,
    cand_h: zml.nn.Linear,

    /// One sign x and the mood h, both {.d} -> the new mood {.d}
    pub fn step(self: Cell, x: zml.Tensor, h: zml.Tensor) zml.Tensor {
        const z = self.gate_x.forward(x, x.dtype()).add(self.gate_h.forward(h, x.dtype())).sigmoid();
        const c = self.cand_x.forward(x, x.dtype()).add(self.cand_h.forward(h, x.dtype())).tanh();
        return h.add(z.mul(c.sub(h)).rename(.{ .dout = .d }));
    }
};

/// The loop: what stays (the fields), what moves (the State).
const Walk = struct {
    // What stays the same during the walk: three fields.
    ???,

    pub const State = struct {
        // What moves: three Tensors.
        ???,
    };

    pub fn cond(self: Walk, state: State) zml.Tensor {
        return ???;
    }

    pub fn body(self: Walk, state: State) State {
        return ???;
    }
};

const Tortoise = struct {
    cell: Cell,

    pub const Memory = struct {
        /// {.d}: the mood after the last sign that was read.
        h: zml.Tensor,
        /// {.s, .d}: the mood after each sign. Zeros where he did not go.
        trail: zml.Tensor,
    };

    /// xs: {.s, .d}, length: i32 rank 0. The loop is in the graph.
    pub fn looped(self: Tortoise, xs: zml.Tensor, length: zml.Tensor) Memory {
        return ???;
    }

    /// xs: {.s, .d} -> the last mood {.d}. The loop is in Zig.
    pub fn unrolled(self: Tortoise, xs: zml.Tensor) zml.Tensor {
        var h = zml.Tensor.zeroes(xs.shape().drop(.s));
        for (0..@intCast(xs.dim(.s))) |i| {
            h = self.cell.step(xs.slice(.s, .single(@intCast(i))), h);
        }
        return h;
    }
};

// A mood is (curious, grumpy). So is a sign.
const gate_x_weight = [2][2]f32{ .{ 2, 0 }, .{ 0, 2 } };
// Sheldon is hard to impress: the gates start half closed...
const gate_x_bias = [2]f32{ -1, -1 };
// ...and a grumpy tortoise gets curious more easily (and back).
const gate_h_weight = [2][2]f32{ .{ 0, 0.5 }, .{ 0.5, 0 } };
const cand_x_weight = [2][2]f32{ .{ 1, 0 }, .{ 0, 1 } };
const cand_h_weight = [2][2]f32{ .{ 0.5, 0 }, .{ 0, 0.5 } };

const signs = [_][]const u8{ "LETTUCE", "NO DOGS", "LETTUCE", "CLOSED " };

/// {.s = 4, .d = 2}
const path = [4][2]f32{
    .{ 2, 0 },
    .{ 0, 2 },
    .{ 2, 0 },
    .{ -1, 3 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const sheldon: Tortoise = .{ .cell = .{
        .gate_x = .init(.init(.{ .dout = 2, .d = 2 }, .f32), .init(.{ .dout = 2 }, .f32), .d),
        .gate_h = .init(.init(.{ .dout = 2, .d = 2 }, .f32), null, .d),
        .cand_x = .init(.init(.{ .dout = 2, .d = 2 }, .f32), null, .d),
        .cand_h = .init(.init(.{ .dout = 2, .d = 2 }, .f32), null, .d),
    } };
    const cell = sheldon.cell;

    var weights: zml.Bufferized(Tortoise) = .{ .cell = .{
        .gate_x = .{
            .weight = try .fromBytes(io, platform, cell.gate_x.weight.shape(), std.mem.sliceAsBytes(&gate_x_weight)),
            .bias = try .fromBytes(io, platform, cell.gate_x.bias.?.shape(), std.mem.sliceAsBytes(&gate_x_bias)),
            .quantization = null,
        },
        .gate_h = .{
            .weight = try .fromBytes(io, platform, cell.gate_h.weight.shape(), std.mem.sliceAsBytes(&gate_h_weight)),
            .bias = null,
            .quantization = null,
        },
        .cand_x = .{
            .weight = try .fromBytes(io, platform, cell.cand_x.weight.shape(), std.mem.sliceAsBytes(&cand_x_weight)),
            .bias = null,
            .quantization = null,
        },
        .cand_h = .{
            .weight = try .fromBytes(io, platform, cell.cand_h.weight.shape(), std.mem.sliceAsBytes(&cand_h_weight)),
            .bias = null,
            .quantization = null,
        },
    } };
    defer zml.Buffer.deinitAll(Tortoise, &weights);

    const xs: zml.Tensor = .init(.{ .s = 4, .d = 2 }, .f32);
    const length: zml.Tensor = .init(.{}, .i32);

    var xs_buffer: zml.Buffer = try .fromBytes(io, platform, xs.shape(), std.mem.sliceAsBytes(&path));
    defer xs_buffer.deinit();

    // The loop in the graph: one Exe, any number of signs.
    ???;

    var last_mood: [2]f32 = undefined;
    for ([_]i32{ 2, 3, 4, 9 }) |n| {
        // `n` as a Buffer, and the run.
        ???;

        last_mood = try memory.h.getValue([2]f32, io);
        try stdout.print("asked for {d} signs, his mood: curious {d:.3}, grumpy {d:.3}\n", .{ n, last_mood[0], last_mood[1] });
        if (n == 9) {
            try stdout.print("the trail {f}:\n", .{memory.trail.shape()});
            for (signs, try memory.trail.getValue([4][2]f32, io)) |sign, mood| {
                try stdout.print("  after {s}  {d:6.3} {d:6.3}\n", .{ sign, mood[0], mood[1] });
            }
        }
    }

    // The loop in Zig: 4 signs, and not one more.
    var unrolled_exe = try platform.compile(gpa, io, sheldon, .unrolled, .{xs}, .{});
    defer unrolled_exe.deinit();

    var unrolled: zml.Buffer = try unrolled_exe.eval(gpa, io, .{ weights, xs_buffer });
    defer unrolled.deinit();
    const mood = try unrolled.getValue([2]f32, io);

    const same = @abs(mood[0] - last_mood[0]) < 1e-5 and @abs(mood[1] - last_mood[1]) < 1e-5;
    try stdout.print("unrolled, {d} inputs against {d}: the same mood: {}\n", .{ unrolled_exe.input_shapes.len, looped_exe.input_shapes.len, same });
    try stdout.flush();
}
