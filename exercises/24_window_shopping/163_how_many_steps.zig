//
// Odette is paid by the glance. (It is in her contract. She
// negotiated it herself.) So before any passer-by sets foot in
// the rue des Vitrines, she wants to know HOW MANY glances a
// walk will produce. Last exercise it was easy: a street of 8,
// a kernel of 3, 8 - 3 + 1 = 6 places to stand. But not
// everybody walks like that.
//
// THE JOGGER moves several windows at a time: the STRIDE.
// With a stride of 2 she stands at windows 0, 2 and 4. At 6
// the kernel would stick out of the street, so that glance
// does not happen, and window 7 is never seen by anybody.
//
//     window_strides: i64 = 1
//
// THE COUNCIL adds blank walls at both ends of the street: the
// PADDING, made of zeros. With one blank wall on each side, a
// kernel of 3 can stand "before" window 0 and "after" window
// 7, and the walk gives 8 glances for 8 windows. That is the
// usual reason to pad: a layer that keeps the size.
//
//     padding: []const i64 = &.{ 0, 0 }
//
// For `conv1d`, a slice of two numbers, (before, after), or of
// ONE number used for both sides. For `conv2d` it will be
// four, (top, bottom, left, right), or one for all sides. Give
// `conv2d` two and you get a Zig panic from inside ZML:
// `index out of bounds: index 2, len 2`.
//
// THE DAYDREAMER looks at one window out of two: his kernel of
// 3 covers windows 0, 2 and 4. That is the DILATION, and it
// buys a wider receptive field for the same three weights. A
// kernel of size k with dilation d is spread over
//
//     span = d * (k - 1) + 1          windows.
//
//     rhs_dilation: i64 = 1
//
// "rhs" as in right-hand side: in `input.conv1d(kernel, ...)`
// the kernel is on the right. There is also an `lhs_dilation`,
// which spreads the INPUT apart instead, zeros in between (the
// `interior` padding of exercise 52: that is how a network
// makes a street LONGER, and it is called a transposed
// convolution). Mixing the two up is not an error. It is a
// different walk.
//
// All together, the formula on the first page of every
// convolution cheat sheet:
//
//     out = (in + pad_before + pad_after - span) / stride + 1
//
// rounded DOWN (the jogger's lost glance), which is exactly
// what Zig's `@divFloor` does. And one more case, which the
// cheat sheets forget: when the spread kernel is wider than
// the padded street, there is no place to stand at all. The
// answer is 0. The formula does not say 0: try a street of 8
// and a kernel of 3 with a dilation of 5.
//
// What does ZML do with a walk of 0 glances? Nothing. No
// panic, no warning: you get a tensor with an axis of size 0,
// which holds no number, and which will happily flow through
// the next ten layers. In a real network, one pooling too many
// or an image smaller than planned gets you there. If a model
// outputs nothing at all, look for a 0 in the shapes.
//
// To make the walks visible, the kernel of this exercise is
// (100, 10, 1) and the windows are numbered 1 to 8: a glance
// at windows 3, 4, 5 reads 345. A blank wall reads as a 0:
// "12" is a blank wall followed by windows 1 and 2.
//
// The walk is described by a plain Zig struct, `Walk`, given
// to the function next to the tensors. That is allowed: in
// the tuple of a compile call, whatever is not a Tensor is
// passed as it is, once, when the graph is built. So the seven
// walks of `main` are seven different Exes.
//
// To do:
//
//  - `glances`: Odette's formula. Plain Zig, no ZML;
//  - `samePad`: the padding (on each side) for which a walk
//    with a stride of 1 gives as many glances as there are
//    windows. `k` is odd. Solve the formula for it;
//  - `walk`: it runs, and Odette's formula disagrees with it
//    on every walk that has a dilation. The formula is right;
//  - `main`: for every walk, compile and run.
//
const std = @import("std");
const zml = @import("zml");

const Walk = struct {
    name: []const u8,
    stride: i64 = 1,
    /// Blank walls added on each side.
    pad: i64 = 0,
    dilation: i64 = 1,
};

/// How many glances a walk gives, on a street of `in` windows
/// with a kernel of `k` weights.
fn glances(in: i64, k: i64, w: Walk) i64 {
    return ???;
}

/// The padding, on each side, that keeps the size of the street
/// when the stride is 1. `k` is odd.
fn samePad(k: i64, dilation: i64) i64 {
    return ???;
}

/// street: {.b, .c, .t}, kernel: {.o, .i, .k} -> {.b, .c, .t}
fn walk(street: zml.Tensor, kernel: zml.Tensor, w: Walk) zml.Tensor {
    return street.conv1d(kernel, .{
        .window_strides = w.stride,
        .padding = &.{w.pad},
        .lhs_dilation = w.dilation,
    });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const street: zml.Tensor = .init(.{ .b = 1, .c = 1, .t = 8 }, .f32);
    const kernel: zml.Tensor = .init(.{ .o = 1, .i = 1, .k = 3 }, .f32);

    const numbers = [8]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var street_buffer: zml.Buffer = try .fromBytes(io, platform, street.shape(), std.mem.sliceAsBytes(&numbers));
    defer street_buffer.deinit();

    const digits = [3]f32{ 100, 10, 1 };
    var kernel_buffer: zml.Buffer = try .fromBytes(io, platform, kernel.shape(), std.mem.sliceAsBytes(&digits));
    defer kernel_buffer.deinit();

    const walks = [_]Walk{
        .{ .name = "the regular" },
        .{ .name = "the jogger", .stride = 2 },
        .{ .name = "the council", .pad = samePad(3, 1) },
        .{ .name = "the daydreamer", .dilation = 2 },
        .{ .name = "the daydreamer, padded", .dilation = 2, .pad = samePad(3, 2) },
        .{ .name = "all of it", .stride = 2, .pad = 1, .dilation = 2 },
        .{ .name = "the astronomer", .dilation = 5 },
    };

    for (walks) |w| {
        // Compile `walk` for this walk, run it: the result is
        // `seen`, a Buffer, and `values` its content as a
        // `zml.Slice` (exercise 13). Free everything.
        ???;

        try stdout.print("{s:<23} Odette says {d}, ZML says {d}: {any}\n", .{
            w.name,
            glances(street.dim(.t), kernel.dim(.k), w),
            seen.shape().dim(.t),
            values.items(f32),
        });
    }
    try stdout.flush();
}
