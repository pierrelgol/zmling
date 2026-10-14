//
// Welcome to the rue des Vitrines, the high street. Odette
// dresses its shop windows. She has done it for thirty years,
// she has opinions about mannequins, and she has a problem:
// she wants to know what a passer-by actually SEES.
//
// Not the whole street at once. Nobody sees a whole street. A
// passer-by walks along, nose against the glass, and at every
// step takes in the few windows right in front of them. That
// is all a CONVOLUTION is:
//
//     a small window of weights (the KERNEL) slides along the
//     input; at every position, the values under it are
//     multiplied by the weights and summed. One number per
//     position.
//
// Why would anyone build a network out of that? Compare with
// the Linear layer of chapter 13, which gives every output its
// own weight for every input:
//
//  - a kernel is SMALL: three weights here, whatever the length
//    of the street. A Linear over a photo of 1000 x 1000 pixels
//    needs a million weights per output;
//  - the SAME weights are used at every position. A kernel that
//    has learned to spot a mannequin spots it at number 3 and
//    at number 42. A Linear would have to learn it twice;
//  - every output depends on a few neighbours only: its
//    RECEPTIVE FIELD. Stack layers and the field grows: an
//    output of the second layer sees three outputs of the
//    first, which together saw five windows.
//
// Tonight the street looks like this (the brightness of each
// window, the bakery and its two neighbours have the lights on):
//
//     street {.t = 8}     1  1  1  6  6  6  2  2
//
// and Odette's first kernel answers "is it getting brighter?":
//
//     kernel {.k = 3}    -1  0  1
//
// Standing at window 0, the passer-by sees (1, 1, 1): that is
// -1*1 + 0*1 + 1*1 = 0. Nothing happens. One step further,
// (1, 1, 6): -1 + 6 = 5. The lights come on!
//
// You have every tool to do this by hand:
//
//     street.slice(.t, .{ .start = 2, .end = 5 })    exercise 54
//     a.dot(b, .k)                                   exercise 61
//     zml.Tensor.stack(&.{ a, b }, 0, .t)            exercise 57
//
// (`dot` wants the same tag on both sides, and a slice of the
// street is still tagged `.t`: a `rename` away, exercise 27.)
//
// How many positions are there? The kernel must fit entirely on
// the street: with 8 windows and a kernel of 3, the last place
// to stand is window 5. Count them.
//
// Then the real thing:
//
//     pub fn conv1d(input: Tensor, kernel: Tensor, opts: struct {
//         window_strides: i64 = 1,
//         padding: []const i64 = &.{ 0, 0 },
//         rhs_dilation: i64 = 1,
//         input_batch_dimension: i64 = 0,
//         input_feature_dimension: i64 = 1,
//         input_spatial_dimensions: i64 = 2,
//         kernel_output_feature_dimension: i64 = 0,
//         kernel_input_feature_dimension: i64 = 1,
//         kernel_spatial_dimensions: i64 = 2,
//         ...
//     }) Tensor
//
// Forget the options for one exercise and read the defaults,
// because they tell you what `conv1d` expects:
//
//     input    rank 3: (batch, channels, length)
//     kernel   rank 3: (output channels, input channels, size)
//
// A batch, because a convolution always runs on several inputs
// at once. Channels, because real inputs have several numbers
// per position (red, green, blue), and a layer usually applies
// several kernels at once: one output channel per kernel, and
// every kernel has weights for every input channel. Our street
// is one input, with one channel, and Odette has one kernel:
// both tensors need two axes of size 1 in FRONT. `insertAxes`,
// exercise 35. The result comes back as (batch, channels,
// length) too.
//
// Two things to know about ZML's convolutions:
//
//  - they do not look at the tags. The axes are found by
//    POSITION (those `..._dimension` options are axis numbers).
//    The result keeps the tags of the INPUT;
//  - mathematicians flip the kernel before sliding it. Like
//    PyTorch and everyone else in machine learning, ZML does
//    not: the first weight meets the first window. What you
//    compute by hand below is what `conv1d` computes.
//
// To do:
//
//  - `glance`: what the passer-by sees from one position;
//  - `stroll`: how many glances there are;
//  - `strollConv`: it panics. Read ZML's message;
//  - `main`: run `strollConv`.
//
const std = @import("std");
const zml = @import("zml");

/// What a passer-by standing at window `at` makes of the windows
/// in front of them: a rank 0 tensor.
/// street: {.t}, kernel: {.k}
fn glance(street: zml.Tensor, kernel: zml.Tensor, at: i64) zml.Tensor {
    return ???;
}

/// The whole walk, by hand: one glance per position where the
/// kernel fits. street: {.t}, kernel: {.k} -> {.t}
fn stroll(street: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    var glances: [16]zml.Tensor = undefined;
    // How many places are there to stand?
    const positions: usize = @intCast(???);
    for (glances[0..positions], 0..) |*g, at| g.* = glance(street, kernel, @intCast(at));
    return zml.Tensor.stack(glances[0..positions], 0, .t);
}

/// The same walk, by `conv1d`. street: {.t}, kernel: {.k} -> {.t}
fn strollConv(street: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    const input = street.insertAxes(0, .{ .b, .c });
    return input.conv1d(kernel, .{}).squeeze(.b).squeeze(.c);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const street: zml.Tensor = .init(.{ .t = 8 }, .f32);
    const kernel: zml.Tensor = .init(.{ .k = 3 }, .f32);

    const brightness = [8]f32{ 1, 1, 1, 6, 6, 6, 2, 2 };
    var street_buffer: zml.Buffer = try .fromBytes(io, platform, street.shape(), std.mem.sliceAsBytes(&brightness));
    defer street_buffer.deinit();

    const getting_brighter = [3]f32{ -1, 0, 1 };
    var kernel_buffer: zml.Buffer = try .fromBytes(io, platform, kernel.shape(), std.mem.sliceAsBytes(&getting_brighter));
    defer kernel_buffer.deinit();

    var by_hand_exe = try platform.compileFn(gpa, io, stroll, .{ street, kernel }, .{});
    defer by_hand_exe.deinit();

    var by_hand = try by_hand_exe.eval(gpa, io, .{ street_buffer, kernel_buffer });
    defer by_hand.deinit();

    // Compile `strollConv` and run it on the same two Buffers:
    // the result is `by_conv`.
    ???;

    const hand = try by_hand.getValue([6]f32, io);
    const conv = try by_conv.getValue([6]f32, io);

    try stdout.print("the street {f}: {any}\n", .{ street.shape(), brightness });
    try stdout.print("by hand    {f}: {any}\n", .{ by_hand.shape(), hand });
    try stdout.print("conv1d     {f}: {any}\n", .{ by_conv.shape(), conv });
    try stdout.print("same: {}\n", .{std.mem.eql(f32, &hand, &conv)});
    try stdout.flush();
}
