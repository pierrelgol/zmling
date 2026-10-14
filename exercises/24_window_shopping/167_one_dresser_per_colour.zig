//
// Business is good, and Odette has hired. The agency sent
// three dressers, and each of them, it turns out, works on ONE
// colour of the photo and refuses to look at the other two:
//
//     Rose      red      changes nothing (she calls it "restraint")
//     Olive     green    adds up everything around
//     Marine    blue     "is it getting brighter?", again
//
// Compare with what you have used since exercise 164. In an
// ordinary convolution every kernel has weights for EVERY input
// channel: from 3 colours to 2 outputs with a 3 x 3 window,
// that is 2 * 3 * 3 * 3 = 54 weights, and every output mixes
// all the colours at once.
//
// The option you have skipped twice changes that:
//
//     feature_group_count: i64 = 1
//
// With `feature_group_count = g`, the input channels are cut
// into g GROUPS, the kernels too, and each group of kernels
// only sees its own group of channels. Three rules follow, and
// two of them are checked by the terse panic of exercise 164:
//
//  - the number of input channels is a multiple of g;
//  - the number of kernels (the `.o` axis) is a multiple of g;
//  - the kernel has weights for the channels of ONE group only:
//    its "in" axis is  input channels / g.  (This is the line
//    1106 you have already visited.)
//
// Push it to the limit, g = the number of channels: every
// channel gets its own kernel and nobody talks to anybody.
// That is a DEPTHWISE convolution. From 3 channels to 3
// channels, 3 x 3: a kernel of (3, 1, 3, 3), 27 weights.
//
// Of course, dressers who never talk do not make a window.
// After them comes the decorator, who looks at ONE pixel at a
// time, across all its channels, and blends:
//
//     result 0  =  1 * red + 1 * green + 1 * blue
//     result 1  =  1 * red + 0 * green - 1 * blue
//
// A weighted sum of the channels, the same weights at every
// pixel. As a convolution, that is a kernel of 1 x 1:
// (2, 3, 1, 1), 6 weights. No neighbours at all. A 1 x 1
// convolution is how a network changes its number of channels,
// and it is nothing new: it is the Linear layer of exercise 89
// applied to each pixel, with the channels as its `.d`.
// `zml.nn.Linear` does not care how many other axes there are:
// it consumes the axis whose tag you give it, and puts `.dout`
// LAST. A vision transformer written with Linears and a
// ConvNet written with 1 x 1 convolutions are, at those
// layers, the same arithmetic. You are going to check it.
//
// Depthwise, then 1 x 1: a SEPARABLE convolution. 27 + 6 = 33
// weights instead of 54 for the same window and the same
// channels, and the gap grows fast: with 256 channels in and
// out it is 67 840 against 589 824. This trade is the whole
// idea of MobileNet, and the reason a ConvNet fits in a phone.
// (In between, 1 < g < channels: "grouped" convolutions, as in
// ResNeXt. And ZML's own LFM2 model, in zml/examples, slides a
// depthwise `conv1d` over its tokens.)
//
// The photo is 4 x 4, in (b, c, h, w) this time (Monsieur Flou
// has been spoken to). After the dressers, 2 x 2 per channel:
//
//     Rose      the pixel in the middle of each window
//     Olive     9 everywhere (the green is 1 everywhere)
//     Marine    3 * (the blue two columns further - the blue here)
//
// To do:
//
//  - `main`: the shape of the dressers' kernels;
//  - `dress`: it panics. You know the line;
//  - `blendLinear`: the decorator, as a `zml.nn.Linear`. Same
//    weights, same result, same layout as `blend`;
//  - `main`: compile and run `windowLinear`.
//
const std = @import("std");
const zml = @import("zml");

/// Each dresser on her own colour.
/// photo: {.b, .c, .h, .w}, kernels: {.o, .i, .kh, .kw} with .i = 1
fn dress(photo: zml.Tensor, kernels: zml.Tensor) zml.Tensor {
    return photo.conv2d(kernels, .{});
}

/// The decorator, as a 1 x 1 convolution.
/// x: {.b, .c, .h, .w}, mixer: {.o, .i, .kh = 1, .kw = 1}
fn blend(x: zml.Tensor, mixer: zml.Tensor) zml.Tensor {
    return x.conv2d(mixer, .{});
}

/// The decorator, as a Linear layer applied to every pixel.
/// x: {.b, .c, .h, .w}, mixer: {.o, .i, .kh = 1, .kw = 1}
/// -> {.b, .c, .h, .w}, like `blend`.
fn blendLinear(x: zml.Tensor, mixer: zml.Tensor) zml.Tensor {
    return ???;
}

/// The whole window: dressers, then decorator.
fn window(photo: zml.Tensor, kernels: zml.Tensor, mixer: zml.Tensor) zml.Tensor {
    return blend(dress(photo, kernels), mixer);
}

fn windowLinear(photo: zml.Tensor, kernels: zml.Tensor, mixer: zml.Tensor) zml.Tensor {
    return blendLinear(dress(photo, kernels), mixer);
}

/// Prints the channels of a {1, channels, 2, 2} image side by side.
fn show(stdout: *std.Io.Writer, comptime channels: usize, image: [1][channels][2][2]f32) !void {
    for (0..2) |row| {
        for (image[0], 0..) |channel, i| {
            if (i > 0) try stdout.print("   ", .{});
            try stdout.print("  {d:>3} {d:>3}", .{ channel[row][0], channel[row][1] });
        }
        try stdout.print("\n", .{});
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const photo: zml.Tensor = .init(.{ .b = 1, .c = 3, .h = 4, .w = 4 }, .f32);
    // One 3 x 3 kernel per dresser: tags .o, .i, .kh, .kw.
    const kernels: zml.Tensor = .init(???, .f32);
    const mixer: zml.Tensor = .init(.{ .o = 2, .i = 3, .kh = 1, .kw = 1 }, .f32);

    const pixels = [3][4][4]f32{
        .{ .{ 0, 0, 0, 0 }, .{ 0, 1, 2, 0 }, .{ 0, 3, 4, 0 }, .{ 0, 0, 0, 0 } }, // red
        @splat(@splat(1)), // green
        @splat(.{ 0, 1, 3, 6 }), // blue
    };
    var photo_buffer: zml.Buffer = try .fromBytes(io, platform, photo.shape(), std.mem.sliceAsBytes(&pixels));
    defer photo_buffer.deinit();

    const habits = [3][3][3]f32{
        .{ .{ 0, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 0 } }, // Rose
        @splat(@splat(1)), // Olive
        @splat(.{ -1, 0, 1 }), // Marine
    };
    var kernels_buffer: zml.Buffer = try .fromBytes(io, platform, kernels.shape(), std.mem.sliceAsBytes(&habits));
    defer kernels_buffer.deinit();

    const recipe = [2][3]f32{ .{ 1, 1, 1 }, .{ 1, 0, -1 } };
    var mixer_buffer: zml.Buffer = try .fromBytes(io, platform, mixer.shape(), std.mem.sliceAsBytes(&recipe));
    defer mixer_buffer.deinit();

    var dressed_exe = try platform.compileFn(gpa, io, dress, .{ photo, kernels }, .{});
    defer dressed_exe.deinit();
    var dressed_result = try dressed_exe.eval(gpa, io, .{ photo_buffer, kernels_buffer });
    defer dressed_result.deinit();

    var window_exe = try platform.compileFn(gpa, io, window, .{ photo, kernels, mixer }, .{});
    defer window_exe.deinit();
    var window_result = try window_exe.eval(gpa, io, .{ photo_buffer, kernels_buffer, mixer_buffer });
    defer window_result.deinit();

    // Compile and run `windowLinear`: the result is `linear_result`.
    ???;

    const by_conv = try window_result.getValue([1][2][2][2]f32, io);
    const by_linear = try linear_result.getValue([1][2][2][2]f32, io);

    try stdout.print("the dressers {f}: Rose, Olive, Marine\n", .{dressed_result.shape()});
    try show(stdout, 3, try dressed_result.getValue([1][3][2][2]f32, io));
    try stdout.print("the decorator, 1 x 1 convolution {f}\n", .{window_result.shape()});
    try show(stdout, 2, by_conv);
    try stdout.print("the decorator, Linear {f}\n", .{linear_result.shape()});
    try show(stdout, 2, by_linear);
    try stdout.print("same: {}\n", .{std.mem.eql(u8, std.mem.asBytes(&by_conv), std.mem.asBytes(&by_linear))});
    try stdout.print("weights: {d} + {d} = {d}, against {d} for one full 3 x 3 convolution\n", .{
        kernels.count(),
        mixer.count(),
        kernels.count() + mixer.count(),
        mixer.dim(.o) * photo.dim(.c) * 3 * 3,
    });
    try stdout.flush();
}
