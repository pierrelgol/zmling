//
// A street has one direction. A photo has two, and colours on
// top. Odette has hired a photographer, Monsieur Flou, to
// shoot the rue des Vitrines, and his photos arrive the way
// photos arrive from every camera, file and image library on
// earth: height, width, then the colour of each pixel.
//
//     photo {.b = 1, .h = 3, .w = 7, .c = 3}        NHWC
//
// You met the two layouts in exercises 8 and 31. Here is where
// it starts to hurt, because a 2D convolution has opinions:
//
//     pub fn conv2d(input: Tensor, kernel: Tensor, opts: struct {
//         window_strides: []const i64 = &.{ 1, 1 },
//         padding: []const i64 = &.{ 0, 0, 0, 0 },
//         rhs_dilation: []const i64 = &.{ 1, 1 },
//         input_batch_dimension: i64 = 0,
//         input_feature_dimension: i64 = 1,
//         input_spatial_dimensions: []const i64 = &.{ 2, 3 },
//         kernel_output_feature_dimension: i64 = 0,
//         kernel_input_feature_dimension: i64 = 1,
//         kernel_spatial_dimensions: []const i64 = &.{ 2, 3 },
//         output_batch_dimension: i64 = 0,
//         output_feature_dimension: i64 = 1,
//         output_spatial_dimensions: []const i64 = &.{ 2, 3 },
//         feature_group_count: i64 = 1,
//         ...
//     }) Tensor
//
// Everything of exercise 163, once per direction (rows first):
// the formula applies to the height and to the width
// separately. And nine options to say where things are. The
// defaults are PyTorch's:
//
//     input    (batch, channels, height, width)          NCHW
//     kernel   (out channels, in channels, height, width) OIHW
//     output   (batch, channels, height, width)          NCHW
//
// "feature" is what ZML (after XLA) calls a channel, "spatial"
// the axes the kernel slides along.
//
// Now the trap of the chapter. These options are axis NUMBERS,
// and `conv2d` never reads a tag. Give it Monsieur Flou's NHWC
// photo with the default options, and it takes axis 1, the
// height, for the channels. A height of 3 and 3 colours: the
// kernel fits. It slides along "height = 7" and "width = 3",
// and returns a tensor that still carries the tags of the
// photo, on the wrong axes. No panic. Look at `careless`
// below: it is in `main` so that you see it once in your life
// before meeting it in production. Same shape as the right
// answer, tag for tag. Wrong numbers.
//
// There are two ways out.
//
// 1. Turn the photo: `transpose` to (b, c, h, w), convolve
//    with the defaults, `transpose` back. You know how: by
//    tags, exercise 30. Two copies of the photo, though.
//
// 2. Tell `conv2d` where things are. Tags do not drive the
//    convolution, but they are still the best way to find an
//    axis number:
//
//        photo.axis(.c)        3, here
//
//    and so
//
//        .input_batch_dimension = photo.axis(.b),
//        .input_feature_dimension = photo.axis(.c),
//        .input_spatial_dimensions = &.{ photo.axis(.h), photo.axis(.w) },
//
//    and the same three for the output, which ZML builds from
//    the shape of the input: keep it in the layout of the
//    photo. This is how ZML's own models do it (see
//    zml/examples/llm/models/lfm2/model.zig), and it works for
//    any layout. The kernel has its three options too: files
//    that come from TensorFlow store kernels as (height, width,
//    in, out).
//
// One more thing before you start. When the kernel does not
// match the input, `conv2d` panics with one of the least
// talkative messages of ZML:
//
//     Invalid inputs tensor.zig@convolution:1106
//
// That is a file, a function and a line. Open
// zml/zml/tensor.zig at that line: every line around it is one
// condition a convolution must satisfy, and the one with the
// number is the one you broke. This one says:
//
//     kernel.dim(kernel_input_feature_dimension)
//         == input.dim(input_feature_dimension) / feature_group_count
//
// (Groups are for exercise 167. Today the count is 1.)
//
// The task. Odette's question is "where does the red begin?".
// The photo shows a red awning in the middle of the street,
// under a uniformly green sky (it was a strange day), with a
// blue that grows from left to right. The kernel is last
// exercise's "is it getting brighter?", three rows high now,
// and it only has weights for the red channel:
//
//     red channel, each row:    0  0  0  1  1  1  0
//     kernel, red, each row:   -1  0  1
//
// The right answer is 3 where the awning begins, -3 where it
// ends: (0, 3, 3, 0, -3).
//
// To do:
//
//  - `main`: the kernel placeholder. The data is (out, in,
//    height, width). Run it and read the panic;
//  - `byOptions`: write it;
//  - `byTranspose`: a step is missing;
//  - `main`: run the two.
//
const std = @import("std");
const zml = @import("zml");

/// What you get when you do not look. Leave it as it is.
/// photo: {.b, .h, .w, .c}, kernel: {.o, .i, .kh, .kw}
fn careless(photo: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    return photo.conv2d(kernel, .{});
}

/// The convolution of an NHWC photo, without moving it.
/// photo: {.b, .h, .w, .c}, kernel: {.o, .i, .kh, .kw} -> {.b, .h, .w, .c}
fn byOptions(photo: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    return ???;
}

/// The same, by turning the photo to NCHW and back.
/// photo: {.b, .h, .w, .c}, kernel: {.o, .i, .kh, .kw} -> {.b, .h, .w, .c}
fn byTranspose(photo: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
    const turned = photo.transpose(.{ .b, .c, .h, .w });
    return turned.conv2d(kernel, .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const photo: zml.Tensor = .init(.{ .b = 1, .h = 3, .w = 7, .c = 3 }, .f32);
    const kernel: zml.Tensor = .init(.{ .i = 3, .o = 1, .kh = 3, .kw = 3 }, .f32);

    // One row of pixels: (red, green, blue) seven times.
    const row = [7][3]f32{ .{ 0, 5, 0 }, .{ 0, 5, 1 }, .{ 0, 5, 2 }, .{ 1, 5, 3 }, .{ 1, 5, 4 }, .{ 1, 5, 5 }, .{ 0, 5, 6 } };
    const pixels = [1][3][7][3]f32{.{ row, row, row }};
    var photo_buffer: zml.Buffer = try .fromBytes(io, platform, photo.shape(), std.mem.sliceAsBytes(&pixels));
    defer photo_buffer.deinit();

    // (out, in, height, width): one kernel, three colours, 3 x 3.
    const brighter = [3][3]f32{ .{ -1, 0, 1 }, .{ -1, 0, 1 }, .{ -1, 0, 1 } };
    const nothing: [3][3]f32 = @splat(@splat(0));
    const weights = [1][3][3][3]f32{.{ brighter, nothing, nothing }};
    var kernel_buffer: zml.Buffer = try .fromBytes(io, platform, kernel.shape(), std.mem.sliceAsBytes(&weights));
    defer kernel_buffer.deinit();

    var careless_exe = try platform.compileFn(gpa, io, careless, .{ photo, kernel }, .{});
    defer careless_exe.deinit();
    var careless_result = try careless_exe.eval(gpa, io, .{ photo_buffer, kernel_buffer });
    defer careless_result.deinit();

    // Compile and run `byOptions` and `byTranspose`: the results
    // are `options_result` and `transpose_result`.
    ???;

    try stdout.print("the photo    {f}\n", .{photo.shape()});
    try stdout.print("careless     {f}: {any}\n", .{ careless_result.shape(), try careless_result.getValue([5]f32, io) });
    try stdout.print("by options   {f}: {any}\n", .{ options_result.shape(), try options_result.getValue([5]f32, io) });
    try stdout.print("by transpose {f}: {any}\n", .{ transpose_result.shape(), try transpose_result.getValue([5]f32, io) });
    try stdout.flush();
}
