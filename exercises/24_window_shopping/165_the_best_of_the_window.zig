//
// The Grand Bazar, at number 12, has the biggest window of the
// street: four panes by four. Odette spent a Saturday counting
// how many people stopped in front of each pane:
//
//     stops {.b = 1, .c = 1, .h = 4, .w = 4}     3  1  0  2
//                                                5  4  1  1
//                                                0  0  7  3
//                                                2  6  8  5
//
// The owner does not want sixteen numbers. He wants four, one
// per quarter of the window. Summarising a neighbourhood by one
// number is called POOLING, and a convolutional network does
// it after almost every convolution, for two reasons:
//
//  - it makes the picture smaller, so the next layers are
//    cheaper and each of their pixels "sees" more of the image;
//  - it forgets WHERE exactly something was. A stroke one pixel
//    to the left gives the same summary. For "is there a cat in
//    this photo", that is what you want.
//
// A pooling slides a window like a convolution does (window
// size, stride, padding: the formula of exercise 163 again),
// but there are no weights. It keeps the MAX of each window,
// or its AVERAGE.
//
// Max pooling has its own functions:
//
//     pub fn maxPool2d(self: Tensor, opts: struct {
//         window_dimensions: [2]i64,
//         window_strides: ?[2]i64 = null,
//         padding: [2][2]i64 = .{ .{ 0, 0 }, .{ 0, 0 } },
//         ...
//     }) MaxPoolRes
//
//     pub fn maxPool1d(self: Tensor, opts: struct {
//         window_dimensions: i64,
//         window_strides: ?i64,
//         padding: [2]i64 = .{ 0, 0 },
//         ...
//     }) MaxPoolRes
//
// Read them slowly, there are four surprises in there:
//
//  1. No stride (`null`) means "as wide as the window": the
//     windows do not overlap. That is what PyTorch does too.
//
//  2. In `maxPool1d`, `window_strides` has no default value.
//     You must write it, even to say `null`. Zig will insist.
//
//  3. The result is not a Tensor. It is the struct you met with
//     `argMax` in exercise 42 (`MaxPoolRes` is another name for
//     `ArgMaxRes`): `.values`, the maxima, and `.indices`. The
//     indices say where each winner was ALONG THE LAST AXIS of
//     the input: for an image, its column in the whole image.
//
//  4. No tags, no axis options at all. `maxPool2d` pools the
//     LAST TWO axes, `maxPool1d` the last one. For
//     (b, c, h, w) that is perfect. An NHWC photo gets its
//     width and its COLOURS pooled together, silently. Turn it
//     first (exercise 164).
//
// Average pooling has no function of its own in ZML. You
// build it, and there are two ways.
//
// By folding. You did it in the quiz of chapter 7, exercise
// 53: cut `.h` and `.w` in two with `splitAxis`, and take the
// `mean` of the small axes. It only works when the windows do
// not overlap and fit exactly.
//
// By window. `maxPool2d` is itself written with one general
// tool, and you can use it too:
//
//     zml.ops.reduceWindow(N, inputs, inits, opts, func)
//
//     pub const ReduceWindowOpts = struct {
//         window_dimensions: []const i64,
//         window_strides: []const i64,
//         base_dilations: []const i64,
//         window_dilations: []const i64,
//         padding: []const [2]i64,
//     };
//
// It slides a window over N tensors at once (N = 1 for us:
// `inputs` is `.{x}`) and reduces each window with `func`,
// starting from `inits` (one rank 0 tensor per input: the
// value an empty window would give, 0 for a sum). The result
// is an array of N tensors. Unlike the max pools, the options
// have NO defaults and want one entry PER AXIS of the input,
// batch and channels included: a window of 1 and a stride of 1
// for the axes you leave alone, dilations of 1 everywhere.
// `func` receives the running result and the next value, and
// returns an array of N tensors; `plus`, below, is the one for
// a sum. A sum over a window of 4, divided by 4: an average.
//
// And at the very end of a modern network, the window is the
// whole image: GLOBAL average pooling, one number per channel,
// whatever the size of the photo. No window needed, `mean`
// does it (exercise 41), and like every reduction it leaves
// axes of size 1 behind (exercise 40).
//
// To do:
//
//  - `best`: it does not compile;
//  - `pairs`: neither does this one, for another reason;
//  - `averageByFolding`: exercise 53, from memory;
//  - `average`: with `reduceWindow`, for any stride;
//  - `overall`: the owner asked for {.b, .c}.
//
const std = @import("std");
const zml = @import("zml");

/// The busiest pane of each quarter of the window.
/// stops: {.b, .c, .h, .w} -> {.b, .c, .h / 2, .w / 2}
fn best(stops: zml.Tensor) zml.Tensor {
    return stops.maxPool2d(.{ .window_dimensions = .{ 2, 2 } });
}

/// In which column of the window that pane was. i32.
fn bestColumn(stops: zml.Tensor) zml.Tensor {
    return stops.maxPool2d(.{ .window_dimensions = .{ 2, 2 } }).indices;
}

/// The busiest pane of each pair, row by row.
/// stops: {.b, .c, .h, .w} -> {.b, .c, .h, .w / 2}
fn pairs(stops: zml.Tensor) zml.Tensor {
    return stops.maxPool1d(.{ .window_dimensions = 2 }).values;
}

/// The average of each quarter, the way of exercise 53.
/// stops: {.b, .c, .h, .w} -> {.b, .c, .h / 2, .w / 2}
fn averageByFolding(stops: zml.Tensor) zml.Tensor {
    return ???;
}

/// What `reduceWindow` needs to compute a sum.
fn plus(args: zml.ops.ReduceArgs) [1]zml.Tensor {
    return .{args.left.add(args.right)};
}

/// The average of every 2 x 2 window, moving `stride` panes at a
/// time in both directions. stops: {.b, .c, .h, .w}
fn average(stops: zml.Tensor, stride: i64) zml.Tensor {
    return ???;
}

/// One number for the whole window: {.b, .c, .h, .w} -> {.b, .c}
fn overall(stops: zml.Tensor) zml.Tensor {
    return stops.mean(.h).mean(.w);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const stops: zml.Tensor = .init(.{ .b = 1, .c = 1, .h = 4, .w = 4 }, .f32);

    const counted = [4][4]f32{
        .{ 3, 1, 0, 2 },
        .{ 5, 4, 1, 1 },
        .{ 0, 0, 7, 3 },
        .{ 2, 6, 8, 5 },
    };
    var stops_buffer: zml.Buffer = try .fromBytes(io, platform, stops.shape(), std.mem.sliceAsBytes(&counted));
    defer stops_buffer.deinit();

    const Line = struct { name: []const u8, exe: zml.Exe };
    var lines = [_]Line{
        .{ .name = "best", .exe = try platform.compileFn(gpa, io, best, .{stops}, .{}) },
        .{ .name = "its column", .exe = try platform.compileFn(gpa, io, bestColumn, .{stops}, .{}) },
        .{ .name = "pairs", .exe = try platform.compileFn(gpa, io, pairs, .{stops}, .{}) },
        .{ .name = "average, folded", .exe = try platform.compileFn(gpa, io, averageByFolding, .{stops}, .{}) },
        .{ .name = "average, stride 2", .exe = try platform.compileFn(gpa, io, average, .{ stops, 2 }, .{}) },
        .{ .name = "average, stride 1", .exe = try platform.compileFn(gpa, io, average, .{ stops, 1 }, .{}) },
        .{ .name = "overall", .exe = try platform.compileFn(gpa, io, overall, .{stops}, .{}) },
    };
    defer for (&lines) |*line| line.exe.deinit();

    for (&lines) |*line| {
        var result = try line.exe.eval(gpa, io, .{stops_buffer});
        defer result.deinit();

        const values = try result.toSliceAlloc(gpa, io);
        defer values.free(gpa);

        try stdout.print("{s:<17} {f}:", .{ line.name, result.shape() });
        if (result.shape().dtype() == .i32) {
            for (values.items(i32)) |v| try stdout.print(" {d}", .{v});
        } else {
            for (values.items(f32)) |v| try stdout.print(" {d}", .{v});
        }
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
