//
// A cake of forty tiers has a problem that a cake of two tiers
// doesn't: every tier leans a tiny bit, the lean adds up, and
// somewhere around tier thirty the whole thing leaves through
// the window. Deep networks do the same. The numbers coming out
// of each layer drift, a little too big here, a little too
// small there, and after enough layers they are either
// astronomical or zero.
//
// The fix, in both trades: every so often, take out the spirit
// level and make things flat again. That's a NORMALIZATION
// layer. The classic one is the layer norm:
//
//     pub const LayerNorm = struct {
//         weight: Tensor,
//         bias: ?Tensor = null,
//         eps: f32 = 1e-5,
//
//         pub fn forward(self: LayerNorm, x: Tensor) Tensor
//     };
//
// For each vector along the LAST axis of x, `forward` computes:
//
//     mean     = average of the vector
//     variance = average of (x - mean)²
//
//     out = (x - mean) / sqrt(variance + eps) * weight + bias
//
// In words: slide the numbers so their mean is 0, squeeze them
// so their spread (standard deviation) is 1, and then let the
// model have the last word: `weight` sets the spread it wants,
// `bias` sets the mean it wants. Both are learned, both hold one
// number per element of the last axis.
//
// Things the source code tells you and the name does not:
//
//  - There is no `init`. It's a plain struct: `.{ .weight = w,
//    .bias = b, .eps = 1e-5 }`.
//
//  - "The last axis" is meant literally: `x.axis(-1)`. Linear
//    looks for a tag; LayerNorm doesn't care about tags at all
//    and levels whatever axis comes last. Features go last.
//
//  - `eps` is an f32, not a Tensor. It is baked into the Exe
//    while tracing, like the activation of the last exercise.
//
// And what is eps FOR? Take a perfectly flat vector: (5, 5, 5,
// 5). Its mean is 5, so x - mean is all zeros, and its variance
// is 0. Without eps we would compute 0 / sqrt(0), and get a
// vector of NaN. With a tiny eps we compute 0 / sqrt(0.00001),
// which is a perfectly respectable 0. One perfectly even cake
// must not burn down the bakery.
//
// On the bench: three cakes of four tiers, heights in cm. Chef
// Babette wants every cake leveled to a MEAN of 10 with a
// SPREAD of 2. The norm on the bench is set to "do nothing in
// particular", and somebody thought eps was optional.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const weight: zml.Tensor = .init(.{ .d = 4 }, .f32);
    const bias: zml.Tensor = .init(.{ .d = 4 }, .f32);

    // A flat cake must come out flat. Not as four NaN.
    const level: zml.nn.LayerNorm = .{ .weight = weight, .bias = bias, .eps = 0 };

    const cakes: zml.Tensor = .init(.{ .cake = 3, .d = 4 }, .f32);

    var exe = try platform.compile(gpa, io, level, .forward, .{cakes}, .{});
    defer exe.deinit();

    // One number per tier. A mean of 10 and a spread of 2, please.
    const spread: [4]f32 = @splat(1);
    const mean: [4]f32 = @splat(0);

    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), std.mem.sliceAsBytes(&spread));
    defer weight_buffer.deinit();
    var bias_buffer: zml.Buffer = try .fromBytes(io, platform, bias.shape(), std.mem.sliceAsBytes(&mean));
    defer bias_buffer.deinit();

    // Look: no `.eps` in here. Only Tensors become Buffers.
    const level_buffers: zml.Bufferized(zml.nn.LayerNorm) = .{ .weight = weight_buffer, .bias = bias_buffer };

    const heights = [3][4]f32{
        .{ 2, 4, 4, 6 }, // the lopsided one
        .{ 10, 20, 30, 40 }, // the staircase
        .{ 5, 5, 5, 5 }, // the perfectly even one
    };
    var cakes_buffer: zml.Buffer = try .fromBytes(io, platform, cakes.shape(), std.mem.sliceAsBytes(&heights));
    defer cakes_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ level_buffers, cakes_buffer });
    defer result.deinit();

    for (heights, try result.getValue([3][4]f32, io)) |before, after| {
        for (before) |h| try stdout.print("{d:3}", .{h});
        try stdout.print("  ->", .{});
        for (after) |h| try stdout.print(" {d:6.3}", .{h});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
