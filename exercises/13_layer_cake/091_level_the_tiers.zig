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
// On a napkin, with the first cake of today, tiers of 2, 4, 4
// and 6 cm, leveled to a spread of 2 around a mean of 10:
//
//     x                  2       4       4       6
//     mean               4
//     x - mean          -2       0       0       2
//     variance          (4 + 0 + 0 + 4) / 4  =  2
//     / sqrt(variance)  -1.414   0       0       1.414
//     * 2 + 10           7.172  10      10      12.828
//
// The shape of the cake survives (short, middle, middle, tall),
// its size does not: (20, 40, 40, 60) would give the very same
// last line.
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
// SPREAD of 2. And she wants it done twice, because she trusts
// the pantry exactly as far as she can throw it:
//
//  1. FROM THE PANTRY, with `zml.nn.LayerNorm`. The norm on the
//     bench is set to "do nothing in particular", somebody
//     thought eps was optional, and nobody made its Buffers.
//
//  2. BY HAND. Write `levelByHand`: the formula above, line by
//     line like the napkin, with the tools of the reduction
//     kitchen and of the diner (chapter 3). Use 1e-5 for eps.
//     Two things to remember from older kitchens:
//
//      - `mean` (exercise 041) keeps the axis it reduces, with
//        a size of 1. That is what you want here: `{cake=3,d=1}`
//        on the right of a `sub` or a `div` stretches along `.d`
//        by itself (exercise 036);
//      - `weight` and `bias` are `{d=4}`, one axis short of
//        `{cake=3,d=4}`. Those do not stretch by themselves:
//        `broad` them to the shape of x (exercise 033), as
//        Linear did with its bias.
//
//     Then compile it and run it: it is a plain function of
//     three tensors.
//
// Same numbers on both sides, or somebody is not going home.
//
const std = @import("std");
const zml = @import("zml");

/// The layer norm of the lesson, by hand, with an eps of 1e-5.
/// x: {.cake, .d}, weight and bias: {.d}. Returns {.cake, .d}.
fn levelByHand(x: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    return ???;
}

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

    // TODO: `hand_exe`, your function, compiled for the same three
    // placeholders.

    // One number per tier. A mean of 10 and a spread of 2, please.
    const spread: [4]f32 = @splat(1);
    const mean: [4]f32 = @splat(0);

    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), std.mem.sliceAsBytes(&spread));
    defer weight_buffer.deinit();
    var bias_buffer: zml.Buffer = try .fromBytes(io, platform, bias.shape(), std.mem.sliceAsBytes(&mean));
    defer bias_buffer.deinit();

    // The Buffers of the norm. Only Tensors become Buffers.
    const level_buffers: zml.Bufferized(zml.nn.LayerNorm) = ???;

    const heights = [3][4]f32{
        .{ 2, 4, 4, 6 }, // the lopsided one
        .{ 10, 20, 30, 40 }, // the staircase
        .{ 5, 5, 5, 5 }, // the perfectly even one
    };
    var cakes_buffer: zml.Buffer = try .fromBytes(io, platform, cakes.shape(), std.mem.sliceAsBytes(&heights));
    defer cakes_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ level_buffers, cakes_buffer });
    defer result.deinit();

    var by_hand: zml.Buffer = ???;
    defer by_hand.deinit();

    const tables = [2][3][4]f32{ try result.getValue([3][4]f32, io), try by_hand.getValue([3][4]f32, io) };
    for ([_][]const u8{ "From the pantry:", "By hand:" }, tables) |title, table| {
        try stdout.print("{s}\n", .{title});
        for (heights, table) |before, after| {
            for (before) |h| try stdout.print("{d:3}", .{h});
            try stdout.print("  ->", .{});
            for (after) |h| try stdout.print(" {d:6.3}", .{h});
            try stdout.print("\n", .{});
        }
    }
    try stdout.flush();
}
