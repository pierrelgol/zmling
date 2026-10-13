//
// Welcome to the Maple Street Origami Club! Motto, painted above
// the door by Coach Brunhilde herself:
//
//     WE DO NOT FOLD UNDER PRESSURE. WE FOLD UNDER CONTROL.
//
// The Regional Championship is eight exercises away, and the
// Northside Paper Cranes have won it three years running. Coach
// has not slept since.
//
// Origami has one sacred rule: no scissors, no glue. You never
// add paper and you never remove any; you only change how it is
// folded. This chapter is about the tensor operations that obey
// that rule. They rearrange a tensor and never touch its
// numbers.
//
// First, a secret. However many axes a tensor claims to have,
// its numbers are stored as ONE long strip:
//
//     shape {2,3}:     { { 0, 1, 2 },
//                        { 3, 4, 5 } }
//
//     the strip:       0 1 2 3 4 5
//
// The strip is read like a book: the LAST axis moves fastest
// ("row-major" order, the same as nested Zig arrays). The shape
// is merely the folding instruction: "crease every 3 numbers".
//
// `reshape` keeps the strip and changes the instruction:
//
//     x.reshape(.{ 3, 2 })       { {0,1}, {2,3}, {4,5} }
//     x.reshape(.{6})            { 0, 1, 2, 3, 4, 5 }
//     x.reshape(.{ 1, 6, 1 })    also fine, if a bit dramatic
//
// It takes what `Shape.init` takes, minus the dtype (reshape
// can't change it), so the new axes can be tagged:
//
//     x.reshape(.{ .rows = 3, .cols = 2 })
//
// The new shape REPLACES the old one entirely. The old tags are
// forgotten: if you want tags on the result, write them.
//
// You may be lazy about ONE size and let ZML do the division:
//
//     x.reshape(.{ .rows = .auto, .cols = 2 })   -> {rows=3,cols=2}
//     x.reshape(.{ -1, 2 })                      -> {3,2}, PyTorch style
//
// And there is a shortcut for "unfold everything":
//
//     x.flatten()        same as x.reshape(.{x.count()})
//
// The one thing reshape will not do is change the number of
// elements. No scissors! Twelve numbers don't fold into rows of
// five, and ZML panics while tracing:
//
//     Can't reshape { 12 } to { 5, -1 }: UnexpectedRemainder
//
// (-1 is how `.auto` looks from the inside.)
//
// Since no number moves, a reshape costs nothing at run time.
// It is the cheapest operation in the whole library, which is
// why you'll see it everywhere.
//
// Warm-up drill! Coach hands you a strip of 12 numbered squares
// and wants three folds, each one starting from the previous:
//
//     grid       3 rows, tagged .rows and .cols
//     booklet    2 pages, each page has rows of 2 squares
//                (tagged .pages, .rows, .cols)
//     unfold     back to a flat strip
//
// Look at the numbers in the output once it works. That's the
// whole lesson.
//
// Four things stand between you and that output: a number Coach
// would not approve of, a fold with no instruction, a fold
// with no name, and a hole in `main` where the strip's Buffer
// should be. You have been making Buffers since exercise 011.
// From this chapter on, nobody holds the paper for you.
//
const std = @import("std");
const zml = @import("zml");

fn grid(strip: zml.Tensor) zml.Tensor {
    // Three rows. Coach said THREE.
    return strip.reshape(.{ .rows = 5, .cols = .auto });
}

fn booklet(strip: zml.Tensor) zml.Tensor {
    // 2 pages, rows of 2 squares, and ZML counts the rows.
    return grid(strip).reshape(???);
}

fn unfold(strip: zml.Tensor) zml.Tensor {
    // The shortcut, please.
    return booklet(strip).???;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const strip: zml.Tensor = .init(.{12}, .i32);

    var squares: [12]i32 = undefined;
    for (&squares, 0..) |*square, i| square.* = @intCast(i);

    // The paper: a Buffer with the placeholder's shape, holding
    // `squares`. Call it `strip_buffer`, and release it when
    // `main` is done.
    ???;

    // One Exe per fold. (`inline for`, because compileFn needs
    // to know each function at compile time.)
    inline for (.{ grid, booklet, unfold }, .{ "grid:   ", "booklet:", "unfold: " }) |fold, name| {
        var exe = try platform.compileFn(gpa, io, fold, .{strip}, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{strip_buffer});
        defer result.deinit();

        const paper: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer paper.free(gpa);

        // The shape, then the numbers exactly as they sit in memory.
        try stdout.print("{s} {f} {any}\n", .{ name, paper.shape, paper.constItems(i32) });
    }
    try stdout.flush();
}
