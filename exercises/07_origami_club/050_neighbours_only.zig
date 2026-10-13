//
// In the last exercise, one word about `merge` was written in
// capitals: it glues ADJACENT axes. Today you find out what
// happens otherwise, because the error you get is one of the
// least helpful in ZML and you should meet it in a safe place.
//
// Why adjacent? Because `merge` is a reshape, and a reshape
// only regroups neighbours. In {layers=2,rows=3,cols=4}:
//
//     .rows and .cols are neighbours: the 4 squares of a row are
//     next to each other in the strip, and the 3 rows of a layer
//     follow each other. Creasing every 12 instead of every 4
//     merges them. Free.
//
//     .layers and .cols are NOT neighbours: .rows sits between
//     them. "Row 0 of layer 0, then row 0 of layer 1" is not a
//     run of the strip. Numbers have to move, and moving numbers
//     is `transpose`'s job.
//
// So the recipe for merging distant axes has two steps: first
// bring them next to each other, then merge.
//
// You could write the transpose by hand. But all you care about
// is "these axes, side by side", and there is a helper that
// says exactly that:
//
//     x.contiguous(.{ .layers, .cols })
//         {layers=2,rows=3,cols=4} -> {rows=3,layers=2,cols=4}
//
// `contiguous` moves the axes you name to the END, in the order
// you name them. The axes you didn't name keep their relative
// order and shift to the front. It is a transpose that computes
// its own permutation.
//
// After that, the merge is legal:
//
//     x.contiguous(.{ .layers, .cols })
//      .merge(.{ .strip = .{ .layers, .cols } })
//         -> {rows=3,strip=8}
//
// The two steps are so often used together that ZML ships them
// as one call (note the different argument order: axes first,
// then the new tag):
//
//     x.mergeTranspose(.{ .layers, .cols }, .strip)
//
// Now the unhelpful error. If you merge non-adjacent axes, ZML
// (at the version pinned by zmling) doesn't check. It computes
// a nonsensical target shape and hands it to the compiler,
// which refuses with:
//
//     error: number of output elements (32) doesn't match
//            expected number of elements (24)
//     error(mlir): Failed to verify MLIR operation:
//     ... "stablehlo.reshape" ... (tensor<2x3x4xi32>) -> tensor<8x4xi32>
//     panic: Failed to create MLIR operation
//
// Not a word about axes or adjacency. The clues are the stack
// trace, which goes through `merge`, and that `8x4`: the merged
// axis (2 * 4 = 8) swallowed the axis next to .layers instead
// of .cols. When a merge dies with an element-count mismatch,
// check your neighbours.
//
// Run the exercise to see it. Then fix it: Coach has a pile of
// 2 layers of paper and wants, for each row, ONE long strip
// that runs through layer 0 and continues into layer 1.
//
const std = @import("std");
const zml = @import("zml");

fn longStrips(pile: zml.Tensor) zml.Tensor {
    return pile.merge(.{ .strip = .{ .layers, .cols } });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const pile: zml.Tensor = .init(.{ .layers = 2, .rows = 3, .cols = 4 }, .i32);

    var exe = try platform.compileFn(gpa, io, longStrips, .{pile}, .{});
    defer exe.deinit();

    // Squares 0..11 are layer 0, squares 12..23 are layer 1.
    var squares: [24]i32 = undefined;
    for (&squares, 0..) |*square, i| square.* = @intCast(i);

    var pile_buffer: zml.Buffer = try .fromBytes(io, platform, pile.shape(), std.mem.sliceAsBytes(&squares));
    defer pile_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{pile_buffer});
    defer result.deinit();

    const strips: zml.Slice = try result.toSliceAlloc(gpa, io);
    defer strips.free(gpa);

    try stdout.print("{f}\n", .{strips.shape});

    const width: usize = @intCast(strips.shape.dim(-1));
    var lines = std.mem.window(i32, strips.constItems(i32), width, width);
    while (lines.next()) |line| try stdout.print("  {any}\n", .{line});
    try stdout.flush();
}
