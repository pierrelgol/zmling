//
// Every pizzeria has a Mr. Henderson. Ours comes in on
// Thursdays, orders the big rectangular pan pizza, and eats
// only the pieces that don't touch the edge. "I don't do
// crust," he says, every Thursday, as if it were news.
//
// The pan is a tensor {rows=4,cols=6}, and "no crust" is a cut
// along TWO axes: drop the first and last row, and the first
// and last column. You already know how to do it by chaining:
//
//     pan.slice(.rows, .{ .start = 1, .end = -1 })
//        .slice(.cols, .{ .start = 1, .end = -1 })
//
// That works and there is nothing wrong with it. When several
// axes are involved there is also a plural form that does all
// the cuts in one operation:
//
//     pan.slices(.{ .rows, .cols }, &.{
//         .{ .start = 1, .end = -1 },
//         .{ .start = 1, .end = -1 },
//     })
//
// The signature:
//
//     pub fn slices(self: Tensor, axes_: anytype, slices_: []const Slice) Tensor
//
// `axes_` is a tuple of axes (tags or indices), and `slices_`
// holds one `zml.Tensor.Slice` per axis listed, IN THE SAME
// ORDER. The first Slice cuts the first axis of the tuple, the
// second cuts the second. Axes that aren't listed are kept
// whole. The order of the tuple is up to you and doesn't have
// to follow the order of the axes in the tensor:
//
//     pan.slices(.{ .cols, .rows }, &.{ cut_for_cols, cut_for_rows })
//
// Note the `&.{ ... }`: the second argument is a real Zig
// slice of Slices. (A slice of Slices, to cut slices. Nonna
// Rosa finds this hilarious.) Because that parameter has a
// proper type, everything inside is known to be a
// `zml.Tensor.Slice`, and all the forms from the previous
// exercises work without spelling the type:
//
//     &.{ .{ .end = 2 }, .single(0), .full }
//
// `.single(i)` keeps its meaning: that axis is cut down to one
// index and then dropped from the result. So `slices` can
// lower the rank by as many axes as you mark single:
//
//     pan.slices(.{ .rows, .cols }, &.{ .single(0), .single(0) })
//         -> a rank-0 tensor: the top-left piece
//
// Tonight's pan has its pieces numbered so that you can read
// the position in the number: piece 23 is row 2, column 3.
//
//   - `noCrust`: Mr. Henderson's order. No piece from an edge.
//
//   - `leftEdge`: Nonna wants to check how the crust is baking.
//     From the left-most column, one piece out of two starting
//     from the top. The result has a single axis, .rows.
//
// Luigi wrote `leftEdge`. Read his two tuples side by side.
//
const std = @import("std");
const zml = @import("zml");

/// Everything but the border: {rows=2,cols=4}.
fn noCrust(pan: zml.Tensor) zml.Tensor {
    return pan.slices(.{ .rows, .cols }, ???);
}

/// Column 0, every other row: {rows=2}.
fn leftEdge(pan: zml.Tensor) zml.Tensor {
    return pan.slices(.{ .rows, .cols }, &.{ .single(0), .{ .step = 2 } });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const pan: zml.Tensor = .init(.{ .rows = 4, .cols = 6 }, .i32);

    var pieces: [4][6]i32 = undefined;
    for (&pieces, 0..) |*row, r| {
        for (row, 0..) |*piece, c| piece.* = @intCast(r * 10 + c);
    }
    var pan_buffer: zml.Buffer = try .fromBytes(io, platform, pan.shape(), std.mem.sliceAsBytes(&pieces));
    defer pan_buffer.deinit();

    inline for (.{ noCrust, leftEdge }, .{ "noCrust", "leftEdge" }) |order, name| {
        var exe = try platform.compileFn(gpa, io, order, .{pan}, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{pan_buffer});
        defer result.deinit();

        const plate: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer plate.free(gpa);

        try stdout.print("{s}: {f}\n", .{ name, plate.shape });

        // One line per run of the last axis.
        const width: usize = @intCast(plate.shape.dim(-1));
        var lines = std.mem.window(i32, plate.constItems(i32), width, width);
        while (lines.next()) |line| try stdout.print("  {any}\n", .{line});
    }
    try stdout.flush();
}
