//
// Every broadcast you did so far was really two steps in a
// trench coat:
//
//   1. give the small tensor the RANK of the big one, by adding
//      axes of size 1 in the right places;
//
//   2. STRETCH those size 1 axes to the size that is wanted.
//
// A size 1 axis is the universal adapter of the tensor world.
// It holds no extra data ({3} and {3,1} and {1,3,1} all contain
// the same 3 numbers), but it gives the tensor a direction:
//
//     {anchor=3,segment=1}     a        a COLUMN: one value per
//                              b        anchor, and room to grow
//                              c        along .segment
//
//     {anchor=1,segment=4}     a b c d  a ROW: one value per
//                                       segment, and room to grow
//                                       along .anchor
//
// And every broadcast function accepts a size 1 axis where the
// target has any size: that's the "1-sized" of the long panic
// message from the previous exercise.
//
// If you come from PyTorch you are now looking for `unsqueeze`.
// It doesn't exist. In ZML an axis deserves a name, so the two
// functions that do step 1 make you give one:
//
//     pub fn insertAxes(self: Tensor, axis_: anytype, tags: anytype) Tensor
//
// Inserts size 1 axes BEFORE the given axis (a tag or a number),
// one per tag in the tuple `tags`. As a special case, the axis
// `.last` means "after everything":
//
//     x: {a=5,b=4}
//     x.insertAxes(.b, .{.c})        -> {a=5,c=1,b=4}
//     x.insertAxes(.a, .{ .c, .d })  -> {c=1,d=1,a=5,b=4}
//     x.insertAxes(.last, .{.c})     -> {a=5,b=4,c=1}
//
//     pub fn appendAxes(self: Tensor, t: anytype) Tensor
//
// The same as `insertAxes(.last, t)`:
//
//     x.appendAxes(.{.c})            -> {a=5,b=4,c=1}
//
// Nothing is copied or computed: this is a relabeling of the
// same data (you'll see in a later chapter that it is a
// `reshape` under the hood).
//
// "But `broad` did both steps for me, by tag!" It did, and when
// it can, let it. Doing step 1 by hand is for the times when
// nobody can do it for you: the axis you need isn't in any shape
// yet, or (next exercise) you want ZML's operators to do step 2
// on their own.
//
// Today: payroll. Each anchor has a fee per minute on air,
// {anchor=3}. Each segment of the show has a length in minutes,
// {segment=4}. Accounting wants the full table: what every
// anchor would cost for every segment, {anchor=3,segment=4}.
// Neither input has the axis of the other, so: make a column,
// make a row, stretch both, multiply.
//
// (If the 1 lands on the wrong side you'll be told, for instance
// "Can't broadcast Tensor({segment=1,anchor=3,f32}) to
// {anchor=3,segment=4,f32}". Notice WHY it could tell, though:
// 3 is not 4. Hold that thought until exercise 037.)
//
const std = @import("std");
const zml = @import("zml");

fn payTable(fees: zml.Tensor, minutes: zml.Tensor) zml.Tensor {
    // You've built Shapes from numbers before. `dim` returns a
    // number. So you can build a Shape from other shapes' sizes.
    const table = zml.Shape.init(.{ .anchor = fees.dim(.anchor), .segment = minutes.dim(.segment) }, .f32);

    // {anchor=3} -> {anchor=3,segment=1}
    const column = fees.appendAxes(???);

    // {segment=4} -> {anchor=1,segment=4}
    const row = minutes.insertAxes(???, ???);

    return column.broad(table).mul(row.broad(table));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const fees: zml.Tensor = .init(.{ .anchor = 3 }, .f32);
    const minutes: zml.Tensor = .init(.{ .segment = 4 }, .f32);

    var exe = try platform.compileFn(gpa, io, payTable, .{ fees, minutes }, .{});
    defer exe.deinit();

    const anchors = [_][]const u8{ "Brock", "Linda", "Gail" };

    // Dollars per minute. Gail also does traffic, for free.
    var fees_buffer: zml.Buffer = try .fromBytes(io, platform, fees.shape(), std.mem.sliceAsBytes(&[_]f32{ 10, 20, 15 }));
    defer fees_buffer.deinit();

    // News, weather, sports, and a cat who can skateboard.
    var minutes_buffer: zml.Buffer = try .fromBytes(io, platform, minutes.shape(), std.mem.sliceAsBytes(&[_]f32{ 12, 3, 5, 2 }));
    defer minutes_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ fees_buffer, minutes_buffer });
    defer result.deinit();

    const pay = try result.getValue([3][4]f32, io);

    try stdout.print("{f}\n", .{result.shape()});
    try stdout.print("       news weather sports cat\n", .{});
    for (anchors, pay) |anchor, row| {
        try stdout.print("{s:<6} {d:>4} {d:>7} {d:>6} {d:>3}\n", .{ anchor, row[0], row[1], row[2], row[3] });
    }
    try stdout.flush();
}
