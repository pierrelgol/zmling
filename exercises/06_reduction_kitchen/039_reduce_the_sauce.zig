//
// Chapter 6: The Reduction Kitchen
//
// Television wasn't for you (the cat on the skateboard got your
// time slot). You are now a commis, the lowest rank there is, in
// the kitchen of La Grande Réduction. It is run by Chef Odile,
// who has one answer to every question: "Reduce it."
//
// Two liters of stock go into the pan. An hour later a few
// spoonfuls of sauce come out, and everything that mattered is
// still in there. Tensors get the same treatment:
//
//     A REDUCTION combines all the values along one axis into a
//     single value, and so makes that axis disappear.
//
// Sum, mean, max, min: all reductions. A model's loss is a
// reduction (thousands of errors boiled down to one number), and
// so is "which word comes next" (fifty thousand scores boiled
// down to one index). The simplest of the family:
//
//     pub fn sum(self: Tensor, axis_: anytype) Tensor
//
// The axis is a tag, or a number (negative numbers count from
// the end, as usual). Tonight's orders, as a table:
//
//     orders: {table=3,dish=4}        soup duck risotto tart
//                            table 0     2    1       0    2
//                            table 1     0    2       2    1
//                            table 2     1    1       3    4
//
//     orders.sum(.table)   ->            3    4       5    7
//     orders.sum(.dish)    ->   5, 5 and 9, one per table
//
// Careful, everybody gets this backwards once: the axis you name
// is the one that gets BOILED AWAY, not the one that stays.
// `sum(.table)` is not "the sum for each table". It is "the
// tables are gone": what remains is one number per dish.
//
// Now the surprise, if you come from numpy or PyTorch. Ask for
// the shape of `orders.sum(.table)`:
//
//     {table=1,dish=4}        and not {dish=4}
//
// In ZML the reduced axis doesn't vanish: it stays, with size 1.
// (PyTorch people: it's always `keepdim=True`.) ZML's own doc
// comment promises it: "Output shape is the input shape with the
// axis_ dim set to 1". Two reasons to like this:
//
//   - the result is still lined up with its input, tag for tag.
//     And you have read the fine print of exercise 036: same
//     tags, size 1 on the right... `orders.div(orders.sum(.dish))`
//     just works.
//
//   - a size 1 axis is cheap to remove (next exercise), and a
//     removed axis is annoying to put back.
//
// 7:15 pm. Chef Odile wants to know how many portions of each
// dish to fire. "The tables, I do not care about. Reduce them."
//
// The ritual needs its axis, the read-back needs a size, and in
// between somebody forgot the most important step of any
// kitchen: nothing was ever put on the fire. `exe` is compiled
// and never run. You have run Exes since exercise 011.
//
const std = @import("std");
const zml = @import("zml");

/// How many portions of each dish, all tables together?
fn toFire(orders: zml.Tensor) zml.Tensor {
    return orders.sum(???);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const orders: zml.Tensor = .init(.{ .table = 3, .dish = 4 }, .i32);

    var exe = try platform.compileFn(gpa, io, toFire, .{orders}, .{});
    defer exe.deinit();

    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[3][4]i32{
        .{ 2, 1, 0, 2 },
        .{ 0, 2, 2, 1 },
        .{ 1, 1, 3, 4 },
    }));
    defer orders_buffer.deinit();

    // Run `exe` on the orders. The Buffer that comes back is
    // called `result`, and it has to be released like the others.

    // Two axes, but how many numbers?
    const portions = try result.getValue([???]i32, io);

    try stdout.print("{f} reduced to {f}\n", .{ orders.shape(), result.shape() });
    for ([_][]const u8{ "soup", "duck", "risotto", "tart" }, portions) |dish, n| {
        try stdout.print("Fire {d} {s}!\n", .{ n, dish });
    }
    try stdout.flush();
}
