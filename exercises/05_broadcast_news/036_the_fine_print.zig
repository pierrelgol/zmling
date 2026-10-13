//
// Time to read the fine print of Channel 5's contract.
//
// "Only scalars broadcast on their own" was a simplification.
// Here is what `a.add(b)` (and `sub`, `mul`, `div`, `pow`,
// `maximum`, ... every two-tensor operation) really checks, in
// this order. It is a dozen lines called `binaryOpHelper` in
// zml/zml/tensor.zig; go and read them, they are short.
//
//   0. Same dtype, or panic. You knew that.
//
//   1. If one side has rank 0, it is broadcast to the other.
//      Left or right, doesn't matter. You knew that too.
//
//   2. If both sides have the same sizes, axis by axis: done.
//      The TAGS are not compared here! {city=3} plus {day=3} is
//      legal, and the result keeps the tags of `a`. Stay awake.
//
//   3. Otherwise, there is ONE more chance:
//
//        - `a` and `b` have exactly the same tags, in the same
//          order (no tags at all on both sides counts), and
//
//        - `b`, the RIGHT side, can be stretched to the shape of
//          `a`: wherever the sizes differ, `b` has size 1.
//
//      Then `b` is broadcast to the shape of `a`, automatically.
//
//   4. Otherwise: "add expects tensor shapes to match, got ...".
//
// Rule 3 is narrow on purpose. Same tags in the same order means
// there is nothing to guess: every axis of `b` sits under its
// twin in `a`, and a size 1 axis can only mean "the same value
// all along". It exists for one very common pattern, "a table,
// and one value per row of that table":
//
//     scores: {anchor=3,night=4}
//     best:   {anchor=3,night=1}       (a column)
//
//     scores.div(best)        fine: `best` is stretched
//
// But read rule 3 again, slowly. It only ever stretches the
// RIGHT side. The result of `a.op(b)` always has the shape of
// `a`: the left side is the boss.
//
//     best.div(scores)        panic: div expects tensor shapes to
//                             match, got {anchor=3,night=1,f32}
//                             and {anchor=3,night=4,f32}
//
// And it won't stretch both sides at once either: a column on
// the left and a row on the right ({anchor=3,night=1} and
// {anchor=1,night=4}) is a panic, not a table. That one was your
// job in the previous exercise.
//
// When the small tensor has to be on the left, you say so
// yourself, with `broad`, like in the old days of exercise 033.
//
// Wardrobe department. Every anchor has a wardrobe budget per
// night, {anchor=3}, and wardrobe wrote down what they really
// spent, {anchor=3,night=4}. Management wants two tables.
//
// FIRST, which fraction of the budget was left, each night
// (below zero when somebody went over):
//
//     left = (budget - spent) / budget
//
// The author of `leftover` made a column out of `budget`, good
// idea, and then believed the slogan about automatic stretching.
// Half of that line is fine and half of it panics. (By the way:
// the panic says "subtract", not "sub". ZML reports the name the
// operation has inside the compiler.)
//
// SECOND, how many dollars each anchor went OVER budget, each
// night. Somebody who stayed under budget is 0 dollars over, not
// "minus 50 dollars over": negative amounts become 0, and Gus's
// diner had the very thing for that on its menu (exercise 020).
//
//     over = spent - budget, but never below 0
//
// Nobody wrote `overspend`. It is yours, and if you put the
// tensors on the right sides, the fine print does all the
// stretching: no broadcast function needed.
//
// Nobody compiled it or ran it either. `main` makes the Buffers
// and prints the two tables; the lines in between, for the
// second table, are for you. It is the same pipeline as for the
// first one, and the same Buffers will do.
//
// (Write what is missing first. The panic waits for you behind
// the compile errors.)
//
const std = @import("std");
const zml = @import("zml");

/// The fraction of the budget that was left, each night.
fn leftover(spent: zml.Tensor, budget: zml.Tensor) zml.Tensor {
    // {anchor=3} -> {anchor=3,night=1}
    const column = budget.appendAxes(.{.night});

    return column.sub(spent).div(column);
}

/// Dollars over budget, each night. Never negative.
fn overspend(spent: zml.Tensor, budget: zml.Tensor) zml.Tensor {
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

    const spent: zml.Tensor = .init(.{ .anchor = 3, .night = 4 }, .f32);
    const budget: zml.Tensor = .init(.{ .anchor = 3 }, .f32);

    var exe = try platform.compileFn(gpa, io, leftover, .{ spent, budget }, .{});
    defer exe.deinit();

    // The second table: `overspend`, compiled for the same two
    // placeholders.

    const anchors = [_][]const u8{ "Brock", "Linda", "Gail" };

    // Brock's Wednesday was a tie. A very expensive tie. Linda's
    // Thursday was the awards dinner.
    var spent_buffer: zml.Buffer = try .fromBytes(io, platform, spent.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 50, 25, 130, 0 },
        .{ 100, 150, 20, 260 },
        .{ 5, 10, 25, 50 },
    }));
    defer spent_buffer.deinit();

    var budget_buffer: zml.Buffer = try .fromBytes(io, platform, budget.shape(), std.mem.sliceAsBytes(&[_]f32{ 100, 200, 50 }));
    defer budget_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ spent_buffer, budget_buffer });
    defer result.deinit();

    // Run the second Exe, and keep its result in `over_result`.

    const left = try result.getValue([3][4]f32, io);
    const over = try over_result.getValue([3][4]f32, io);

    try stdout.print("Budget left:\n", .{});
    for (anchors, left) |anchor, row| {
        try stdout.print("{s:<5}", .{anchor});
        for (row) |fraction| try stdout.print(" {d:.2}", .{fraction});
        try stdout.print("\n", .{});
    }
    try stdout.print("Dollars over budget:\n", .{});
    for (anchors, over) |anchor, row| {
        try stdout.print("{s:<5}", .{anchor});
        for (row) |dollars| try stdout.print(" {d:>3}", .{dollars});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
