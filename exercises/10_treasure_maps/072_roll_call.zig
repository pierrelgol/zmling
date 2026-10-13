//
// Payday. By tradition the crew lines up richest first, and the
// Captain reads the roll.
//
//     x.sort(axis, .{})                       ascending
//     x.sort(axis, .{ .descending = true })
//
// sorts along one axis (each row on its own, if there are
// others) and returns not a Tensor but a small struct of two:
//
//     pub const SortRes = struct {
//         values: Tensor,     // the sorted elements
//         indices: Tensor,    // where each one CAME FROM, as .i32
//     };
//
// Both have the shape of `x`. `indices` is the interesting half.
// It is the permutation that sorts `x`: indices[0] is the
// position, in the original tensor, of the element that ended
// up first:
//
//     loot      { 30, 80,  5, 80, 12 }
//     values    { 80, 80, 30, 12,  5 }     descending
//     indices   {  1,  3,  0,  4,  2 }     where each came from
//
// When that's the only half you want, there is
//
//     x.argsort(axis, .{ .descending = true })      // a Tensor
//
// which is `sort(...).indices`, word for word.
//
// The sort is stable: elements that compare equal keep the
// order they had. Two pirates with 80 doubloons each stay in
// the order they signed on.
//
// Why would you want a permutation? Because you rarely sort one
// thing. The crew has loot AND ages AND hooks for hands, in
// separate tensors. Sort the loot, and the ages have to follow.
// Positions in, elements out: that is `gather`.
//
//     const order = loot.argsort(.pirate, .{ .descending = true });
//     age.gather(.{ .pirate = order }, .{})
//
// ...and that panics:
//
//     gather expects axes to appear at most twice. Axis pirate
//     has been found both in 'self=...', in 'idx_axes=...' and
//     in 'indices=...'
//
// Think about what `order` is. It has the shape of `loot`, tags
// included: {pirate=5,i32}. You are asking to gather along
// .pirate with an index tensor that itself has an axis called
// .pirate, which by the batching rule of the previous exercise
// means "for each pirate separately". It can't be both. And ZML
// has a point: the axis of `order` isn't "pirate" any more, it
// is "place in the line". Call it that, .rank, before it goes
// into `gather` (a `rename`, as for Chad's Synergy Matrix in
// exercise 63), and the result comes out as {rank=5}, which is
// what it is.
//
// Since `exe.eval` hands back one Buffer, the functions below
// each return a single Tensor. (Returning the whole SortRes is
// the next exercise.) The cabin boy sorted the wrong way round
// in one of them, and didn't rename in the other.
//
// The ship's surgeon would like one more number: the average
// age of the THREE richest pirates. That is the front of what
// `agesInLine` returns (call it, then take a slice along .rank:
// chapter 8), turned into .f32 because a mean of integers is an
// integer, and reduced to a single number with no axis left.
// `ageOfTheTopThree` is yours.
//
// The cabin boy has also dropped the middle of `main` over
// board. The placeholders and the numbers survived, and so did
// the prints. In between: three Exes and two input Buffers
// (name them as you like), and three results that the prints
// call `richest`, `ages` and `average`.
//
const std = @import("std");
const zml = @import("zml");

/// The loot, richest pirate first.
fn richestFirst(loot: zml.Tensor) zml.Tensor {
    return loot.sort(.pirate, .{}).values;
}

/// The ages of the pirates, in the order of the pay line (richest first).
fn agesInLine(loot: zml.Tensor, age: zml.Tensor) zml.Tensor {
    const order = loot.argsort(.pirate, .{ .descending = true });
    return age.gather(.{ .pirate = order }, .{});
}

/// The average age of the first three in the pay line: one .f32, no axis.
fn ageOfTheTopThree(loot: zml.Tensor, age: zml.Tensor) zml.Tensor {
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

    const loot: zml.Tensor = .init(.{ .pirate = 5 }, .i32);
    const age: zml.Tensor = .init(.{ .pirate = 5 }, .i32);

    // In the order they signed on: Anne, Bart, Calico, Dread, Edna.
    const loot_data = [_]i32{ 30, 80, 5, 80, 12 };
    const age_data = [_]i32{ 41, 19, 63, 27, 35 };

    // Over board: three Exes, a Buffer for each of the two lists
    // above, and three results called `richest`, `ages` and
    // `average`.
    ???;

    try stdout.print("the pay line {f}: {any}\n", .{ richest.shape(), try richest.getValue([5]i32, io) });
    try stdout.print("their ages   {f}: {any}\n", .{ ages.shape(), try ages.getValue([5]i32, io) });
    try stdout.print("the first three are {d:.1} years old on average.\n", .{try average.getValue(f32, io)});
    try stdout.flush();
}
