//
// Scandal at the Maple Street Origami Club. The scoreboard says
// that Timmy, who last week glued a crane to his own elbow,
// scored an 8 in round one.
//
// Nobody cheated. Somebody reshaped.
//
// Here is the most classic bug in all of tensor programming,
// and you are going to commit it on purpose, once, here, so
// that you never commit it again at 2 a.m. inside a model.
//
// The scores come as one row per member:
//
//     scores  {members=2,rounds=3}       round: 1  2   3
//                                 Ingrid    { { 9, 8, 10 },
//                                 Timmy       { 2, 3,  1 } }
//
// The scoreboard wants one row per ROUND: {rounds=3,members=2}.
// There are two ways to obtain a tensor of that shape.
//
//     scores.reshape(.{ .rounds = 3, .members = 2 })
//
//         Keeps the strip 9 8 10 2 3 1 and creases it every 2:
//
//                   { {  9, 8 },
//                     { 10, 2 },
//                     {  3, 1 } }
//
//     scores.transpose(.{ .rounds, .members })
//
//         Keeps every score attached to its (member, round) and
//         reorders the axes:
//
//                   { {  9, 2 },
//                     {  8, 3 },
//                     { 10, 1 } }
//
// SAME shape. SAME tags. SAME dtype. Same six numbers, even.
// Every shape check in the world passes on both. The compiler
// is happy, ZML is happy, nothing panics, and one of the two is
// garbage: it took Ingrid's round-2 score and filed it under
// Timmy.
//
// And notice that the tags did not save us. `reshape` lets you
// write any tags you like on the result; it has no idea what
// they mean. Writing `.rounds` on an axis doesn't make it the
// rounds axis.
//
// How to tell which one you need:
//
//   - `reshape` regroups NEIGHBOURS. Numbers that were next to
//     each other in the strip stay next to each other. Use it
//     to split an axis in two, to merge two adjacent axes, to
//     add or drop axes of size 1.
//
//   - `transpose` changes WHO IS NEIGHBOUR OF WHOM. Use it
//     whenever the order of the axes changes.
//
// If the sentence in your head is "I want the same axes in a
// different order", the answer is never reshape.
//
// (For the curious: there is one case where both agree. Moving
// an axis of size 1 around doesn't reorder the strip, and
// ZML's `transpose` quietly emits a free reshape when it
// notices that.)
//
// Part one: fix the scoreboard. Timmy would like you to know
// that he is fine with his real scores, and that the elbow
// thing was a design choice. Whoever wrote `main` had the old
// layout in mind too: check how the board is read back.
//
// Part two: the judges also want, for each round, the GAP
// between the best and the worst score of that round, as a
// {rounds=3}. Write `gaps`, starting from the scoreboard. You
// met `max`, `min` and the axis of size 1 they leave behind in
// the Reduction Kitchen (exercises 040 and 041); bring them
// back. Then give `gaps` its own Exe in `main`: compile, run,
// read back. Only the print is waiting for you there.
//
// When it all works, try this for fun: make `gaps` start from
// `scores` instead of the scoreboard. Same answer! A reduction
// finds its axis by TAG and doesn't care where the axis sits.
// Now put the reshape back in `scoreboard` and look at the
// gaps again. Tags are only as honest as whoever wrote them.
//
const std = @import("std");
const zml = @import("zml");

/// One row per member in, one row per round out.
fn scoreboard(scores: zml.Tensor) zml.Tensor {
    return scores.reshape(.{ .rounds = 3, .members = 2 });
}

/// For each round, best score minus worst score: {rounds=3}.
fn gaps(scores: zml.Tensor) zml.Tensor {
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

    const scores: zml.Tensor = .init(.{ .members = 2, .rounds = 3 }, .i32);

    var exe = try platform.compileFn(gpa, io, scoreboard, .{scores}, .{});
    defer exe.deinit();

    const sheet = [2][3]i32{
        .{ 9, 8, 10 }, // Ingrid
        .{ 2, 3, 1 }, // Timmy
    };
    var scores_buffer: zml.Buffer = try .fromBytes(io, platform, scores.shape(), std.mem.sliceAsBytes(&sheet));
    defer scores_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{scores_buffer});
    defer result.deinit();

    // A nested Zig array is row-major too: one row per ROUND.
    const board = try result.getValue([2][3]i32, io);

    // The same three steps for `gaps`: an Exe called `gaps_exe`,
    // its result, and the three numbers in an array called `gap`.
    ???;

    try stdout.print("Scoreboard {f}\n", .{exe.output_shapes[0]});
    for (board, 1..) |round, i| {
        try stdout.print("round {d}: Ingrid {d}, Timmy {d}\n", .{ i, round[0], round[1] });
    }
    try stdout.print("Gaps {f}: {any}\n", .{ gaps_exe.output_shapes[0], gap });
    try stdout.flush();
}
