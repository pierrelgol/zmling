//
// Quiz! The Off-By-One has reached an archipelago of six
// islands, and Captain Scurvy has a plan with three steps.
// (There are four. It is that kind of ship.)
//
//   1. An island with MORE THAN 3 sharks around it is out of
//      the question. Its gold counts as 0.
//
//   2. Of what remains, pick the 2 islands with the most gold.
//      The Captain wants to know how much, and where.
//
//   3. Each island needs a landing party: so many diggers and so
//      many lookouts, listed in `parties` {island=6,role=2}. Look
//      up the parties for the two islands chosen, best island
//      first.
//
//   4. Add those two parties up: how many diggers and how many
//      lookouts leave the ship, in all.
//
// The tensors:
//
//     gold     {island=6,i32}
//     sharks   {island=6,i32}
//     parties  {island=6,role=2,i32}
//
// Three functions, and none of them is written:
//
//     plan(gold, sharks)                     zml.Tensor.SortRes
//         Steps 1 and 2: the pair of tensors you get from
//         `topK`, with the new axis named .best.
//
//     landingParties(gold, sharks, parties)  {best=2,role=2}
//         Step 3. It calls `plan`.
//
//     crew(gold, sharks, parties)            {role=2}
//         Step 4. It calls `landingParties`.
//
// `plan` has two outputs: it needs the runner of the previous
// exercise, and a `zml.Bufferized(zml.Tensor.SortRes)` called
// `res`. The other two return one tensor each, and plain old
// `exe.eval` is enough: call their results `landing` and
// `party_total`.
//
// `main` knows the numbers and what to print. Everything in
// between is yours: placeholders, Exes, Buffers, the runner,
// and something to free each of them.
//
// Your tools:
//
//     a.cmp(.GT, b)   .EQ .NE .GE .LT .LE     -> bool tensor
//     x.mask(pred, value)                     keeps where true
//     pred.select(on_true, on_false)
//     x.topK(.{ .new = .old }, k, .{})        .values, .indices
//     t.gather(.{ .axis = indices }, .{})
//
// Polly has been asked for a hint and said "false".
//
const std = @import("std");
const zml = @import("zml");

/// Steps 1 and 2: the two best islands that are safe to land on.
fn plan(gold: zml.Tensor, sharks: zml.Tensor) zml.Tensor.SortRes {
    return ???;
}

// `landingParties` and `crew` go here.

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const gold_data = [_]i32{ 40, 90, 15, 70, 85, 99 };
    const sharks_data = [_]i32{ 1, 7, 0, 3, 2, 9 };
    // For each island: diggers, lookouts.
    const parties_data = [6][2]i32{
        .{ 2, 1 },
        .{ 9, 4 },
        .{ 1, 1 },
        .{ 5, 2 },
        .{ 6, 3 },
        .{ 8, 8 },
    };

    // Your code goes here. It ends with `res` (filled in by a
    // runner), `landing` and `party_total`.
    ???;

    try stdout.print("landing parties: {f}\n", .{landing.shape()});
    try stdout.flush();

    const doubloons = try res.values.getValue([2]i32, io);
    const which = try res.indices.getValue([2]i32, io);
    const party = try landing.getValue([2][2]i32, io);
    for (doubloons, which, party) |d, i, p| {
        try stdout.print("Island {d}: {d} doubloons. Send {d} diggers and {d} lookouts.\n", .{ i, d, p[0], p[1] });
    }
    const total = try party_total.getValue([2]i32, io);
    try stdout.print("That makes {d} diggers and {d} lookouts. Polly stays on the ship.\n", .{ total[0], total[1] });
    try stdout.flush();
}
