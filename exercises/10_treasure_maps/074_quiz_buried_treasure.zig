//
// Quiz! The Off-By-One has reached an archipelago of six
// islands, and Captain Scurvy has a plan with three steps.
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
// The tensors:
//
//     gold     {island=6,i32}
//     sharks   {island=6,i32}
//     parties  {island=6,role=2,i32}
//
// `plan` does steps 1 and 2 and returns the pair of tensors you
// get from `topK`, with the new axis named .best. The small
// functions under it each return ONE tensor, so that plain old
// `exe.eval` is enough. Two of them are already written; step 3
// is yours.
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

fn howMuch(gold: zml.Tensor, sharks: zml.Tensor) zml.Tensor {
    return plan(gold, sharks).values;
}

fn where(gold: zml.Tensor, sharks: zml.Tensor) zml.Tensor {
    return plan(gold, sharks).indices;
}

/// Step 3: {best=2,role=2}, the landing party for each chosen island.
fn landingParties(gold: zml.Tensor, sharks: zml.Tensor, parties: zml.Tensor) zml.Tensor {
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

    const gold: zml.Tensor = .init(.{ .island = 6 }, .i32);
    const sharks: zml.Tensor = .init(.{ .island = 6 }, .i32);
    const parties: zml.Tensor = ???;

    var how_much_exe = try platform.compileFn(gpa, io, howMuch, .{ gold, sharks }, .{});
    defer how_much_exe.deinit();
    var where_exe = try platform.compileFn(gpa, io, where, .{ gold, sharks }, .{});
    defer where_exe.deinit();
    var parties_exe = try platform.compileFn(gpa, io, landingParties, .{ gold, sharks, parties }, .{});
    defer parties_exe.deinit();

    var gold_buffer: zml.Buffer = try .fromBytes(io, platform, gold.shape(), std.mem.sliceAsBytes(&[_]i32{ 40, 90, 15, 70, 85, 99 }));
    defer gold_buffer.deinit();
    var sharks_buffer: zml.Buffer = try .fromBytes(io, platform, sharks.shape(), std.mem.sliceAsBytes(&[_]i32{ 1, 7, 0, 3, 2, 9 }));
    defer sharks_buffer.deinit();
    // For each island: diggers, lookouts.
    var parties_buffer: zml.Buffer = try .fromBytes(io, platform, parties.shape(), std.mem.sliceAsBytes(&[6][2]i32{
        .{ 2, 1 },
        .{ 9, 4 },
        .{ 1, 1 },
        .{ 5, 2 },
        .{ 6, 3 },
        .{ 8, 8 },
    }));
    defer parties_buffer.deinit();

    var how_much: zml.Buffer = try how_much_exe.eval(gpa, io, .{ gold_buffer, sharks_buffer });
    defer how_much.deinit();
    var islands: zml.Buffer = try where_exe.eval(gpa, io, .{ gold_buffer, sharks_buffer });
    defer islands.deinit();
    var landing: zml.Buffer = try parties_exe.eval(gpa, io, .{ gold_buffer, sharks_buffer, parties_buffer });
    defer landing.deinit();

    try stdout.print("landing parties: {f}\n", .{landing.shape()});
    try stdout.flush();

    const doubloons = try how_much.getValue([2]i32, io);
    const which = try islands.getValue([2]i32, io);
    const party = try landing.getValue([2][2]i32, io);
    for (doubloons, which, party) |d, i, p| {
        try stdout.print("Island {d}: {d} doubloons. Send {d} diggers and {d} lookouts.\n", .{ i, d, p[0], p[1] });
    }
    try stdout.flush();
}
