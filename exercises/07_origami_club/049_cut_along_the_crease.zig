//
// "No scissors" is the rule, but Coach Brunhilde allows TEARING
// along a crease, "as our ancestors did". Good news: that is
// exactly the legitimate use of reshape from the last exercise.
// Regrouping neighbours.
//
// Doing it with a bare `reshape` is tedious, though. You must
// restate every axis of the tensor, including all the ones you
// didn't want to touch, and retype all their tags:
//
//     // x: {batch=16,seq=128,d=512}; cut .d into 8 pieces
//     x.reshape(.{ .batch = 16, .seq = 128, .h = 8, .hd = 64 })
//
// One day `seq` becomes 256 and this line is wrong. ZML has two
// tools that only talk about the axes they change.
//
// `splitAxis` cuts ONE axis into several, in place:
//
//     x.splitAxis(.d, .{ .h = 8, .hd = .auto })
//         {batch=16,seq=128,d=512} -> {batch=16,seq=128,h=8,hd=64}
//
// First argument: the axis to cut (tag or index). Second: what
// replaces it, written like a little shape. The sizes must
// multiply to the size of the cut axis, and one of them may be
// `.auto`. The other axes are not mentioned and not touched.
//
// Which element goes where? It's a reshape, so the strip
// doesn't move: the first 64 values of `.d` become h=0, the
// next 64 become h=1, and so on. On a tensor small enough to
// see:
//
//     t {a=2,b=4}                    { { 0, 1, 2, 3 },
//                                      { 4, 5, 6, 7 } }
//
//     t.splitAxis(.b, .{ .c = 2, .d = 2 })
//
//       {a=2,c=2,d=2}                { { {0,1}, {2,3} },
//                                      { {4,5}, {6,7} } }
//
// Every row was torn down the middle, and no number crossed
// from one row to the other: .a was not mentioned, so .a was
// not touched.
//
// `merge` is the inverse. It glues ADJACENT axes into one:
//
//     y.merge(.{ .d = .{ .h, .hd } })
//         {batch=16,seq=128,h=8,hd=64} -> {batch=16,seq=128,d=512}
//
// Read it as "new axis .d is made of .h and .hd". Name the old
// axes in the order they have in the tensor: `merge` starts at
// the first one you name and swallows its neighbours to the
// right. The struct can hold several merges at once:
//
//     img.merge(.{ .pixels = .{ .h, .w }, .bc = .{ .b, .c } })
//
// If these shapes ring a bell: this is how the "multi-head" in
// multi-head attention is done. One axis of 512 features is cut
// into 8 heads of 64, the heads do their thing separately, and
// a merge sews them back together. You will write exactly that
// later in the course; today it's paper.
//
// A split that doesn't divide evenly panics during tracing:
//
//     Can't split { 3, 8 } along axis 1 into { 3, 3, -1 }: UnexpectedRemainder
//
// Now, the drill. Each of the 3 members folded a strip of 8
// squares, and the judges gave a score to every square. A crane
// uses FOUR squares, so each strip makes 2 cranes, and a
// crane's score is the sum of its squares.
//
//     strips    {members=3,squares=8}
//     cut       {members=3,cranes=2,folds=4}
//     score     {members=3,cranes=2}       (sum over .folds)
//     entries   {entries=6}                all cranes in one line
//
// Reminder from the reductions chapter: `sum(.folds)` leaves a
// `.folds` axis of size 1 behind, and `squeeze(.folds)` drops
// it.
//
// That is three functions, each one built on the previous:
//
//   - `craneScores` cuts and scores: {members=3,cranes=2}.
//     Somebody miscounted the cranes.
//
//   - `judge` lines the crane scores up on one axis, .entries.
//     It is yours to write.
//
//   - `shares` says how much each crane weighs in its MEMBER's
//     total: Ingrid's two cranes scored 37 and 37, so each is
//     0.5 of her total. A share is a fraction, so the scores
//     are converted to f32 first (exercise 021). The division
//     is the trick from the Reduction Kitchen (exercise 039):
//     divide a table by its own sum along an axis, and the
//     size-1 axis the sum left behind is stretched for you
//     (rule 3 of exercise 036). It only works if you sum along
//     the right axis. Somebody didn't.
//
// And `shares` has an Exe in `main`, but nobody runs it.
//
const std = @import("std");
const zml = @import("zml");

/// The score of every crane: {members=3,cranes=2}.
fn craneScores(strips: zml.Tensor) zml.Tensor {
    // Tear each strip into cranes of 4 squares.
    const cranes = strips.splitAxis(.squares, .{ .cranes = 3, .folds = .auto });

    return cranes.sum(.folds).squeeze(.folds);
}

/// Every crane of every member on a single axis: {entries=6}.
fn judge(strips: zml.Tensor) zml.Tensor {
    return ???;
}

/// Each crane's share of its member's total: {members=3,cranes=2}, f32.
fn shares(strips: zml.Tensor) zml.Tensor {
    const scores = craneScores(strips).convert(.f32);

    return scores.div(scores.sum(.members));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const strips: zml.Tensor = .init(.{ .members = 3, .squares = 8 }, .i32);

    var exe = try platform.compileFn(gpa, io, judge, .{strips}, .{});
    defer exe.deinit();

    const marks = [3][8]i32{
        .{ 9, 9, 10, 9, 8, 10, 10, 9 }, // Ingrid
        .{ 7, 6, 7, 8, 5, 6, 6, 6 }, // Coach's nephew
        .{ 1, 2, 1, 0, 3, 3, 3, 3 }, // Timmy (improving!)
    };
    var strips_buffer: zml.Buffer = try .fromBytes(io, platform, strips.shape(), std.mem.sliceAsBytes(&marks));
    defer strips_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{strips_buffer});
    defer result.deinit();

    const entries: zml.Slice = try result.toSliceAlloc(gpa, io);
    defer entries.free(gpa);

    try stdout.print("{f} {any}\n", .{ entries.shape, entries.constItems(i32) });

    var shares_exe = try platform.compileFn(gpa, io, shares, .{strips}, .{});
    defer shares_exe.deinit();

    // Run `shares_exe` on the same strips, and read the answer
    // back as a nested array called `share`: one row per member.
    ???;

    try stdout.print("shares {f}\n", .{shares_exe.output_shapes[0]});
    for ([_][]const u8{ "Ingrid", "Coach's nephew", "Timmy" }, share) |member, row| {
        try stdout.print("  {s}: {d:.3} {d:.3}\n", .{ member, row[0], row[1] });
    }
    try stdout.flush();
}
