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
// next 64 become h=1, and so on.
//
// `merge` is the inverse. It glues ADJACENT axes into one:
//
//     y.merge(.{ .d = .{ .h, .hd } })
//         {batch=16,seq=128,h=8,hd=64} -> {batch=16,seq=128,d=512}
//
// Read it as "new axis .d is made of .h and .hd". The struct
// can hold several merges at once:
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
const std = @import("std");
const zml = @import("zml");

fn judge(strips: zml.Tensor) zml.Tensor {
    // Tear each strip into cranes of 4 squares.
    const cranes = strips.splitAxis(.squares, .{ .cranes = 3, .folds = .auto });

    const scores = cranes.sum(.folds).squeeze(.folds);

    // The judges want every crane of every member on a single
    // axis called .entries.
    return scores;
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
    try stdout.flush();
}
