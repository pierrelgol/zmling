//
// The hold of the Off-By-One has room for three chests. Not
// six: three. The Captain wants the three best, and he wants to
// know which ones they are.
//
// You could sort everything and slice off the front. That's
// such a common move that it has a name:
//
//     x.topK(axis, k, .{})
//
// It returns the same struct as `sort`, a `zml.Tensor.SortRes`
// with `.values` and `.indices` (.i32), where the axis has
// shrunk to size `k`. Three things to know.
//
// 1. BEST FIRST. Unlike `sort`, the default is descending,
//    because "top" means top. `.{ .descending = false }` gives
//    the k smallest instead.
//
// 2. THE AXIS ARGUMENT CAN RENAME. A plain number works:
//    `x.topK(0, 3, .{})`, or `-1` for the last axis. But you
//    just saw what happens when a tensor of indices keeps the
//    name of the axis it points into. So `topK` takes a
//    one-field struct, "new name = old name":
//
//        loot.topK(.{ .best = .chest }, 3, .{})
//
//        loot     {chest=6,i32}
//        values   {best=3,i32}
//        indices  {best=3,i32}      positions along .chest
//
//    The result is ready to be handed to `gather`. (A bare tag
//    like `.chest` is refused at compile time; the error message
//    shows the two accepted spellings.)
//
// 3. `k` is a Zig number, baked into the Exe, and it can't be
//    larger than the axis.
//
// Now, about getting BOTH tensors out. A function you compile
// may return a struct of tensors, and the Exe then has one
// output per tensor: `exe.output_shapes.len` is 2 here. But
// `exe.eval` returns a single Buffer, and it tells you so:
//
//     eval expects one output, got 2; use a runner for
//     multiple outputs
//
// So here is a runner. A whole chapter is devoted to it later;
// for now, read along:
//
//     var res: zml.Bufferized(zml.Tensor.SortRes) = undefined;
//
// `Bufferized(T)` is "T, with every Tensor replaced by a
// Buffer": a struct with two Buffers called `values` and
// `indices`.
//
//     var runner = try exe.runner(gpa);
//     defer runner.deinit(gpa);
//     runner.run(io, .{loot_buffer}, .{&res}, .{ .wait = true });
//
// `run` takes the inputs like `eval` does, then a tuple of
// POINTERS to where the outputs should be written, and fills
// them in. The Buffers in `res` are now yours to deinit.
//
// The cabin boy filled in the `topK` call. He got the axis
// argument wrong, and then came back with the three WORST
// chests.
//
const std = @import("std");
const zml = @import("zml");

/// The three richest chests: their values, and where they are.
fn threeBest(loot: zml.Tensor) zml.Tensor.SortRes {
    return loot.topK(.chest, 3, .{ .descending = false });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const loot: zml.Tensor = .init(.{ .chest = 6 }, .i32);

    var exe = try platform.compileFn(gpa, io, threeBest, .{loot}, .{});
    defer exe.deinit();

    var loot_buffer: zml.Buffer = try .fromBytes(io, platform, loot.shape(), std.mem.sliceAsBytes(&[_]i32{ 30, 5, 80, 12, 95, 1 }));
    defer loot_buffer.deinit();

    var res: zml.Bufferized(zml.Tensor.SortRes) = undefined;
    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);
    runner.run(io, .{loot_buffer}, .{&res}, .{ .wait = true });
    defer res.values.deinit();
    defer res.indices.deinit();

    try stdout.print("The Exe has {d} outputs: {f} and {f}.\n", .{ exe.output_shapes.len, exe.output_shapes[0], exe.output_shapes[1] });
    const values = try res.values.getValue([3]i32, io);
    const indices = try res.indices.getValue([3]i32, io);
    for (values, indices) |doubloons, chest| {
        try stdout.print("Load chest {d}: {d} doubloons.\n", .{ chest, doubloons });
    }
    try stdout.flush();
}
