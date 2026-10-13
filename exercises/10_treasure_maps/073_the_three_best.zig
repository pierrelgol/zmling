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
// chests. He didn't dare touch the runner: those lines are
// missing from `main`, and the prints are waiting for `res`.
//
// Meanwhile, below deck, the crew has been BETTING. Each of the
// four pirates named the chest he believes to be among the
// three best: `bets` {pirate=4,i32}, positions along .chest.
// The quartermaster wants the share of the crew that got it
// right, a number that people who train models call the TOP-3
// ACCURACY. `topThreeAccuracy` goes like this:
//
//   - Compare every bet with every one of the three winning
//     indices (`threeBest` has them: call it, keep the half you
//     need): a grid of bools, {pirate=4,best=3}. `cmp` wants
//     the same shape on both sides (exercise 68), so both have
//     to be stretched to that grid first, with `broad`
//     (chapter 5). `broad` wants a `zml.Shape` to aim at, and
//     no tensor around here has that one: build it yourself,
//     with the `zml.Shape.init(.{ .tag = size, ... }, dtype)`
//     of chapter 1. Only its tags and sizes are read; the
//     result keeps the dtype of the tensor being stretched.
//   - A pirate was right when one of his three bools is true.
//     At most one can be: count them along .best (exercise 68).
//   - The accuracy is the mean over .pirate of those 0's and
//     1's, as an .f32, with no axis left.
//
// Tonight the winning chests turn out to be 4, 2 and 0, and
// the bets are 4, 1, 0 and 3. Anne and Calico get to gloat:
//
//     hits    Anne    { true,  false, false }     1
//             Bart    { false, false, false }     0
//             Calico  { false, false, true  }     1
//             Dread   { false, false, false }     0
//
// It has a single output, so `exe.eval` will do. Compile it and
// run it into a Buffer called `accuracy`.
//
const std = @import("std");
const zml = @import("zml");

/// The three richest chests: their values, and where they are.
fn threeBest(loot: zml.Tensor) zml.Tensor.SortRes {
    return loot.topK(.chest, 3, .{ .descending = false });
}

/// The share of the bets that are among the three best chests: one .f32.
fn topThreeAccuracy(loot: zml.Tensor, bets: zml.Tensor) zml.Tensor {
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

    const loot: zml.Tensor = .init(.{ .chest = 6 }, .i32);
    const bets: zml.Tensor = .init(.{ .pirate = 4 }, .i32);

    var exe = try platform.compileFn(gpa, io, threeBest, .{loot}, .{});
    defer exe.deinit();

    // The quartermaster's number needs an Exe as well.
    ???;

    var loot_buffer: zml.Buffer = try .fromBytes(io, platform, loot.shape(), std.mem.sliceAsBytes(&[_]i32{ 30, 5, 80, 12, 95, 1 }));
    defer loot_buffer.deinit();
    // Anne says chest 4, Bart chest 1, Calico chest 0, Dread chest 3.
    var bets_buffer: zml.Buffer = try .fromBytes(io, platform, bets.shape(), std.mem.sliceAsBytes(&[_]i32{ 4, 1, 0, 3 }));
    defer bets_buffer.deinit();

    // The runner goes here: `exe` has two outputs, and both of them
    // end up in `res`. Its two Buffers are yours to free.
    ???;

    // Then the bets, with plain `eval`, into a Buffer called `accuracy`.
    ???;

    try stdout.print("The Exe has {d} outputs: {f} and {f}.\n", .{ exe.output_shapes.len, exe.output_shapes[0], exe.output_shapes[1] });
    const values = try res.values.getValue([3]i32, io);
    const indices = try res.indices.getValue([3]i32, io);
    for (values, indices) |doubloons, chest| {
        try stdout.print("Load chest {d}: {d} doubloons.\n", .{ chest, doubloons });
    }
    try stdout.print("Top-3 accuracy of the crew: {d:.2}\n", .{try accuracy.getValue(f32, io)});
    try stdout.flush();
}
