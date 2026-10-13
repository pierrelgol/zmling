//
// The club room needs decorating before the Championship, and
// Coach Brunhilde has found a loophole in "no scissors, no
// glue": the rule says nothing about the PHOTOCOPIER.
//
// So, a small exception in this chapter. The next operations
// don't change any number either, but they do make the tensor
// bigger, by copying what is already there.
//
// There are two ways to "repeat [1, 2, 3] twice", and they have
// different names because confusing them is a bug:
//
//     x.repeat1d(axis, n)     the whole thing, n times
//
//         [1, 2, 3]  ->  [1, 2, 3, 1, 2, 3]
//
//     x.stutter1d(axis, n)    each element, n times in a row
//
//         [1, 2, 3]  ->  [1, 1, 2, 2, 3, 3]
//
// Both multiply the size of ONE axis by n and leave the other
// axes alone. On a tensor with more axes, "element" means a
// whole slab along that axis:
//
//     x {rows=2,cols=2}          { { 1, 2 },
//                                  { 3, 4 } }
//
//     x.repeat1d(.rows, 2)       { {1,2}, {3,4}, {1,2}, {3,4} }
//     x.stutter1d(0, 2)          { {1,2}, {1,2}, {3,4}, {3,4} }
//
//     x.repeat1d(.cols, 2)       { {1,2,1,2}, {3,4,3,4} }
//     x.stutter1d(1, 2)          { {1,1,2,2}, {3,3,4,4} }
//
// A wart you should know about, visible in the examples above.
// The signatures are
//
//     pub fn repeat1d(self: Tensor, axis_: anytype, n_rep: u63) Tensor
//     pub fn stutter1d(self: Tensor, axis_: i8, n_rep: u32) Tensor
//
// `repeat1d` takes a tag or an index, like nearly everything
// else. `stutter1d` wants an INTEGER (negative is fine), and a
// tag is a compile error. If all you have is a tag, convert it
// with `x.axis(.rows)`, which you met in the tags chapter.
//
// (PyTorch people: `repeat1d` is `torch.Tensor.repeat` on one
// axis, `stutter1d` is `repeat_interleave`. Stuttering is what
// nearest-neighbour upscaling of an image is made of, and a
// very small image is waiting for you below.)
//
// While we're at the copier, one more: the mirror.
//
//     x.reverse(.{.cols})        { { 2, 1 },
//                                  { 4, 3 } }
//
// `reverse` takes a TUPLE of axes (tags or indices) and flips
// the order of the elements along each of them. The shape
// doesn't change. Mind the tuple: one axis is still `.{.cols}`.
//
// The task: a paper garland. Coach designed a motif of three
// beads, numbered 1, 2, 3, and wants it processed like so:
//
//     1. every bead twice in a row      1 1 2 2 3 3
//     2. mirrored                       3 3 2 2 1 1
//     3. the whole thing twice          3 3 2 2 1 1 3 3 2 2 1 1
//
// Timmy operated the copier. The garland has the right length.
// That is the nicest thing one can say about it.
//
// Second job, and this one is yours from the first line to the
// last: the POSTER. Coach has a stamp of 2 by 2 squares,
//
//     stamp {rows=2,cols=2}        { 1, 2 },
//                                  { 3, 4 }
//
// and wants it twice as tall and twice as wide, every square
// becoming a block of 2 by 2:
//
//     poster {rows=4,cols=4}       { 1, 1, 2, 2 },
//                                  { 1, 1, 2, 2 },
//                                  { 3, 3, 4, 4 },
//                                  { 3, 3, 4, 4 }
//
// That is the nearest-neighbour upscaling mentioned above, and
// the reason photocopied posters look the way they do. Write
// `poster`.
//
// Then write its trip through the machine in `main`. The
// placeholder, the numbers and the prints are there. The Exe,
// the Buffer, the run and the read-back are not: four steps
// you have done in every exercise since chapter 2.
//
const std = @import("std");
const zml = @import("zml");

fn garland(motif: zml.Tensor) zml.Tensor {
    // 1. Every bead twice in a row.
    const doubled = motif.repeat1d(.beads, 2);

    // 2. Mirrored.
    const mirrored = doubled;

    // 3. The whole thing twice.
    return mirrored.repeat1d(.beads, 2);
}

/// Every square of the stamp becomes a 2x2 block: {rows=4,cols=4}.
fn poster(stamp: zml.Tensor) zml.Tensor {
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

    const motif: zml.Tensor = .init(.{ .beads = 3 }, .i32);

    var exe = try platform.compileFn(gpa, io, garland, .{motif}, .{});
    defer exe.deinit();

    var motif_buffer: zml.Buffer = try .fromBytes(io, platform, motif.shape(), std.mem.sliceAsBytes(&[_]i32{ 1, 2, 3 }));
    defer motif_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{motif_buffer});
    defer result.deinit();

    const beads: zml.Slice = try result.toSliceAlloc(gpa, io);
    defer beads.free(gpa);

    try stdout.print("Garland {f}:", .{beads.shape});
    for (beads.constItems(i32)) |bead| try stdout.print(" {d}", .{bead});
    try stdout.print("\n", .{});

    const stamp: zml.Tensor = .init(.{ .rows = 2, .cols = 2 }, .i32);
    const ink = [2][2]i32{
        .{ 1, 2 },
        .{ 3, 4 },
    };

    // Compile `poster` for the stamp, put `ink` in a Buffer, run,
    // and copy the result to a zml.Slice called `squares`.
    // Release everything you create.
    ???;

    try stdout.print("Poster {f}\n", .{squares.shape});

    const width: usize = @intCast(squares.shape.dim(-1));
    var lines = std.mem.window(i32, squares.constItems(i32), width, width);
    while (lines.next()) |line| try stdout.print("  {any}\n", .{line});
    try stdout.flush();
}
