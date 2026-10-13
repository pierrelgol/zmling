//
// A customer calls Tensorino's: "I'd like ONE pizza." Simple
// enough, until Luigi asks the question that separates the
// professionals from the amateurs: "in a box, or in your
// hands?"
//
// You know the difference from Zig:
//
//     const tray = [3]u8{ 10, 20, 30 };
//
//     tray[1]        a u8          the element itself
//     tray[1..2]     a []const u8  a slice holding ONE element
//
// Same number, different TYPE, and a function that wants one
// won't take the other. Tensors have exactly the same split,
// except that "type" reads "shape". On a
// tray {pizzas=3,slices=4}:
//
//     tray.slice(.pizzas, .{ .start = 1, .end = 2 })
//
//         -> {pizzas=1,slices=4}     still a tray. It holds a
//                                    single pizza, in its box.
//
//     tray.slice(.pizzas, .single(1))
//
//         -> {slices=4}              the pizza itself. The
//                                    .pizzas axis is GONE.
//
// `zml.Tensor.Slice.single(i)` is a constructor: it builds a
// Slice that selects index `i` and is marked as "and then drop
// the axis". The four numbers selected are the same in both
// cases. Negative indices work in both forms too:
// `.single(-1)` is the last pizza, out of its box, and
// `.{ .start = -1 }` is the last pizza, boxed.
//
// (About that `.single(1)` with a leading dot: the parameter
// has type `Slice`, so Zig looks up `single` inside `Slice`
// for you. It's the same "decl literal" trick as
// `const t: zml.Tensor = .init(...)`.)
//
// Why would you care? Because ZML does not forgive shapes that
// are "nearly the same". Remember the rule from the arithmetic
// chapter: `add` wants two tensors of the SAME shape (or a
// scalar). A {pizzas=1,slices=4} and a {slices=4} hold the same
// amount of pizza, and they are not the same shape:
//
//     add expects tensor shapes to match, got {pizzas=1,slices=4,i32} and {slices=4,i32}
//
// In the other direction, some operations want the axis to
// still be there: gluing trays together along .pizzas (coming
// up in two exercises) needs every piece to HAVE a .pizzas
// axis.
//
// So the question is always Luigi's question:
//
//     "one element OF the axis"       .single(i)   axis dropped
//     "a shorter axis"                start/end    axis kept
//
// Three orders tonight:
//
//   - `extraBasil`: take pizza number 1 out of its box and add
//     the basil leaves, slice by slice. `basil` is {slices=4}.
//
//   - `lastBoxed`: the LAST pizza of the tray, still in its
//     box, so the delivery bag can stack it: {pizzas=1,slices=4}.
//
//   - `comparedToFirst`: Nonna's quality control. For every
//     pizza of the tray and every slice position, how many
//     MORE olives it has than the same slice of pizza number
//     0: {pizzas=3,slices=4}. It is one subtraction, and
//     Luigi's question decides whether it works. Think back to
//     the newsroom (exercise 036): what kind of right-hand
//     side does ZML agree to stretch? Answer wrong and you
//     will be told so, in the usual words:
//
//         subtract expects tensor shapes to match, got {pizzas=3,slices=4,i32} and {slices=4,i32}
//
// Luigi prepared the first two. He got both questions right,
// and then answered each of them on the wrong order. The third
// he never touched: neither the function, nor its block in
// `main`, where only the print is left.
//
const std = @import("std");
const zml = @import("zml");

/// Pizza number 1, plus basil. Result: {slices=4}.
fn extraBasil(tray: zml.Tensor, basil: zml.Tensor) zml.Tensor {
    return tray.slice(.pizzas, .{ .start = 1, .end = 2 }).add(basil);
}

/// The last pizza, in its box. Result: {pizzas=1,slices=4}.
fn lastBoxed(tray: zml.Tensor) zml.Tensor {
    return tray.slice(.pizzas, .single(-1));
}

/// Every pizza minus pizza number 0. Result: {pizzas=3,slices=4}.
fn comparedToFirst(tray: zml.Tensor) zml.Tensor {
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

    const tray: zml.Tensor = .init(.{ .pizzas = 3, .slices = 4 }, .i32);
    const basil: zml.Tensor = .init(.{ .slices = 4 }, .i32);

    // Olives per slice.
    const olives = [3][4]i32{
        .{ 1, 2, 3, 4 },
        .{ 5, 6, 7, 8 },
        .{ 9, 10, 11, 12 },
    };
    var tray_buffer: zml.Buffer = try .fromBytes(io, platform, tray.shape(), std.mem.sliceAsBytes(&olives));
    defer tray_buffer.deinit();
    var basil_buffer: zml.Buffer = try .fromBytes(io, platform, basil.shape(), std.mem.sliceAsBytes(&[_]i32{ 100, 100, 200, 200 }));
    defer basil_buffer.deinit();

    {
        var exe = try platform.compileFn(gpa, io, extraBasil, .{ tray, basil }, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ tray_buffer, basil_buffer });
        defer result.deinit();

        const plate: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer plate.free(gpa);

        try stdout.print("extraBasil: {f} {any}\n", .{ plate.shape, plate.constItems(i32) });
    }
    {
        var exe = try platform.compileFn(gpa, io, lastBoxed, .{tray}, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{tray_buffer});
        defer result.deinit();

        const plate: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer plate.free(gpa);

        try stdout.print("lastBoxed:  {f} {any}\n", .{ plate.shape, plate.constItems(i32) });
    }
    {
        // Same journey as the block above, for `comparedToFirst`:
        // an Exe, a result, and a zml.Slice called `plate`.
        ???;

        try stdout.print("compared:   {f} {any}\n", .{ plate.shape, plate.constItems(i32) });
    }
    try stdout.flush();
}
