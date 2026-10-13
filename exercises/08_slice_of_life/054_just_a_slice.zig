//
// Benvenuti at Tensorino's, the only pizzeria in town where the
// menu is zero-indexed. Nonna Rosa has run the oven since
// before you were born, and Luigi, her grandson, started on
// deliveries this week. He means well.
//
// The last chapter rearranged whole tensors. This one is about
// taking PARTS of a tensor, and about putting parts together.
// We start with the dish of the house: the slice.
//
//     x.slice(axis, .{ .start = 2, .end = 5 })
//
// keeps, along `axis` (a tag or an index), the elements from
// `start` to `end`, `end` EXCLUDED: exactly like Zig's own
// `x[2..5]`. The other axes are left alone, and the result has
// the same rank and the same tags; only that one axis shrinks.
//
// The second argument is a `zml.Tensor.Slice`. Stop right
// there, says the attentive student, we already have a Slice!
// Yes. They are two different things, and the names are
// unfortunate:
//
//     zml.Slice           host memory holding numbers
//                         (chapter 2: toSliceAlloc, items, ...)
//
//     zml.Tensor.Slice    a tiny struct DESCRIBING a cut:
//                         "from here to there". No numbers in it.
//
// In kitchen terms: one is the slice on your plate, the other
// is the line Nonna draws in the air with her knife before
// cutting. Here is the struct, at least the part of it you need
// today:
//
//     pub const Slice = struct {
//         start: i64 = 0,
//         end: i64 = to_the_end,
//         step: i32 = 1,
//         ...
//     };
//
// Every field has a default, so you only write what you mean.
// On a pizza `p` of 8 slices holding 0, 1, 2, ... 7:
//
//     p.slice(.s, .{ .end = 3 })                 0 1 2
//     p.slice(.s, .{ .start = 5 })               5 6 7
//     p.slice(.s, .{ .start = 2, .end = 5 })     2 3 4
//     p.slice(.s, .{ .step = 3 })                0 3 6
//     p.slice(.s, .{ .start = 1, .step = 2 })    1 3 5 7
//     p.slice(.s, .full)                         everything
//
// Negative `start` and `end` count from the END of the axis,
// which Zig's slices can't do:
//
//     p.slice(.s, .{ .start = -3 })              5 6 7
//     p.slice(.s, .{ .end = -1 })                0 1 2 3 4 5 6
//     p.slice(.s, .{ .start = 1, .end = -1 })    1 2 3 4 5 6
//
// `step` must be positive. (To walk backwards, `reverse` from
// the last chapter.) The new size of the axis is
// (end - start) / step, rounded up.
//
// Bounds are checked while tracing, since shapes are known
// then. Ask for more pizza than there is and you get, at
// compile time of the model, not at run time:
//
//     Slice [7..9] is invalid for axis of dimension 8 (resolved to [7..9])
//
// One warning before you go. The struct also has a `len` field,
// and the doc comments suggest `.{ .start = 2, .len = 3 }`. At
// the ZML version pinned by zmling, `slice` IGNORES `len` when
// the start is an ordinary number: you silently get everything
// from `start` to the end of the axis. Use `end`. (`len` does
// have its moment of glory, in exercise 059.)
//
// Four orders are waiting. `firstThree` was written by Nonna.
// The others were written by Luigi, who counts slices starting
// from 1 "like a normal person".
//
const std = @import("std");
const zml = @import("zml");

fn firstThree(pizza: zml.Tensor) zml.Tensor {
    return pizza.slice(.slices, .{ .end = 3 });
}

/// The last two slices of the pizza.
fn lastTwo(pizza: zml.Tensor) zml.Tensor {
    return pizza.slice(.slices, .{ .start = 7, .end = 9 });
}

/// Everything except the first two and the last two.
fn middle(pizza: zml.Tensor) zml.Tensor {
    return pizza.slice(.slices, .{ .start = 3, .end = -2 });
}

/// One slice out of two, starting with slice number 1 (so: 1, 3, 5, 7).
fn everyOther(pizza: zml.Tensor) zml.Tensor {
    return pizza.slice(.slices, .{ .step = 2 });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const pizza: zml.Tensor = .init(.{ .slices = 8 }, .i32);

    // Number of olives on each slice. Nonna is generous but not consistent.
    const olives = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var pizza_buffer: zml.Buffer = try .fromBytes(io, platform, pizza.shape(), std.mem.sliceAsBytes(&olives));
    defer pizza_buffer.deinit();

    // One Exe per order. (`inline for`, because compileFn needs
    // to know each function at compile time.)
    inline for (
        .{ firstThree, lastTwo, middle, everyOther },
        .{ "firstThree:", "lastTwo:   ", "middle:    ", "everyOther:" },
    ) |order, name| {
        var exe = try platform.compileFn(gpa, io, order, .{pizza}, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{pizza_buffer});
        defer result.deinit();

        // A zml.Slice (the plate) of what the zml.Tensor.Slice (the knife) selected.
        const plate: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer plate.free(gpa);

        try stdout.print("{s} {f} {any}\n", .{ name, plate.shape, plate.constItems(i32) });
    }
    try stdout.flush();
}
