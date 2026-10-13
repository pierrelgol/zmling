//
// `max` tells you HOW MUCH the biggest value is. Very often the
// question is WHERE it is. A classifier doesn't care that the
// best score is 0.93: it wants to know that it's entry number
// 281, "tabby cat". The index of the maximum is its argmax, and
// it is probably the most used reduction in machine learning.
//
//     pub fn argMax(x: Tensor, axis_: anytype) ArgMaxRes
//
// For the first time, a tensor operation that doesn't return a
// Tensor. Since it had to find the maximum anyway, it gives you
// both answers, in a struct:
//
//     pub const ArgMaxRes = struct {
//         values: Tensor,      // the maxima: same as x.max(axis)
//         indices: Tensor,     // where they were, along the axis
//     };
//
//     sales: {day=2,dish=3,f32}        4  9  2
//                                      6  6  1
//
//     const best = sales.argMax(.dish);
//     best.values     {day=2,dish=1,f32}     9, 6
//     best.indices    {day=2,dish=1,i32}     1, 0
//
// Things to notice:
//
//   - `indices` is an INTEGER tensor, .i32, whatever the dtype
//     of `x` was. (It becomes .i64 for axes with more than two
//     billion entries. May your kitchen never need it.)
//
//   - This is a reduction like the others: both fields keep the
//     reduced axis, with size 1.
//
//   - Second row: 6 and 6, a tie. The FIRST one wins. That's a
//     promise, there's a test for it in tensor.zig.
//
// Now, what happens to a struct of tensors when it is returned
// from a ritual? `compileFn` is fine with it: a ritual may return
// a Tensor, or a struct or an array of Tensors, and the Exe
// simply gets one output per tensor (`exe.output_shapes.len`
// would be 2 here). It is `eval` that objects. `eval` is the
// simple way to call an Exe, and simple means: ONE output.
//
// Running an Exe with several outputs takes more ceremony (if
// you are curious, `test argMax` in zml/zml/tensor.zig does it,
// with `exe.runner`). For today the way out is easier: you
// rarely want both fields anyway. Return the one you want.
// (The ritual then hands back a Tensor and not an `ArgMaxRes`
// any more: its return type is yours to change, too.)
//
// La Grande Réduction puts a "dish of the day" on the blackboard:
// the best-selling dish of the same weekday last week. Chef
// Odile wants, for each of the 5 days, the NUMBER of the dish (0
// is soup, 1 is duck, ...). Jean-Luc will copy the names onto
// the blackboard, in his beautiful handwriting.
//
// You'll have to run this more than once. First to meet `eval`'s
// complaint, and then... well, read the first line it prints.
// Jean-Luc wants a plain list, one entry per day.
//
// He also wants to chalk the number of portions next to each
// name ("duck, 16 sold" makes the neighbours order duck). That
// is the other half of the same answer, so it gets a ritual of
// its own, `winningSales`: a plain list again, one entry per
// day. (The `.values` of an `argMax` and a plain `max` are the
// same numbers, the lesson said so: take whichever you prefer.
// Both leave the same size 1 axis behind, and the first line of
// the output shows the shape of both lists.) The ritual is
// empty, and although its Exe is compiled, nobody runs it:
// `main` reads a Buffer called `sold` that doesn't exist yet.
// (Zig refuses to build anything while those two holes are
// open, so `eval`'s complaint comes after them.)
//
const std = @import("std");
const zml = @import("zml");

/// For each day, the index of the dish that sold the most: {day=5}.
fn dishOfTheDay(sales: zml.Tensor) zml.Tensor.ArgMaxRes {
    return sales.argMax(.dish);
}

/// For each day, how many portions the winner sold: {day=5}.
fn winningSales(sales: zml.Tensor) zml.Tensor {
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

    const sales: zml.Tensor = .init(.{ .day = 5, .dish = 4 }, .f32);

    var exe = try platform.compileFn(gpa, io, dishOfTheDay, .{sales}, .{});
    defer exe.deinit();

    var sold_exe = try platform.compileFn(gpa, io, winningSales, .{sales}, .{});
    defer sold_exe.deinit();

    const days = [_][]const u8{ "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
    const dishes = [_][]const u8{ "soup", "duck", "risotto", "tart" };

    // Portions sold. On Thursday the duck and the tart tied.
    var sales_buffer: zml.Buffer = try .fromBytes(io, platform, sales.shape(), std.mem.sliceAsBytes(&[5][4]f32{
        .{ 12, 9, 7, 10 },
        .{ 8, 8, 14, 9 },
        .{ 5, 11, 6, 11 },
        .{ 9, 16, 15, 13 },
        .{ 10, 18, 17, 21 },
    }));
    defer sales_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{sales_buffer});
    defer result.deinit();

    // Run `sold_exe` on the same sales: that's `sold`.

    const winners = try result.getValue([5]i32, io);
    const portions = try sold.getValue([5]f32, io);

    try stdout.print("The blackboard is a {f}, with a {f} of portions.\n", .{ result.shape(), sold.shape() });
    for (days, winners, portions) |day, winner, n| {
        try stdout.print("{s}: {s}, {d} sold\n", .{ day, dishes[@intCast(winner)], n });
    }
    try stdout.flush();
}
