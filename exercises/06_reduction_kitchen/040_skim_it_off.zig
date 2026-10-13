//
// A reduction leaves a size 1 axis behind, like the skin on a
// sauce that sat too long. Sometimes you want it there (it keeps
// the result lined up with the original). Sometimes you want a
// plain list of numbers, or a true scalar. Then you skim it off:
//
//     pub fn squeeze(self: Tensor, axis_: anytype) Tensor
//
// "Drops a 1-dim axis at the given index", says the source. The
// axis is a tag or a number, and no data moves: it is the exact
// opposite of the `insertAxes` you used at Channel 5.
//
//     x: {table=1,dish=4}
//     x.squeeze(.table)    -> {dish=4}
//
// It only removes axes of size 1. It has to: an axis of size 4
// holds 4 different things, and dropping it would mean throwing
// 3 of them away without saying which. ZML won't:
//
//     x.squeeze(.dish)
//     panic: squeeze expects axis to be squeezed to have a
//            dimension of 1, got 4
//
// (That message is stingy: no shape, no tag. When you meet it,
// the stack trace tells you which line of YOUR code did it.)
//
// So the usual way to write "the total for each table" has two
// steps, reduce and skim:
//
//     orders.sum(.dish).squeeze(.dish)      {table=3,dish=4}
//                                        -> {table=3,dish=1}
//                                        -> {table=3}
//
// What about reducing over SEVERAL axes? `sum` takes one axis
// per call, so you chain. Each step keeps its size 1 axis, which
// is what keeps the tags valid for the next step:
//
//     orders.sum(.table).sum(.dish)         {table=3,dish=4}
//                                        -> {table=1,dish=4}
//                                        -> {table=1,dish=1}
//
// That is ONE number, but it is not a scalar yet. A sock in a
// box in another box is still not a sock. A true scalar has rank
// 0, and you get there by squeezing every axis.
//
// Does the difference matter? `getValue(i32, io)` can't tell: it
// only counts bytes. But the next operation can. A rank 0 tensor
// combines with anything, on either side; a {table=1,dish=1} one
// is just a small table, with all the fine print that goes with
// it.
//
// Two rituals tonight, so two Exes: you may call `compileFn` as
// often as you like, and use the same placeholder in several
// compilations. Jean-Luc, the maître d', asks how many plates
// each table is getting (he carries them). Chef Odile asks for
// ONE number: how many plates leave her kitchen tonight.
//
const std = @import("std");
const zml = @import("zml");

/// Plates for each table: {table=3}.
fn platesPerTable(orders: zml.Tensor) zml.Tensor {
    return orders.sum(.dish).squeeze(.table);
}

/// Every plate of the night, as a true scalar.
fn platesTonight(orders: zml.Tensor) zml.Tensor {
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

    const orders: zml.Tensor = .init(.{ .table = 3, .dish = 4 }, .i32);

    var per_table_exe = try platform.compileFn(gpa, io, platesPerTable, .{orders}, .{});
    defer per_table_exe.deinit();

    var tonight_exe = try platform.compileFn(gpa, io, platesTonight, .{orders}, .{});
    defer tonight_exe.deinit();

    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[3][4]i32{
        .{ 2, 1, 0, 2 },
        .{ 0, 2, 2, 1 },
        .{ 1, 1, 3, 4 },
    }));
    defer orders_buffer.deinit();

    var per_table: zml.Buffer = try per_table_exe.eval(gpa, io, .{orders_buffer});
    defer per_table.deinit();

    var tonight: zml.Buffer = try tonight_exe.eval(gpa, io, .{orders_buffer});
    defer tonight.deinit();

    try stdout.print("Jean-Luc carries {f}: {any}\n", .{ per_table.shape(), try per_table.getValue([3]i32, io) });
    try stdout.print("Chef Odile gets {f}: {d} plates\n", .{ tonight.shape(), try tonight.getValue(i32, io) });
    try stdout.flush();
}
