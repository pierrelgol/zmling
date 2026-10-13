//
// Quiz! Saturday night, 11:40 pm. The last table has left, the
// pans are soaking, and Chef Odile is doing the accounts on the
// back of a menu. She gives up after two minutes and hands you
// the pencil. "You. The one who reduces. Reduce my Saturday."
//
// You have:
//
//     orders   {table=3,dish=4,f32}   portions of each dish that
//                                     each table ordered
//     prices   {dish=4,f32}           the price of one portion
//
// She wants five things. Each is one ritual, compiled into its
// own Exe below. The rituals are yours, all five, and so are a
// few lines of `main` that she didn't get to: the two
// placeholders, the Buffer with the prices, and everything that
// happens to the fifth ritual between its definition and the
// printing. The other four show the way.
//
//   1. THE BILLS. What each table owes: portions times price,
//      added up over the dishes. A plain {table=3}.
//
//   2. THE AVERAGE BILL. One number, a true scalar. (Rituals
//      are Zig functions: nothing stops one from calling
//      another.)
//
//   3. THE BEST SELLER. The index of the dish with the most
//      portions sold, all tables together. One number, a true
//      scalar, and not a float.
//
//   4. THE SHARES. Which fraction of the night's MONEY each dish
//      brought in: a plain {dish=4} that adds up to 1. First the
//      money per dish, then divide it by its own total. If you
//      keep your size 1 axes until the last moment, the fine
//      print of exercise 036 does that division for you.
//
//   5. AGAINST THE AVERAGE. Each bill divided by the average
//      bill: 1 is an average table, 2 spent twice as much. A
//      plain {table=3}. You have rituals for both halves, and
//      one of them returns a true scalar: this is what true
//      scalars are for.
//
// Things to keep in mind, from this chapter and the previous
// one:
//
//   - `orders` and `prices` don't have the same shape.
//   - a reduced axis stays, with size 1, until you squeeze it.
//   - one axis per reduction, one axis per squeeze.
//   - the left side of an operation is the boss.
//
// Chef Odile has already left. She said "bonne nuit", which
// from her is a standing ovation.
//
const std = @import("std");
const zml = @import("zml");

/// 1. What each table owes: {table=3}.
fn bills(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
    return ???;
}

/// 2. The mean of the bills: a scalar.
fn averageBill(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
    return ???;
}

/// 3. The index of the most ordered dish: an integer scalar.
fn bestSeller(orders: zml.Tensor) zml.Tensor {
    return ???;
}

/// 4. The fraction of the money that each dish brought in: {dish=4}.
fn shares(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
    return ???;
}

/// 5. Each bill, divided by the average bill: {table=3}.
fn againstAverage(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
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

    // The two placeholders. Shapes and dtypes are in the lesson.
    const orders: zml.Tensor = ???;
    const prices: zml.Tensor = ???;

    var bills_exe = try platform.compileFn(gpa, io, bills, .{ orders, prices }, .{});
    defer bills_exe.deinit();
    var average_exe = try platform.compileFn(gpa, io, averageBill, .{ orders, prices }, .{});
    defer average_exe.deinit();
    var best_exe = try platform.compileFn(gpa, io, bestSeller, .{orders}, .{});
    defer best_exe.deinit();
    var shares_exe = try platform.compileFn(gpa, io, shares, .{ orders, prices }, .{});
    defer shares_exe.deinit();

    // The fifth Exe, `against_exe`, for `againstAverage`.

    const dishes = [_][]const u8{ "soup", "duck", "risotto", "tart" };

    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 2, 1, 0, 2 },
        .{ 0, 2, 2, 1 },
        .{ 1, 1, 3, 4 },
    }));
    defer orders_buffer.deinit();

    // The menu: soup 8, duck 26, risotto 18, tart 9.

    var bills_result: zml.Buffer = try bills_exe.eval(gpa, io, .{ orders_buffer, prices_buffer });
    defer bills_result.deinit();
    var average_result: zml.Buffer = try average_exe.eval(gpa, io, .{ orders_buffer, prices_buffer });
    defer average_result.deinit();
    var best_result: zml.Buffer = try best_exe.eval(gpa, io, .{orders_buffer});
    defer best_result.deinit();
    var shares_result: zml.Buffer = try shares_exe.eval(gpa, io, .{ orders_buffer, prices_buffer });
    defer shares_result.deinit();

    // Run the fifth Exe: its result is `against_result`.

    try stdout.print("bills {f}: {any}\n", .{ bills_result.shape(), try bills_result.getValue([3]f32, io) });
    try stdout.print("average bill {f}: {d:.2}\n", .{ average_result.shape(), try average_result.getValue(f32, io) });

    const best: usize = @intCast(try best_result.getValue(i32, io));
    try stdout.print("best seller {f}: {s}\n", .{ best_result.shape(), dishes[best] });

    try stdout.print("shares {f}:", .{shares_result.shape()});
    for (dishes, try shares_result.getValue([4]f32, io)) |dish, share| {
        try stdout.print(" {s} {d:.3}", .{ dish, share });
    }
    try stdout.print("\n", .{});

    try stdout.print("against the average {f}:", .{against_result.shape()});
    for (try against_result.getValue([3]f32, io)) |ratio| try stdout.print(" {d:.3}", .{ratio});
    try stdout.print("\n", .{});
    try stdout.flush();
}
