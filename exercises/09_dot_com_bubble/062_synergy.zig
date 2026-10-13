//
// "We need SYNERGY between the order book and the price list."
// Nobody knows what Chad means. Brenda decides he means a
// matrix times a vector.
//
// `dot` doesn't care how many axes its operands have. The rule
// is always the same, and it has three parts:
//
//   1. The axis you name is CONTRACTED: multiplied pair by pair,
//      summed, and removed from the result.
//   2. Every other axis SURVIVES.
//   3. The survivors come out in this order: first the remaining
//      axes of the left operand, then the remaining axes of the
//      right operand.
//
// (There's a rule 0 about axes that have the same name on both
// sides. It's the next exercise. For now, no shared names
// except the contracted one.)
//
// Matrix times vector:
//
//     orders {days=3,products=4} . prices {products=4}
//                                             -> {days=3}
//
// For each day, the dot product of that day's row with the
// price list. Notice what we did NOT have to say: that
// .products is the last axis of `orders`. It could be the first,
// or the fifth. The tag finds it.
//
// Matrix times matrix:
//
//     a {n=2,k=3} . b {k=3,m=5}     -> {n=2,m=5}
//
// which is the classic "rows of a times columns of b". And here
// is where tags really pay their rent:
//
//     a {n=2,k=3} . b {m=5,k=3}     -> {n=2,m=5}
//
// Same result, and `b` is "the wrong way around"! In NumPy you'd
// need a transpose. Here .k is found wherever it lives. The
// question "which axis?" has gone away.
//
// What does NOT go away is rule 3:
//
//     b.dot(a, .k)                  -> {m=5,n=2}
//
// Same numbers, different order of the axes: the left operand's
// survivors always go first. The memory layout follows, so if
// you read that result into a `[2][5]f32`, it compiles, it
// runs, and it is scrambled.
//
// The warehouse wants two reports. Each product is assembled
// from parts (rocks, and googly eyes), listed in `recipes`:
//
//     revenue per day       {days=3}
//     parts needed per day  {days=3,parts=2}
//
// Chad wrote both. One of them panics, with a message worth
// reading. The other one runs fine, which is worse.
//
const std = @import("std");
const zml = @import("zml");

/// Dollars per day.
fn dailyRevenue(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
    return orders.dot(prices, .days);
}

/// For each day, how many of each part the warehouse must ship.
fn partsNeeded(orders: zml.Tensor, recipes: zml.Tensor) zml.Tensor {
    return recipes.dot(orders, .products);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const orders: zml.Tensor = .init(.{ .days = 3, .products = 4 }, .f32);
    const prices: zml.Tensor = .init(.{ .products = 4 }, .f32);
    const recipes: zml.Tensor = .init(.{ .products = 4, .parts = 2 }, .f32);

    var revenue_exe = try platform.compileFn(gpa, io, dailyRevenue, .{ orders, prices }, .{});
    defer revenue_exe.deinit();
    var parts_exe = try platform.compileFn(gpa, io, partsNeeded, .{ orders, recipes }, .{});
    defer parts_exe.deinit();

    // Units ordered: Pet Rock, Deluxe Rock, Family Pack, Enterprise Rock.
    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 10, 2, 0, 0 },
        .{ 20, 5, 1, 0 },
        .{ 40, 10, 2, 1 },
    }));
    defer orders_buffer.deinit();
    var prices_buffer: zml.Buffer = try .fromBytes(io, platform, prices.shape(), std.mem.sliceAsBytes(&[_]f32{ 5, 20, 50, 1000 }));
    defer prices_buffer.deinit();
    // For each product: rocks, googly eyes.
    var recipes_buffer: zml.Buffer = try .fromBytes(io, platform, recipes.shape(), std.mem.sliceAsBytes(&[4][2]f32{
        .{ 1, 0 },
        .{ 1, 2 },
        .{ 4, 8 },
        .{ 100, 0 },
    }));
    defer recipes_buffer.deinit();

    var revenue: zml.Buffer = try revenue_exe.eval(gpa, io, .{ orders_buffer, prices_buffer });
    defer revenue.deinit();
    var parts: zml.Buffer = try parts_exe.eval(gpa, io, .{ orders_buffer, recipes_buffer });
    defer parts.deinit();

    try stdout.print("revenue {f}: {any}\n", .{ revenue.shape(), try revenue.getValue([3]f32, io) });
    try stdout.print("parts {f}:\n", .{parts.shape()});
    for (try parts.getValue([3][2]f32, io), 1..) |day, i| {
        try stdout.print("  day {d}: {d} rocks, {d} googly eyes\n", .{ i, day[0], day[1] });
    }
    try stdout.flush();
}
