//
// PetRocks.com is going global: an office in San Francisco and
// an office in London. Each has its own order book and, because
// rocks are heavy, its own price list.
//
// Here is the rule 0 that was promised. `dot` looks at every
// axis of the left operand and sorts it into one of three bins:
//
//     CONTRACTED   you named it. Multiplied, summed, gone.
//
//     BATCHING     you did not name it, but the right operand
//                  has an axis with the SAME TAG. The two are
//                  paired up, and the axis appears ONCE in the
//                  result. "For each office, separately."
//
//     FREE         everything else. Survives as it is.
//
// And the full order of the result is:
//
//     batching axes, then free axes of the left operand,
//     then free axes of the right operand.
//
// So with one order book and one price list per office:
//
//     orders {office=2,days=3,products=4}
//     prices {office=2,products=4}
//
//     orders.dot(prices, .products)  ->  {office=2,days=3}
//
// London's orders meet London's prices and nobody else's. On
// its third day London sold 3 Pet Rocks and 1 Deluxe Rock, at
// London's prices of 4 and 15: 3*4 + 1*15 = 27 dollars. The
// classic "batched matrix multiplication" is the same thing with
// one more free axis on each side:
//
//     {batch=8,n=2,k=3} . {batch=8,k=3,m=5} -> {batch=8,n=2,m=5}
//
// No `bmm`, no special function. Same name: same thing: pair
// them up. (Batching axes must have the same size on both
// sides, and it only works with tags. Two untagged axes are
// never paired up behind your back.)
//
// It's lovely, until you WANT every combination. Chad asks for
// the Synergy Matrix (he has a slide for it): for every pair of
// offices, the dot product of their two price lists. The
// obvious code is
//
//     prices.dot(prices, .products)
//
// Sort the axes into bins: .products is contracted, and .office
// is on both sides, so it's batching. You get {office=2}: each
// office with ITSELF. No panic, no warning, and not a matrix.
// To get all the pairs, the two sides must DISAGREE on the name.
// So one of them gets a new one before the `dot`: you met
// `x.rename(.{ .old = .new })` at the airport (exercise 27).
// Call the offices of the second copy .partner, and sort the
// axes into bins again: .office is free on the left, .partner
// is free on the right,
//
//     {office=2,products=4} . {partner=2,products=4}
//                                    ->  {office=2,partner=2}
//
// four numbers, London-with-San-Francisco included. `rename`
// costs nothing at run time. It only changes what the axis is
// called while the graph is being built.
//
// Three functions today. `localRevenue` is the batched product
// above, and nobody has written it. `synergyMatrix` is Chad's
// obvious code: once the rest compiles, watch what it does to
// the last lines of `main`, then deliver his slide.
//
// The third one is for the accountants, who don't care about
// days: `officeTotals` is the local revenue added up over the
// days, {office=2} and nothing else (a reduction leaves a
// souvenir behind: chapter 6). It may call `localRevenue`. It
// also needs an Exe in `main` (name it as you like) and a
// result, which the print expects to be a Buffer called
// `totals`.
//
const std = @import("std");
const zml = @import("zml");

/// Dollars per office and per day.
fn localRevenue(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
    return ???;
}

/// For every (office, partner) pair, the dot product of their price lists.
fn synergyMatrix(prices: zml.Tensor) zml.Tensor {
    return prices.dot(prices, .products);
}

/// Dollars per office, all days together: {office}.
fn officeTotals(orders: zml.Tensor, prices: zml.Tensor) zml.Tensor {
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

    const orders: zml.Tensor = .init(.{ .office = 2, .days = 3, .products = 4 }, .f32);
    const prices: zml.Tensor = .init(.{ .office = 2, .products = 4 }, .f32);

    var revenue_exe = try platform.compileFn(gpa, io, localRevenue, .{ orders, prices }, .{});
    defer revenue_exe.deinit();
    var synergy_exe = try platform.compileFn(gpa, io, synergyMatrix, .{prices}, .{});
    defer synergy_exe.deinit();

    // The accountants are waiting: `officeTotals` needs an Exe too.
    ???;

    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[2][3][4]f32{
        // San Francisco
        .{ .{ 10, 2, 0, 0 }, .{ 20, 5, 1, 0 }, .{ 40, 10, 2, 1 } },
        // London
        .{ .{ 1, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 3, 1, 0, 0 } },
    }));
    defer orders_buffer.deinit();
    var prices_buffer: zml.Buffer = try .fromBytes(io, platform, prices.shape(), std.mem.sliceAsBytes(&[2][4]f32{
        .{ 5, 20, 50, 1000 },
        .{ 4, 15, 40, 700 },
    }));
    defer prices_buffer.deinit();

    var revenue: zml.Buffer = try revenue_exe.eval(gpa, io, .{ orders_buffer, prices_buffer });
    defer revenue.deinit();
    var synergy: zml.Buffer = try synergy_exe.eval(gpa, io, .{prices_buffer});
    defer synergy.deinit();

    // Run it. The second print reads a Buffer called `totals`.
    ???;

    try stdout.print("revenue {f}: {any}\n", .{ revenue.shape(), try revenue.getValue([2][3]f32, io) });
    try stdout.print("totals {f}: {any}\n", .{ totals.shape(), try totals.getValue([2]f32, io) });
    try stdout.print("synergy {f}:\n", .{synergy.shape()});
    // Printed now, in case the next line doesn't go as planned.
    try stdout.flush();

    for (try synergy.getValue([2][2]f32, io)) |row| try stdout.print("  {any}\n", .{row});
    try stdout.flush();
}
