//
// San Francisco, 1999. PetRocks.com ("Rocks. On the Internet.")
// has just raised forty million dollars. Chad, founder and Chief
// Visionary Officer, has a foosball table, a Super Bowl ad, and
// one engineer. Her name is Brenda. You work for Brenda.
//
// Chad would like to know the revenue. He has two lists:
//
//     prices   { 5, 20, 50, 1000 }     dollars, per product
//     units    { 120, 30, 4, 0 }       sold this quarter
//
// Multiply them pair by pair, add everything up: 1400 dollars.
// That operation is the DOT PRODUCT, and it is the single most
// important thing a neural network does. A "linear layer" is
// dot products. "Attention" is dot products. Whatever the
// Visionary says the product is this week: dot products.
//
// If you come from NumPy or PyTorch, you'd now reach for one of
// `a @ b`, `matmul`, `mm`, `bmm`, `dot`, `inner`, `tensordot`,
// `einsum`... each with its own rules about WHICH axis of `a`
// meets WHICH axis of `b` ("the last one of a and the second to
// last of b, unless b is 1-D, in which case..."). People put
// those rules on a sticky note. Then they transpose until the
// error goes away.
//
// ZML has none of them. There is no `matmul`. There is ONE
// method, and you tell it the axis BY NAME:
//
//     a.dot(b, .k)
//
// It means: "`a` and `b` both have an axis tagged .k, of the
// same size. Walk along it in both, multiply the pairs, and sum
// the products." The technical word is CONTRACTION: .k is
// contracted, and it is gone from the result. For two vectors,
//
//     {k=4,f32} . {k=4,f32}   ->   {f32}
//
// there was only one axis, it got contracted, and what's left
// has rank 0. A scalar: one number, no box. That's why this
// program reads the result with `getValue(f32, io)`.
//
// Things `dot` insists on, each with a panic while tracing:
//
//   - both tensors have an axis with that tag;
//   - the two axes have the same size;
//   - same dtype on both sides (no promotion, remember).
//
// Three things are wrong today, and they are three different
// kinds of wrong.
//
// Chad copied `revenue` from a NumPy tutorial he found on
// GeoCities. Zig has opinions about it.
//
// Then Brenda asks the question Chad didn't: what did those
// rocks COST? There is a third list,
//
//     costs    { 2, 8, 20, 400 }       dollars, per product
//
// and the profit is the dot product of the MARGINS with the
// units, a margin being a price minus a cost (element by
// element: chapter 3). Nobody wrote `profit`. You do.
//
// When all of that compiles, `dot` will panic about one of the
// placeholders in `main`. Read what it says: it checks exactly
// the three things listed above, and it names the one it
// didn't find.
//
const std = @import("std");
const zml = @import("zml");

/// Total dollars: sum over products of price * units.
fn revenue(prices: zml.Tensor, units: zml.Tensor) zml.Tensor {
    return prices.matmul(units);
}

/// Dollars left once the rocks are paid for:
/// sum over products of (price - cost) * units.
fn profit(prices: zml.Tensor, costs: zml.Tensor, units: zml.Tensor) zml.Tensor {
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

    // Pet Rock, Deluxe Rock, Family Pack, Enterprise Rock (TM).
    const prices: zml.Tensor = .init(.{ .products = 4 }, .f32);
    const costs: zml.Tensor = .init(.{ .products = 4 }, .f32);
    const units: zml.Tensor = .init(.{ .units = 4 }, .f32);

    var exe = try platform.compileFn(gpa, io, revenue, .{ prices, units }, .{});
    defer exe.deinit();
    var profit_exe = try platform.compileFn(gpa, io, profit, .{ prices, costs, units }, .{});
    defer profit_exe.deinit();

    var prices_buffer: zml.Buffer = try .fromBytes(io, platform, prices.shape(), std.mem.sliceAsBytes(&[_]f32{ 5, 20, 50, 1000 }));
    defer prices_buffer.deinit();
    var costs_buffer: zml.Buffer = try .fromBytes(io, platform, costs.shape(), std.mem.sliceAsBytes(&[_]f32{ 2, 8, 20, 400 }));
    defer costs_buffer.deinit();
    var units_buffer: zml.Buffer = try .fromBytes(io, platform, units.shape(), std.mem.sliceAsBytes(&[_]f32{ 120, 30, 4, 0 }));
    defer units_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ prices_buffer, units_buffer });
    defer result.deinit();
    var profit_result: zml.Buffer = try profit_exe.eval(gpa, io, .{ prices_buffer, costs_buffer, units_buffer });
    defer profit_result.deinit();

    try stdout.print("The result is a {f}: no axes left.\n", .{exe.output_shapes[0]});
    try stdout.print("Revenue: ${d}. Cost of the Super Bowl ad: $2000000.\n", .{try result.getValue(f32, io)});
    try stdout.print("Profit: ${d}. Chad has already spent it on beanbags.\n", .{try profit_result.getValue(f32, io)});
    try stdout.flush();
}
