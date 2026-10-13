//
// PetRocks.com has made its first acquisition: Gravel.net, for
// nine million dollars in stock. Their whole codebase is now
// yours. It was written in 1996. Nothing in it has a tag.
//
// `a.dot(b, .k)` is a thin, friendly wrapper. Here is all of it:
//
//     pub fn dot(lhs: Tensor, rhs: Tensor, args: anytype) Tensor {
//         const dot_axes = lhs.dotAxes(rhs, args);
//         return lhs.dotGeneral(
//             rhs,
//             dot_axes.contracting.constSlice(),
//             dot_axes.batching.constSlice(),
//         );
//     }
//
// `dotAxes` turns tags into axis numbers and finds the batching
// axes. The work is done by the function underneath:
//
//     pub fn dotGeneral(
//         lhs: Tensor,
//         rhs: Tensor,
//         contracting_axes: []const [2]i8,
//         batching_axes: []const [2]i8,
//     ) Tensor
//
// Both lists are made of PAIRS: `.{ axis of lhs, axis of rhs }`.
// Contract the pairs of the first list, pair up the pairs of the
// second. Nothing is inferred, nothing is implicit, and no tag
// is needed. The result is laid out the way you know:
//
//     batching axes, free axes of lhs, free axes of rhs.
//
// An ordinary matrix product of {2,3} by {3,5}: axis 1 of the
// left meets axis 0 of the right, and there's nothing to batch.
//
//     a.dotGeneral(b, &.{.{ 1, 0 }}, &.{})          -> {2,5}
//
// (`&.{ ... }` is how Zig spells a slice literal; here, a slice
// holding one `[2]i8`.) Negative numbers count from the end, as
// everywhere: `.{ -1, 0 }` is the same pair.
//
// When should you use it? When the tensors really have no tags
// (a quick script, weights straight from a file), or when
// you're writing generic code that can't know them. Otherwise
// stay with `dot`: every number in those pairs is a chance to be
// wrong, and the panic you get isn't as helpful as usual:
//
//     dotGeneral expects contracting dimensions to be equal,
//     got 2 and 2 in Tensor({2,3,4,f32}) and Tensor({2,4,2,f32})
//
// Careful: "2 and 2" are not the sizes that differ. They are the
// two AXIS NUMBERS you passed. Look the sizes up yourself in the
// shapes printed after them.
//
// Gravel.net has two quarries. Each has an order book
// {quarry, days, products} and its own recipes
// {quarry, products, parts}. The parts report should be
// {quarry, days, parts}, each quarry on its own.
//
// One more thing before you run it. If you forget the batching
// pair, it's not an error: quarry stays a free axis on BOTH
// sides and you get every quarry's orders against every
// quarry's recipes, a rank 4 tensor.
//
// Gravel.net's function is below, with the pairs as they left
// them. The second function is how legacy code is brought home:
// `partsPerQuarry` takes that anonymous {2,3,2}, gives its axes
// their names (.quarry, .days, .parts: chapter 4, all three in
// one call), and adds up the days, so that the rest of the
// company gets a {quarry=2,parts=2} and no souvenir.
//
// Gravel.net never got round to running anything, either. The
// placeholders and the Buffers are in `main`, the prints are
// there, and what goes in between is on you: two Exes, and two
// results named `result` and `per_quarry`.
//
const std = @import("std");
const zml = @import("zml");

/// For each quarry and each day, the parts to ship. No tags allowed.
fn partsNeeded(orders: zml.Tensor, recipes: zml.Tensor) zml.Tensor {
    return orders.dotGeneral(recipes, &.{.{ 2, 2 }}, &.{});
}

/// The same report with its tags on, days added up: {quarry,parts}.
fn partsPerQuarry(orders: zml.Tensor, recipes: zml.Tensor) zml.Tensor {
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

    // quarry, days, products
    const orders: zml.Tensor = .init(.{ 2, 3, 4 }, .f32);
    // quarry, products, parts
    const recipes: zml.Tensor = .init(.{ 2, 4, 2 }, .f32);

    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[2][3][4]f32{
        .{ .{ 10, 2, 0, 0 }, .{ 20, 5, 1, 0 }, .{ 40, 10, 2, 1 } },
        .{ .{ 1, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 3, 1, 0, 0 } },
    }));
    defer orders_buffer.deinit();
    // Per product: bags of gravel, shovels.
    var recipes_buffer: zml.Buffer = try .fromBytes(io, platform, recipes.shape(), std.mem.sliceAsBytes(&[2][4][2]f32{
        .{ .{ 1, 0 }, .{ 1, 2 }, .{ 4, 8 }, .{ 100, 0 } },
        .{ .{ 2, 1 }, .{ 2, 1 }, .{ 2, 1 }, .{ 2, 1 } },
    }));
    defer recipes_buffer.deinit();

    // Gravel.net stopped here. Compile the two functions, then run
    // them: `partsNeeded` into a Buffer called `result`, and
    // `partsPerQuarry` into one called `per_quarry`.
    ???;

    try stdout.print("per quarry {f}: {any}\n", .{ per_quarry.shape(), try per_quarry.getValue([2][2]f32, io) });
    try stdout.print("parts {f}:\n", .{result.shape()});
    try stdout.flush();

    for (try result.getValue([2][3][2]f32, io), 1..) |quarry, i| {
        try stdout.print("  quarry {d}: {any}\n", .{ i, quarry });
    }
    try stdout.flush();
}
