//
// Marketing has discovered discounts. They want a grid: every
// product at every discount tier. With two tiers, list price
// and "friends of Chad", it looks like this:
//
//     prices    { 5, 20, 1000 }
//     tiers     { 1, 0.5 }
//
//     grid      {  5,   2.5 }
//               { 20,  10   }
//               { 1000, 500 }
//
// Every element of one vector times every element of the other.
// Nothing is summed, no axis disappears: this is the OUTER
// PRODUCT, `dot`'s extroverted cousin.
//
//     a.outer(b)        {a=3} outer {b=2}  ->  {a=3,b=2}
//
// The axes of the left operand come first, then the axes of the
// right one. It isn't limited to vectors either:
// {a=2,c=2} outer {b=3} gives {a=2,c=2,b=3}.
//
// How it works is worth thirty seconds, because you already
// know all the pieces. This is the body of `outer` in ZML's
// tensor.zig, minus a special case for scalars:
//
//     const res_shape = self.shape().outer(other.shape());
//     return self.broad(res_shape).mul(other.broad(res_shape));
//
// Compute the shape of the result, broadcast both operands to
// it, and multiply element by element. And how does `broad`
// know that `prices` goes along the rows and `tiers` along the
// columns? You know this one too: by matching TAGS.
//
// Which means that `outer` has requirements that the textbook
// outer product doesn't:
//
//   - Untagged operands don't work. `broad` is handed a {3,f32}
//     and a target {3,2,f32} and can't know where the 3 goes.
//     It panics, and the message talks about broadcasting; it
//     doesn't mention `outer` at all.
//
//   - The two operands need DIFFERENT tags. `v.outer(v)` builds
//     a shape with the same tag twice, and neither a panic nor
//     anything sensible comes out of that. Rename one side
//     first, like you did for the Synergy Matrix.
//
// Chad built the price grid on a napkin and then in Zig. The
// napkin was fine. Look at his placeholders before you look at
// anything else.
//
// Then Marketing has a follow-up question: if the whole STOCK
// (so many units of each product) sold at one tier, how many
// dollars would that be? One number per tier. That is the grid,
// with its .products axis contracted against the stock: an
// `outer` followed by a `dot`, in a function called
// `revenuePerTier` that is still empty. At list price, with the
// stock of `main`: 5*100 + 20*10 + 1000*1 = 1700 dollars, and
// the other tiers likewise. Nobody compiled it or ran it:
// `main` has a hole for its Exe (any name will do) and one for
// its result, a Buffer that the print calls `sold_out`.
//
// Mind the third placeholder while you are down there. `stock`
// is a list over the PRODUCTS, just like `prices`, and `dot`
// has only one way of finding that out.
//
const std = @import("std");
const zml = @import("zml");

/// Every product at every discount tier.
fn priceGrid(prices: zml.Tensor, tiers: zml.Tensor) zml.Tensor {
    return prices.outer(tiers);
}

/// Dollars per tier if the whole stock sold at that tier: {tiers}.
fn revenuePerTier(prices: zml.Tensor, tiers: zml.Tensor, stock: zml.Tensor) zml.Tensor {
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

    // Three products and three tiers. The grid should come out
    // as {products=3,tiers=3,f32}@replicated.
    const prices: zml.Tensor = .init(.{3}, .f32);
    const tiers: zml.Tensor = .init(.{3}, .f32);
    // Units in the warehouse, per product.
    const stock: zml.Tensor = .init(.{3}, .f32);

    var exe = try platform.compileFn(gpa, io, priceGrid, .{ prices, tiers }, .{});
    defer exe.deinit();

    // Marketing's follow-up question needs an Exe of its own.
    ???;

    var prices_buffer: zml.Buffer = try .fromBytes(io, platform, prices.shape(), std.mem.sliceAsBytes(&[_]f32{ 5, 20, 1000 }));
    defer prices_buffer.deinit();
    // List price, friends of Chad, employees.
    var tiers_buffer: zml.Buffer = try .fromBytes(io, platform, tiers.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 0.5, 0.1 }));
    defer tiers_buffer.deinit();
    var stock_buffer: zml.Buffer = try .fromBytes(io, platform, stock.shape(), std.mem.sliceAsBytes(&[_]f32{ 100, 10, 1 }));
    defer stock_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ prices_buffer, tiers_buffer });
    defer result.deinit();

    // ...and an answer: a Buffer called `sold_out`.
    ???;

    try stdout.print("price grid {f}:\n", .{result.shape()});
    for (try result.getValue([3][3]f32, io)) |row| {
        for (row) |price| try stdout.print(" {d:>7.2}", .{price});
        try stdout.print("\n", .{});
    }
    try stdout.print("stock sold out, per tier {f}:", .{sold_out.shape()});
    for (try sold_out.getValue([3]f32, io)) |dollars| try stdout.print(" ${d:.2}", .{dollars});
    try stdout.print("\n", .{});
    try stdout.flush();
}
