//
// Quiz! The big one. Five layers, the tallest thing La Couche
// has ever attempted, and Monsieur Perte has announced that he
// will be "dropping by".
//
// The till must guess the OCCASION from an order, so that the
// right box comes out before the customer has finished talking.
// An order is four menu items. The occasions are:
//
//     0: breakfast     1: tea time     2: a wedding
//
// The whole cake, bottom to top. `orders` comes in as
// `{.order, .s}`, item ids, several orders at once:
//
//  1. `menu`     a TokenEmbedding. Each item becomes its three
//                flavors (sweet, buttery, crunchy): `.d = 3`.
//
//  2.            The taste of an order is the AVERAGE of its
//                four items: reduce `.s` and get rid of it.
//                (No layer for this one. It's chapter 6.)
//
//  3. `hidden`   a Linear, 3 flavors in, 3 hunches out...
//
//  4. `oven`     ...through the activation. Chef Babette's
//                three hunches, once baked:
//                    relu(crunchy - sweet)          crusty
//                    relu(sweet - buttery)          dainty
//                    relu(sweet + buttery - 1.2)    rich
//
//  5. `level`    a LayerNorm over the three hunches.
//
//  6. `verdict`  a Linear, 3 hunches in, 3 occasions out.
//
//  7.            The index of the best occasion of each order.
//                The result is `{.order}`, i32.
//
// The weights are written down for you, as Buffers. Everything
// in between is yours: three `???` in the model, the body of
// `forward`, and the `zml.Bufferized(Till)` that hands every
// Buffer to its Tensor.
//
// You know everything you need. A few reminders, in no
// particular order, because Chef Babette is watching and it
// would look bad if we helped too much:
//
//  - a Linear only works on tagged tensors, and what it
//    returns is not tagged like what it takes;
//  - LayerNorm levels the last axis;
//  - of all the things in a model, only Tensors get a Buffer.
//    And one field of Linear wants a `null` it doesn't deserve.
//
// If the tea time order gets a wedding box, a layer is missing
// from your `forward`. All of them matter. That's the point of
// a wedding cake.
//
const std = @import("std");
const zml = @import("zml");

const Till = struct {
    menu: zml.nn.TokenEmbedding,
    hidden: zml.nn.Linear,
    oven: zml.nn.Activation,
    level: zml.nn.LayerNorm,
    verdict: zml.nn.Linear,

    /// orders: {.order, .s} item ids -> {.order} occasion ids
    pub fn forward(self: Till, orders: zml.Tensor) zml.Tensor {
        return ???;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const till: Till = .{
        .menu = .{ .weight = .init(.{ .voc = 5, .d = 3 }, .f32) },
        // A Linear: 3 flavors in, 3 hunches out, with a bias.
        .hidden = ???,
        .oven = .relu,
        // A LayerNorm over 3 hunches: a weight, no bias, the default eps.
        .level = ???,
        // A Linear: 3 hunches in, 3 occasions out, with a bias.
        .verdict = ???,
    };

    // Three orders of four items.
    const orders: zml.Tensor = .init(.{ .order = 3, .s = 4 }, .i32);

    var exe = try platform.compile(gpa, io, till, .forward, .{orders}, .{});
    defer exe.deinit();

    var menu_weight: zml.Buffer = try .fromBytes(io, platform, till.menu.weight.shape(), std.mem.sliceAsBytes(&[5][3]f32{
        //  sweet, buttery, crunchy
        .{ 0.2, 0.9, 0.7 }, // 0: croissant
        .{ 0.0, 0.1, 0.9 }, // 1: baguette
        .{ 0.8, 0.6, 0.1 }, // 2: eclair
        .{ 0.9, 0.2, 0.5 }, // 3: macaron
        .{ 1.0, 0.7, 0.2 }, // 4: brownie
    }));
    defer menu_weight.deinit();

    var hidden_weight: zml.Buffer = try .fromBytes(io, platform, till.hidden.weight.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ -1, 0, 1 }, // crusty: crunchy - sweet
        .{ 1, -1, 0 }, // dainty: sweet - buttery
        .{ 1, 1, 0 }, // rich: sweet + buttery (the -1.2 is in the bias)
    }));
    defer hidden_weight.deinit();
    var hidden_bias: zml.Buffer = try .fromBytes(io, platform, till.hidden.bias.?.shape(), std.mem.sliceAsBytes(&[3]f32{ 0, 0, -1.2 }));
    defer hidden_bias.deinit();

    var level_weight: zml.Buffer = try .fromBytes(io, platform, till.level.weight.shape(), std.mem.sliceAsBytes(&[3]f32{ 1, 1, 1 }));
    defer level_weight.deinit();

    var verdict_weight: zml.Buffer = try .fromBytes(io, platform, till.verdict.weight.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ 1, 0, 0 }, // breakfast: crusty
        .{ 0, 1, 0 }, // tea time: dainty
        .{ -1, 0, 2 }, // a wedding: very rich, and nothing crusty
    }));
    defer verdict_weight.deinit();
    // Weddings get a head start. Weddings pay.
    var verdict_bias: zml.Buffer = try .fromBytes(io, platform, till.verdict.bias.?.shape(), std.mem.sliceAsBytes(&[3]f32{ 0, 0, 0.6 }));
    defer verdict_bias.deinit();

    const till_buffers: zml.Bufferized(Till) = ???;

    const order_names = [_][]const u8{
        "2 croissants, 2 baguettes",
        "2 eclairs, 2 macarons",
        "3 brownies, 1 eclair",
    };
    var orders_buffer: zml.Buffer = try .fromBytes(io, platform, orders.shape(), std.mem.sliceAsBytes(&[3][4]i32{
        .{ 0, 1, 0, 1 },
        .{ 2, 3, 3, 2 },
        .{ 4, 4, 2, 4 },
    }));
    defer orders_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ till_buffers, orders_buffer });
    defer result.deinit();

    try stdout.print("occasions {f}\n", .{result.shape()});
    const occasions = [_][]const u8{ "breakfast", "tea time", "a wedding" };
    for (order_names, try result.getValue([3]i32, io)) |name, occasion| {
        try stdout.print("  {s:<25} -> {s}\n", .{ name, occasions[@intCast(occasion)] });
    }
    try stdout.flush();
}
