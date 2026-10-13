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
//  7.            Scores into probabilities, then the best
//                occasion of each order: its index, and how sure
//                the till is of it. Two results, both `{.order}`.
//
// The weights and the orders are written down for you, as plain
// arrays. The printing at the bottom is written too, and tells
// you which names it expects. Everything in between is yours:
//
//  - the `Till` struct: five fields, named as in the list above,
//    a `Verdict` struct for the two results (the printing calls
//    them `occasion` and `confidence`), and `forward`;
//  - in `main`: the placeholders of the model, with the sizes
//    read off the arrays; the placeholder of the orders; the
//    Exe; every weight uploaded once and freed at the end; the
//    run.
//
// You know everything you need. A few reminders, in no
// particular order, because Chef Babette is watching and it
// would look bad if we helped too much:
//
//  - a Linear only works on tagged tensors, and what it
//    returns is not tagged like what it takes;
//  - LayerNorm levels the last axis. This one has a weight, no
//    bias, and the default eps;
//  - the oven is a relu, but `forward` must not know that;
//  - of all the things in a model, only Tensors get a Buffer.
//    And one field of Linear wants a `null` it doesn't deserve;
//  - an Exe with two outputs does not `eval` (exercise 083);
//  - `zml.Buffer.deinitAll(T, &buffers)` frees a whole flat
//    struct of Buffers at once (exercise 085).
//
// If the tea time order gets a wedding box, a layer is missing
// from your `forward`. All of them matter. That's the point of
// a wedding cake.
//
const std = @import("std");
const zml = @import("zml");

const Till = struct {
    // TODO: the five fields.

    // TODO: `Verdict`, the two results of `forward`.

    // TODO: `forward`.
    //   orders: {.order, .s} item ids
    //   -> {.order} occasion ids (i32) and {.order} confidences (f32)
};

const menu_weight = [5][3]f32{
    //  sweet, buttery, crunchy
    .{ 0.2, 0.9, 0.7 }, // 0: croissant
    .{ 0.0, 0.1, 0.9 }, // 1: baguette
    .{ 0.8, 0.6, 0.1 }, // 2: eclair
    .{ 0.9, 0.2, 0.5 }, // 3: macaron
    .{ 1.0, 0.7, 0.2 }, // 4: brownie
};

const hidden_weight = [3][3]f32{
    .{ -1, 0, 1 }, // crusty: crunchy - sweet
    .{ 1, -1, 0 }, // dainty: sweet - buttery
    .{ 1, 1, 0 }, // rich: sweet + buttery (the -1.2 is in the bias)
};
const hidden_bias = [3]f32{ 0, 0, -1.2 };

const level_weight = [3]f32{ 1, 1, 1 };

const verdict_weight = [3][3]f32{
    .{ 1, 0, 0 }, // breakfast: crusty
    .{ 0, 1, 0 }, // tea time: dainty
    .{ -1, 0, 2 }, // a wedding: very rich, and nothing crusty
};
// Weddings get a head start. Weddings pay.
const verdict_bias = [3]f32{ 0, 0, 0.6 };

const order_names = [_][]const u8{
    "2 croissants, 2 baguettes",
    "2 eclairs, 2 macarons",
    "3 brownies, 1 eclair",
};
const order_items = [3][4]i32{
    .{ 0, 1, 0, 1 },
    .{ 2, 3, 3, 2 },
    .{ 4, 4, 2, 4 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // TODO: `till`, the model, as placeholders. The oven is a relu.
    // TODO: the three orders of four items, as a placeholder.
    // TODO: the Exe.
    // TODO: the weights, uploaded once, and freed at the end.
    // TODO: the orders, as a Buffer.
    // TODO: the run. The two results go in `verdicts`.

    try stdout.print("occasions {f}, confidence {f}\n", .{ verdicts.occasion.shape(), verdicts.confidence.shape() });
    const occasions = [_][]const u8{ "breakfast", "tea time", "a wedding" };
    const confidences = try verdicts.confidence.getValue([3]f32, io);
    for (order_names, try verdicts.occasion.getValue([3]i32, io), confidences) |name, occasion, confidence| {
        try stdout.print("  {s:<25} -> {s:<9} ({d:.1}% sure)\n", .{ name, occasions[@intCast(occasion)], confidence * 100 });
    }
    try stdout.flush();
}
