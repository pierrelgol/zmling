//
// Chef Babette catches you stacking a second Linear right on top
// of the first one, with nothing in between.
//
// "That is not two tiers," she says. "That is one tall, sad
// tier." And she is right, it's plain algebra:
//
//     W2·(W1·x)  =  (W2·W1)·x
//
// Two linear layers in a row do exactly what ONE linear layer
// (with the weight W2·W1) would do. Stack a hundred of them and
// you have still baked a single Linear. For the stack to be
// worth anything, something that is NOT linear has to happen
// between the tiers. In a bakery this is called the oven. In a
// neural network it is called the activation.
//
// You met the activations in chapter 3, as Tensor methods:
// `x.relu()`, `x.tanh()`, `x.sigmoid()`, `x.leakyReLU(slope)`...
// `zml.nn` adds one small thing on top of them:
//
//     pub const Activation = union(enum) {
//         sigmoid,
//         tanh,
//         relu,
//         leakyReLU: f32,
//         elu: f32,
//         silu,
//         gelu,
//         quick_gelu,
//
//         pub fn forward(self: Activation, x: Tensor) Tensor
//     };
//
// A tagged union, with a `forward` that is one big `switch`
// calling the Tensor method of the same name. Two variants carry
// a payload: the slope of `leakyReLU` and the alpha of `elu`.
// In Zig you write values of this type like so:
//
//     const a: zml.nn.Activation = .gelu;
//     const b: zml.nn.Activation = .{ .elu = 1.0 };
//
// Why bother, when `x.relu()` is shorter? Because it turns the
// choice of activation into DATA. A model struct can have a
// field `oven: zml.nn.Activation`, filled in from a config file
// at start-up, and its `forward` never needs to know which one
// it got. Real model families do differ by exactly that.
//
// Now the part to think about for a second. WHEN does that
// `switch` run?
//
// Not when numbers flow. `forward` is called once, while ZML
// traces the graph, on ghosts. The switch picks a branch, that
// branch writes "relu" into the recipe, and the compiled Exe
// contains a relu and nothing else. There is no "if" left at
// run time and no cost for the flexibility. Also: an Activation
// contains no Tensor, so there is nothing of it to upload.
// `zml.Bufferized(Tier)` doesn't even have an `oven` field.
// (More about that vanishing act in a few exercises.)
//
// Tonight's experiment: one batter, four ovens. The batter is a
// Linear WITHOUT bias (`null`), which turns the cake (1, 2) into
// the four values -2, -0.5, 0.5 and 2. Then each oven setting
// gets its own Exe, because each one is a different graph.
//
// What to expect from the four ovens, for a value v (you cooked
// with all of them in exercise 020):
//
//     relu          v if v > 0, else 0            -2  ->   0
//     leakyReLU     v if v > 0, else slope·v      -2  ->  -0.2
//     tanh          squashed between -1 and 1     -2  ->  -0.964
//     sigmoid       squashed between  0 and 1     -2  ->   0.119
//
// None of them is a straight line, and that is the one thing
// asked of an oven.
//
// From this exercise on, the model is yours to write. Chef
// Babette left notes where things go:
//
//  - the `Tier` struct: its two fields and its `forward`. An
//    Activation is called like any layer: `oven.forward(x)`.
//    The Linear still wants its `out_dtype` (exercise 089);
//  - the four oven settings (one of them carries a payload);
//  - the Buffers of a Tier. Ask yourself what an oven is made
//    of before you type `.oven`;
//  - in the loop: the Tier of this setting, and its Exe. You
//    compiled a model with a `forward` all through chapter 12.
//
// If all four ovens return the same, suspiciously raw cake:
// somebody mixed the batter and forgot to bake it.
//
const std = @import("std");
const zml = @import("zml");

const Tier = struct {
    // TODO: two fields. `batter`, a Linear, and `oven`, an Activation.

    /// The batter, then the oven.
    pub fn forward(self: Tier, x: zml.Tensor) zml.Tensor {
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

    const weight: zml.Tensor = .init(.{ .dout = 4, .d = 2 }, .f32);
    const cake: zml.Tensor = .init(.{ .d = 2 }, .f32);

    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), std.mem.sliceAsBytes(&[4][2]f32{
        .{ 1, -1.5 },
        .{ 0.5, -0.5 },
        .{ -0.5, 0.5 },
        .{ 0, 1 },
    }));
    defer weight_buffer.deinit();

    var cake_buffer: zml.Buffer = try .fromBytes(io, platform, cake.shape(), std.mem.sliceAsBytes(&[2]f32{ 1, 2 }));
    defer cake_buffer.deinit();

    // The Buffers of a Tier. The batter has no bias.
    const tier_buffers: zml.Bufferized(Tier) = ???;

    // relu, leaky relu with a slope of 0.1, tanh, sigmoid.
    const settings = [_]zml.nn.Activation{ .relu, ???, .tanh, .sigmoid };

    for (settings) |setting| {
        // An ordinary run-time loop: the model is an ordinary
        // value, and each setting traces a different graph.
        // TODO: `tier`, a Tier with this setting and a batter that
        // has a weight, no bias, and contracts `.d`. Then `exe`.

        var result: zml.Buffer = try exe.eval(gpa, io, .{ tier_buffers, cake_buffer });
        defer result.deinit();

        try stdout.print("{s:<10}", .{@tagName(setting)});
        for (try result.getValue([4]f32, io)) |v| try stdout.print(" {d:7.3}", .{v});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
