//
// Chapter 13: Layer Cake
//
// Welcome to La Couche, the bakery of Chef Babette. "Couche" is
// French for "layer", and Chef Babette builds everything in
// layers: cakes, lasagna, arguments with suppliers. You are the
// new apprentice. The previous one "over-fitted his apron".
// Nobody will tell you what that means.
//
// So far you have written every bit of math yourself. That is
// about to stop. A neural network is a stack of LAYERS, and a
// layer is a very ordinary thing:
//
//     a struct that holds some Tensors (the "weights")
//     and a `forward` method that uses them.
//
// You built such structs by hand in the last chapter. `zml.nn`
// is the pantry where ZML keeps the ready-made ones, and the
// flour-and-water of that pantry is the linear layer:
//
//     y = W·x + b
//
// Here it is, straight from zml/nn.zig (minus the quantization
// machinery, which we'll ignore for the whole course):
//
//     pub const Linear = struct {
//         weight: Tensor,
//         bias: ?Tensor = null,
//         tag: Shape.Tag,
//
//         pub fn init(
//             weight: Tensor,
//             bias: ?Tensor,
//             tag: anytype,
//         ) Linear
//
//         pub fn forward(
//             self: Linear,
//             x: Tensor,
//             out_dtype: DataType,
//         ) Tensor
//     };
//
// And `forward` is two statements you could have written
// yourself (re-wrapped here to fit the page):
//
//     const y = x
//         .dot(self.weight.convert(x.dtype()), self.tag)
//         .convert(out_dtype);
//     return if (self.bias) |bias|
//         y.add(bias.convert(y.dtype()).broad(y.shape()))
//     else
//         y;
//
// Read them slowly, everything about Linear is in there:
//
//  1. `x.dot(weight, self.tag)`. It is the `dot` of chapter 9,
//     and it contracts over a TAG. So the input AND the weight
//     must both carry an axis with that tag. Untagged tensors
//     cannot go through a Linear. No label, no cake.
//
//  2. The convention, used by every model in ZML's examples, is:
//
//         weight   {.dout, .d}    "d" for dimension, in and out
//         bias     {.dout}
//         tag      .d
//
//     `.d` is what comes in and gets consumed, `.dout` is what
//     comes out. This is the layout PyTorch saves its weights in
//     ([out_features, in_features]), which is no coincidence:
//     that's where the weights you'll load later come from.
//
//  3. The result keeps the other axes of x, in order, and then
//     gains the leftover axis of the weight:
//
//         {.cake, .d}  through  {.dout, .d}   ->   {.cake, .dout}
//
//     Note: `.dout`, not `.d`. Remember that for later.
//
//  4. The bias is found by `broad`, which matches axes by tag:
//     it must be tagged `.dout` as well. It's optional: pass
//     `null` to `init` and the layer is just the dot.
//
//  5. `out_dtype` is the dtype you want back: the dtype of the
//     input, unless you have a good reason.
//
// A Linear is a model like any other: it compiles with
// `platform.compile(gpa, io, layer, .forward, .{ x, .f32 }, .{})`
// (the `.f32` is a plain Zig argument: it is baked in and does
// NOT appear when you run the Exe), and it runs with a
// `zml.Bufferized(zml.nn.Linear)` holding one Buffer per Tensor.
//
// Your first tier. Chef Babette judges a cake by two scores,
// sweetness and richness, and she computes both from three
// measurements: cups of flour, sugar and butter. Three numbers
// in, two numbers out: a Linear with `.d = 3` and `.dout = 2`.
// The recipe card (the weights) is already on the wall:
//
//                  flour  sugar  butter     bias
//     sweetness      0      2     0.5        1
//     richness      0.5     0      3        -1
//
// One row of the weight per OUTPUT, one column per INPUT: that is
// what `{.dout = 2, .d = 3}` looks like when you write it down.
// Bake one by hand before the machine does. The sponge is 2 cups
// of flour, 1 of sugar, half a cup of butter:
//
//     sweetness = 0·2   + 2·1 + 0.5·0.5 + 1  =  3.25
//     richness  = 0.5·2 + 0·1 + 3·0.5   - 1  =  1.5
//
// Each output is one row of the weight, multiplied term by term
// with the input, summed up, plus its bias. That sum over the
// shared `.d` axis is all that "contracting" ever meant.
//
// "Before you take anything off MY shelf," says Chef Babette,
// "you show me you can make it yourself." So the tier gets baked
// twice:
//
//  1. BY HAND. Write `byHand`: the dot and the bias, as in the
//     `layer` helper you wrote in exercise 067 (the bias has
//     fewer axes than the result, so it gets broadcast first,
//     exactly as in the `forward` quoted above). Three tensors go
//     in, in the order of the signature. It is compiled with
//     `compileFn`, like any plain function.
//
//  2. FROM THE PANTRY. The same placeholders, wrapped in a
//     `zml.nn.Linear`, compiled with `platform.compile`, and run
//     with a `zml.Bufferized(zml.nn.Linear)`. That struct has a
//     third field, `quantization`: the machinery we said we'd
//     ignore. It has no default value, and it still wants its
//     `null`.
//
// The two must agree to the last crumb, and the last line of the
// program checks that they do.
//
// The previous apprentice left this half done. (He also left in
// a hurry.) What is missing, top to bottom: the body of
// `byHand`, the tag of the layer, the labels on the cakes (a
// `.cake` axis, and the axis the tier consumes), the Exe of the
// pantry tier, something in its Buffers, and the run of the
// hand-made one.
//
const std = @import("std");
const zml = @import("zml");

/// The tier, by hand: {.cake, .d} through {.dout, .d}, plus the
/// bias {.dout}. Returns {.cake, .dout}.
fn byHand(x: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor) zml.Tensor {
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

    // The layer: placeholders for its weights, and the tag of the
    // axis that `forward` will contract.
    const weight: zml.Tensor = .init(.{ .dout = 2, .d = 3 }, .f32);
    const bias: zml.Tensor = .init(.{ .dout = 2 }, .f32);
    const tier: zml.nn.Linear = .init(weight, bias, ???);

    // Three cakes, three measurements each: flour, sugar, butter.
    // A tier only accepts labeled goods: see point 1 of the lesson.
    const cakes: zml.Tensor = .init(.{ 3, 3 }, .f32);

    var hand_exe = try platform.compileFn(gpa, io, byHand, .{ cakes, weight, bias }, .{});
    defer hand_exe.deinit();

    // TODO: `exe`, the pantry tier, compiled: a model, its method,
    // and the arguments of that method.

    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 0, 2, 0.5 }, // sweetness: sugar counts double, butter helps
        .{ 0.5, 0, 3 }, // richness: it's the butter. It's always the butter.
    }));
    defer weight_buffer.deinit();

    var bias_buffer: zml.Buffer = try .fromBytes(io, platform, bias.shape(), std.mem.sliceAsBytes(&[2]f32{ 1, -1 }));
    defer bias_buffer.deinit();

    // One Buffer per Tensor of the layer, and one `null`.
    const tier_buffers: zml.Bufferized(zml.nn.Linear) = .{
        .weight = weight_buffer,
        .bias = bias_buffer,
    };

    var cakes_buffer: zml.Buffer = try .fromBytes(io, platform, cakes.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ 2, 1, 0.5 }, // sponge
        .{ 3, 0.5, 2 }, // brioche
        .{ 0, 2, 0 }, // meringue
    }));
    defer cakes_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ tier_buffers, cakes_buffer });
    defer result.deinit();

    // The hand-made tier takes its three Buffers one by one.
    var by_hand: zml.Buffer = ???;
    defer by_hand.deinit();

    try stdout.print("scores {f}\n", .{result.shape()});
    const names = [_][]const u8{ "sponge  ", "brioche ", "meringue" };
    for (names, try result.getValue([3][2]f32, io)) |name, scores| {
        try stdout.print("  {s}  sweetness {d:5.2}  richness {d:5.2}\n", .{ name, scores[0], scores[1] });
    }
    const same = std.meta.eql(try result.getValue([3][2]f32, io), try by_hand.getValue([3][2]f32, io));
    try stdout.print("By hand {f}: {s}\n", .{ by_hand.shape(), if (same) "the same six numbers." else "NOT the same numbers!" });
    try stdout.flush();
}
