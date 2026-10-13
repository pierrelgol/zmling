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
// The recipe card (the weights) is already on the wall.
//
// The previous apprentice left this half done. (He also left
// in a hurry.)
//
const std = @import("std");
const zml = @import("zml");

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
    const cakes: zml.Tensor = .init(.{ 3, 3 }, .f32);

    var exe = try platform.compile(gpa, io, tier, .forward, .{ cakes, .f32 }, .{});
    defer exe.deinit();

    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 0, 2, 0.5 }, // sweetness: sugar counts double, butter helps
        .{ 0.5, 0, 3 }, // richness: it's the butter. It's always the butter.
    }));
    defer weight_buffer.deinit();

    var bias_buffer: zml.Buffer = try .fromBytes(io, platform, bias.shape(), std.mem.sliceAsBytes(&[2]f32{ 1, -1 }));
    defer bias_buffer.deinit();

    // One Buffer per Tensor of the layer. (`quantization` is the
    // machinery we said we'd ignore. It still wants its `null`.)
    const tier_buffers: zml.Bufferized(zml.nn.Linear) = .{
        .weight = weight_buffer,
        .bias = bias_buffer,
        .quantization = null,
    };

    var cakes_buffer: zml.Buffer = try .fromBytes(io, platform, cakes.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ 2, 1, 0.5 }, // sponge
        .{ 3, 0.5, 2 }, // brioche
        .{ 0, 2, 0 }, // meringue
    }));
    defer cakes_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ tier_buffers, cakes_buffer });
    defer result.deinit();

    try stdout.print("scores {f}\n", .{result.shape()});
    const names = [_][]const u8{ "sponge  ", "brioche ", "meringue" };
    for (names, try result.getValue([3][2]f32, io)) |name, scores| {
        try stdout.print("  {s}  sweetness {d:5.2}  richness {d:5.2}\n", .{ name, scores[0], scores[1] });
    }
    try stdout.flush();
}
