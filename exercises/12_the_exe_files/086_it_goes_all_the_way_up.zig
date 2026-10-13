//
// Molder has covered the wall with photographs and red string.
// "It's not one detector, Scally. It's layers. Every one of them
// reports to the next. It goes all the way up."
//
// For once, he is right. Real models are structs made of
// structs: a network holds blocks, a block holds layers, a layer
// holds a weight and a bias. And when there are N of the same
// thing, they sit in a slice:
//
//     const Layer = struct {
//         weight: zml.Tensor,
//         bias: zml.Tensor,
//
//         pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor { ... }
//     };
//
//     const Network = struct {
//         layers: []const Layer,
//
//         pub fn forward(self: Network, x: zml.Tensor) zml.Tensor {
//             var h = x;
//             for (self.layers) |layer| h = layer.forward(h);
//             return h;
//         }
//     };
//
// Look at that `for` with your Loopsville eyes. It is a Zig
// loop over a Zig slice: it runs at trace time and UNROLLS. The
// number of layers is frozen into the Exe, each trip uses
// different weights, and `layer.forward(h)` is an ordinary
// function call that happens to record a few operations.
// Calling a sub-model is just calling a method. Only the
// outermost `forward` gets compiled.
//
// The walk that finds Tensors goes through all of it: nested
// structs, arrays, slices, optionals, in order. Three layers
// here, so the Exe has 3 * 2 + 1 = 7 inputs:
//
//     layers[0].weight, layers[0].bias, layers[1].weight, ...,
//     layers[2].bias, x
//
// And `Bufferized` mirrors all of it. `zml.Bufferized(Network)`
// is
//
//     struct { layers: []const zml.Bufferized(Layer) }
//
// A slice of Bufferized layers. A slice needs memory to point
// at, and who owns that memory is your business, as always in
// Zig. Here an array on the stack does it:
//
//     var layer_buffers: [3]zml.Bufferized(Layer) = undefined;
//     ...fill them in...
//     const weights: zml.Bufferized(Network) = .{ .layers = &layer_buffers };
//
// Freeing has one catch. `zml.Buffer.deinitAll` only handles
// FLAT structs, the kind whose size is known at compile time. It
// refuses to follow a slice, and it refuses at compile time:
//
//     zml.meta.visitStruct only handles flat struct, found
//     pointer in ...
//
// It works fine on each `Layer`, though. So: loop over the
// layers yourself. (Deleting the cleanup would shut the compiler
// up just as well, and you know from the stakeout that nobody
// would notice six leaked Buffers. Scally would know.)
//
// The case: a three-layer chain of command. Each layer computes
//
//     relu(weight . x + bias)
//
// and renames its output axis so that the next layer can take
// it: `weight.dot(x, .d)` leaves an axis called .d_out, and the
// next layer wants a .d. (`t.withTags(.{.d})` relabels the axes
// of t, chapter 04. And `t.relu()` is from the diner, exercise
// 020: negative numbers become 0.)
//
// With the numbers of this file, the report { 3, 1 } goes up the
// chain as
//
//     { 4, 2 }   ->   { 1, 0 }   ->   { 2.5, 4.5 }
//
// and the 0 was a -1 before middle management's relu got to it.
//
// The file is thin. From the bottom of the chain to the top:
//
// 1. `Layer` is an empty struct. It needs its two fields and
//    its `forward`.
//
// 2. `Network.forward`.
//
// 3. The placeholders. Every layer needs its OWN weight and
//    bias: a placeholder serves one input. The two Shapes are
//    there; `zml.Tensor.fromShape(shape)` makes a fresh
//    placeholder from a Shape (chapter 02).
//
// 4. The compile call.
//
// 5. Each layer's two Buffers. In that loop `w` and `b` are
//    already POINTERS to arrays, which is what
//    `std.mem.sliceAsBytes` wants: no `&` this time.
//
// 6. And whoever wrote the cleanup hadn't read this far. (Once
//    nothing takes `&weights` any more, Zig will point out that
//    `weights` never changes, and ask for a `const`. Fair.)
//
const std = @import("std");
const zml = @import("zml");

const Layer = struct {
    // TODO: the two fields, and
    //     pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor
    // which hands the next layer an axis called .d
};

const Network = struct {
    layers: []const Layer,

    pub fn forward(self: Network, x: zml.Tensor) zml.Tensor {
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

    const weight_shape: zml.Shape = .init(.{ .d_out = 2, .d = 2 }, .f32);
    const bias_shape: zml.Shape = .init(.{ .d_out = 2 }, .f32);

    // Every layer needs its OWN placeholders: a placeholder may
    // only be used for one input.
    var layers: [3]Layer = undefined;
    for (&layers) |*layer| layer.* = ???;
    const network: Network = .{ .layers = &layers };
    const x: zml.Tensor = .init(.{ .d = 2 }, .f32);

    var exe = ???;
    defer exe.deinit();

    try stdout.print("{d} layers, {d} inputs.\n", .{ network.layers.len, exe.input_shapes.len });

    // From the field office up to the man in the corner office.
    const weight_data = [3][2][2]f32{
        .{ .{ 1, 1 }, .{ 1, -1 } },
        .{ .{ 0.5, 0 }, .{ 0, 2 } },
        .{ .{ 2, 3 }, .{ 4, 5 } },
    };
    const bias_data = [3][2]f32{
        .{ 0, 0 },
        .{ -1, -5 },
        .{ 0.5, 0.5 },
    };

    var layer_buffers: [3]zml.Bufferized(Layer) = undefined;
    for (&layer_buffers, &weight_data, &bias_data) |*layer_buffer, *w, *b| {
        // This layer's two Buffers, made from `w` and `b`.
        layer_buffer.* = ???;
    }

    var weights: zml.Bufferized(Network) = .{ .layers = &layer_buffers };
    defer zml.Buffer.deinitAll(Network, &weights);

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[_]f32{ 3, 1 }));
    defer x_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ weights, x_buffer });
    defer result.deinit();

    try stdout.print("What reaches the top: {any}\n", .{try result.getValue([2]f32, io)});
    try stdout.flush();
}
