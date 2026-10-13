//
// Monsieur Perte, the food critic, has been in again. He writes
// for a magazine called "Loss" and has never once left a tip.
// This time he complained that La Couche's leveling is
// "needlessly baroque". Annoyingly, he has a point.
//
// A layer norm does two things: it CENTERS (subtract the mean)
// and it SCALES (divide by the spread). The people who build
// large language models tried dropping the centering, found
// that nothing got worse, and kept the cheaper version. It is
// called RMS norm, for "root mean square":
//
//     rms = sqrt(mean(x²) + eps)
//     out = x / rms
//
// No mean to subtract. Llama, Mistral, Qwen and most of their
// cousins are leveled this way. So you open the pantry, look
// for `zml.nn.RMSNorm`... and there is no such struct. There is
// a function:
//
//     pub fn rmsNorm(x: Tensor, axis: anytype, eps: f32) Tensor
//
// Differences with LayerNorm, all visible in that one line:
//
//  - you say WHICH axis, by tag or by number. Nothing is
//    assumed to be last;
//  - eps is an argument, with no default;
//  - there is no weight. The function only normalizes.
//
// But every real model has a learned weight on its RMS norm!
// Indeed. And every real model in ZML's examples simply writes
// the six lines itself. The Llama example has a struct with two
// fields, the learned `weight` and an `eps`, and a `forward`
// that calls `zml.nn.rmsNorm` on the `.d` axis and multiplies
// the result by the weight, broadcast to the shape of x.
//
// This is the real lesson of the exercise. `zml.nn` is a
// pantry, not a franchise contract. A layer is just a struct
// with Tensors and a `forward`, so when the pantry doesn't
// stock yours, you bake it. Nobody will check your license.
//
// While you are in there, a close relative:
//
//     pub fn normalizeL2(input: Tensor, eps: f32) Tensor
//
//     out = x / sqrt(sum(x²) + eps)        (last axis, no choice)
//
// `sum` where rmsNorm has `mean`. The result has LENGTH 1: it
// keeps the direction of a vector and forgets its size. That's
// what you want when comparing embeddings, where "pointing the
// same way" means "similar". (The two only differ by a constant
// factor, sqrt(N). Check it on the second number of each cake,
// where the weight is 1.)
//
// Three cakes, described by two numbers each. Watch the first
// two: one is the other, ten times bigger. A normalization worth
// its salt cannot tell them apart.
//
// To do, and it is most of the program:
//
//  - the `RmsNorm` struct: the two fields described above (the
//    weight is a Tensor, eps is not) and its `forward`. Each
//    CAKE is leveled on its own: mind which axis that is;
//  - `unitLength`, a plain function;
//  - one Exe for each: one is a model with a `forward`, the
//    other a function. Chapter 12 taught you both calls;
//  - the Buffers of YOUR struct. `zml.Bufferized` works on it
//    like on any other: what survives in there?
//  - the two runs.
//
const std = @import("std");
const zml = @import("zml");

const RmsNorm = struct {
    // TODO: two fields, `weight` and `eps`.

    /// x: {.cake, .d}. Every cake leveled on its own, times the weight.
    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

/// Each cake, scaled to a length of 1. Use 1e-12 for eps.
fn unitLength(x: zml.Tensor) zml.Tensor {
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

    const level: RmsNorm = .{ .weight = .init(.{ .d = 2 }, .f32), .eps = 1e-6 };
    const cakes: zml.Tensor = .init(.{ .cake = 3, .d = 2 }, .f32);

    // TODO: `rms_exe`, the model `level` compiled for `cakes`, and
    // `unit_exe`, the function `unitLength` compiled for `cakes`.

    // The learned weight: double the first number, keep the second.
    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, level.weight.shape(), std.mem.sliceAsBytes(&[2]f32{ 2, 1 }));
    defer weight_buffer.deinit();

    const recipes = [3][2]f32{
        .{ 3, 4 }, // a cupcake
        .{ 30, 40 }, // the same cupcake, for ten people
        .{ 0, -5 }, // Monsieur Perte's opinion of it
    };
    var cakes_buffer: zml.Buffer = try .fromBytes(io, platform, cakes.shape(), std.mem.sliceAsBytes(&recipes));
    defer cakes_buffer.deinit();

    // Our own struct, our own Bufferized.
    const level_buffers: zml.Bufferized(RmsNorm) = ???;

    // The two runs. A model wants its Buffers first.
    var rms: zml.Buffer = ???;
    defer rms.deinit();

    var unit: zml.Buffer = ???;
    defer unit.deinit();

    const rms_values = try rms.getValue([3][2]f32, io);
    const unit_values = try unit.getValue([3][2]f32, io);
    for (recipes, rms_values, unit_values) |recipe, r, u| {
        try stdout.print("({d:3},{d:3})   rms norm {d:6.3} {d:6.3}   unit length {d:6.3} {d:6.3}\n", .{ recipe[0], recipe[1], r[0], r[1], u[0], u[1] });
    }
    try stdout.flush();
}
