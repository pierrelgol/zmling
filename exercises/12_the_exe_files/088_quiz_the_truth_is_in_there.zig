//
// Quiz! Assistant Director Skimmer wants the basement office to
// justify its budget by Friday: one model, both agents, five
// open cases, a verdict for each.
//
// The model is the Bureau's two-layer network. A case file is
// three numbers (glow, hum, speed).
//
//     hidden = relu(hidden_layer(x))          {d=2}
//     score  = output_layer(hidden)           {d=1}
//     alien  = score > doubt                  {d=1}, a bool
//
// `doubt` is a setting, an f32 in the model struct: how much
// evidence an agent needs. Molder needs none (doubt 0). Scally
// needs a lot (doubt 10). Two agents, two Exes, ONE set of
// weights.
//
// Part 1. `Layer` is given. `Agent` is yours, all of it: its
// fields (the two layers, `hidden` and `output`, and the
// setting), a `Verdict` struct for its two outputs (`score` and
// `alien`), and a `forward` that returns one. A Layer knows how
// to run itself, and `t.relu()` you have known since the diner.
//
// So is everything in `main`, from the placeholders to the
// calls. The weights are below, as arrays; the `print` lines
// tell you which names they expect.
//
// Part 2. Friday, 4 PM. Skimmer has "reviewed" the output layer
// himself: his version of its weight and bias is below, and the
// hidden layer is untouched. Put all five cases through
// SCALLY's Exe again, with Skimmer's weights, and count how
// often it says alien. No third compile: the weights are inputs
// (exercise 087), and other Buffers of the same shapes will do.
// That makes two complete sets of weights to upload. You may
// find that a small function saves typing.
//
// What you will need, all from this chapter:
//
//     platform.compile(gpa, io, model, .forward, .{x}, .{})
//     zml.Bufferized(T)        zml.Buffer.fromBytes(...)
//     zml.Buffer.deinitAll(T, &buffers)
//     exe.runner(gpa)          runner.run(io, inputs, outputs, opts)
//
// The shapes: a layer's weight is {d_out=?,d=?} and its bias
// {d_out=?}; read the sizes off the arrays. The case file is
// {d=3}. All f32.
//
// And the fine print, also from this chapter: the model goes
// first; a placeholder serves ONE input; each call makes new
// output Buffers, and they are yours.
//
// The truth is in there.
//
const std = @import("std");
const zml = @import("zml");

const Layer = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        return self.weight.dot(x, .d).add(self.bias).withTags(.{.d});
    }
};

// TODO: `const Agent = struct { ... };` with
//   - the fields `hidden` and `output` (two Layers) and `doubt`,
//   - `pub const Verdict`, a struct of two Tensors: `score`
//     and `alien`,
//   - `pub fn forward(self: Agent, x: zml.Tensor) Verdict`.

//                                  glow hum speed
const hidden_weight = [2][3]f32{ .{ 1, 0, 1 }, .{ 0, 2, -1 } };
const hidden_bias = [2]f32{ 0, -1 };
const output_weight = [1][2]f32{.{ 2, 1 }};
const output_bias = [1]f32{-1};

// Part 2: the output layer, as reviewed by the Assistant Director.
const skimmer_output_weight = [1][2]f32{.{ 0, 0 }};
const skimmer_output_bias = [1]f32{11};

const cases = [5][3]f32{
    .{ 1, 1, 1 }, // lights over a cornfield
    .{ 4, 0, 2 }, // the Nevada tape
    .{ 0, 3, 0 }, // a humming fridge in Tulsa
    .{ 0, 0, 5 }, // something fast over the Atlantic
    .{ -1, 0, -1 }, // a weather balloon, signed and dated
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // TODO, part 1:
    //   - the two agents and the case file, as placeholders;
    //   - `molder_exe` and `scally_exe`;
    //   - the weights, uploaded once, and freed at the end;
    //   - a runner for each Exe.
    //
    // TODO, part 2: Skimmer's set of weights, same treatment.

    try stdout.print("Each agent's Exe has {d} inputs and {d} outputs.\n", .{ molder_exe.input_shapes.len, molder_exe.output_shapes.len });

    // TODO, part 2: `approved` counts Skimmer's aliens.

    for (cases, 1..) |case, n| {
        // TODO, part 1: this case as a Buffer; then ask both agents,
        // and put their answers in `molder_says` and `scally_says`.
        //
        // TODO, part 2: the same case through Scally's Exe, with
        // Skimmer's weights. Is it an alien? (It is.)

        const score = try molder_says.score.getValue([1]f32, io);
        const molder_believes = try molder_says.alien.getValue([1]bool, io);
        const scally_believes = try scally_says.alien.getValue([1]bool, io);

        try stdout.print("Case {d}: score {d}. Molder: {s} Scally: {s}\n", .{
            n,
            score[0],
            if (molder_believes[0]) "Aliens!" else "Inconclusive.",
            if (scally_believes[0]) "...Aliens." else "Swamp gas.",
        });
    }

    try stdout.print("With Skimmer's weights, Scally's Exe finds aliens in {d} cases out of {d}. Budget approved.\n", .{ approved, cases.len });
    try stdout.flush();
}
