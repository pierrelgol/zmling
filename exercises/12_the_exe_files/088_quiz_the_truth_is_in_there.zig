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
// `forward` returns both the score and the verdict, as a
// `Verdict` struct. It is yours to write: a Layer knows how to
// run itself, and `t.relu()` you have known since the diner.
//
// So is everything in `main` from the placeholders to the call. The weights are below, as arrays; the printing
// at the bottom tells you which names it expects. What you will
// need, all from this chapter:
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

const Agent = struct {
    hidden: Layer,
    output: Layer,
    doubt: f32,

    pub const Verdict = struct {
        score: zml.Tensor,
        alien: zml.Tensor,
    };

    pub fn forward(self: Agent, x: zml.Tensor) Verdict {
        return ???;
    }
};

//                                  glow hum speed
const hidden_weight = [2][3]f32{ .{ 1, 0, 1 }, .{ 0, 2, -1 } };
const hidden_bias = [2]f32{ 0, -1 };
const output_weight = [1][2]f32{.{ 2, 1 }};
const output_bias = [1]f32{-1};

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

    // TODO: the two agents and the case file, as placeholders.
    // TODO: `molder_exe` and `scally_exe`.
    // TODO: the weights, uploaded once, and freed at the end.
    // TODO: a runner for each Exe.

    try stdout.print("Each agent's Exe has {d} inputs and {d} outputs.\n", .{ molder_exe.input_shapes.len, molder_exe.output_shapes.len });

    for (cases, 1..) |case, n| {
        // TODO: this case as a Buffer; then ask both agents, and
        // put their answers in `molder_says` and `scally_says`.

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
    try stdout.flush();
}
