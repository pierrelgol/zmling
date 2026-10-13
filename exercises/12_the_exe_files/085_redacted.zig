//
// The file comes back from upstairs with half of it blacked out.
//
//     const Polygraph = struct {
//         weight: zml.Tensor,
//         bias: zml.Tensor,
//         threshold: f32,
//         paranoid: bool,
//         ...
//     };
//
// A model struct may hold more than Tensors. Sizes, flags,
// names, an epsilon, the number of attention heads: the
// SETTINGS of the model live in the struct, next to its weights,
// and `forward` uses them through `self`.
//
// You know from chapter 11 what happens to a Zig value during
// tracing: it is read once, and what it caused is frozen into
// the graph. A model's settings come in two flavours of that.
//
//     if (self.paranoid) ...
//
// is a Zig `if` (exercise 076): whatever it guards is in the
// Exe, or it isn't.
//
//     zml.Tensor.scalar(self.threshold, .f32)
//
// turns the f32 into a constant of the graph: a rank-0 tensor,
// ready to be compared with another f32 tensor, and 5 forever.
//
// Neither is an input. The Exe of a Polygraph has 3 slots:
// weight, bias, x.
//
// Which raises a question about `zml.Bufferized(Polygraph)`.
// What should it do with `threshold`? Keep an f32 that nobody
// will ever read? No:
//
//     Bufferized(T) keeps the fields that contain Tensors, and
//     turns those Tensors into Buffers. Every other field is
//     REMOVED from the type.
//
// So `Bufferized(Polygraph)` is `struct { weight: Buffer, bias:
// Buffer }`. Two fields. The settings are redacted: they belong
// to trace time, the Bufferized struct belongs to run time, and
// the two never meet. Try to give it a `.threshold` and Zig will
// tell you there is no such field. (It will call the struct by
// its birth name, something with `meta.MapRestrict(tensor.Tensor,
// buffer.Buffer)` in it. That is the machine `Bufferized` is
// built with: map every Tensor to a Buffer, drop the rest.)
//
// A pleasant consequence: models that differ only in their
// settings have the SAME Bufferized type, and can share the same
// weight Buffers. One upload, several Exes. That's what this
// case needs.
//
// Freeing. Writing one `defer x.deinit()` per weight gets old
// around the fourth weight. For a struct of Buffers there is
//
//     zml.Buffer.deinitAll(Polygraph, &weights);
//
// The first argument is the MODEL type (not the Bufferized one),
// the second a pointer to its Bufferized value. It deinits every
// Buffer inside, nested structs included. (It stops at slices.
// Next exercise.)
//
// The case. A witness statement is three numbers: sweating,
// stammering, and looking at the door. The Bureau's polygraph
// says "lying" when the score passes the threshold. Scally uses
// it as built. Molder is convinced everybody is hiding
// something, and runs it in paranoid mode. Same machine, same
// weights, two Exes.
//
// What upstairs left of the file:
//
// 1. `forward` is blacked out. The score is weight . x + bias,
//    with a single output this time: the weight is {d=3}, the
//    bias a scalar. In paranoid mode the score counts double.
//    The answer is one bool: is the score above the threshold?
//
//    In numbers: the weights are { 1, 2, 0.5 } and the bias -1.
//    The man who "fell asleep in a field" states { 2, 1, 0 },
//    which scores 1*2 + 2*1 + 0.5*0 - 1 = 3. Under 5, so Scally
//    lets him go. Molder's machine makes that a 6.
//
// 2. Molder's own polygraph isn't set up like Molder at all.
//
// 3. Somebody tried to pass the settings in with the weights.
//
// 4. Nobody is questioning the witnesses. In the loop, each
//    statement becomes a Buffer and goes through both Exes. One
//    output each, so `eval` will do. The model goes first.
//
const std = @import("std");
const zml = @import("zml");

const Polygraph = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
    threshold: f32,
    paranoid: bool,

    /// One bool: is the witness lying?
    pub fn forward(self: Polygraph, x: zml.Tensor) zml.Tensor {
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

    const scally: Polygraph = .{
        .weight = .init(.{ .d = 3 }, .f32),
        .bias = .init(.{}, .f32),
        .threshold = 5,
        .paranoid = false,
    };
    // Molder's machine: the same placeholders, another setting.
    const molder: Polygraph = .{
        .weight = scally.weight,
        .bias = scally.bias,
        .threshold = 5,
        .paranoid = false,
    };
    const x: zml.Tensor = .init(.{ .d = 3 }, .f32);

    var scally_exe = try platform.compile(gpa, io, scally, .forward, .{x}, .{});
    defer scally_exe.deinit();
    var molder_exe = try platform.compile(gpa, io, molder, .forward, .{x}, .{});
    defer molder_exe.deinit();

    try stdout.print("A Polygraph has {d} fields, its Bufferized twin has {d}. Each Exe has {d} inputs.\n", .{
        @typeInfo(Polygraph).@"struct".field_names.len,
        @typeInfo(zml.Bufferized(Polygraph)).@"struct".field_names.len,
        scally_exe.input_shapes.len,
    });

    // The weights: uploaded once, used by both Exes.
    var weights: zml.Bufferized(Polygraph) = .{
        .weight = try .fromBytes(io, platform, scally.weight.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 2, 0.5 })),
        .bias = try .scalar(io, platform, -1, .f32),
        .threshold = 5,
        .paranoid = false,
    };
    defer zml.Buffer.deinitAll(Polygraph, &weights);

    const witnesses = [3][3]f32{
        .{ 1, 0, 2 }, // a park ranger
        .{ 2, 1, 0 }, // a man who "fell asleep in a field"
        .{ 3, 2, 4 }, // a senator
    };

    for (witnesses, 1..) |statement, n| {
        var x_buffer: zml.Buffer = ???;
        defer x_buffer.deinit();

        var s: zml.Buffer = ???;
        defer s.deinit();
        var m: zml.Buffer = ???;
        defer m.deinit();

        try stdout.print("Witness {d}: Scally's polygraph says lying={}, Molder's says lying={}.\n", .{
            n,
            try s.getValue(bool, io),
            try m.getValue(bool, io),
        });
    }
    try stdout.flush();
}
