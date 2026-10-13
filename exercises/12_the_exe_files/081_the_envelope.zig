//
// THE EXE FILES
//
// Washington, D.C. A basement office of the Bureau of Unexplained
// Computations. On the wall, a poster of a blurry Buffer over a
// forest, and the words: I WANT TO COMPILE.
//
// Agent Molder believes. Agent Scally, a scientist, has been
// assigned to keep an eye on him, and checks.
//
// For ten chapters, the Exe has been the thing you rush through
// on the way to your numbers: compile, `eval`, print. This
// chapter opens the file on the Exe itself. Because real models
// don't take one tensor and return one tensor. They take
// hundreds, and they hand back several.
//
// First, what an Exe knows about itself:
//
//     exe.input_shapes     []const zml.Shape
//     exe.output_shapes    []const zml.Shape
//
// Two FLAT lists of shapes. That's all there is. However
// elaborate the arguments of your function were, the Exe has
// numbered slots, and each slot takes one Buffer.
//
// How do arguments become slots? ZML walks through the argument
// tuple you gave to `compileFn`, in order, looking for Tensors:
//
//   - a Tensor is one slot;
//   - a struct is searched field by field, in declaration
//     order (structs inside structs too);
//   - anything that contains no Tensor (a bool, a usize, a
//     string) is no slot at all. You saw this in Loopsville:
//     such values are baked in at trace time.
//
// The same walk is done over whatever the function RETURNS. And
// it may return more than a Tensor:
//
//     fn f(...) zml.Tensor                          1 output
//     fn f(...) [2]zml.Tensor                       2 outputs
//     fn f(...) struct { zml.Tensor, zml.Tensor }   2 outputs
//     fn f(...) Report                              one output per
//                                                   Tensor in Report
//
// A named struct is the civilized choice: two years from now,
// `report.loudest` will still mean something. `result[1]` won't.
//
// The case. An unmarked envelope has been slid under the office
// door. In it: the function `analyze`, below. Before running
// anything that came in an unmarked envelope, Scally wants to
// know exactly what goes in and what comes out.
//
// Give `analyze` its placeholders, and print the two lists.
// What the function needs:
//
//     the photo        4 frames of 6 pixels each, f32,
//                      axes tagged .frames and .pixels
//     the tape         10 seconds of sound, f32, tagged .seconds
//     the sensitivity  one f32 number
//
// Nothing is executed in this exercise. An Exe can be questioned
// without being run. Scally approves.
//
const std = @import("std");
const zml = @import("zml");

const Evidence = struct {
    photo: zml.Tensor,
    tape: zml.Tensor,
};

const Report = struct {
    /// The brightest pixel of each frame.
    brightest: zml.Tensor,
    /// The loudest second on the tape.
    loudest: zml.Tensor,
    /// The whole photo, with the contrast turned up.
    enhanced: zml.Tensor,
};

fn analyze(evidence: Evidence, sensitivity: zml.Tensor, night_vision: bool) Report {
    var photo = evidence.photo.mul(sensitivity);
    if (night_vision) photo = photo.scale(4);

    return .{
        .brightest = photo.max(.pixels).squeeze(.pixels),
        .loudest = evidence.tape.max(.seconds).squeeze(.seconds),
        .enhanced = photo,
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const evidence: Evidence = .{
        .photo = ???,
        .tape = ???,
    };
    const sensitivity: zml.Tensor = ???;

    // Three arguments: a struct, a Tensor, a bool.
    var exe = try platform.compileFn(gpa, io, analyze, .{ evidence, sensitivity, true }, .{});
    defer exe.deinit();

    try stdout.print("This Exe takes {d} Buffers:\n", .{exe.input_shapes.len});
    for (exe.input_shapes, 0..) |shape, i| {
        try stdout.print("  input {d}: {f}\n", .{ i, shape });
    }

    try stdout.print("and gives back {d}:\n", .{???});
    for (???, 0..) |shape, i| {
        try stdout.print("  output {d}: {f}\n", .{ i, shape });
    }
    try stdout.flush();
}
