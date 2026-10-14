//
// On the north face, somebody carved "CAMP 1200 m" into the
// ice, years ago. The camp has moved twice since. The ice does
// not care.
//
// You now know that a program has ARGUMENTS: the Tensors found
// in the args of the compile call, one Buffer each at run time.
// Everything else a program needs has to be written INTO the
// program. Those are the constants, and you have used them
// since chapter 03 without looking at what they do:
//
//     pub fn scalar(val: anytype, dt: DataType) Tensor
//         a rank-0 constant. It ends in `Tensor.constant`:
//
//     pub fn constant(val: DataType.Value) Tensor {
//         const op = dialects.stablehlo.constant(
//             mlirCtx(), &.{}, ..., val.asBytes(), currentLoc(),
//         ).appendTo(currentBlock());
//         return _result(.init(&.{}, val.dtype()), op.result(0));
//     }
//
// One line of program, `stablehlo.constant`, with the BYTES of
// the value inside it. And for more than one number:
//
//     pub fn constantTensor(sh: Shape, bytes_: []const u8) Tensor
//         "Embeds a buffer with concrete values into an Mlir
//         program." Same op, with a shape and all the bytes.
//
//     const table = zml.Tensor.constantTensor(
//         .init(.{ .gait = 4 }, .f32),
//         std.mem.sliceAsBytes(&lengths),
//     );
//
// `addConstant`, `scale`, `zeroes`: all of them are a constant
// and one more op. Three consequences.
//
//   - A constant is an op: it needs a current Compiler. It can
//     only be made inside the traced function, never in `main`.
//   - A constant is not an input. `exe.input_shapes.len` does
//     not count it, and no Buffer is ever passed for it.
//   - A constant made from a Zig variable is a PHOTOGRAPH of
//     that variable, taken while the snow fell. Change the
//     variable afterwards and the Exe does not know, does not
//     panic, and goes on answering with the old number. To
//     change a constant, you trace again.
//
// Then why are weights arguments? They are fixed too, after
// all, and `constantTensor` takes any bytes. Three reasons, and
// they are the reasons ZML is built the way it is:
//
//   - Size. The program is a text that MLIR and XLA read,
//     rewrite and optimise many times over. With 4 GB of
//     weights written in it, every pass drags 4 GB along.
//     With arguments, the program of a large model is a few
//     hundred lines, whatever the size of the checkpoint.
//   - One Exe, any checkpoint. The Exe depends on the SHAPES of
//     the weights, not on their values: load other weights,
//     same Exe, no compile.
//   - Sharing. Weights are Buffers, on the device, once. The
//     two Exes of a generation loop (exercise 158) receive the
//     same Buffers; neither holds a copy.
//
// So: constants for what the ARCHITECTURE fixes and that is
// small (a mask, the positions of exercise 172, `sqrt(d)`, an
// epsilon); arguments for everything a file or a user decides.
//
// The Beast has four gaits: creeping, walking, trotting,
// bounding, with a stride of 0.5, 1, 1.5 and 3 meters. That is
// anatomy: a constant. A trail is the gait of each print,
// {.print} i32, and the model returns the marker of each print:
// the marker of the camp, plus the distance walked so far
// (`cumulativeSum`, exercise 043).
//
// `carved` reads the camp marker from a Zig variable. `surveyed`
// receives it as an input.
//
// To do:
//
//  1. `strides`: the table as a constant, then one stride per
//     print (`gather`, exercise 070).
//  2. `surveyed`.
//  3. In `main`, the camp moves. Three lines are printed after
//     that; the last one is the carved Exe "carved again", and
//     as it stands it is the old ice.
//
const std = @import("std");
const zml = @import("zml");

/// The altitude marker of the camp, in meters. It moves.
var camp_marker: f32 = 1200;

const gait_strides = [4]f32{ 0.5, 1, 1.5, 3 };

/// gaits: {.print} i32 -> {.print} f32, the stride of each print.
fn strides(gaits: zml.Tensor) zml.Tensor {
    const table = ???;
    return ???;
}

/// gaits: {.print} i32 -> {.print} f32
fn carved(gaits: zml.Tensor) zml.Tensor {
    return strides(gaits).cumulativeSum(.print).addConstant(camp_marker);
}

/// gaits: {.print} i32, camp: {} f32 -> {.print} f32
fn surveyed(gaits: zml.Tensor, camp: zml.Tensor) zml.Tensor {
    return ???;
}

const trail = [5]i32{ 1, 1, 3, 0, 2 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const gaits: zml.Tensor = .init(.{ .print = 5 }, .i32);
    const camp: zml.Tensor = .init(.{}, .f32);

    var carved_exe = try platform.compileFn(gpa, io, carved, .{gaits}, .{});
    defer carved_exe.deinit();
    var surveyed_exe = ???;
    defer surveyed_exe.deinit();

    try stdout.print("carved: {d} input. surveyed: {d} inputs.\n", .{ carved_exe.input_shapes.len, surveyed_exe.input_shapes.len });

    var gaits_buffer: zml.Buffer = try .fromBytes(io, platform, gaits.shape(), std.mem.sliceAsBytes(&trail));
    defer gaits_buffer.deinit();

    try stdout.print("the camp is at {d} m\n", .{camp_marker});
    {
        var camp_buffer: zml.Buffer = try .scalar(io, platform, camp_marker, .f32);
        defer camp_buffer.deinit();

        var a: zml.Buffer = try carved_exe.eval(gpa, io, .{gaits_buffer});
        defer a.deinit();
        var b: zml.Buffer = try surveyed_exe.eval(gpa, io, .{ gaits_buffer, camp_buffer });
        defer b.deinit();

        try stdout.print("  carved:        {any}\n", .{try a.getValue([5]f32, io)});
        try stdout.print("  surveyed:      {any}\n", .{try b.getValue([5]f32, io)});
    }

    camp_marker = 1500;
    try stdout.print("the camp moves to {d} m\n", .{camp_marker});
    {
        var camp_buffer: zml.Buffer = try .scalar(io, platform, camp_marker, .f32);
        defer camp_buffer.deinit();

        var a: zml.Buffer = try carved_exe.eval(gpa, io, .{gaits_buffer});
        defer a.deinit();
        var b: zml.Buffer = try surveyed_exe.eval(gpa, io, .{ gaits_buffer, camp_buffer });
        defer b.deinit();

        try stdout.print("  carved:        {any}\n", .{try a.getValue([5]f32, io)});
        try stdout.print("  surveyed:      {any}\n", .{try b.getValue([5]f32, io)});

        // A constant changes when...
        var c: zml.Buffer = try carved_exe.eval(gpa, io, .{gaits_buffer});
        defer c.deinit();

        try stdout.print("  carved again:  {any}\n", .{try c.getValue([5]f32, io)});
    }
    try stdout.flush();
}
