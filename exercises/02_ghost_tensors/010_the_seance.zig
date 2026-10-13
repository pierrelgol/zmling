//
// To make a ghost do something, you hold a séance. In ZML, a
// séance needs three things.
//
// 1. A MEDIUM: the `zml.Platform`. It represents the hardware
//    that will run your computations, and it owns the compiler
//    for that hardware.
//
//        const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
//        defer platform.deinit(gpa, io);
//
//    `.cpu` is the target. Others are .cuda, .rocm, .metal, ...
//    and `zml.Platform.auto(gpa, io, .{})` picks the best one
//    available. Your code doesn't change, only this line does.
//    (zmling sticks to .cpu so that everybody gets the same
//    answers.)
//
// 2. A RITUAL: a plain Zig function that takes Tensors and
//    returns Tensors.
//
//        fn flicker(candles: zml.Tensor) zml.Tensor {
//            return candles.add(candles);
//        }
//
//    `a.add(b)` is the element-wise sum of two tensors of the
//    same shape and dtype. Inside a ritual, calling it is
//    allowed; and remember that it doesn't add anything! It
//    writes "add these two" into the recipe and returns a new
//    ghost standing for the future result.
//
// 3. The séance itself: compiling.
//
//        var exe = try platform.compileFn(gpa, io, flicker, .{candles}, .{});
//        defer exe.deinit();
//
//    The arguments, in order: an allocator, the io, the ritual,
//    a tuple with one placeholder per parameter of the ritual,
//    and compile options (the defaults are fine).
//
//    During this call ZML runs your function ONCE, with the
//    placeholders as arguments, and records every tensor
//    operation it performs. This is called TRACING. The recorded
//    graph is then optimized and compiled to machine code. What
//    you get back is a `zml.Exe`: an executable.
//
// The Exe knows what it wants and what it gives:
//
//     exe.input_shapes    []const zml.Shape
//     exe.output_shapes   []const zml.Shape
//
// Notice we've said nothing about actual candle flames yet. A
// model is compiled for SHAPES, before any data exists. That
// comes next. For now, complete the ritual and the séance. Then
// read what the Exe says about itself: the apprentice set the
// table for the wrong number of candles.
//
// (Curious about what ZML recorded? Add this line below the
// imports:
//
//     pub const std_options: std.Options = .{ .log_level = .debug };
//
// and run this exercise by hand:
//
//     cd zml && bazel run //zmling/02_ghost_tensors:010
//
// The debug log shows the "ZML generated MLIR": your function,
// as the compiler sees it. zmling only checks stdout, and logs
// go to stderr, so you can do this in any exercise.)
//
const std = @import("std");
const zml = @import("zml");

/// Every candle flame burns twice as bright.
fn flicker(candles: zml.Tensor) zml.Tensor {
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

    // One placeholder per parameter of the ritual. Seven candles.
    const candles: zml.Tensor = .init(.{ .candles = 13 }, .f32);

    var exe = try platform.compileFn(gpa, io, flicker, .{???}, .{});
    defer exe.deinit();

    try stdout.print("The seance takes {d} input {f} and returns {d} output {f}.\n", .{
        exe.input_shapes.len,
        exe.input_shapes[0],
        exe.output_shapes.len,
        exe.output_shapes[0],
    });
    try stdout.flush();
}
