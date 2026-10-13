//
// 6:00 AM. The clock radio clicks on and plays the same song as
// yesterday. Phil Tracer, weatherman, is in the small town of
// Loopsville to cover the Groundhog Festival, and he has now
// woken up on February 2nd more often than he cares to count.
//
// You know the feeling. You have been living with a strange
// fact since chapter 02 and it is time to look it in the eye:
//
//     During `compileFn`, ZML runs your function ONCE, with
//     placeholders, and records the tensor operations.
//     After that, your Zig function is never called again.
//
// So a compiled function lives in two different times.
//
//     TRACE TIME   your Zig code runs. Once. Tensors have shapes
//                  but no values.
//     RUN TIME     the Exe runs. As often as you like. Values
//                  exist, but your Zig code is long gone: only
//                  the recorded graph is executed.
//
// Everything in your function that is NOT a tensor operation
// belongs to trace time: `std.debug.print`, incrementing a
// counter, an `if`, a `for`. Take this:
//
//     fn triple(x: zml.Tensor) zml.Tensor {
//         var y = x;
//         for (0..2) |_| y = y.add(x);
//         return y;
//     }
//
// The graph has no loop in it. ZML never saw a `for`: it saw
// `add` being called twice, and wrote down
//
//     y1 = add(x, x)
//     y2 = add(y1, x)
//
// The loop has been UNROLLED. Zig is the macro language of your
// graph: loops and helper functions are ways of stamping out
// operations, and the stamp is gone when tracing is over.
//
// Next strange fact: the argument tuple of `compileFn` may hold
// things that are not tensors.
//
//     fn relive(mood: zml.Tensor, days: usize) zml.Tensor
//
//     platform.compileFn(gpa, io, relive, .{ mood, 7 }, .{});
//
// ZML looks through the arguments for Tensors, and those become
// the inputs of the Exe. Everything else (ints, bools, enums,
// structs of settings) is simply handed to your function during
// tracing. That 7 is BAKED IN: this Exe relives seven days,
// forever, and `exe.input_shapes.len` is 1. To relive eight you
// compile another Exe.
//
// Phil's mood halves with every repeated day (`x.scale(0.5)`,
// you know that one). Three things to do:
//
//   1. Write `relive`: one trip per day, and every trip rings
//      the alarm (`alarm_rings += 1`) and halves the mood.
//   2. Compile it for THREE days. Somebody copied the 7 from
//      the example above.
//   3. In the loop of `main`, this morning's mood has to become
//      a scalar Buffer and go through the Exe. That is chapter
//      02, and from this chapter on it is more and more often
//      yours to write.
//
// `alarm_rings` is an ordinary Zig global, counting how often
// the loop body really runs. Look at what it says after the Exe
// has been called four times. That number is this exercise.
//
const std = @import("std");
const zml = @import("zml");

var alarm_rings: usize = 0;

fn relive(mood: zml.Tensor, days: usize) zml.Tensor {
    // One trip per day. Every trip rings the alarm and halves the mood.
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

    const mood: zml.Tensor = .init(.{}, .f32);

    // One tensor, and one plain Zig number: three days.
    var exe = try platform.compileFn(gpa, io, relive, .{ mood, 7 }, .{});
    defer exe.deinit();

    try stdout.print("The Exe has {d} input. While tracing, the alarm rang {d} times.\n", .{ exe.input_shapes.len, alarm_rings });

    for ([_]f32{ 80, 64, 8, 0 }) |morning_mood| {
        // This morning's mood, as a scalar Buffer.
        var mood_buffer: zml.Buffer = ???;
        defer mood_buffer.deinit();

        // One day in the life of an Exe.
        var result: zml.Buffer = ???;
        defer result.deinit();

        try stdout.print("Phil wakes up at {d} and goes to bed at {d}.\n", .{ morning_mood, try result.getValue(f32, io) });
    }

    try stdout.print("Four runs later, the alarm has rung {d} times in total.\n", .{alarm_rings});
    try stdout.flush();
}
