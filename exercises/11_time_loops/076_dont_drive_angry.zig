//
// Day whatever. Phil has decided to stop reading the forecast
// and start arguing with it.
//
// Yesterday (it is always yesterday) you saw that a Zig `for`
// runs at trace time. So does a Zig `if`:
//
//     fn forecast(temps: zml.Tensor, pessimist: bool) zml.Tensor {
//         var t = temps;
//         if (pessimist) t = t.addConstant(-5);
//         return t;
//     }
//
// `pessimist` is a Zig bool, known while tracing. If it is
// true, a subtraction gets recorded; if it is false, nothing
// does. The Exe contains ONE of the two roads and has no idea
// there ever was a fork. That is not a limitation, it is a
// feature: it's how one `forward` function can build the small
// and the large variant of a model, with or without dropout,
// with 6 layers or 60. You pay nothing at run time for options
// you didn't take.
//
// But here is what Phil tried next:
//
//     if (temps.cmp(.LT, zero)) ...
//
// "If it is freezing." Reasonable! Impossible. `cmp` returns a
// Tensor, and at trace time a Tensor has no value: it is a
// ghost, a promise that a bool will exist when the Exe runs.
// Zig wants a real `bool`, right now, to decide which code to
// run, and refuses to compile:
//
//     error: expected type 'bool', found 'tensor.Tensor'
//
// And even if it didn't: there are four temperatures in there.
// Which one would decide?
//
// A decision that depends on DATA has to be recorded into the
// graph, as an operation. You already own the simplest one,
// from chapter 10: `pred.select(on_true, on_false)` takes a
// bool tensor and picks, element by element, from two tensors
// of the same shape. Both sides are computed in full; `select`
// only chooses.
//
//     Zig `if`   decides once, at trace time, from Zig values.
//                The other branch doesn't exist in the Exe.
//     `select`   decides at run time, per element, from tensor
//                values. Both branches are in the Exe.
//
// The task, in two parts.
//
// 1. Wind chill in Loopsville is simple: any temperature below
//    zero feels twice as cold, the others feel like what they
//    are. And a pessimist reads every thermometer 5 degrees low
//    before the wind even starts. Phil's draft of `feelsLike`
//    is below. The pessimist switch is fine as it is, a Zig
//    `if`. The wind chill has to decide at run time.
//
// 2. `frostbiteHours` counts the hours that feel colder than
//    -5, as an i32 scalar. It must not repeat the wind chill:
//    it CALLS `feelsLike`. A helper function is a stamp
//    (exercise 075): calling it while tracing stamps its
//    operations into the graph of the caller. The counting is
//    old news: bools become i32 (exercise 021), get summed over
//    .hours, and the axis is squeezed away (chapter 06).
//
// `main` compiles each function twice, once per kind of
// weatherman. The optimist is taken care of. The pessimist's
// two Exes, and their two results, are yours.
//
const std = @import("std");
const zml = @import("zml");

fn feelsLike(temps: zml.Tensor, pessimist: bool) zml.Tensor {
    var t = temps;

    // Trace time: a pessimist reads every thermometer 5 degrees low.
    if (pessimist) t = t.addConstant(-5);

    // Run time: below zero, it feels twice as cold.
    const zero = zml.Tensor.scalar(0, .i32);
    if (t.cmp(.LT, zero)) t = t.scale(2);
    return t;
}

/// How many hours feel colder than -5? An i32 scalar.
fn frostbiteHours(temps: zml.Tensor, pessimist: bool) zml.Tensor {
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

    const temps: zml.Tensor = .init(.{ .hours = 4 }, .i32);

    var optimist_exe = try platform.compileFn(gpa, io, feelsLike, .{ temps, false }, .{});
    defer optimist_exe.deinit();
    var pessimist_exe = ???;
    defer pessimist_exe.deinit();

    var optimist_count_exe = try platform.compileFn(gpa, io, frostbiteHours, .{ temps, false }, .{});
    defer optimist_count_exe.deinit();
    var pessimist_count_exe = ???;
    defer pessimist_count_exe.deinit();

    const thermometers = [_]i32{ -4, 0, 3, 7 };
    var temps_buffer: zml.Buffer = try .fromBytes(io, platform, temps.shape(), std.mem.sliceAsBytes(&thermometers));
    defer temps_buffer.deinit();

    var optimist: zml.Buffer = try optimist_exe.eval(gpa, io, .{temps_buffer});
    defer optimist.deinit();
    var pessimist: zml.Buffer = ???;
    defer pessimist.deinit();

    var optimist_count: zml.Buffer = try optimist_count_exe.eval(gpa, io, .{temps_buffer});
    defer optimist_count.deinit();
    var pessimist_count: zml.Buffer = ???;
    defer pessimist_count.deinit();

    try stdout.print("The thermometers say   {any}\n", .{thermometers});
    try stdout.print("An optimist feels      {any}\n", .{try optimist.getValue([4]i32, io)});
    try stdout.print("A pessimist feels      {any}\n", .{try pessimist.getValue([4]i32, io)});
    try stdout.print("Hours of frostbite: {d} for the optimist, {d} for the pessimist.\n", .{
        try optimist_count.getValue(i32, io),
        try pessimist_count.getValue(i32, io),
    });
    try stdout.flush();
}
