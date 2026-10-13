//
// Quiz! Phil has a theory about how to get out of February 2nd.
// He got it from Ned, the insurance salesman, who got it from a
// mathematician, who has been stuck on it since 1937.
//
// Take a number n. Then, over and over:
//
//     if n is even:  n = n / 2
//     if n is odd:   n = 3 * n + 1
//
// The day you reach 1, you are free. Phil wants to know how
// many STEPS it takes, for a starting number that he will only
// pick in the morning. (6 takes 8 steps: 6, 3, 10, 5, 16, 8, 4,
// 2, 1. Nobody has ever found a number that doesn't get there,
// and nobody has proved that there isn't one.)
//
// Phil is not a patient man, and a loop that might never end is
// exactly what he's trying to get out of. So there is a second
// condition: he gives up after `patience` steps. `patience` is
// part of Phil's character, not of the day: it is a Zig integer,
// known when you compile.
//
// So, in the words of this chapter:
//
//   - the number of trips depends on run-time data: a graph loop;
//   - even or odd is a run-time, per-value question;
//   - `patience` is trace-time, and gets baked in. `main`
//     compiles two Phils.
//
// Part 1. Write `Escape`, the While struct: its setting, its
// State, `cond` and `body`. Then `stepsToFreedom(start,
// patience)`, which returns the number of steps taken, an i32
// scalar. Go around while n is not 1 AND steps is below
// patience.
//
// Part 2. The same walk, driven from Zig, the way you drove the
// piano in exercise 078. `nextNumber(n)` is ONE step and no
// loop. `main` compiles it once and calls it again and again,
// starting from 27, each result becoming the next input. This
// time Zig has to LOOK at every result, for two reasons: to know
// when to stop (the number is 1), and to remember the highest
// number met on the way. That is the price and the point of a
// host loop: one call per step, but Zig sees everything.
//
// A step is a step: `Escape.body` has no business spelling out
// the even/odd rule a second time.
//
// Below the imports there is nothing but the data and the
// `print` lines, which tell you the names they expect. Your
// tools, all seen before:
//
//     zml.ops.@"while"(While, context, initial_state)
//     a.cmp(.NE, b)    .EQ  .LT  ...        a.remainder(b)
//     p.logical(.AND, q)                    a.div(b)
//     pred.select(on_true, on_false)        a.scale(3)
//     zml.Tensor.scalar(value, .i32)        a.addConstant(1)
//
// Everything here is an i32. And remember the house rule of
// Loopsville: a tensor has no value at trace time, so if you
// catch yourself typing a Zig `if` or a Zig `while` in there,
// it's 6:00 AM again.
//
const std = @import("std");
const zml = @import("zml");

// TODO, part 2: `fn nextNumber(n: zml.Tensor) zml.Tensor`.
// One step, no loop: the number that comes after n.

// TODO, part 1: `const Escape = struct { ... };` and
// `fn stepsToFreedom(start: zml.Tensor, patience: i32) zml.Tensor`.

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // TODO: a placeholder for the starting number, and two Phils.
    // `calm_exe` will try 1000 steps. `grumpy_exe`, Phil before
    // coffee, 20.

    for ([_]i32{ 1, 6, 7, 27, 97 }) |n| {
        // TODO: `n` goes through both Exes. The results are
        // `calm` and `grumpy`.

        try stdout.print("Starting from {d}: free after {d} steps. Before coffee, Phil stops at {d}.\n", .{
            n,
            try calm.getValue(i32, io),
            try grumpy.getValue(i32, io),
        });
    }

    // Part 2: one call per step, and Zig watches every number go by.
    //
    // TODO: compile `nextNumber`, then walk from 27 down to 1,
    // one call at a time. `calls` counts the calls, `highest` is
    // the highest number met on the way (27 counts).

    try stdout.print("By hand from 27: {d} calls to the Exe, and the highest number on the way was {d}.\n", .{ calls, highest });
    try stdout.flush();
}
