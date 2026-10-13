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
// Write the While struct and its use. Go around while n is not
// 1 AND steps is below patience. Your tools, all seen before:
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

const Escape = struct {
    patience: i32,

    pub const State = struct {
        n: zml.Tensor,
        steps: zml.Tensor,
    };

    pub fn cond(self: Escape, state: State) zml.Tensor {
        return ???;
    }

    pub fn body(self: Escape, state: State) State {
        _ = self;
        return ???;
    }
};

/// The number of steps taken from `start`, at most `patience`.
fn stepsToFreedom(start: zml.Tensor, patience: i32) zml.Tensor {
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

    const start: zml.Tensor = .init(.{}, .i32);

    // Phil on a good day will try 1000 steps. Phil before coffee: 20.
    var calm_exe = try platform.compileFn(gpa, io, stepsToFreedom, ???, .{});
    defer calm_exe.deinit();
    var grumpy_exe = try platform.compileFn(gpa, io, stepsToFreedom, ???, .{});
    defer grumpy_exe.deinit();

    for ([_]i32{ 1, 6, 7, 27, 97 }) |n| {
        var start_buffer: zml.Buffer = try .scalar(io, platform, n, .i32);
        defer start_buffer.deinit();

        var calm: zml.Buffer = try calm_exe.eval(gpa, io, .{start_buffer});
        defer calm.deinit();
        var grumpy: zml.Buffer = try grumpy_exe.eval(gpa, io, .{start_buffer});
        defer grumpy.deinit();

        try stdout.print("Starting from {d}: free after {d} steps. Before coffee, Phil stops at {d}.\n", .{
            n,
            try calm.getValue(i32, io),
            try grumpy.getValue(i32, io),
        });
    }
    try stdout.flush();
}
