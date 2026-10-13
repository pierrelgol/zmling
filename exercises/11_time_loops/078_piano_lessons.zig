//
// With eternity on his hands, Phil takes up the piano. On his
// first day of lessons he practices 1 minute, on the second day
// 2 minutes, on day d, d minutes. How much has he practiced
// after n days?
//
// If n is a Zig number, you know what to do: a Zig `for`,
// unrolled at trace time into n additions. But nobody knows how
// long Phil is going to be stuck. n arrives at RUN time, in a
// Buffer, and a Zig `for` cannot count up to a number that
// doesn't exist yet.
//
// The graph itself needs a loop. Meet the biggest machine in
// `zml.ops`:
//
//     zml.ops.@"while"(While, context, initial_state)
//
// (`while` is a keyword too, hence the quotes. `zml.ops.while_`
// is the same function, for the quote-averse.)
//
// `While` is a struct TYPE that you write, and it must contain
// three things:
//
//     const Countdown = struct {
//         // 1. What changes from one iteration to the next.
//         pub const State = struct {
//             n: zml.Tensor,
//         };
//
//         // 2. Shall we go around again? Returns a SCALAR
//         //    tensor of dtype .bool.
//         pub fn cond(self: Countdown, state: State) zml.Tensor {
//             _ = self;
//             return state.n.cmp(.GT, zml.Tensor.scalar(0, .i32));
//         }
//
//         // 3. One trip around: from a State, make the next.
//         pub fn body(self: Countdown, state: State) State {
//             _ = self;
//             return .{ .n = state.n.addConstant(-1) };
//         }
//     };
//
//     const end = zml.ops.@"while"(Countdown, .{}, .{ .n = start });
//
// `context` is a value of type `While` (an empty one here; the
// next exercise puts things in it). `initial_state` is a
// `While.State`. The call returns the final State: the one for
// which `cond` said false. `end.n` is an ordinary Tensor again,
// and the graph goes on from there.
//
// It behaves like the Zig you would write by hand:
//
//     var state = initial_state;
//     while (cond(state)) state = body(state);
//     return state;
//
// with one enormous difference. At TRACE time, `cond` and
// `body` are each called exactly ONCE, to record what one check
// and one trip look like. At RUN time, the Exe goes around as
// many times as the data says: zero, four, a million.
//
// Note what `body` does NOT do: it doesn't change the State it
// was given. It builds the next one, and every field of the new
// State is computed from the OLD values. There is no "first
// this, then that" inside a trip.
//
// Three rules:
//
//   - `cond` returns a scalar bool Tensor. Not a Zig bool!
//   - `body` returns a State with the same shapes and dtypes it
//     was given. A loop can't change the shape of its state
//     halfway through.
//   - If `cond` never turns false, the Exe never returns. No
//     panic, no error, just February 2nd again. (Ctrl-C.)
//
// The task, part 1. The State is ready: `day` counts down from
// n to 0, `minutes` is the running total, and today's practice
// time is the value of `day`. For n = 4, the State goes through
//
//     day        4    3    2    1    0
//     minutes    0    4    7    9   10     <- `cond` says no
//
// Write `cond`, write `body`, and write `totalPractice`, which
// starts the loop and hands back the total.
//
// Part 2. There is a third place to put a loop: OUTSIDE the
// Exe, in `main`. Compile a function that does ONE trip, and
// call that Exe from an ordinary Zig `while`, feeding the result
// of each call into the next one:
//
//     var total: zml.Buffer = ...zero...;
//     defer total.deinit();
//
//     while (...) {
//         const next = try exe.eval(gpa, io, .{ total, ... });
//         total.deinit();
//         total = next;
//     }
//
// An output Buffer is a perfectly good input Buffer. The total
// never comes back to Zig; only the handle changes hands. Every
// call makes a NEW Buffer, so every trip frees the one it
// replaces, and the `defer` frees the last.
//
// One call per trip instead of one call in all: it sounds like
// the worst of the three loops. It is also how every text
// generator in the later chapters runs, because between two
// calls Zig can do what a graph cannot: print the result so far,
// look at it, decide to stop.
//
// `practiceDay` is the one trip: the total so far and a day go
// in, the new total comes out. Write it, then write its part of
// `main`: two scalar i32 placeholders, an Exe, and the loop over
// the days 100, 99, ... 1, counting the calls. The `print` at
// the bottom tells you which names it expects.
//
const std = @import("std");
const zml = @import("zml");

const Practice = struct {
    pub const State = struct {
        day: zml.Tensor,
        minutes: zml.Tensor,
    };

    /// Keep going as long as there are days left.
    pub fn cond(self: Practice, state: State) zml.Tensor {
        _ = self;
        return ???;
    }

    pub fn body(self: Practice, state: State) State {
        _ = self;
        return ???;
    }
};

fn totalPractice(days: zml.Tensor) zml.Tensor {
    // Start the loop: all the days still to go, no minutes yet.
    return ???;
}

/// Part 2. One trip, no loop: the total, `day` minutes later.
fn practiceDay(minutes: zml.Tensor, day: zml.Tensor) zml.Tensor {
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

    const days: zml.Tensor = .init(.{}, .i32);

    // Compiled once, for any number of days.
    var exe = try platform.compileFn(gpa, io, totalPractice, .{days}, .{});
    defer exe.deinit();

    for ([_]i32{ 0, 4, 100, 3650 }) |n| {
        var days_buffer: zml.Buffer = try .scalar(io, platform, n, .i32);
        defer days_buffer.deinit();

        var result: zml.Buffer = ???;
        defer result.deinit();

        try stdout.print("After {d} days, Phil has practiced {d} minutes.\n", .{ n, try result.getValue(i32, io) });
    }

    // Part 2: the loop is in Zig, and the Exe does one day per call.
    //
    // TODO: two placeholders and `day_exe`, compiled from `practiceDay`.
    //
    // TODO: `total`, a Buffer that starts at 0, and `calls`. Then
    // the days 100, 99, ... 1, each one as a Buffer, through
    // `day_exe`. What comes out is the next `total`.

    try stdout.print("By hand, 100 days: {d} minutes, in {d} calls. The loop Exe needs 1.\n", .{ try total.getValue(i32, io), calls });
    try stdout.flush();
}
