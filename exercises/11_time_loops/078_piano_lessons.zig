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
// Three rules:
//
//   - `cond` returns a scalar bool Tensor. Not a Zig bool!
//   - `body` returns a State with the same shapes and dtypes it
//     was given. A loop can't change the shape of its state
//     halfway through.
//   - If `cond` never turns false, the Exe never returns. No
//     panic, no error, just February 2nd again. (Ctrl-C.)
//
// The task. The State is ready: `day` counts down from n to 0,
// `minutes` is the running total. Today's practice time is the
// value of `day`. Fill in the two blanks.
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
        return .{
            .day = state.day.addConstant(-1),
            .minutes = ???,
        };
    }
};

fn totalPractice(days: zml.Tensor) zml.Tensor {
    const end = zml.ops.@"while"(Practice, .{}, .{
        .day = days,
        .minutes = zml.Tensor.scalar(0, .i32),
    });
    return end.minutes;
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

        var result: zml.Buffer = try exe.eval(gpa, io, .{days_buffer});
        defer result.deinit();

        try stdout.print("After {d} days, Phil has practiced {d} minutes.\n", .{ n, try result.getValue(i32, io) });
    }
    try stdout.flush();
}
