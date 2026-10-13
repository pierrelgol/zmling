//
// 7:20 AM, Gobbler's Knob. Gus the groundhog is lifted out of
// his burrow, as he is every day. If Gus sees his shadow, it's
// six more weeks of winter. If he doesn't, spring comes early.
//
// This is not a job for `select`. Nobody is choosing element by
// element here: ONE yes-or-no answer decides which of two whole
// computations happens. For that, ZML has a real, recorded,
// run-time `if`. It lives in `zml.ops`, and since `if` is a Zig
// keyword, its name needs the @"..." quoting:
//
//     zml.ops.@"if"(If, if_captures, pred)
//
//     If            a struct TYPE with two methods:
//                       pub fn onTrue(self: If) T
//                       pub fn onFalse(self: If) T
//     if_captures   a VALUE of that type: its fields are the
//                   tensors (and Zig values) the branches need
//     pred          a Tensor of dtype .bool with exactly ONE
//                   element
//
// and it returns a T, whatever `onTrue` returns: a Tensor, or a
// struct of Tensors. Both methods must return the same type,
// with the same shapes.
//
// Why a struct and not two closures? Because Zig has no
// closures. The struct IS the closure: the fields are what the
// branches may touch, written down in plain sight.
//
//     const Coin = struct {
//         x: zml.Tensor,
//
//         pub fn onTrue(self: Coin) zml.Tensor {
//             return self.x.scale(2);
//         }
//         pub fn onFalse(self: Coin) zml.Tensor {
//             return self.x.scale(-1);
//         }
//     };
//
//     const y = zml.ops.@"if"(Coin, .{ .x = x }, heads);
//
// Now think in two times, like a local.
//
//   - At TRACE time, ZML calls BOTH `onTrue` and `onFalse`, once
//     each, and records each into its own little sub-graph.
//   - At RUN time, the Exe looks at `pred` and executes only ONE
//     of them. The other costs nothing. (With `select`, both
//     sides are always computed.)
//
// There is a shorter spelling for small cases:
//
//     zml.ops.if2(pred, on_true, on_false)
//
// Here `on_true` and `on_false` are not functions but values of
// the same type (Tensors, or structs of Tensors) that you have
// already computed with ordinary operations. The `if` then only
// picks one. Less ceremony; in exchange you rely on the backend
// to be clever enough not to compute the loser.
//
// About `pred`: one element, dtype .bool, checked while tracing.
// Hand it anything else and both functions panic with
//
//     zml.ops.if expects the condition to have exactly one
//     element of dtype .bool, got ...
//
// (yes, `if2` says "zml.ops.if" too). Remember where bools come
// from: `cmp`.
//
// The task. The station's light meter reports the minutes of
// sunshine this morning, as an i32 scalar. Any sunshine at all
// means a shadow. A shadow means 10 degrees colder for the three
// weeks ahead; no shadow means 5 degrees warmer. The forecast on
// file is { 1, 3, 10 }: with a shadow that makes { -9, -7, 0 },
// without one { 6, 8, 15 }.
//
//   1. Write the two branches of `Burrow`.
//   2. `prediction` doesn't survive tracing: run it and read the
//      panic.
//   3. `prediction2` must give the same answer using `if2`.
//   4. In `main`, `prediction2` is never compiled and never
//      run. Gus's lawyer is waiting.
//
// (Zig will insist on 1, 3 and 4 before it lets you meet the
// panic of 2. Zig is like that.)
//
const std = @import("std");
const zml = @import("zml");

const Burrow = struct {
    temps: zml.Tensor,

    /// Gus saw his shadow.
    pub fn onTrue(self: Burrow) zml.Tensor {
        return ???;
    }

    /// No shadow.
    pub fn onFalse(self: Burrow) zml.Tensor {
        return ???;
    }
};

fn prediction(sunshine: zml.Tensor, temps: zml.Tensor) zml.Tensor {
    return zml.ops.@"if"(Burrow, .{ .temps = temps }, sunshine);
}

/// Same prediction, with `zml.ops.if2` and no struct.
fn prediction2(sunshine: zml.Tensor, temps: zml.Tensor) zml.Tensor {
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

    const sunshine: zml.Tensor = .init(.{}, .i32);
    const temps: zml.Tensor = .init(.{ .weeks = 3 }, .i32);

    var exe = try platform.compileFn(gpa, io, prediction, .{ sunshine, temps }, .{});
    defer exe.deinit();
    var exe2 = ???;
    defer exe2.deinit();

    var temps_buffer: zml.Buffer = try .fromBytes(io, platform, temps.shape(), std.mem.sliceAsBytes(&[_]i32{ 1, 3, 10 }));
    defer temps_buffer.deinit();

    // Two mornings, two skies. Same two Exes.
    for ([_]i32{ 0, 45 }) |minutes| {
        var sunshine_buffer: zml.Buffer = try .scalar(io, platform, minutes, .i32);
        defer sunshine_buffer.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ sunshine_buffer, temps_buffer });
        defer result.deinit();
        var result2: zml.Buffer = ???;
        defer result2.deinit();

        try stdout.print("{d} minutes of sun. Gus says {any}, and Gus's lawyer says {any}.\n", .{
            minutes,
            try result.getValue([3]i32, io),
            try result2.getValue([3]i32, io),
        });
    }
    try stdout.flush();
}
