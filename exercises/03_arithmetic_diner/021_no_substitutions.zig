//
// The second line of the sign in Flo's window: SAME DTYPE.
//
// In Zig, `@as(i32, 3) + @as(f32, 1.5)` is a compile error. You
// have to say which one you meant. Python shrugs and gives you
// 4.5; numpy and PyTorch have pages of "type promotion rules"
// deciding what an int8 plus a float16 should be.
//
// ZML sides with Zig: there is NO dtype promotion. None. An .i32
// tensor and an .f32 tensor cannot be added, multiplied, compared
// or anything else. It's not a Zig compile error, though (dtypes
// are runtime values, remember?). It's a panic while tracing:
//
//     panic: add expects tensor to be of same type, got
//            Tensor({3,i32}) and Tensor({3,f32})
//
// You say what you mean with ONE method:
//
//     x.convert(.f32)      -> a tensor of the same shape, with
//                             each element converted to f32
//
// `convert` is a real operation: it goes in the graph and costs
// time at run time. (Converting to the dtype a tensor already has
// costs nothing, ZML notices and returns the tensor unchanged.)
// What it does to the values:
//
//     int   -> float    the nearest float.
//     float -> int      TRUNCATES toward zero: 2.9 becomes 2,
//                       -2.9 becomes -2. If you wanted rounding,
//                       round first: `x.round().convert(.i32)`.
//
//     x                          { -2.9, -0.5, 0.5, 2.9 }
//     x.convert(.i32)            { -2,    0,   0,   2   }
//     x.round().convert(.i32)    { -3,    0,   0,   3   }
//
// The no-promotion rule also applies to the constant forms from
// two exercises ago, in a sneakier way. Remember that
// `x.scale(c)` is `x.mul(.scalar(c, x.dtype()))`: the constant is
// made in the dtype of the tensor. For an .i32 tensor, `scale(2)`
// is fine. `scale(1.5)` asks for an .i32 constant holding 1.5,
// and:
//
//     panic: Could not create Value of type i32 from value of
//            type comptime_float
//
// ZML refuses to quietly turn your 1.5 into a 1. Good.
//
// Why so strict? Because in machine learning the dtype IS the
// performance. One accidental promotion to f64 in the middle of
// a model and it runs at half speed with twice the memory, and
// nothing tells you. Here, nothing happens that you didn't write.
//
// Payday at the diner. For each of the three cooks:
//
//   - `hours`  worked this Sunday, whole hours:        .i32
//   - `wage`   dollars per hour:                       .f32
//   - `bonus`  whole dollars, from the tip jar:        .i32
//
// Sundays pay time and a half. Flo pays in whole dollars: the
// cents are dropped (not rounded, dropped; she has a heart, not
// two), THEN the bonus is added. The envelope is .i32.
//
// The third cook, on a napkin: 6 hours count as 9. At 15.25 an
// hour that is 137.25 dollars, so 137 in the envelope, plus 20
// from the tip jar: 157.
//
// The dishwasher wrote the formula as if dtypes were a
// suggestion. You'll meet both panics from above, one after the
// other: the one about the constant first, and once that is
// cured, the one about two tensors. Each time, ask what dtype
// the chain has AT THAT POINT, and what dtype it should have.
//
// Then Gus wants to know what that came to PER HOUR, in dollars
// and cents this time: the envelope divided by the hours. Two
// .i32 tensors in, an .f32 tensor out, and remember what `div`
// does to integers (exercise 017). The recipe `hourly` is yours.
//
// main() runs it the way exercise 018 ran the register: the
// envelopes are still on the device, so the `result` of the first
// Exe is the first input of the second one, next to the hours
// Buffer that is already there. Two lines of main() are for you:
// the placeholder for the envelopes (what did `payday` return?),
// and the run itself.
//
const std = @import("std");
const zml = @import("zml");

/// trunc(hours * 1.5 * wage) + bonus, as whole dollars.
fn payday(hours: zml.Tensor, wage: zml.Tensor, bonus: zml.Tensor) zml.Tensor {
    return hours.scale(1.5).mul(wage).add(bonus);
}

/// pay / hours, in dollars and cents.
fn hourly(pay: zml.Tensor, hours: zml.Tensor) zml.Tensor {
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

    const hours: zml.Tensor = .init(.{ .cooks = 3 }, .i32);
    const wage: zml.Tensor = .init(.{ .cooks = 3 }, .f32);
    const bonus: zml.Tensor = .init(.{ .cooks = 3 }, .i32);

    var exe = try platform.compileFn(gpa, io, payday, .{ hours, wage, bonus }, .{});
    defer exe.deinit();

    var hours_buffer: zml.Buffer = try .fromBytes(io, platform, hours.shape(), std.mem.sliceAsBytes(&[_]i32{ 8, 10, 6 }));
    defer hours_buffer.deinit();
    var wage_buffer: zml.Buffer = try .fromBytes(io, platform, wage.shape(), std.mem.sliceAsBytes(&[_]f32{ 12.5, 11, 15.25 }));
    defer wage_buffer.deinit();
    var bonus_buffer: zml.Buffer = try .fromBytes(io, platform, bonus.shape(), std.mem.sliceAsBytes(&[_]i32{ 5, 0, 20 }));
    defer bonus_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ hours_buffer, wage_buffer, bonus_buffer });
    defer result.deinit();

    // The Exe knows the dtype of what it returns.
    try stdout.print("Pay envelopes are {f}: {any}\n", .{
        exe.output_shapes[0],
        try result.getValue([3]i32, io),
    });

    // Per hour. A placeholder for the envelopes, as `payday`
    // returns them.
    const pay: zml.Tensor = ???;

    // `hours` again? Yes. "One ghost per chair" (exercise 015) is
    // about one compile call: a placeholder may sit in as many
    // seances as you like, one chair in each.
    var hourly_exe = try platform.compileFn(gpa, io, hourly, .{ pay, hours }, .{});
    defer hourly_exe.deinit();

    // Run it on the envelopes and the hours. Both are on the
    // device already.
    var hourly_result: zml.Buffer = ???;
    defer hourly_result.deinit();

    try stdout.print("Per hour, as {f}:", .{hourly_exe.output_shapes[0]});
    for (try hourly_result.getValue([3]f32, io)) |dollars| try stdout.print(" {d:.3}", .{dollars});
    try stdout.print("\n", .{});
    try stdout.flush();
}
