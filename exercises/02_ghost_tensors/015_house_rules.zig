//
// Graymoor Manor has more house rules than rooms. Two of them
// trip up every new apprentice, so let's break them on purpose.
//
// HOUSE RULE: one ghost per chair.
//
// Each placeholder you pass to `compileFn` becomes one input of
// the compiled Exe, and ZML tells placeholders apart by a hidden
// id that every `Tensor.init` call hands out. "Two tensors of
// the same shape" are two separate ghosts. "The same tensor
// twice" is one ghost asked to sit on two chairs:
//
//     const x: zml.Tensor = .init(.{4}, .f32);
//     platform.compileFn(gpa, io, f, .{ x, x }, .{})
//
//     panic: Tensor with id ... has already been used once as
//            an argument
//
// If a function really must see the same input twice, it should
// take one parameter and use it twice. Otherwise: one `init`
// per parameter.
//
// (Bodies are easier going. The same Buffer may be passed for
// several inputs of an Exe: it's just data, and nothing gets
// modified.)
//
// HOUSE RULE: the body must fit the ghost.
//
// An Exe is compiled for specific shapes. When you run it, each
// Buffer must have precisely the shape of its placeholder:
//
//     panic: Expected argument 0 to have shape {4,f32}, got {5,f32}
//
// (Arguments are counted from 0, in the order of the tuple you
// gave to `eval`. So "argument 0" is the first Buffer.)
//
// Not "something that could be reshaped", not "the same number
// of bytes". Need a different size? Compile another Exe. This
// sounds strict, and it is! It's also why the compiler can
// optimize so aggressively: it knows every size in advance.
//
// A neat trick to stay out of trouble: don't repeat shapes.
// Build Buffers from `placeholder.shape()`, like the previous
// exercises did.
//
// Tonight two spirits, Agatha and Bartholomew, are to be
// compared, element by element: `a.maximum(b)` keeps the larger
// of each pair. The apprentice broke both house rules. You'll
// have to run this twice to meet both panics.
//
// But first: the last two steps, running the séance and reading
// the answer, were never written. Those are yours, and the
// compiler won't let you meet anybody before they are done.
//
const std = @import("std");
const zml = @import("zml");

/// Who is spookier, hour by hour?
fn spookiest(agatha: zml.Tensor, bartholomew: zml.Tensor) zml.Tensor {
    return agatha.maximum(bartholomew);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Spookiness for each of the four hours after midnight.
    const agatha: zml.Tensor = .init(.{ .hours = 4 }, .i32);
    const bartholomew: zml.Tensor = .init(.{ .hours = 4 }, .i32);

    var exe = try platform.compileFn(gpa, io, spookiest, .{ agatha, agatha }, .{});
    defer exe.deinit();

    var agatha_buffer: zml.Buffer = try .fromBytes(io, platform, agatha.shape(), std.mem.sliceAsBytes(&[_]i32{ 3, 9, 4, 1 }));
    defer agatha_buffer.deinit();

    // Bartholomew logged a fifth hour. Nobody asked him to; the
    // seance covers four. Drop his last entry.
    const bartholomew_log = bartholomew.shape().setDim(.hours, 5);
    var bartholomew_buffer: zml.Buffer = try .fromBytes(io, platform, bartholomew_log, std.mem.sliceAsBytes(&[_]i32{ 5, 2, 4, 8, 6 }));
    defer bartholomew_buffer.deinit();

    // Run the seance: both bodies go in, one body comes out.
    var result: zml.Buffer = ???;
    defer result.deinit();

    // Bring the four hours back as a Zig array.
    const hours: [4]i32 = ???;

    try stdout.print("Spookiest, hour by hour: {any}\n", .{hours});
    try stdout.flush();
}
