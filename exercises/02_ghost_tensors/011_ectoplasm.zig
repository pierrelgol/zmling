//
// A compiled séance is useless until something with a BODY
// walks in. Meet the other half of ZML:
//
//     zml.Tensor   a ghost. Shape, dtype, no numbers.
//                  Lives while the model is being compiled.
//
//     zml.Buffer   a body. Shape, dtype, and actual numbers,
//                  stored in the memory of the device that runs
//                  the model (your RAM for .cpu, the video
//                  memory for .cuda, ...).
//                  Lives while the model is being run.
//
// They mirror each other: for each placeholder Tensor you gave
// to `compileFn`, you must supply one Buffer of the same shape
// when you run the Exe.
//
// Since a Buffer may live on a GPU, you can't just poke at its
// bytes. You copy data IN when creating it, and copy data OUT
// when you want to look at a result.
//
// In:
//
//     const data = [_]f32{ 1, 2, 3 };
//     var body: zml.Buffer = try .fromBytes(
//         io,
//         platform,
//         shape,                        // a zml.Shape
//         std.mem.sliceAsBytes(&data),  // raw bytes, []const u8
//     );
//     defer body.deinit();
//
// ZML doesn't care what Zig type your data has, it wants bytes;
// the Shape says how to interpret them. If the number of bytes
// doesn't match `shape.byteSize()`, it panics. (Three f32 are
// 3 x 4 = 12 bytes: exercise 001's formula, back from the dead.)
//
// Run:
//
//     var result: zml.Buffer = try exe.eval(gpa, io, .{body});
//     defer result.deinit();
//
// `eval` is the simplest way to call an Exe: a tuple of input
// Buffers goes in, it waits for the computation to finish, and
// hands you the (single) output Buffer. You own it: deinit it.
//
// Out:
//
//     const numbers = try result.getValue([3]f32, io);
//
// `getValue` copies the Buffer's content into an ordinary Zig
// value of the type you ask for. That type must have exactly
// the right size, and it's on you to pick one that makes sense:
// `[3]f32` for a {3,f32} buffer, `[2][3]f32` for {2,3,f32},
// plain `f32` for a scalar.
//
// Time to light actual candles. Two blanks are yours: the body
// itself, and what walks into the séance. The compiler wants
// those first. After that you'll find that Madame Zelda's
// apprentice (you) miscounted the candles somewhere. ZML will
// panic about it. The message describes the Buffer at length
// first; the part that matters comes after, with two numbers of
// bytes that ought to be equal.
//
const std = @import("std");
const zml = @import("zml");

/// Every candle flame burns twice as bright.
fn flicker(candles: zml.Tensor) zml.Tensor {
    return candles.add(candles);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const candles: zml.Tensor = .init(.{ .candles = 7 }, .f32);

    var exe = try platform.compileFn(gpa, io, flicker, .{candles}, .{});
    defer exe.deinit();

    // Flame heights, in centimeters.
    const flames = [_]f32{ 1, 2, 3, 4, 5, 6, 7 };

    // The body: same shape as the placeholder, plus the numbers.
    // Write the whole `fromBytes` call. Ask the placeholder for
    // its shape rather than spelling it out again.
    var flames_buffer: zml.Buffer = ???;
    defer flames_buffer.deinit();

    // Run the seance on the body.
    var result: zml.Buffer = try exe.eval(gpa, io, .{???});
    defer result.deinit();

    // Bring the result back to the land of the living.
    const brighter = try result.getValue([6]f32, io);

    try stdout.print("The flames rise: {any}\n", .{brighter});
    try stdout.flush();
}
