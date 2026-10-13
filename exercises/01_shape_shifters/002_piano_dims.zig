//
// Your second job at Shifty & Sons: a piano. Well, a crate of
// piano parts. The manifest reads:
//
//     3 octave-blocks x 88 keys x 2 hammers per key
//
// which you already know how to write up:
//
//     const piano = zml.Shape.init(.{ 3, 88, 2 }, .i16);
//
// Each of those three numbers is the size of one AXIS. Axes are
// numbered from zero, left to right, exactly like array indices:
//
//     axis:   0   1   2
//     dims: { 3, 88,  2 }
//
// You can ask a Shape about one axis or about all of them:
//
//     piano.dim(0)   -> 3            (an i64)
//     piano.dims()   -> { 3, 88, 2 } (a []const i64)
//
// Yes, dimensions are i64, not usize. ZML sits on top of MLIR and
// StableHLO (the compiler machinery doing the heavy lifting), and
// that's the integer type they speak. You'll see `@intCast` now
// and then because of it.
//
// One more trick. Borrowed from Python, negative axes count from
// the END:
//
//     axis:   -3  -2  -1
//     dims: {  3, 88,  2 }
//
// So `dim(-1)` always means "the last axis", whatever the rank.
// This turns out to be surprisingly handy: in machine learning
// the interesting axis ("features", "channels", ...) is very often
// the last one, while the number of axes in front of it varies.
//
// Old Shifty's nephew wanted the number of keys and the number
// of hammers per key. He got one of them wrong, and the other one
// REALLY wrong. He also left a blank at the bottom, which the
// compiler will point out before anything else: fill it in, it
// is exercise 001 all over again. THEN run the program and look
// at how it fails!
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const piano = zml.Shape.init(.{ 3, 88, 2 }, .i16);

    // The number of keys, please. Not the number of octave-blocks.
    const keys = piano.dim(0);

    // The size of the LAST axis. This must keep working when
    // somebody adds another axis in front, so use a negative axis.
    const hammers = piano.dim(3);

    // For the customs form: how many parts are in the crate in
    // total? You met the method for that in exercise 001.
    const parts = piano.???;

    try stdout.print("{d} keys, {d} hammers each. Full manifest: {any}\n", .{ keys, hammers, piano.dims() });
    try stdout.print("{d} parts in the crate, {d} bytes.\n", .{ parts, piano.byteSize() });
    try stdout.flush();
}
