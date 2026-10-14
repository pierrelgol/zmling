//
// The Grand Complication has been on Mr. Tick's bench for
// eleven years: twenty-four wheels in a row, each one nudging
// the same eight hands a little further. A collector wants
// it "in bf16, to save weight". Mr. Tick puts down his loupe.
// "Tock. You decide. Show me the numbers."
//
// One wheel is a residual block, the shape of every block of
// every transformer you have ported:
//
//     hands = hands + 0.05 * tanh(W . hands)
//
// A big stream, and a small update added to it. Twenty-four
// times. You have seen, one at a time, everything that can
// happen to it:
//
//  - the weights W, stored in a cheaper dtype, are rounded
//    once (148);
//  - the stream, kept in a cheaper dtype, is rounded after
//    EVERY wheel, and a small update added to a big number
//    is the slow watch of exercise 150;
//  - `reducePrecision` lets one .f32 graph behave like any
//    of these (153).
//
// Three movements are on trial:
//
//     reference   weights .f32,  stream .f32
//     cheap       weights .bf16, stream .bf16
//     mixed       weights .bf16, stream .f32
//
// "Mixed" is what real deployments do, and what ZML does
// inside `softmax` and `rmsNorm`: the gigabytes (weights) in
// the small type, the arithmetic that accumulates in the
// big one.
//
// To do: everything below the constants.
//
//  1. `asSeenBy`, from exercise 153.
//  2. `movement(w, hands, weights_dtype, stream_dtype)`:
//     runs the 24 wheels and returns the hands after EACH
//     wheel, as {.wheel = 24, .d = 8} .f32. The weights are
//     seen by `weights_dtype` once. The stream is seen by
//     `stream_dtype` after every wheel, the first hands
//     included. `w` is {.dout, .d}: `dot` then rename
//     (exercise 66), `Tensor.stack` (57).
//  3. `reference`, `cheap` and `mixed`: three functions that
//     `compileFn` accepts.
//  4. `worst(a, b, wheel)`: plain Zig. The largest absolute
//     difference between the hands of two movements after
//     wheel `wheel` (rows of 8 in a flat slice, exercise 13).
//  5. `main`: compile the three, run them on the same
//     Buffers, and print, with `tolerance`:
//
//     Wheels: 24
//     The cheap movement leaves the reference at wheel: N
//     The mixed movement stays with the reference: true/false
//     At the last wheel, cheap is further off than mixed: true/false
//     The verdict: <mixed|cheap|neither> goes to the collector
//
//     N counts from 1; print "never" if it never leaves. A
//     movement "leaves" at the first wheel where `worst` is
//     above `tolerance`. The verdict is "cheap" if it never
//     leaves, else "mixed" if that one never leaves, else
//     "neither".
//
const std = @import("std");
const zml = @import("zml");

const wheels = 24;
const d = 8;
const tolerance = 0.045;

/// The 8 x 8 gear train, row by row.
const gear_train: [d * d]f32 = blk: {
    var w: [d * d]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i + 1)) * 1.7);
    break :blk w;
};

/// Where the eight hands start.
const noon: [d]f32 = .{ 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8 };

// asSeenBy, movement, reference, cheap, mixed, worst: yours.

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The three movements, compiled and run: yours. The prints below
    // expect `ref`, `cheap_hands` and `mixed_hands` ([]const f32), and
    // `cheap_leaves` and `mixed_leaves` (?usize, counted from 1).

    try stdout.print("Wheels: {d}\n", .{wheels});
    if (cheap_leaves) |wheel| {
        try stdout.print("The cheap movement leaves the reference at wheel: {d}\n", .{wheel});
    } else {
        try stdout.print("The cheap movement leaves the reference at wheel: never\n", .{});
    }
    try stdout.print("The mixed movement stays with the reference: {}\n", .{mixed_leaves == null});
    try stdout.print("At the last wheel, cheap is further off than mixed: {}\n", .{
        worst(cheap_hands, ref, wheels - 1) > worst(mixed_hands, ref, wheels - 1),
    });
    const verdict = if (cheap_leaves == null) "cheap" else if (mixed_leaves == null) "mixed" else "neither";
    try stdout.print("The verdict: {s} goes to the collector\n", .{verdict});
    try stdout.flush();
}
