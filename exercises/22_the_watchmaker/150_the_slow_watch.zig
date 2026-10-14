//
// A customer is back, furious. "I wind it one tenth of a turn,
// a thousand times. It should show 100. It shows 32.1. It has
// shown 32.1 since Tuesday."
//
// Mr. Tick does not even open the case. "It is a .bf16. It
// stopped at 32, did it not?"
//
// Remember the spacing rule of exercise 148: with m mantissa
// bits, the numbers in [2^e, 2^(e+1)) are 2^(e-m) apart.
// .bf16 has m = 7:
//
//     in [16, 32)   neighbours are 0.125 apart
//     in [32, 64)   neighbours are 0.25  apart
//
// Add 0.1 to 32.0. The exact answer, 32.1, does not exist: it
// must become 32.0 or 32.25. The nearest one wins, and 32.0
// is nearer. The sum did not move. Add 0.1 again: same
// numbers, same rounding, same result. Forever.
//
// This is called STAGNATION, and it is the most common way
// low precision ruins a model. Nothing overflows, nothing is
// NaN, every single operation is "correctly rounded". The
// error of one addition is tiny. But a neural network is
// made of long sums: a `dot` over 4096 dimensions, a `mean`
// over a sequence, the residual stream that every block adds
// into (x = x + block(x), eighty times in a row).
//
// The cure is as old as accounting:
//
//     STORE IN THE SMALL TYPE. ADD UP IN THE BIG ONE.
//
// Convert up, accumulate in .f32, convert the RESULT down
// once. One rounding instead of a thousand. ZML does this for
// you in the places that matter. Go and read them in
// `zml/zml/tensor.zig` and `zml/zml/nn.zig`:
//
//     softmax:   self.sub(max_val).convert(.f32).exp() ...
//                                  ... .convert(self.dtype())
//     rmsNorm:   var x = x_.convert(.f32); // upcast to improve
//                                          // precision
//
// When you write a layer by hand in a port, nobody does it
// for you. (This exercise adds with a loop, one addition
// after the other, on purpose. `Tensor.sum` is free to add in
// another order, pairs first for instance, and would stall
// elsewhere. Same disease, different symptoms: do not rely
// on either.)
//
// The loop below is a Zig `for`: it runs while the graph is
// being built and leaves a thousand `add` in it (exercise 75).
//
// To do:
//
//  1. Write `spacing`: the distance between `value` and the
//     next number, for a float with `mantissa_bits`. Plain
//     Zig, no tensors: `std.math.log2`, `@floor`,
//     `std.math.pow(f32, 2, e)`.
//  2. `windCheap` is the customer's watch. Leave it alone: it
//     is the evidence.
//  3. Write `windWell`: the same thousand turns, the same
//     .bf16 result type, and the right time on the dial.
//
const std = @import("std");
const zml = @import("zml");

const turns = 1000;

/// The gap between `value` and the next representable number.
fn spacing(value: f32, mantissa_bits: f32) f32 {
    return ???;
}

/// step: {} f32 -> {} bf16. Everything in .bf16.
fn windCheap(step: zml.Tensor) zml.Tensor {
    const tenth = step.convert(.bf16);
    var dial: zml.Tensor = .scalar(0, .bf16);
    for (0..turns) |_| dial = dial.add(tenth);
    return dial;
}

/// step: {} f32 -> {} bf16. The step is still stored in .bf16.
fn windWell(step: zml.Tensor) zml.Tensor {
    return ???;
}

/// Only here to read a .bf16 result from `main`.
fn readCheap(step: zml.Tensor) zml.Tensor {
    return windCheap(step).convert(.f32);
}

fn readWell(step: zml.Tensor) zml.Tensor {
    return windWell(step).convert(.f32);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    try stdout.print("bf16 ticks near 16: {d}\n", .{spacing(16, 7)});
    try stdout.print("bf16 ticks near 32: {d}\n", .{spacing(32, 7)});
    try stdout.print("f32 ticks near 32: {e}\n", .{spacing(32, 23)});

    const step: zml.Tensor = .init(.{}, .f32);
    var step_buffer: zml.Buffer = try .scalar(io, platform, @as(f32, 0.1), .f32);
    defer step_buffer.deinit();

    var cheap_exe = try platform.compileFn(gpa, io, readCheap, .{step}, .{});
    defer cheap_exe.deinit();
    var cheap = try cheap_exe.eval(gpa, io, .{step_buffer});
    defer cheap.deinit();
    try stdout.print("The customer's watch: {d:.1}\n", .{try cheap.getValue(f32, io)});

    var well_exe = try platform.compileFn(gpa, io, readWell, .{step}, .{});
    defer well_exe.deinit();
    var well = try well_exe.eval(gpa, io, .{step_buffer});
    defer well.deinit();
    try stdout.print("After the repair: {d:.1}\n", .{try well.getValue(f32, io)});
    try stdout.flush();
}
