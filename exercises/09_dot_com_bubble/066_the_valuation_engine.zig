//
// The venture capitalists are back, and they'd like to know
// what PetRocks.com is worth. Chad promises them "an
// AI-powered valuation engine". Brenda writes down what that
// is going to be:
//
//     y = W x + b
//
// A LINEAR LAYER (also: dense layer, fully connected layer). It
// has a few inputs and a few outputs, and every output is a
// weighted sum of all the inputs, plus a constant of its own.
// Stack a hundred of these with something non-linear in between
// and you have most of deep learning. This one has
//
//     3 inputs   employees, foosball tables, and how often the
//                press release says "synergy"
//     2 outputs  valuation and monthly burn, in million dollars
//
// and three tensors:
//
//     x     {startup=2,d_in=3}   one row of inputs per startup
//     w     {d_out=2,d_in=3}     the weights: one row PER OUTPUT
//     bias  {d_out=2}            one constant per output
//
// (Weights stored "outputs first" is what you'll find in model
// files from PyTorch. With tags, you don't have to care.)
//
// "A weighted sum over the inputs" is a contraction over .d_in:
//
//     const y = x.dot(w, .d_in);       // {startup=2,d_out=2}
//
// .d_in is contracted, .startup is a free axis of `x`, .d_out a
// free axis of `w`: left first, then right. If you're thinking
// "but W x means w.dot(x)": that works just as well, and gives
// {d_out=2,startup=2}. Putting `x` on the left keeps the batch
// axis in front, where everybody expects it.
//
// Then the bias. `y.add(bias)` is {startup=2,d_out=2} plus
// {d_out=2}, and you know what ZML thinks of that: only rank 0
// tensors are broadcast automatically. It is one bias per
// OUTPUT, repeated for every startup, and you have to say so:
//
//     y.add(bias.broad(y.shape()))
//
// `broad` matches by tags, finds .d_out, and repeats along
// .startup. That's the whole layer. `zml.nn.Linear`, which
// you'll meet in a later chapter, has the same two lines in its
// `forward`, plus a few dtype conversions.
//
// By hand, for PetRocks.com and the weights of `main`:
// 0.1*12 employees + 5*1 foosball table + 20*7 synergies is
// 146.2, plus a bias of 10: 156.2 million dollars. For rocks.
//
// Chad typed the formula straight from the whiteboard, and it
// panics.
//
// The VCs' second question is the one that matters in 1999: how
// long until the money runs out? `runway` is the valuation
// divided by the monthly burn, one number per startup. Both
// numbers are in the output of the layer, side by side along
// .d_out (the valuation at index 0, the burn at index 1), and
// you know how to take one of them out of its box, axis gone
// (the pizzeria, exercise 55). `runway` may call `valuation`.
//
// As for `main`: Brenda left the data and the prints, and went
// to lunch. There are no placeholders (the three shapes are in
// the table above, all of them .f32; the Buffers are waiting
// for them), no Exes and no results. Name the Exes as you
// like. The prints call the two result Buffers `result` and
// `months`.
//
const std = @import("std");
const zml = @import("zml");

/// y = W x + b, for every startup at once.
fn valuation(x: zml.Tensor, w: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    return x.dot(w, .d_in).add(bias);
}

/// Months of money left, valuation / burn: {startup}.
fn runway(x: zml.Tensor, w: zml.Tensor, bias: zml.Tensor) zml.Tensor {
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

    // Three placeholders, `x`, `w` and `bias`, and one Exe for each
    // of the two functions.
    ???;

    // Employees, foosball tables, mentions of "synergy".
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 12, 1, 7 }, // PetRocks.com
        .{ 300, 0, 0 }, // a company that makes actual things
    }));
    defer x_buffer.deinit();
    var w_buffer: zml.Buffer = try .fromBytes(io, platform, w.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 0.1, 5, 20 }, // valuation: what each input is "worth"
        .{ 0.01, 0.5, 0.25 }, // burn: what each input costs per month
    }));
    defer w_buffer.deinit();
    // It's 1999: you're worth 10 million for showing up.
    var bias_buffer: zml.Buffer = try .fromBytes(io, platform, bias.shape(), std.mem.sliceAsBytes(&[_]f32{ 10, 0.5 }));
    defer bias_buffer.deinit();

    // Run both: `result` for the valuation, `months` for the runway.
    ???;

    try stdout.print("output {f}, runway {f}\n", .{ result.shape(), months.shape() });
    for (try result.getValue([2][2]f32, io), try months.getValue([2]f32, io), [_][]const u8{ "PetRocks.com", "Boring Inc." }) |y, left, name| {
        try stdout.print("{s}: valued at ${d:.2}M, burning ${d:.2}M a month, {d:.1} months to live\n", .{ name, y[0], y[1], left });
    }
    try stdout.flush();
}
