//
// Two black suitcases come down the carousel. Same size, same
// weight, same little wheels. Are they the same suitcase?
//
// Depends who you ask. A Shape has three ways of comparing
// itself to another one:
//
//     a.eqlDims(b)       same dims.
//     a.eql(b)           same dims AND same dtype.
//     a.eqlWithTags(b)   same dims, same dtype AND the same tags
//                        on the same axes.
//
// Look at the middle one again: plain `eql` IGNORES TAGS. To
// `eql`, `{bags=4,f32}` and `{flights=4,f32}` are the same
// suitcase.
//
// Now the uncomfortable truth of this chapter, and better you
// hear it from us. Which of the three do the element-wise
// operations (`add`, `mul`, `maximum`, ...) use to check their
// operands? Here is the line, from ZML's tensor.zig:
//
//     var same_shape = self._shape.eql(other._shape);
//
// Plain `eql`. So:
//
//     ELEMENT-WISE OPERATIONS DO NOT CHECK TAGS.
//
// Add a `{bags=4,f32}` to a `{flights=4,f32}` and you get no
// panic and no warning. Four numbers plus four numbers. The
// result keeps the tags of the tensor the method was called on
// (the left one):
//
//     bags.add(flights)     -> {bags=4,f32}
//     flights.add(bags)     -> {flights=4,f32}
//
// Same story at run time: when you call an Exe, each Buffer is
// compared with its placeholder by dims and dtype only. Tags
// exist while tracing and are gone afterwards. (That's the "tags
// cost nothing when your model runs" from exercise 005, seen
// from the other side.)
//
// So what ARE tags for, if `add` shrugs at them? For everything
// that names an axis: `dim`, `axis`, `rename`, and nearly every
// operation in the chapters ahead. There, a wrong or missing tag
// is a panic with both shapes printed in full. Tags don't make
// mistakes impossible. They make axes say what they are, in your
// code and in every error message.
//
// And when you do want element-wise operands checked, you check.
// A recipe is ordinary Zig code that runs once, during the
// compile call, so an ordinary assert in it is a check at
// "model-building time":
//
//     std.debug.assert(a.shape().eqlWithTags(b.shape()));
//
// At the office: `weights` has the weight of each of four bags.
// `fees` has the handling fee of each of four FLIGHTS. Kevin
// added them, because four is four. He also filled in the
// incident report below, using the only comparison he knows,
// and the only order of addition he believes in.
//
// Brenda has since looked into it. Each of the four flights lost
// exactly one bag today, so fee number i really is the fee of bag
// number i, and the sum is legitimate. But in this office you
// SAY so: `checkedSum` relabels the fees per bag (the careful
// method of exercise 027), checks the labels, and only then adds.
// Kevin "simplified" its first line. Run it, and find out what a
// failed assert looks like when it happens inside a compile call.
//
// Last, somebody took the line that runs Kevin's Exe (two lines,
// with the cleanup). Exercise 010. The Buffers are waiting.
//
const std = @import("std");
const zml = @import("zml");

/// weights + fees, in that order.
fn weightsPlusFees(weights: zml.Tensor, fees: zml.Tensor) zml.Tensor {
    return weights.add(fees);
}

/// fees + weights, in that order. (Kevin: "same thing.")
fn feesPlusWeights(weights: zml.Tensor, fees: zml.Tensor) zml.Tensor {
    return weights.add(fees);
}

/// Brenda's sum: fees relabeled per bag, labels checked, then
/// fees + weights.
fn checkedSum(weights: zml.Tensor, fees: zml.Tensor) zml.Tensor {
    const per_bag = fees;
    std.debug.assert(weights.shape().eqlWithTags(per_bag.shape()));
    return per_bag.add(weights);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const weights: zml.Tensor = .init(.{ .bags = 4 }, .f32);
    const fees: zml.Tensor = .init(.{ .flights = 4 }, .f32);
    const serial_numbers: zml.Tensor = .init(.{ .bags = 4 }, .u32);

    // Each line must use a different comparison. The expected
    // answers are: true, true, false. Then: true, false, false.
    try stdout.print("weights vs fees:    dims {}, dims+dtype {}, dims+dtype+tags {}\n", .{
        weights.shape().eql(fees.shape()),
        weights.shape().eql(fees.shape()),
        weights.shape().eql(fees.shape()),
    });
    try stdout.print("weights vs serials: dims {}, dims+dtype {}, dims+dtype+tags {}\n", .{
        weights.shape().eql(serial_numbers.shape()),
        weights.shape().eql(serial_numbers.shape()),
        weights.shape().eql(serial_numbers.shape()),
    });

    // Kevin's sum compiles. Both ways around.
    var kevin = try platform.compileFn(gpa, io, weightsPlusFees, .{ weights, fees }, .{});
    defer kevin.deinit();
    var nivek = try platform.compileFn(gpa, io, feesPlusWeights, .{ weights, fees }, .{});
    defer nivek.deinit();

    // Which tags does each result carry? Don't guess: the first
    // must print a shape tagged .bags, the second .flights.
    try stdout.print("weights + fees is {f}\n", .{kevin.output_shapes[0]});
    try stdout.print("fees + weights is {f}\n", .{nivek.output_shapes[0]});

    // Brenda's sum has the fees on the left as well. Look at
    // what it prints once the check passes.
    var brenda = try platform.compileFn(gpa, io, checkedSum, .{ weights, fees }, .{});
    defer brenda.deinit();
    try stdout.print("checked sum is    {f}\n", .{brenda.output_shapes[0]});

    // It even runs. The Buffers below are built from the WRONG
    // placeholders on purpose (weights data labeled as fees, and
    // the other way around): at run time, nobody reads labels.
    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, fees.shape(), std.mem.sliceAsBytes(&[_]f32{ 23, 8, 15, 31 }));
    defer weights_buffer.deinit();
    var fees_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), std.mem.sliceAsBytes(&[_]f32{ 50, 50, 75, 120 }));
    defer fees_buffer.deinit();

    // Run Kevin's Exe on the two Buffers.

    try stdout.print("Kilograms plus dollars: {any}. Unit: unknown.\n", .{try result.getValue([4]f32, io)});
    try stdout.flush();
}
