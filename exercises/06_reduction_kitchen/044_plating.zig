//
// A neural network doesn't answer "the duck". It answers with a
// row of raw scores, one per possible answer, called LOGITS:
//
//     soup 2.0   duck 1.0   risotto -1.0   tart 0.0
//
// Bigger means "more likely", and that's all they mean. They
// can be negative, they don't add up to anything, and you can't
// put them on a plate. The step that turns logits into
// probabilities (all positive, adding up to 1) closes nearly
// every classifier and every language model:
//
//     pub fn softmax(self: Tensor, axis_: anytype) Tensor
//
//     softmax(x)[i] = exp(x[i]) / sum over k of exp(x[k])
//
// `exp` makes everything positive, and dividing by the total
// makes it add up to 1. The output has the SAME shape as the
// input; the axis you name is the one along which the values
// compete with each other and add up to 1. Name the wrong axis
// and you still get nice-looking numbers between 0 and 1. They
// just answer a different question.
//
// With what you know, you could write it yourself:
//
//     const e = x.exp();
//     return e.div(e.sum(.dish));      // rule 3 does the stretching
//
// Don't. exp(100) is already too big for an f32, and inf / inf
// is NaN. ZML's `softmax` first subtracts the max of each row,
// which changes nothing to the result and keeps `exp` out of
// trouble.
//
// The same worry gave birth to softmax's twin:
//
//     pub fn logSumExp(self: Tensor, axis_: anytype) Tensor
//
//     logSumExp(x) = log( sum over k of exp(x[k]) )
//
// computed with the same subtract-the-max trick. This one IS a
// reduction: the axis comes out with size 1. Think of it as a
// "soft" max: for 1, 2, 3 it gives 3.41; for 1, 2, 30 it gives
// 30.00. Always a bit above the true max, and very close to it
// when one value dominates.
//
// The twins are related. Take the log of the softmax formula:
//
//     log(softmax(x)) = x - logSumExp(x)
//
// Log-probabilities are what a model's loss is computed from,
// and this is how they are made in practice: one subtraction,
// no division, no overflow.
//
// Two guests sit down at La Grande Réduction. A (suspiciously
// well informed) waiter has scored how much each of them craves
// each dish: {guest=2,dish=4}. Chef Odile wants probabilities:
// for each guest, the four dishes must add up to 1. She wants
// them twice, from `softmax` and "by hand" with `logSumExp`
// (`x.exp()` undoes a log), because she trusts no one. The two
// tables must agree.
//
// Right now they don't, and one of the two rituals isn't even
// finished. The last column printed is the sum of each row.
//
const std = @import("std");
const zml = @import("zml");

/// For each guest, the probability of ordering each dish.
fn cravings(logits: zml.Tensor) zml.Tensor {
    return logits.softmax(.guest);
}

/// The same thing: exp(x - logSumExp(x)).
fn cravingsByHand(logits: zml.Tensor) zml.Tensor {
    return logits.sub(logits.???(.dish)).exp();
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const logits: zml.Tensor = .init(.{ .guest = 2, .dish = 4 }, .f32);

    var softmax_exe = try platform.compileFn(gpa, io, cravings, .{logits}, .{});
    defer softmax_exe.deinit();

    var by_hand_exe = try platform.compileFn(gpa, io, cravingsByHand, .{logits}, .{});
    defer by_hand_exe.deinit();

    // Soup, duck, risotto, tart. The second guest came for the tart.
    var logits_buffer: zml.Buffer = try .fromBytes(io, platform, logits.shape(), std.mem.sliceAsBytes(&[2][4]f32{
        .{ 2, 1, -1, 0 },
        .{ 0, 0.5, 0, 4 },
    }));
    defer logits_buffer.deinit();

    var from_softmax: zml.Buffer = try softmax_exe.eval(gpa, io, .{logits_buffer});
    defer from_softmax.deinit();

    var by_hand: zml.Buffer = try by_hand_exe.eval(gpa, io, .{logits_buffer});
    defer by_hand.deinit();

    const tables = [_][2][4]f32{
        try from_softmax.getValue([2][4]f32, io),
        try by_hand.getValue([2][4]f32, io),
    };

    for ([_][]const u8{ "softmax", "by hand" }, tables) |name, table| {
        try stdout.print("{s}:\n", .{name});
        for (table, 0..) |row, guest| {
            var total: f32 = 0;
            try stdout.print("  guest {d}:", .{guest});
            for (row) |p| {
                try stdout.print(" {d:.3}", .{p});
                total += p;
            }
            try stdout.print("  sum {d:.2}\n", .{total});
        }
    }
    try stdout.flush();
}
