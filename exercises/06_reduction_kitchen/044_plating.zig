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
// every classifier and every language model. It is called
// SOFTMAX:
//
//     softmax(x)[i] = exp(x[i]) / sum over k of exp(x[k])
//
// `exp` makes everything positive, and dividing by the total
// makes it add up to 1. The output has the SAME shape as the
// input. The sum runs along one axis: the one along which the
// values compete with each other, and add up to 1 afterwards.
//
// ZML has a machine for this, of course. But Chef Odile has a
// rule: nobody touches the machine before they can do the job
// with a knife. So tonight has four rituals, and the first two
// are yours from the first line to the last.
//
// Three guests sit down at La Grande Réduction. A (suspiciously
// well informed) waiter has scored how much each of them craves
// each dish: {guest=3,dish=4}. For each guest, the four dishes
// must add up to 1.
//
// RITUAL 1, `byHand`. The formula above and nothing else. You
// have `exp` from Gus's griddle (exercise 019), a reduction
// that keeps its size 1 axis, and the fine print of exercise
// 036 to divide a table by a column.
//
// It serves guests 0 and 1 perfectly. Guest 2 is a food critic,
// and the waiter got carried away: 100, 101, 102, 99. Now,
// exp(100) is a number with 44 digits, and the largest f32 has
// 39. The result is `inf`, the total is `inf`, and inf / inf is
// NaN, "not a number". Your first table will end with a row of
// `nan`. That row is CORRECT, in the sense that it is the
// expected output of this exercise: don't fix ritual 1. Look at
// it, and remember it the day a model of yours prints `nan`.
//
// RITUAL 2, `byHandStable`. Look at the critic's logits again.
// What matters is not that they are big, but how far apart they
// are: 100, 101, 102, 99 should give the very same probabilities
// as -2, -1, 0, -3. And they do, on paper: subtract any number m
// from a whole row, and every exp is divided by exp(m), above
// and below the fraction bar. It cancels out.
//
// So choose m to be the largest logit OF EACH ROW. The biggest
// value becomes 0, the others are negative, and `exp` never
// sees anything it can't hold. That's the whole trick, used by
// everybody: shift each row by its max, then do ritual 1. (A
// ritual is a Zig function, it can call another one.)
//
// RITUAL 3, `cravings`. Now you may use the machine:
//
//     pub fn softmax(self: Tensor, axis_: anytype) Tensor
//
// Open zml/zml/core/tensor.zig and read it: it is your ritual 2,
// plus two details that can wait (it computes in f32 whatever
// the dtype, and it returns zeros for a row that is all -inf).
// Somebody used it here already, and named the wrong axis. No
// panic: you still get numbers between 0 and 1. They just answer
// a different question (which guest wants the soup most?), and
// the sums give it away.
//
// RITUAL 4, `viaLogSumExp`. Softmax has a twin:
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
// no division, no overflow. And since `exp` undoes a log,
// exp(x - logSumExp(x)) is softmax once more. One word is
// missing in that ritual.
//
// Four tables are printed. The last column is the sum of each
// row. Tables 2, 3 and 4 must be identical; table 1 must agree
// with them for the first two guests.
//
const std = @import("std");
const zml = @import("zml");

/// 1. Softmax along .dish, straight from the formula.
fn byHand(logits: zml.Tensor) zml.Tensor {
    return ???;
}

/// 2. The same, after shifting each guest's row by its max.
fn byHandStable(logits: zml.Tensor) zml.Tensor {
    return ???;
}

/// 3. For each guest, the probability of ordering each dish.
fn cravings(logits: zml.Tensor) zml.Tensor {
    return logits.softmax(.guest);
}

/// 4. The same thing: exp(x - logSumExp(x)).
fn viaLogSumExp(logits: zml.Tensor) zml.Tensor {
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

    const logits: zml.Tensor = .init(.{ .guest = 3, .dish = 4 }, .f32);

    var by_hand_exe = try platform.compileFn(gpa, io, byHand, .{logits}, .{});
    defer by_hand_exe.deinit();

    var stable_exe = try platform.compileFn(gpa, io, byHandStable, .{logits}, .{});
    defer stable_exe.deinit();

    var softmax_exe = try platform.compileFn(gpa, io, cravings, .{logits}, .{});
    defer softmax_exe.deinit();

    var lse_exe = try platform.compileFn(gpa, io, viaLogSumExp, .{logits}, .{});
    defer lse_exe.deinit();

    // Soup, duck, risotto, tart. The second guest came for the
    // tart. The third one is the critic.
    var logits_buffer: zml.Buffer = try .fromBytes(io, platform, logits.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 2, 1, -1, 0 },
        .{ 0, 0.5, 0, 4 },
        .{ 100, 101, 102, 99 },
    }));
    defer logits_buffer.deinit();

    var by_hand: zml.Buffer = try by_hand_exe.eval(gpa, io, .{logits_buffer});
    defer by_hand.deinit();

    var stable: zml.Buffer = try stable_exe.eval(gpa, io, .{logits_buffer});
    defer stable.deinit();

    var from_softmax: zml.Buffer = try softmax_exe.eval(gpa, io, .{logits_buffer});
    defer from_softmax.deinit();

    var from_lse: zml.Buffer = try lse_exe.eval(gpa, io, .{logits_buffer});
    defer from_lse.deinit();

    const tables = [_][3][4]f32{
        try by_hand.getValue([3][4]f32, io),
        try stable.getValue([3][4]f32, io),
        try from_softmax.getValue([3][4]f32, io),
        try from_lse.getValue([3][4]f32, io),
    };

    for ([_][]const u8{ "1. by hand", "2. by hand, stable", "3. softmax", "4. logSumExp" }, tables) |name, table| {
        try stdout.print("{s}:\n", .{name});
        for (table, 0..) |row, guest| {
            var total: f32 = 0;
            try stdout.print("  guest {d}:", .{guest});
            for (row) |p| {
                total += p;
                // (A NaN has a sign, and which one you get depends
                // on your CPU. Not something to print.)
                if (std.math.isNan(p)) try stdout.print(" nan", .{}) else try stdout.print(" {d:.3}", .{p});
            }
            if (std.math.isNan(total)) try stdout.print("  sum nan\n", .{}) else try stdout.print("  sum {d:.2}\n", .{total});
        }
    }
    try stdout.flush();
}
