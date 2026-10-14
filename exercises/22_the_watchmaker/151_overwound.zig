//
// CRACK. Tock has wound a mainspring until it snapped, and
// now the hands of the watch point at nothing at all.
//
// Floats have two ways of pointing at nothing:
//
//     inf   too big to hold. exp(1000) in .f32 (anything
//           above exp(88.7)), 1/0. It is still ordered:
//           inf > everything, and inf + 1 = inf.
//     NaN   "not a number": a question without an answer.
//           inf - inf, inf / inf, 0 / 0, 0 * inf.
//
// NaN is the dangerous one, because it is CONTAGIOUS. Any
// operation that touches a NaN returns a NaN. One NaN in one
// attention score, and after the softmax the whole row is
// NaN; after the next `dot`, the whole sequence; one block
// later, the whole model. The port prints token 0 forever,
// and the bug is forty layers upstream.
//
// How to catch them in a graph:
//
//     x.isFinite()      bool tensor, false for inf, -inf, NaN
//     x.cmp(.NE, x)     bool tensor, true ONLY for NaN:
//                       NaN is the one value not equal to
//                       itself. (`cmp` is exercise 68.)
//
// To count the `true` of a bool tensor: convert it to .i32
// and `sum`. (`Tensor.any` and `Tensor.all` exist, and do
// not compile in this version of ZML.)
//
// The classic place where a model overwinds is the softmax:
//
//     softmax(x)_i = exp(x_i) / sum_j exp(x_j)
//
// Written like that, scores of 1000, 999 and 998 give
// inf / inf = NaN, three times. The cure costs nothing:
// softmax does not change when the same number is subtracted
// from every score (the exp of it cancels between top and
// bottom). Subtract the LARGEST score:
//
//     m = max(x);  softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)
//
// Now the biggest exponent is exp(0) = 1 and the others are
// smaller: nothing can overflow, and the sum is at least 1.
// The same trick saves log(sum(exp(x))), which appears in
// every loss and every perplexity:
//
//     logsumexp(x) = m + log(sum_j exp(x_j - m))
//
// `Tensor.softmax(axis)` and `Tensor.logSumExp(axis)` are
// written this way (you used them in exercise 44 without
// being told). `softmax` knows one more trick: a row of
// attention scores that is ENTIRELY -inf, a query with
// nobody to listen to. There m = -inf, x - m = -inf + inf =
// NaN. ZML's softmax returns a row of zeros instead. A
// hand-written one, yours included, does not.
//
// To do:
//
//  1. Write `countNan` and `countBad` (bad = not finite).
//  2. `naiveSoftmax` is the snapped mainspring. Keep it.
//  3. Write `stableSoftmax` and `stableLogSumExp`. `max` and
//     `sum` keep the reduced axis with size 1 (chapter 6):
//     `sub` and `div` stretch it for you, because it is the
//     right operand and the tags match (exercise 36).
//  4. The last two lines of the output are about the silent
//     row. Explain them to Tock before you move on.
//
const std = @import("std");
const zml = @import("zml");

/// How many NaN in `x`: {} i32
fn countNan(x: zml.Tensor) zml.Tensor {
    return ???;
}

/// How many values of `x` are not finite: {} i32
fn countBad(x: zml.Tensor) zml.Tensor {
    return ???;
}

/// scores: {.score} -> {.score}
fn naiveSoftmax(scores: zml.Tensor) zml.Tensor {
    const e = scores.exp();
    return e.div(e.sum(.score));
}

/// scores: {.score} -> {.score}
fn stableSoftmax(scores: zml.Tensor) zml.Tensor {
    return ???;
}

/// scores: {.score} -> {}
fn stableLogSumExp(scores: zml.Tensor) zml.Tensor {
    return ???;
}

const Report = struct {
    naive_nan: zml.Tensor,
    naive_bad: zml.Tensor,
    stable: zml.Tensor,
    stable_bad: zml.Tensor,
    library: zml.Tensor,
    library_bad: zml.Tensor,
    naive_lse_bad: zml.Tensor,
    stable_lse: zml.Tensor,
    library_lse: zml.Tensor,
};

fn inspect(scores: zml.Tensor) Report {
    const naive = naiveSoftmax(scores);
    const stable = stableSoftmax(scores);
    const library = scores.softmax(.score);
    return .{
        .naive_nan = countNan(naive),
        .naive_bad = countBad(naive),
        .stable = stable,
        .stable_bad = countBad(stable),
        .library = library,
        .library_bad = countBad(library),
        .naive_lse_bad = countBad(scores.exp().sum(.score).log()),
        .stable_lse = stableLogSumExp(scores),
        .library_lse = scores.logSumExp(.score).squeeze(.score),
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const scores: zml.Tensor = .init(.{ .score = 4 }, .f32);
    var exe = try platform.compileFn(gpa, io, inspect, .{scores}, .{});
    defer exe.deinit();
    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);

    const minus_inf = -std.math.inf(f32);
    const rows = [_][4]f32{
        .{ 1000, 999, 998, -5 },
        .{ minus_inf, minus_inf, minus_inf, minus_inf },
    };
    const names = [_][]const u8{ "The overwound row", "The silent row" };

    for (rows, names) |row, name| {
        var row_buffer: zml.Buffer = try .fromBytes(io, platform, scores.shape(), std.mem.sliceAsBytes(&row));
        defer row_buffer.deinit();
        var report: zml.Bufferized(Report) = undefined;
        runner.run(io, .{row_buffer}, &report, .{ .wait = true });
        defer zml.Buffer.deinitAll(Report, &report);

        try stdout.print("{s}:\n", .{name});
        try stdout.print("  naive: {d} NaN, {d} not finite\n", .{
            try report.naive_nan.getValue(i32, io), try report.naive_bad.getValue(i32, io),
        });
        try stdout.print("  by hand: {d} not finite\n", .{try report.stable_bad.getValue(i32, io)});
        try stdout.print("  library: {d} not finite\n", .{try report.library_bad.getValue(i32, io)});
        if (try report.stable_bad.getValue(i32, io) == 0) {
            const stable = try report.stable.toSliceAlloc(gpa, io);
            defer stable.free(gpa);
            const library = try report.library.toSliceAlloc(gpa, io);
            defer library.free(gpa);
            try stdout.print("  by hand:", .{});
            for (stable.items(f32)) |v| try stdout.print(" {d:.4}", .{v});
            try stdout.print("\n  same as the library: {}\n", .{std.mem.eql(f32, stable.items(f32), library.items(f32))});
            try stdout.print("  naive logsumexp: {d} not finite\n", .{try report.naive_lse_bad.getValue(i32, io)});
            try stdout.print("  logsumexp by hand: {d:.4}, library: {d:.4}\n", .{
                try report.stable_lse.getValue(f32, io), try report.library_lse.getValue(f32, io),
            });
        }
    }
    try stdout.flush();
}
