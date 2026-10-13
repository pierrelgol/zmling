//
// There is a problem with yesterday's herd. Look at the first
// row of weights: Tina, who stands FIRST in line, spent half
// her attention on Kuzco, who stands behind her.
//
// On a hillside, fine. In a language model, that's cheating.
// A language model writes one token at a time, left to right,
// and each token is predicted from the ones before it. While it
// is being written, the rest of the sentence does not exist
// yet. A token that listens to a later token is reading
// tomorrow's newspaper.
//
// So, the law of the herd: gossip only travels DOWN the line.
// You hear the llamas in front of you, and yourself. Nobody
// behind you.
//
//     listener \ speaker   Tina  Kuzco  Paca  Dolly
//     Tina                  ok     -     -      -
//     Kuzco                 ok    ok     -      -
//     Paca                  ok    ok    ok      -
//     Dolly                 ok    ok    ok     ok
//
// That triangle is the CAUSAL MASK (causes come before
// effects). And here is the neat trick to enforce it without a
// single `if`: softmax turns a score s into exp(s) / sum, and
// exp(-inf) is exactly 0. So ADD -inf to every forbidden score,
// add 0 to the others, and the softmax does the rest: forbidden
// speakers get a weight of precisely zero, and the allowed ones
// still share 100%.
//
// `zml.nn` builds the triangle for you:
//
//     pub fn causalAttnMask(
//         attn_shape_: anytype,
//         dtype: DataType,
//         attn_window_len: ?u32,
//     ) Tensor
//
// From its source:
//
//  - `attn_shape_` is the shape of the SCORES: exactly two
//    axes, listeners then speakers, like `.{ .q = 4, .k = 4 }`.
//    One cell per pair. (Any other rank panics.) The triangle
//    is drawn by POSITION, first axis against second: the tags
//    are only names, and nothing will warn you if you swap them.
//
//  - With a float `dtype`, a cell holds 0 where k <= q and -inf
//    elsewhere: ready to be added. (With an integer dtype you
//    get 1 and 0 instead, which is handy for looking at it.)
//
//  - `attn_window_len` is for herds with a short memory: with
//    a window of 2, a llama hears herself and the ONE llama
//    just before her, and that's all. `null` means no limit.
//
// It contains only constants. It's built while tracing and is
// baked into the Exe, like `Tensor.arange`.
//
// The order of operations is the whole point: scores, THEN the
// mask, THEN the softmax. The mask must knock speakers out
// before the 100% gets shared.
//
// The function below gets two things wrong. The first one, ZML
// will complain about. The second one prints `-inf` all over
// the hillside.
//
const std = @import("std");
const zml = @import("zml");

/// Attention weights {.q, .k}, where nobody hears the llamas behind her,
/// nor those more than `window` places in front (when there is a window).
fn causalWeights(q: zml.Tensor, k: zml.Tensor, window: ?u32) zml.Tensor {
    const hd: f32 = @floatFromInt(q.dim(.hd));
    const scores = q.dot(k, .hd).scale(1.0 / @sqrt(hd));

    // One cell per (listener, speaker) pair.
    const mask = zml.nn.causalAttnMask(.{ .q = q.dim(.q), .hd = q.dim(.hd) }, scores.dtype(), window);

    return scores.softmax(.k).add(mask);
}

const names = [_][]const u8{ "Tina ", "Kuzco", "Paca ", "Dolly" };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const q: zml.Tensor = .init(.{ .q = 4, .hd = 2 }, .f32);
    const k: zml.Tensor = .init(.{ .k = 4, .hd = 2 }, .f32);

    // Same herd as before.
    var q_buffer: zml.Buffer = try .fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 0, 0 } }));
    defer q_buffer.deinit();
    var k_buffer: zml.Buffer = try .fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 0, 2 }, .{ 2, 0 }, .{ 1, 1 }, .{ 0, 0 } }));
    defer k_buffer.deinit();

    // `window` is a plain Zig value: it is baked into each Exe.
    for ([_]?u32{ null, 2 }) |window| {
        var exe = try platform.compileFn(gpa, io, causalWeights, .{ q, k, window }, .{});
        defer exe.deinit();

        var weights: zml.Buffer = try exe.eval(gpa, io, .{ q_buffer, k_buffer });
        defer weights.deinit();

        if (window) |w| {
            try stdout.print("a memory of {d} llamas:\n", .{w});
        } else {
            try stdout.print("no spoilers:\n", .{});
        }
        try stdout.print("        Tina  Kuzco Paca  Dolly\n", .{});
        for (names, try weights.getValue([4][4]f32, io)) |name, row| {
            try stdout.print("  {s}", .{name});
            for (row) |w| try stdout.print(" {d:.3}", .{w});
            try stdout.print("\n", .{});
        }
    }
    try stdout.flush();
}
