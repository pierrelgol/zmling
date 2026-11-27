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
// Today you draw the triangle yourself. It is one table of
// shape {.q, .k}, and each cell only needs to know its row and
// its column:
//
//  - The row numbers 0, 1, 2... are an `arange` (exercise 22).
//    Tag it `.q` (`withTags`, exercise 25) and stretch it over
//    the whole table with `broad` (exercise 33): every cell now
//    holds the position of its LISTENER. Same again tagged
//    `.k`: every cell holds the position of its SPEAKER.
//    Integers will do.
//
//        listener        speaker
//        0 0 0 0         0 1 2 3
//        1 1 1 1         0 1 2 3
//        2 2 2 2         0 1 2 3
//        3 3 3 3         0 1 2 3
//
//    `broad` wants the Shape of the table, and no tensor has
//    that shape yet. Make it: a `zml.Shape` is built the way a
//    placeholder is, `.init`, the sizes with their tags, a
//    dtype.
//
//  - A cell is allowed when its speaker does not stand behind
//    its listener. That is a comparison, the `cmp` of chapter 10
//    (exercise 68), and it gives a table of bools.
//
//  - 0 where allowed, -inf elsewhere: the `select` of exercise
//    69. In Zig, minus infinity is `-std.math.inf(f32)`.
//
// Some herds also have a short memory: with a WINDOW of 2, a
// llama hears herself and the ONE llama just before her, and
// that's all. One more condition on the same two tables,
//
//     listener < speaker + window
//
// (a tensor plus a plain number is `addConstant`, exercise 18),
// glued to the first one with `logical(.AND, ...)`, exercise 68
// again. `window` is a plain Zig optional, `null` meaning "no
// limit": an ordinary Zig `if (window) |w|` decides, while
// tracing, whether the second condition is in the graph at all.
//
// The mask contains only constants. It's built while tracing
// and is baked into the Exe: `causalMask` takes no Tensor, so
// its Exe has no input, and `eval` gets an empty tuple, `.{}`.
//
// The order of operations is the whole point: scores, THEN the
// mask, THEN the softmax. The mask must knock speakers out
// before the 100% gets shared.
//
// Of course `zml.nn` has the triangle on the shelf:
//
//     pub fn causalAttnMask(
//         attn_shape_: anytype,
//         dtype: DataType,
//         attn_window_len: ?u32,
//     ) Tensor
//
// `attn_shape_` is the shape of the SCORES: exactly two axes,
// listeners then speakers, like `.{ .q = 4, .k = 4 }`. (Any
// other rank panics.) The triangle is drawn by POSITION, first
// axis against second: the tags are only names, and nothing
// will warn you if you swap them. With a float `dtype` you get
// 0 and -inf, ready to be added. Its body is the one you are
// about to write, give or take a line: go and compare, in
// zml/nn/nn.zig, once yours works.
//
// To do:
//
//  1. Write `causalMask`, by hand, window included.
//
//  2. Write `officialMask` with the helper. The program checks
//     that the two agree, -inf for -inf.
//
//  3. `causalWeights` is the `attentionWeights` of the previous
//     exercise, with the mask. It compiles and runs, and prints
//     `-inf` all over the hillside. Something in it happens too
//     late.
//
//  4. In `main`: three Exes per kind of herd, to compile and to
//     run. Mind what each function takes: Tensors become
//     placeholders, and plain Zig values are just passed along
//     in the compile tuple, like the 7 of `.{ mood, 7 }` in
//     exercise 75. Once baked in, they are NOT in the tuple of
//     `eval`: only Buffers go there.
//
const std = @import("std");
const zml = @import("zml");

/// The size of the herd.
const herd = 4;

/// The causal mask, by hand: {.q = herd, .k = herd}, f32.
/// 0 where the listener may hear the speaker, -inf elsewhere.
/// Nobody hears the llamas behind her, nor those more than `window`
/// places in front (when there is a window).
fn causalMask(window: ?u32) zml.Tensor {
    return ???;
}

/// The same mask, from the shelf.
fn officialMask(window: ?u32) zml.Tensor {
    return ???;
}

/// Attention weights {.q, .k}, with no spoilers.
fn causalWeights(q: zml.Tensor, k: zml.Tensor, window: ?u32) zml.Tensor {
    const hd: f32 = @floatFromInt(q.dim(.hd));
    const scores = q.dot(k, .hd).scale(1.0 / @sqrt(hd));
    return scores.softmax(.k).add(causalMask(window));
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

    const q: zml.Tensor = .init(.{ .q = herd, .hd = 2 }, .f32);
    const k: zml.Tensor = .init(.{ .k = herd, .hd = 2 }, .f32);

    // Same herd as before.
    var q_buffer: zml.Buffer = try .fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 0, 0 } }));
    defer q_buffer.deinit();
    var k_buffer: zml.Buffer = try .fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 0, 2 }, .{ 2, 0 }, .{ 1, 1 }, .{ 0, 0 } }));
    defer k_buffer.deinit();

    // `window` is a plain Zig value: it is baked into each Exe.
    for ([_]?u32{ null, 2 }) |window| {
        // Three Exes, for `causalMask`, `officialMask` and
        // `causalWeights`, and their three results: `mask`,
        // `official` and `weights`, the Buffers printed below.
        ???;

        if (window) |w| {
            try stdout.print("a memory of {d} llamas:\n", .{w});
        } else {
            try stdout.print("no spoilers:\n", .{});
        }

        const by_hand = try mask.getValue([herd][herd]f32, io);
        try stdout.print("  the mask {f}:\n", .{mask.shape()});
        for (by_hand) |row| {
            try stdout.print("   ", .{});
            for (row) |m| try stdout.print(" {d:5}", .{m});
            try stdout.print("\n", .{});
        }
        try stdout.print("  the official mask agrees: {}\n", .{std.meta.eql(by_hand, try official.getValue([herd][herd]f32, io))});

        try stdout.print("  the weights:\n", .{});
        try stdout.print("        Tina  Kuzco Paca  Dolly\n", .{});
        for (names, try weights.getValue([herd][herd]f32, io)) |name, row| {
            try stdout.print("  {s}", .{name});
            for (row) |w| try stdout.print(" {d:.3}", .{w});
            try stdout.print("\n", .{});
        }
    }
    try stdout.flush();
}
