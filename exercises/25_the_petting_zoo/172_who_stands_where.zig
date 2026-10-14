//
// Fourth enclosure: the reading corner, where Marge reads the
// incident reports. This morning's: "goat bites boy". Last
// week's: "boy bites goat". (The boy was provoked.)
//
// The Reader is a tiny model: look the words up, one round of
// self-attention, then mean pooling. And it files both reports
// under the SAME vector. You saw why in exercise 107: a score
// is a dot product of one query and one key, nothing in it
// knows who stands where. Shuffle the words and every word
// hears exactly the same blend as before; pool, and the order
// is gone. Attention reads a BAG of words.
//
// So every transformer INJECTS the positions somewhere, and a
// port must find out where and how. There are four families,
// and you have already met two:
//
//   - LEARNED ABSOLUTE (BERT, GPT-2, Neo: exercises 123, 129).
//     A second table, {.pos, .d}, one trained row per
//     position, added to the word vectors. Simple, and the
//     model cannot read past the last row of the table.
//
//   - ROTARY (the llamas: exercises 107, 143). Queries and
//     keys are rotated by an angle that grows with the
//     position, inside every attention. No weights.
//
// Here are the other two.
//
//   - SINUSOIDAL (the original Transformer, Whisper's encoder,
//     the time step of every diffusion model). Also a table
//     added to the word vectors, but COMPUTED, not trained.
//     Position p, features 2i and 2i+1:
//
//         angle(p, i) = p / 10000^(2i / d)
//         pe[p, 2i]   = sin(angle(p, i))
//         pe[p, 2i+1] = cos(angle(p, i))
//
//     Each pair of features is the hand of a clock, and each
//     clock is slower than the one before: the first turns one
//     radian per position, the last takes tens of thousands of
//     positions to go round. No weights in the file: if you
//     look for `position_embeddings` in the checkpoint and it
//     is not there, this is one of the reasons.
//
//     In ZML it is a handful of ops you know. `arange` for the
//     positions and for `i` (as f32), `outer` for all the
//     angles at once (exercise 064), `sin` and `cos`, and to
//     INTERLEAVE them, sin cos sin cos: `stack` them on a new
//     last axis, {.s, .i, .pair = 2}, then `merge` `.i` and
//     `.pair` into `.d` (exercises 057 and 050). The divisor is
//     easier as a factor: 1 / 10000^(2i/d) = exp(-ln(10000) *
//     2i / d). `d` is a Zig number while the graph is built:
//     `@log(10000.0)` runs at trace time, the `exp` in the graph.
//
//   - RELATIVE BIAS (T5, Swin, DeBERTa's cousins). Nothing is
//     added to the words. Instead, a small TRAINED table gives
//     one number per DISTANCE between the listener and the
//     speaker, and that number is added to the score, before
//     the softmax: "I listen more to my left neighbour".
//
//         distance = position of the key - position of the query
//         bucket   = clamp(distance, -max, max) + max
//         bias[q, k] = table[bucket]
//
//     Clamping means "max or further, same thing", and the
//     `+ max` makes an index out of a signed distance: bucket 0
//     is "far behind me", the middle one is "myself", the last
//     is "far ahead". (T5 has a fancier bucket function, with
//     a logarithm for the far ones, and one table column per
//     head. The principle is this one.) Two grids of positions
//     as you drew them in exercise 104, `sub`, `clamp` with two
//     scalar Tensors, and a `gather` on `.bucket` (exercise
//     070): the indices are {.q, .k}, so is the result.
//
//     The corner's table has 3 buckets: +2 for a speaker
//     BEHIND the listener (earlier in the sentence), 0 for
//     herself, -2 for a speaker ahead.
//
// Which family the Reader uses is a field of the struct that
// is NOT a Tensor: a Zig enum, read while the graph is being
// built (exercise 076). One struct, three graphs. Real ports
// do the same with a string of `config.json`; ZML's LFM2
// example picks a whole block type per layer that way.
//
// To do:
//
//  1. `sinusoidal`.
//  2. `relativeBias` runs. Its picture, printed by `main`, has
//     the +2 on the wrong side of the diagonal.
//  3. `Reader.forward` ignores `self.positions`. Add the table
//     to the words when it says `.sinusoidal`, the bias to the
//     scores when it says `.relative`.
//  4. In `main`, the loop: for each family, a Reader and its
//     Exe. The weights are the same Buffers for the three.
//
const std = @import("std");
const zml = @import("zml");

const Positions = enum { none, sinusoidal, relative };

/// The sinusoidal table {.s = len, .d = d}, f32. `d` is even.
fn sinusoidal(len: i64, d: i64) zml.Tensor {
    return ???;
}

/// table: {.bucket = 2 * max + 1}, one number per clamped distance.
/// Returns the bias {.q = len, .k = len}.
fn relativeBias(table: zml.Tensor, len: i64) zml.Tensor {
    const max = @divExact(table.dim(.bucket) - 1, 2);
    const grid: zml.Shape = .init(.{ .q = len, .k = len }, .i32);
    const q = zml.Tensor.arange(.{ .end = len }, .i32).withTags(.{.q}).broad(grid);
    const k = zml.Tensor.arange(.{ .end = len }, .i32).withTags(.{.k}).broad(grid);

    const distance = q.sub(k).clamp(.scalar(-max, .i32), .scalar(max, .i32));
    return table.gather(.{ .bucket = distance.addConstant(max) }, .{});
}

const Reader = struct {
    words: zml.nn.TokenEmbedding,
    /// {.bucket}: only used by the `.relative` family.
    bias: zml.Tensor,
    positions: Positions,

    /// tokens: {.s} word ids -> {.d}, one vector for the sentence.
    pub fn forward(self: Reader, tokens: zml.Tensor) zml.Tensor {
        const x = self.words.forward(tokens);

        const d: f32 = @floatFromInt(x.dim(.d));
        const q = x.rename(.{ .s = .q });
        const k = x.rename(.{ .s = .k });
        const scores = q.dot(k, .d).scale(1.0 / @sqrt(d));

        const heard = scores.softmax(.k).dot(k, .k);
        return heard.mean(.q).squeeze(.q);
    }
};

/// {.voc = 3, .d = 4}
const words_weight = [3][4]f32{
    .{ 2, 0, 1, 0 }, // 0: goat
    .{ 0, 2, 0, 1 }, // 1: bites
    .{ 1, 1, 0, 0 }, // 2: boy
};

/// {.bucket = 3}: a speaker behind me, myself, a speaker ahead.
const bias_weight = [3]f32{ 2, 0, -2 };

const reports = [2][3]i32{
    .{ 0, 1, 2 }, // goat bites boy
    .{ 2, 1, 0 }, // boy bites goat
};

const families = [_]Positions{ .none, .sinusoidal, .relative };
const family_names = [_][]const u8{ "no positions", "sinusoidal", "relative bias" };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The two tables, for the eye.
    {
        var exe = try platform.compileFn(gpa, io, sinusoidal, .{ 3, 4 }, .{});
        defer exe.deinit();
        var table: zml.Buffer = try exe.eval(gpa, io, .{});
        defer table.deinit();

        try stdout.print("the clocks {f}:\n", .{table.shape()});
        for (try table.getValue([3][4]f32, io), 0..) |row, pos| {
            try stdout.print("  position {d}: {d:6.3} {d:6.3} {d:6.3} {d:6.3}\n", .{ pos, row[0], row[1], row[2], row[3] });
        }
    }

    const bias: zml.Tensor = .init(.{ .bucket = 3 }, .f32);
    var bias_buffer: zml.Buffer = try .fromBytes(io, platform, bias.shape(), std.mem.sliceAsBytes(&bias_weight));
    defer bias_buffer.deinit();
    {
        var exe = try platform.compileFn(gpa, io, relativeBias, .{ bias, 3 }, .{});
        defer exe.deinit();
        var picture: zml.Buffer = try exe.eval(gpa, io, .{bias_buffer});
        defer picture.deinit();

        try stdout.print("the bias {f}, one row per listener:\n", .{picture.shape()});
        for (try picture.getValue([3][3]f32, io)) |row| {
            try stdout.print("  {d:2} {d:2} {d:2}\n", .{ row[0], row[1], row[2] });
        }
    }

    const words: zml.Tensor = .init(.{ .voc = 3, .d = 4 }, .f32);
    var words_buffer: zml.Buffer = try .fromBytes(io, platform, words.shape(), std.mem.sliceAsBytes(&words_weight));
    defer words_buffer.deinit();

    const tokens: zml.Tensor = .init(.{ .s = 3 }, .i32);

    try stdout.print("\"goat bites boy\" and \"boy bites goat\":\n", .{});
    for (families, family_names) |family, name| {
        // A Reader of this family, and its Exe: `exe`.
        ???;

        // The Buffers belong to `main`: nothing to free here.
        const weights: zml.Bufferized(Reader) = .{ .words = .{ .weight = words_buffer }, .bias = bias_buffer };

        var filed: [2][4]f32 = undefined;
        for (reports, &filed) |report, *vector| {
            var tokens_buffer: zml.Buffer = try .fromBytes(io, platform, tokens.shape(), std.mem.sliceAsBytes(&report));
            defer tokens_buffer.deinit();

            var pooled: zml.Buffer = try exe.eval(gpa, io, .{ weights, tokens_buffer });
            defer pooled.deinit();
            vector.* = try pooled.getValue([4]f32, io);
        }

        var same = true;
        for (filed[0], filed[1]) |a, b| {
            if (@abs(a - b) > 1e-4) same = false;
        }
        try stdout.print("  {s:<13}", .{name});
        for (filed) |vector| {
            try stdout.print(" ({d:.3} {d:.3} {d:.3} {d:.3})", .{ vector[0], vector[1], vector[2], vector[3] });
        }
        try stdout.print("  the same file: {}\n", .{same});
    }
    try stdout.flush();
}
