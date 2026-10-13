//
// So far the whole herd was on the hillside from the start.
// That's not how a language model writes. It writes one token,
// then runs again to get the next one, then again. The llamas
// ARRIVE ONE AT A TIME, and each newcomer wants to hear
// everything that was said before she got there.
//
// The naive way: at every arrival, run the full attention again
// over everybody present. For the 1000th llama, that means
// recomputing the keys and values of the 999 others. Which,
// thanks to the causal mask, are EXACTLY what they were last
// time: an earlier llama never hears a later one, so nothing
// about her can change when someone new shows up.
//
// The herd's way: write things down. Keep an archive of every
// key and every value produced so far. When a llama arrives:
//
//  1. compute HER key and HER value, and file them;
//  2. match HER query against the whole archive;
//  3. that's all. One listener per step instead of the whole
//     herd. Nothing old gets recomputed, ever.
//
// This is the KV CACHE, and it is the reason your favorite LLM
// answers in seconds rather than in geological time.
//
// One difficulty, and it is very ZML: the archive GROWS, and an
// Exe is compiled for fixed shapes. A tensor cannot be 3 rows
// long on Tuesday and 4 on Wednesday. So the archive is built
// at its maximum size from the start, mostly empty,
//
//     k, v:  {.k = max_len, .hd}
//
// and each arrival overwrites one row, at a position which is
// only known at RUN time. You have swapped a pizza on a tray
// exactly that way, in exercise 59:
//
//     archive.dynamicUpdateSlice(.{ .k = pos }, row)
//
// "Return a copy of `archive` where `row` has been written at
// offset `pos` along `.k`." What matters here:
//
//  - `pos` is a rank 0 integer TENSOR: a real input of the Exe,
//    different on every run;
//  - a single row `{.hd}` has no `.k` axis, and that's fine: it
//    is treated as a slice of size 1 at the given offset;
//  - like every Tensor op, it RETURNS the result. Nothing is
//    modified in place. The new archive is an output of the
//    Exe, and must be handed back as an input on the next run.
//
// And one trap. The archive is mostly empty rows, full of
// zeros. A key of (0, 0) has a score of 0 with anybody, and
// exp(0) is 1: without precautions, every llama spends a good
// part of her attention listening to llamas WHO HAVE NOT
// ARRIVED YET. They have nothing to say and they say it loudly.
// You know the cure: a mask. It is ONE ROW of the triangle you
// drew in exercise 104, the row of the newcomer: 0 for the
// rows of the archive up to `pos` (herself included), -inf for
// the others. The same three ops, except that this time the
// listener's position is not an `arange`, it is `pos`.
//
// With the mask, the rest is exercise 103 for a single
// listener: scores, mask, softmax, blend. The newcomer's query
// is a lone `{.hd}` vector: against `{.k, .hd}` keys, `dot`
// gives her one score per row of the archive.
//
// The step function returns two things, what the newcomer
// heard and the new archive, so it runs with the `Runner` of
// chapter 12: `runner.run(io, inputs, outputs, .{ .wait = true })`.
// The Archive is a struct of Tensors: on the Buffer side it is
// a `zml.Bufferized(Archive)`, going in and coming out.
//
// (ZML's Llama example has the same struct, named `KvCache`,
// with one more axis for the layers. It writes with a more
// general op, `scatterSlices`. Same idea.)
//
// To do, top to bottom:
//
//  1. `Archive`: its two fields, and `updateAt`.
//  2. `arrived`: the mask.
//  3. `step`: file the newcomer FIRST (she hears herself), then
//     let her listen to the updated archive.
//  4. `main`: compile `step` and get a runner. Then, in the
//     loop, run one step, and make the archive that came out
//     the archive that goes in next. The old one is yours to
//     free: nothing was donated.
//
// If you get it right, the llamas hear, one by one, exactly
// what the causal attention of exercise 105 computed for the
// whole herd at once.
//
const std = @import("std");
const zml = @import("zml");

const max_len = 4;

const Archive = struct {
    // Every key so far, and every value so far: {.k = max_len, .hd}.
    ???,

    /// The archive, with one key and one value {.hd} filed at row `pos`.
    pub fn updateAt(self: Archive, new_k: zml.Tensor, new_v: zml.Tensor, pos: zml.Tensor) Archive {
        return ???;
    }
};

/// A mask over the rows of the archive, {.k}: 0 for the rows 0 to `pos`
/// (included: a llama hears herself), -inf for the rows that are still empty.
fn arrived(pos: zml.Tensor) zml.Tensor {
    return ???;
}

/// One llama arrives at position `pos`, with her q, k and v: {.hd}.
/// Returns what she hears, {.hd}, and the updated archive.
fn step(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, archive: Archive, pos: zml.Tensor) struct { zml.Tensor, Archive } {
    return ???;
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

    const q: zml.Tensor = .init(.{ .hd = 2 }, .f32);
    const k: zml.Tensor = .init(.{ .hd = 2 }, .f32);
    const v: zml.Tensor = .init(.{ .hd = 2 }, .f32);
    const archive: Archive = .{
        .k = .init(.{ .k = max_len, .hd = 2 }, .f32),
        .v = .init(.{ .k = max_len, .hd = 2 }, .f32),
    };
    const pos: zml.Tensor = .init(.{}, .i32);

    // Compile `step`. Two outputs: it will need a `runner`.
    ???;

    // The archive starts empty: all zeros.
    const empty: [max_len][2]f32 = @splat(.{ 0, 0 });
    var archive_k: zml.Buffer = try .fromBytes(io, platform, archive.k.shape(), std.mem.sliceAsBytes(&empty));
    defer archive_k.deinit();
    var archive_v: zml.Buffer = try .fromBytes(io, platform, archive.v.shape(), std.mem.sliceAsBytes(&empty));
    defer archive_v.deinit();

    // Same herd as before, but now they show up one by one.
    const queries = [4][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 0, 0 } };
    const keys = [4][2]f32{ .{ 0, 2 }, .{ 2, 0 }, .{ 1, 1 }, .{ 0, 0 } };
    const values = [4][2]f32{ .{ 10, 0 }, .{ 0, 10 }, .{ 5, 5 }, .{ 1, 2 } };

    for (names, queries, keys, values, 0..) |name, query, key, value, i| {
        var q_buffer: zml.Buffer = try .fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(&query));
        defer q_buffer.deinit();
        var k_buffer: zml.Buffer = try .fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(&key));
        defer k_buffer.deinit();
        var v_buffer: zml.Buffer = try .fromBytes(io, platform, v.shape(), std.mem.sliceAsBytes(&value));
        defer v_buffer.deinit();
        var pos_buffer: zml.Buffer = try .scalar(io, platform, @as(i32, @intCast(i)), .i32);
        defer pos_buffer.deinit();

        // Run one step: the newcomer and the archive
        // (`archive_k`, `archive_v`) go in, `heard` and a new
        // archive come out. Then the new archive replaces the old
        // one, for the next llama.
        ???;

        const h = try heard.getValue([2]f32, io);
        try stdout.print("{s} arrives at row {d} and hears {d:6.3} {d:6.3}\n", .{ name, i, h[0], h[1] });
    }

    try stdout.print("the archive of keys {f}:", .{archive_k.shape()});
    for (try archive_k.getValue([max_len][2]f32, io)) |row| try stdout.print(" ({d},{d})", .{ row[0], row[1] });
    try stdout.print("\n", .{});
    try stdout.flush();
}
