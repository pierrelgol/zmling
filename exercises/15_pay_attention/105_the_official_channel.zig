//
// You have now written attention by hand, twice. Like the
// Linear of chapter 13, it's a recipe everybody needs, so the
// pantry stocks it:
//
//     pub fn sdpa(
//         q: Tensor,
//         k: Tensor,
//         v: Tensor,
//         opts: SdpaOpts,
//     ) Tensor
//
//     pub const SdpaOpts = struct {
//         attn_mask: ?Tensor = null,
//         scale: ?Tensor = null,
//         softmax_bias: ?Tensor = null,
//     };
//
// "sdpa" is Scaled Dot Product Attention. Its doc comment is
// short and worth knowing by heart:
//
//     q, result:  .{ .h, .q, .hd }
//     k, v:       .{ .h, .k, .hd }
//
//     .h is the number of heads, .q the number of queries,
//     .k the number of keys, .hd the head dimension.
//
// Those tags are not a suggestion. `sdpa` finds its axes BY
// NAME, and its first three lines are asserts that end with
// things like "q is missing tags {.h, .q, .hd}".
//
// After the asserts, the body is what you wrote: scale (it
// multiplies the keys by 1/sqrt(hd), which gives the same
// scores), `q.dot(k, .hd)`, add `attn_mask` if there is one,
// `softmax(.k)`, `dot(v, .k)`. The default `scale` is the
// 1/sqrt(hd) you want; ignore `softmax_bias`.
//
// Two things are new.
//
// THE HEADS. `.h` is mandatory. A "head" is one independent
// copy of the attention mechanism, and real models run many
// side by side: that's the next exercise. Today the herd has a
// single head, so the tensors simply get an `.h` axis of size 1.
//
// THE RENAMES. So far the listeners arrived tagged `.q` and
// the speakers `.k`, as if they were two different herds. In a
// real model they are ONE sequence playing both parts, which
// is why this is called SELF-attention: queries, keys and
// values are all computed from the same tokens, and they all
// arrive with the same tag, `.s`, for "sequence":
//
//     q, k, v:  {.h, .s, .hd}
//
// Assigning the roles is your job, with `rename` from chapter 4:
//
//     q.rename(.{ .s = .q })     the sequence, as listeners
//     k.rename(.{ .s = .k })     the sequence, as speakers
//     v.rename(.{ .s = .k })     values belong to speakers
//
// And when the result comes back tagged `.q`, it is the
// sequence again: rename it back to `.s`.
//
// The mask goes in `opts.attn_mask`. `sdpa` matches it against
// the scores by tag (with `broad`), so `{.q, .k}` is all it
// needs: the same mask is used for every head.
//
// Below, `byHand` is the previous exercise (complete, and
// trusted). `official` should return the very same numbers by
// calling `zml.nn.sdpa`. At the moment it hands over tensors
// with the wrong tags, and it lets everybody read tomorrow's
// newspaper.
//
const std = @import("std");
const zml = @import("zml");

/// Causal self-attention, the long way.
/// q, k, v: {.h, .s, .hd} -> {.h, .s, .hd}
fn byHand(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
    const hd: f32 = @floatFromInt(q.dim(.hd));
    const listeners = q.rename(.{ .s = .q });
    const speakers = k.rename(.{ .s = .k });

    const scores = listeners.dot(speakers, .hd).scale(1.0 / @sqrt(hd));
    const mask = zml.nn.causalAttnMask(.{ .q = q.dim(.s), .k = k.dim(.s) }, q.dtype(), null);
    const weights = scores.add(mask.broad(scores.shape())).softmax(.k);

    return weights.dot(v.rename(.{ .s = .k }), .k).rename(.{ .q = .s });
}

/// Causal self-attention, the short way.
/// q, k, v: {.h, .s, .hd} -> {.h, .s, .hd}
fn official(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
    return zml.nn.sdpa(q, k, v, .{});
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

    // One head, four llamas, two numbers.
    const q: zml.Tensor = .init(.{ .h = 1, .s = 4, .hd = 2 }, .f32);
    const k: zml.Tensor = .init(.{ .h = 1, .s = 4, .hd = 2 }, .f32);
    const v: zml.Tensor = .init(.{ .h = 1, .s = 4, .hd = 2 }, .f32);

    // Same herd as before.
    var q_buffer: zml.Buffer = try .fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 0, 0 } }));
    defer q_buffer.deinit();
    var k_buffer: zml.Buffer = try .fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 0, 2 }, .{ 2, 0 }, .{ 1, 1 }, .{ 0, 0 } }));
    defer k_buffer.deinit();
    var v_buffer: zml.Buffer = try .fromBytes(io, platform, v.shape(), std.mem.sliceAsBytes(&[4][2]f32{ .{ 10, 0 }, .{ 0, 10 }, .{ 5, 5 }, .{ 1, 2 } }));
    defer v_buffer.deinit();

    inline for (.{
        .{ "by hand ", byHand },
        .{ "official", official },
    }) |row| {
        const title, const recipe = row;

        var exe = try platform.compileFn(gpa, io, recipe, .{ q, k, v }, .{});
        defer exe.deinit();

        var heard: zml.Buffer = try exe.eval(gpa, io, .{ q_buffer, k_buffer, v_buffer });
        defer heard.deinit();

        try stdout.print("{s} {f}:\n", .{ title, heard.shape() });
        for (names, (try heard.getValue([1][4][2]f32, io))[0]) |name, heard_by| {
            try stdout.print("  {s} {d:6.3} {d:6.3}\n", .{ name, heard_by[0], heard_by[1] });
        }
    }
    try stdout.flush();
}
