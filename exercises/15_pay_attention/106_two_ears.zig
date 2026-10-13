//
// Paca has a complaint. She wants grass news AND drama, and
// with one softmax she can't have both. A softmax hands out
// 100% once: every bit of attention she gives to Kuzco (grass)
// is taken from Tina (drama). She ends up with a lukewarm
// average of the two, and a lukewarm average of gossip is the
// saddest thing on the hillside.
//
// Fortunately, a llama has two ears.
//
// MULTI-HEAD attention runs several small attentions side by
// side, the "heads", each with its own scores and its own
// softmax, so each with a full 100% to hand out. One ear on the
// grass, one ear on the drama.
//
// The surprise is how little it takes. There are no new
// weights and no new ops, only a change of point of view on
// the vectors. A token's vector has `.d` numbers. With 2 heads
// you declare that the first half belongs to head 0 and the
// second half to head 1:
//
//     {.s, .d = 4}    ->    {.s, .h = 2, .hd = 2}
//
// and this is where the name "head dimension" finally makes
// sense: hd = d / heads. You know the op that does this, it is
// the `splitAxis` of chapter 7:
//
//     x.splitAxis(.d, .{ .h = 2, .hd = .auto })
//
// (`.auto`: "you do the division".) Nothing is moved or copied.
// The same numbers get a second axis.
//
// And then... you call `sdpa`, exactly as before. It was
// expecting `.h` all along. In its `dot` calls, `.h` is present
// on both sides and is not the contracted axis, so it is a
// BATCHING axis: head 0 of the queries only ever meets head 0
// of the keys. Two completely independent attentions, from the
// same three lines of code.
//
// `sdpa` returns its result laid out like the queries, here
// `{.q, .h, .hd}`. The rest of the model wants plain vectors
// again, so the heads are glued back with the inverse of the
// split, `merge`:
//
//     heard.merge(.{ .d = .{ .h, .hd } })       -> {.q, .d}
//
// To see the effect, `listen` takes the number of heads as a
// plain argument and gets compiled twice. To keep things bare,
// each llama's vector `x` is used both as her query and as her
// key: its first two numbers are about grass, its last two
// about drama.
//
// The values are rigged so that you can read the result. They
// are name tags: (is it Tina?, is it Kuzco?), written twice,
// once for each ear. So in the output, each half of a row says
// how much of Tina and how much of Kuzco went into that ear.
//
// Watch Paca's row (the third one). With one head, both ears
// get the same thin blur of everybody. With two, one ear is
// nearly all Kuzco and the other nearly all Tina.
//
const std = @import("std");
const zml = @import("zml");

/// Self-attention with `heads` heads. x, v: {.s, .d} -> {.s, .d}
/// x is used for the queries and for the keys, v for the values.
fn listen(x: zml.Tensor, v: zml.Tensor, heads: u32) zml.Tensor {
    // {.s, .d} -> {.s, .h, .hd}
    const x_heads = x.splitAxis(.d, ???);
    const v_heads = v.splitAxis(.d, ???);

    const heard = zml.nn.sdpa(
        x_heads.rename(.{ .s = .q }),
        x_heads.rename(.{ .s = .k }),
        v_heads.rename(.{ .s = .k }),
        .{},
    );

    // {.q, .h, .hd} -> {.s, .d}
    return heard.merge(???).rename(.{ .q = .s });
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

    const x: zml.Tensor = .init(.{ .s = 4, .d = 4 }, .f32);
    const v: zml.Tensor = .init(.{ .s = 4, .d = 4 }, .f32);

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[4][4]f32{
        //  grass,    drama
        .{ 0, 0, 3, 0 }, // Tina: all drama
        .{ 3, 0, 0, 0 }, // Kuzco: all grass
        .{ 2, 0, 2, 0 }, // Paca: both, please
        .{ 0, 1, 0, 1 }, // Dolly: something else entirely
    }));
    defer x_buffer.deinit();

    var v_buffer: zml.Buffer = try .fromBytes(io, platform, v.shape(), std.mem.sliceAsBytes(&[4][4]f32{
        // one ear, the other ear: (Tina?, Kuzco?)
        .{ 1, 0, 1, 0 }, // Tina
        .{ 0, 1, 0, 1 }, // Kuzco
        .{ 0, 0, 0, 0 }, // Paca
        .{ 0, 0, 0, 0 }, // Dolly
    }));
    defer v_buffer.deinit();

    for ([_]u32{ 1, 2 }) |heads| {
        var exe = try platform.compileFn(gpa, io, listen, .{ x, v, heads }, .{});
        defer exe.deinit();

        var heard: zml.Buffer = try exe.eval(gpa, io, .{ x_buffer, v_buffer });
        defer heard.deinit();

        try stdout.print("{d} head(s), heard {f}:\n", .{ heads, heard.shape() });
        try stdout.print("         Tina  Kuzco | Tina  Kuzco\n", .{});
        for (names, try heard.getValue([4][4]f32, io)) |name, row| {
            try stdout.print("  {s}  {d:.3} {d:.3} | {d:.3} {d:.3}\n", .{ name, row[0], row[1], row[2], row[3] });
        }
    }
    try stdout.flush();
}
