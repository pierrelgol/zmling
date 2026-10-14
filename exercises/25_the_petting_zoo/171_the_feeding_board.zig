//
// Third enclosure: the keepers' hut. On the wall, the FEEDING
// BOARD, where the night shift pins its notes. In the morning
// three keepers come in, each with something on her mind, and
// each of them reads the board.
//
// Until now, attention was a herd talking to itself: queries,
// keys and values all came from the SAME sequence. That is
// SELF-attention. Here the ones who ask (the keepers) and the
// ones who answer (the notes) are two different sequences:
//
//     queries        from the keepers   {.s = 3, .d = 4}
//     keys, values   from the board     {.m = 4, .dm = 3}
//
// That is CROSS-ATTENTION, and it is how one model reads the
// output of another: the decoder of Whisper reads the encoded
// audio with it, T5 and every translation model read the source
// sentence, a vision-language model reads the image patches.
// The formula does not change by one symbol. Three things do.
//
// 1. TWO LENGTHS, TWO WIDTHS. The scores are {.q = 3, .k = 4}:
//    not square. And the notes are not vectors of the keepers'
//    kind: 3 numbers, where a keeper has 4. So the projections
//    of the keys and of the values are Linears from `.dm` to
//    the keepers' width, and only the queries' one is d to d.
//    In a checkpoint you see it in the shapes: `encoder_attn.
//    k_proj.weight` is (d, d_encoder).
//
// 2. NO CAUSAL MASK. The board was written last night, all of
//    it: there is no future to hide. The only mask is a
//    padding mask over the KEYS (exercise 170), when the board
//    has empty slots. Today it has one, and Biscuit has
//    drawn on it.
//
// 3. NO KV CACHE THAT GROWS. While the keepers arrive one by
//    one (exercise 108), the board does not move. Its keys and
//    values are computed ONCE and reused at every step.
//
// The pantry's `zml.nn.sdpa` (exercise 105) does cross-attention
// as it is. Its contract, from its doc comment, is about NAMES:
//
//     q, result:  .{ .h, .q, .hd }
//     k, v:       .{ .h, .k, .hd }
//
// `.q` and `.k` are two different axes, which is the whole
// point of naming them: in self-attention you renamed `.s` to
// `.q` on one side and to `.k` on the other (exercise 109). Now
// the two sides really are different, and each has its own
// axis to rename: `.s` for the keepers, `.m` for the board. The
// mask is `.attn_mask`, added to the scores before the softmax:
// {.k} is enough, `sdpa` broadcasts it by name.
//
// sdpa CHECKS the names, and stops with a message when one is
// missing. So does every op that is given a tag: a name that
// is not there is a panic, never a guess. Dewey declared the
// board in `main`, and Dewey tags every sequence `.s`, out of
// habit. Once your `forward` is written, ZML will tell you
// about it: the message names the Tensor and the tag it was
// looking for.
//
// The numbers, so that you can check the result by eye. A note
// is (about food, about water, buckets). A keeper is two heads
// of two numbers, both asking (food?, water?):
//
//     k_proj   both heads: key = (about food, about water)
//     v_proj   head 0: value = (buckets, 0)
//              head 1: value = (about food, about water)
//     q_proj   the identity
//
// So in a keeper's answer, the first number is how many buckets
// she read, and the last two say what they were buckets of.
// Ada thinks about food, Bo about water. Cy thinks about
// nothing at all, and reads every note equally: every REAL
// note. If Cy reads 4.5 buckets, she has read Biscuit's.
//
// To do:
//
//  1. The fields of `CrossAttention`: three Linears and the
//     number of heads, a plain `u32`.
//  2. `forward`. Split the heads as in exercise 109, rename,
//     mask, `sdpa`, merge the heads back.
//  3. In `main`: Dewey's tag, and the weights. The model, the
//     three matrices and the Buffers of the inputs are given;
//     the `zml.Bufferized(CrossAttention)` is the one of
//     exercise 096 (no bias here: `.bias = null`).
//
const std = @import("std");
const zml = @import("zml");

const CrossAttention = struct {
    // `q_proj`, `k_proj`, `v_proj` and `heads`.
    ???,

    /// x: {.s, .d}, the ones who ask.
    /// memory: {.m, .dm}, what they read. memory_mask: {.m}, i32, 0 for an empty slot.
    /// Returns {.s, .d}.
    pub fn forward(self: CrossAttention, x: zml.Tensor, memory: zml.Tensor, memory_mask: zml.Tensor) zml.Tensor {
        return ???;
    }
};

/// {.dout = 4, .d = 4}: a keeper asks what she thinks.
const q_weight = [4][4]f32{
    .{ 1, 0, 0, 0 },
    .{ 0, 1, 0, 0 },
    .{ 0, 0, 1, 0 },
    .{ 0, 0, 0, 1 },
};

/// {.dout = 4, .dm = 3}: for both heads, a note answers to
/// (about food, about water).
const k_weight = [4][3]f32{
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
};

/// {.dout = 4, .dm = 3}: head 0 reads the buckets, head 1 reads
/// what the note is about.
const v_weight = [4][3]f32{
    .{ 0, 0, 1 },
    .{ 0, 0, 0 },
    .{ 1, 0, 0 },
    .{ 0, 1, 0 },
};

const keepers = [_][]const u8{ "Ada", "Bo ", "Cy " };

/// {.s = 3, .d = 4}: (food?, water?) for head 0, and again for head 1.
const thoughts = [3][4]f32{
    .{ 4, 0, 4, 0 }, // Ada: the hay, the apples
    .{ 0, 4, 0, 4 }, // Bo: the trough
    .{ 0, 0, 0, 0 }, // Cy: her holidays
};

/// {.m = 4, .dm = 3}: (about food, about water, buckets)
const board = [4][3]f32{
    .{ 1, 0, 3 }, // "hay: 3 buckets"
    .{ 0, 1, 5 }, // "water: 5 buckets"
    .{ 1, 0, 1 }, // "apples: 1 bucket"
    .{ 9, 9, 9 }, // an empty slot, with a drawing of a goat
};
const board_mask = [4]i32{ 1, 1, 1, 0 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const hut: CrossAttention = .{
        .q_proj = .init(.init(.{ .dout = 4, .d = 4 }, .f32), null, .d),
        .k_proj = .init(.init(.{ .dout = 4, .dm = 3 }, .f32), null, .dm),
        .v_proj = .init(.init(.{ .dout = 4, .dm = 3 }, .f32), null, .dm),
        .heads = 2,
    };

    const x: zml.Tensor = .init(.{ .s = 3, .d = 4 }, .f32);
    const memory: zml.Tensor = .init(.{ .s = 4, .dm = 3 }, .f32);
    const memory_mask: zml.Tensor = .init(.{ .s = 4 }, .i32);

    var exe = try platform.compile(gpa, io, hut, .forward, .{ x, memory, memory_mask }, .{});
    defer exe.deinit();

    // The weights of the hut, in a `zml.Bufferized(CrossAttention)`
    // named `weights`. Free them at the end.
    ???;

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&thoughts));
    defer x_buffer.deinit();
    var memory_buffer: zml.Buffer = try .fromBytes(io, platform, memory.shape(), std.mem.sliceAsBytes(&board));
    defer memory_buffer.deinit();
    var mask_buffer: zml.Buffer = try .fromBytes(io, platform, memory_mask.shape(), std.mem.sliceAsBytes(&board_mask));
    defer mask_buffer.deinit();

    var read: zml.Buffer = try exe.eval(gpa, io, .{ weights, x_buffer, memory_buffer, mask_buffer });
    defer read.deinit();

    try stdout.print("{f} keepers read {f} notes: {f}\n", .{ x.shape(), memory.shape(), read.shape() });
    try stdout.print("     buckets         food  water\n", .{});
    for (keepers, try read.getValue([3][4]f32, io)) |keeper, row| {
        try stdout.print("{s}  {d:6.3} {d:6.3} {d:6.3} {d:6.3}\n", .{ keeper, row[0], row[1], row[2], row[3] });
    }
    try stdout.flush();
}
