//
// Quiz! A crate was left at the gate during the night. No
// papers. On the lid, in pencil: "soft. purrs." Marge needs to
// know which enclosure the new arrival goes in, and Marge has
// a zoo full of building blocks and somebody who has petted
// every one of them.
//
// The plan: read the note on the crate and the signs of the
// three residents with ONE model, get one vector per sentence,
// and put the newcomer with the resident whose sign points the
// same way. Four sentences in one batch, {.b = 4, .s = 3}: row
// 0 is the crate, rows 1 to 3 are the rabbit, the goose and
// the tortoise. Short sentences are padded on the RIGHT with
// token 0, and `sentence_masks` says where.
//
// The model, top to bottom. Nothing in it is new.
//
//     Zoo
//       words     a `zml.nn.TokenEmbedding` (exercise 093)
//                     tokens {.b, .s} -> {.b, .s, .d}
//       cell      Sheldon's `Cell`, the one of exercise 174,
//                 walked along `.s` from a mood of zeros. What
//                 goes on is the TRAIL, every mood: {.b, .s, .d}
//       clinic    the `Clinic` of exercise 173: 4 vets, top 2,
//                 relu. Its output is ADDED to its input (the
//                 residual of exercise 131: a block refines,
//                 it does not replace)
//       then      masked mean pooling over `.s` (exercise 169)
//                 and `zml.nn.normalizeL2`: {.b, .d}
//
// and `forward` ends with the ranking: the cosine of row 0 with
// each of the three others, and the index of the best. `main`
// reads them from a struct with two fields:
//
//     scores   {.b = 3}, f32    one cosine per resident
//     best     rank 0, i32      0 rabbit, 1 goose, 2 tortoise
//
// Notes from Marge, who has watched Dewey do this:
//
//  - The cell now walks FOUR sentences at once. Nothing to
//    change in `Cell.step`: a Linear does not care about one
//    more axis in front. The mood is {.b, .d}; and a mood that
//    has no `.s` axis is filed into a trail {.b, .s, .d} with
//    `dynamicUpdateSlice(.{ .s = step }, h)` all the same.
//
//  - Loop in the graph, or loop in Zig? Here every row has the
//    same 3 positions and the number is known when the graph
//    is built: both work, both give the same numbers. Yours to
//    choose. (With a loop in Zig, the index of
//    `dynamicUpdateSlice` is still a Tensor: a scalar one.)
//
//  - The padding. In the barn (exercise 170) it had to be
//    masked inside the attention. Sheldon needs nothing: he
//    walks left to right, the padding is on the right, so by
//    the time he steps on it every real mood is already on the
//    trail. The vets see one token at a time. The ONLY place
//    where the mask is needed is the pooling. (Padding on the
//    left would be another story: think about which.)
//
//  - The clinic by hand or `zml.moe.forwardMoe` with
//    `.stablehlo`: same numbers. The pantry wants its axes
//    named `.b`, `.s`, `.d`: they are.
//
//  - `zml.mem.bufferize` and `zml.Buffer.deinitAll` do not
//    walk through arrays (exercise 102), and nobody asked for
//    arrays: every weight below has its own name. The Linears
//    without a bias get `.bias = null`, and all of them
//    `.quantization = null`, in the Buffers.
//
//  - Build it leaves first, and you have most of the leaves
//    already: bring `Cell` and `Clinic` from the last two
//    exercises, and the pooling from the first.
//
// The weights are the ones of exercises 173 and 174, so that
// you know them. A word, a mood and a prescription are all
// (fluffy, loud) today; nobody said a zoo had to make sense.
//
// The data and the printing are given. The rest of the page is
// blank, and it is yours.
//
const std = @import("std");
const zml = @import("zml");

// TODO: `Cell`, with something that walks a whole batch.

// TODO: `Clinic`.

// TODO: `Zoo`, its `forward`, and the struct `forward` returns.

/// {.voc = 7, .d = 2}: (fluffy, loud)
const words_weight = [7][2]f32{
    .{ 9, -9 }, // 0: the padding. Biscuit has been at it.
    .{ 1, 0 }, // 1: soft
    .{ 0.5, -1 }, // 2: quiet
    .{ 0, 2 }, // 3: honks
    .{ -2, -1 }, // 4: hard
    .{ -0.5, 1 }, // 5: hisses
    .{ 1, 0.5 }, // 6: purrs
};

// The cell of exercise 174.
const gate_x_weight = [2][2]f32{ .{ 2, 0 }, .{ 0, 2 } };
const gate_x_bias = [2]f32{ -1, -1 };
const gate_h_weight = [2][2]f32{ .{ 0, 0.5 }, .{ 0.5, 0 } };
const cand_x_weight = [2][2]f32{ .{ 1, 0 }, .{ 0, 1 } };
const cand_h_weight = [2][2]f32{ .{ 0.5, 0 }, .{ 0, 0.5 } };

// The clinic of exercise 173.
/// {.expert = 4, .d = 2}
const router_weight = [4][2]f32{ .{ 2, 0 }, .{ 0, 2 }, .{ 1, 1 }, .{ -1, -1 } };
/// {.expert = 4, .hid = 2, .d = 2}
const up_weight = [4][2][2]f32{
    .{ .{ 1, 0 }, .{ 0, 1 } },
    .{ .{ 1, 0 }, .{ 0, 1 } },
    .{ .{ 1, -1 }, .{ -1, 1 } },
    .{ .{ -1, 0 }, .{ 0, -1 } },
};
/// {.expert = 4, .dout = 2, .hid = 2}
const down_weight = [4][2][2]f32{
    .{ .{ 2, 0 }, .{ 0, 0 } },
    .{ .{ 0, 0 }, .{ 0, 2 } },
    .{ .{ 1, 0 }, .{ 0, 1 } },
    .{ .{ 0, 0 }, .{ 0, 0 } },
};

const resident_names = [_][]const u8{ "rabbit", "goose", "tortoise" };

/// {.b = 4, .s = 3}
const sentence_tokens = [4][3]i32{
    .{ 1, 6, 0 }, // the crate: "soft. purrs."
    .{ 1, 2, 1 }, // the rabbit: "soft, quiet, soft"
    .{ 3, 5, 0 }, // the goose: "honks, hisses"
    .{ 4, 4, 0 }, // the tortoise: "hard. hard."
};
const sentence_masks = [4][3]i32{
    .{ 1, 1, 0 },
    .{ 1, 1, 1 },
    .{ 1, 1, 0 },
    .{ 1, 1, 0 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // TODO: the zoo: every weight, as a placeholder with its tags.
    // TODO: the placeholders of the two inputs, and the Exe.
    // TODO: the weights, as Buffers. Free them all at the end.
    // TODO: `sentence_tokens` and `sentence_masks`, as Buffers.
    // TODO: the run. The two results go in `verdict`.

    try stdout.print("{f} sentences, {f} cosines\n", .{ tokens.shape(), verdict.scores.shape() });
    for (resident_names, try verdict.scores.getValue([3]f32, io)) |resident, score| {
        try stdout.print("  like the {s:<8} {d:6.3}\n", .{ resident, score });
    }
    const best = try verdict.best.getValue(i32, io);
    try stdout.print("the new arrival moves in with the {s}\n", .{resident_names[@intCast(best)]});
    try stdout.flush();
}
