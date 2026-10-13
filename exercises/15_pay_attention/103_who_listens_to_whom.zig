//
// Chapter 15: Pay Attention
//
// High in the Andes, on a hillside with an unreasonable view,
// lives a herd of llamas. Llamas spit, hum, and carry things,
// but mostly, llamas GOSSIP. All day. About everything.
//
// Four of them stand in a line: Tina, Kuzco, Paca and Dolly.
// (Dolly insists she is a llama. Nobody has the heart.)
//
// This chapter is about ATTENTION, the mechanism at the center
// of every transformer, and therefore of every LLM. It answers
// one question: when each element of a sequence gets to listen
// to all the others, WHO LISTENS TO WHOM, and how much? Which
// is the only question a herd of llamas has ever cared about.
//
// Each llama carries three vectors:
//
//     a QUERY   what she wants to hear about
//     a KEY     what she knows about
//     a VALUE   what she'd actually say if you listened
//
// Here they have two numbers each: (grass, drama). Tina's query
// is (1, 0): she wants grass news, and no drama, thank you.
//
// The whole mechanism is three steps, and there is not a single
// new op in them. Which is why, today, nobody writes them for
// you.
//
// STEP 1: SCORES. How well does each listener's query match
// each speaker's key? "How well do two vectors match" is a dot
// product, the `dot` of chapter 9 (exercise 61). Tag the
// listeners `.q`, the speakers `.k`, and the two numbers `.hd`
// (it stands for "head dimension", you'll see why in a few
// exercises):
//
//     q: {.q, .hd}    k: {.k, .hd}    ->    scores: {.q, .k}
//
// One axis is contracted, and since `.q` and `.k` are different
// tags, we get the full table: one score for every (listener,
// speaker) pair. That's why the SAME herd gets two different
// tags. Had both been tagged `.llama`, `dot` would have seen a
// batching axis (exercise 63) and compared each llama with
// herself only.
//
// By hand, for Tina, whose query is (1, 0): against the four
// keys (0,2), (2,0), (1,1) and (0,0) she scores 0, 2, 1 and 0.
// Kuzco's grass is what she is after, and it shows.
//
// These scores get divided by sqrt(hd). The longer the vectors,
// the larger their dot products tend to be, and large scores
// make the next step far too decisive. Hence "scaled dot
// product attention", the name of this whole recipe. `hd` is
// not data, it is a size: plain Zig arithmetic on `q.dim(.hd)`
// (an i64, so `@floatFromInt` it before the `@sqrt`), then a
// `scale` (exercise 18) by the f32 that comes out.
//
// STEP 2: WEIGHTS. A listener has only so much attention: 100%.
// Turn each ROW of scores into shares that sum to 1. Turning
// scores into shares is the `softmax` of chapter 6 (exercise
// 44). It takes one argument, an axis: the one you WALK ALONG
// while the shares add up to 1. On a small table `{.row, .col}`,
// `softmax(.col)` walks from column to column, and so it is
// each ROW that ends up summing to 1:
//
//     scores          softmax(.col)
//      2  0  1        0.665  0.090  0.245     = 1
//      0  0  0        0.333  0.333  0.333     = 1
//
// So, which axis here? Each LISTENER has 100% to hand out. To
// meet everyone she could give it to, you walk along...
//
//     scores: {.q, .k}    ->    weights: {.q, .k}
//
// STEP 3: LISTEN. Each llama hears a blend of all the values,
// mixed according to her weights. A weighted sum over the
// speakers... is a dot product again:
//
//     weights: {.q, .k}    v: {.k, .hd}    ->    heard: {.q, .hd}
//
// That's it. That is attention: dot, softmax, dot. The rest of
// this chapter is refinements.
//
// To do:
//
//  1. Write `attentionWeights` (steps 1 and 2) and `gossip`
//     (step 3, on top of `attentionWeights`).
//
//  2. `main` has the herd's Buffers and the printing. What
//     goes in between is the routine of chapter 2, and from
//     this chapter on it is yours: compile both functions, run
//     both.
//
//  3. Once all of that is written, ZML has one more complaint.
//     Somebody declared the placeholders in a hurry, and one of
//     the three stands in the wrong herd. The panic names the
//     tag it was looking for.
//
// Predict before you run: Tina wants grass. Kuzco is the one
// whose key says grass. Who will Tina mostly listen to? And
// Dolly, whose query is (0, 0), wants nothing in particular:
// how will she split her attention?
//
const std = @import("std");
const zml = @import("zml");

/// How much attention each listener gives to each speaker: {.q, .k}.
/// Every listener has exactly 100% to give.
fn attentionWeights(q: zml.Tensor, k: zml.Tensor) zml.Tensor {
    return ???;
}

/// What each listener ends up hearing: {.q, .hd}.
fn gossip(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
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

    const q: zml.Tensor = .init(.{ .q = 4, .hd = 2 }, .f32);
    const k: zml.Tensor = .init(.{ .k = 4, .hd = 2 }, .f32);
    const v: zml.Tensor = .init(.{ .q = 4, .hd = 2 }, .f32);

    // Two Exes: one for `attentionWeights`, one for `gossip`.
    ???;

    //                                        (grass, drama)
    var q_buffer: zml.Buffer = try .fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(&[4][2]f32{
        .{ 1, 0 }, // Tina wants grass news
        .{ 0, 1 }, // Kuzco wants drama
        .{ 1, 1 }, // Paca wants everything
        .{ 0, 0 }, // Dolly is just happy to be here
    }));
    defer q_buffer.deinit();

    var k_buffer: zml.Buffer = try .fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(&[4][2]f32{
        .{ 0, 2 }, // Tina knows the drama
        .{ 2, 0 }, // Kuzco knows where the grass is
        .{ 1, 1 }, // Paca knows a bit of both
        .{ 0, 0 }, // Dolly knows nothing, cheerfully
    }));
    defer k_buffer.deinit();

    var v_buffer: zml.Buffer = try .fromBytes(io, platform, v.shape(), std.mem.sliceAsBytes(&[4][2]f32{
        .{ 10, 0 }, // what Tina says (in llama, it loses in translation)
        .{ 0, 10 }, // what Kuzco says
        .{ 5, 5 }, // what Paca says
        .{ 1, 2 }, // what Dolly says
    }));
    defer v_buffer.deinit();

    // Run them: `weights` and `heard` are the two Buffers printed below.
    ???;

    try stdout.print("weights {f}, each row is a listener:\n", .{weights.shape()});
    try stdout.print("        Tina  Kuzco Paca  Dolly\n", .{});
    for (names, try weights.getValue([4][4]f32, io)) |name, row| {
        try stdout.print("  {s}", .{name});
        for (row) |w| try stdout.print(" {d:.3}", .{w});
        try stdout.print("\n", .{});
    }

    try stdout.print("heard {f}:\n", .{heard.shape()});
    for (names, try heard.getValue([4][2]f32, io)) |name, row| {
        try stdout.print("  {s} {d:6.3} {d:6.3}\n", .{ name, row[0], row[1] });
    }
    try stdout.flush();
}
