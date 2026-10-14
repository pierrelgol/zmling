//
// Chapter 25: The Petting Zoo
//
// Welcome to Marge's Petting Zoo. Small animals, small fences,
// and in every enclosure ONE building block that the models of
// chapters 17 to 21 did not need, and that the next model you
// port probably will. You may pet them all. Mind Biscuit, the
// goat: Biscuit eats padding.
//
// First enclosure, at the gate. Visitors ask a question ("which
// one is loud?") and the gate must point at an animal. Every
// animal has a SIGN, a short sentence. An encoder (think of
// Bertie, chapter 18) has read the question and the signs, and
// gives back what encoders give: ONE VECTOR PER TOKEN.
//
//     hidden:  {.s, .d}          for the question
//              {.doc, .s, .d}    for the three signs
//
// But to compare a question with a sign you need one vector per
// SENTENCE. Going from "one per token" to "one per sentence" is
// called POOLING, and an "embedding model" is nothing more than
// an encoder, a pooling, and a rule for comparing. There are
// two poolings in the wild.
//
// 1. CLS POOLING. Keep the vector of token 0, the [CLS] that
//    the port adds in front (exercise 122), and throw the rest
//    away. It works when the model was TRAINED to put the
//    meaning of the sentence there, and not otherwise.
//
// 2. MEAN POOLING. Average the vectors of all the tokens. And
//    here is the trap of this enclosure. The signs do not have
//    the same length, and a batch is a rectangle (exercise 119),
//    so the short ones are PADDED to the length of the longest.
//    The encoder still writes a vector on a padded position. It
//    means nothing: it is whatever fell out of the layers.
//    `hidden.mean(.s)` (exercise 096 did exactly that) averages
//    the garbage with the meaning. So the tokenizer also hands
//    you the ATTENTION MASK, {.doc, .s}, integers, 1 for a real
//    token and 0 for padding, and the mean becomes:
//
//                    sum over .s of (hidden * mask)
//        pooled  =  --------------------------------
//                         sum over .s of mask
//
//    Two details. The mask is made of integers and there is no
//    dtype promotion: convert it. And it has no `.d` axis:
//    `broad` it to the shape of `hidden` (it finds the axes by
//    their tags). Divide by the COUNT of real tokens, not by
//    the length of the row.
//
// Which pooling is right? Neither: the one the model was
// trained with. A sentence-transformers checkpoint says it in
// `1_Pooling/config.json` ("pooling_mode_cls_token",
// "pooling_mode_mean_tokens"). It is part of the model, as
// much as the weights. Marge's card says: mean.
//
// Then, the comparison. Two vectors "mean the same" when they
// point the same way, however long they are: the COSINE of
// their angle, 1 for the same direction, 0 for unrelated, -1
// for opposite.
//
//     cos(a, b) = (a . b) / (|a| * |b|)
//
// Nobody computes it that way. You make every vector of length
// 1 once, and from then on a cosine IS a dot product:
//
//     pub fn normalizeL2(input: Tensor, eps: f32) Tensor
//
// in `zml.nn`. It is two lines, read them:
//
//     const inv_norm = input.powByConst(2).sum(-1).addConstant(eps).rsqrt();
//     return input.mul(inv_norm);
//
// `eps` keeps a vector of zeros from dividing by zero (PyTorch's
// `F.normalize` uses 1e-12). And see the `-1`: it works along
// the LAST axis, whatever its tag. Keep `.d` last, or it will
// happily normalize your tokens instead of your features.
//
// With unit vectors, ranking N documents for a query is one
// `dot` over `.d`, {.doc, .d} against {.d}, and the best one is
// an `argMax` (exercise 043).
//
// The data. Vectors have 2 numbers: how FLUFFY, how LOUD. The
// question has 3 tokens and one padding; the rabbit's sign is
// full, the goose's has two paddings, the tortoise's one. Look
// at what the encoder left on the padded positions.
//
// To do:
//
//  1. `maskedMeanPool`.
//  2. `cosine` runs, and its numbers are not cosines: a cosine
//     never leaves [-1, 1]. BOTH sides must have length 1.
//  3. In `main`: the two masks as Buffers, and the run. Four
//     inputs, a struct of four outputs: that is the `runner` of
//     chapter 12.
//
// Three poolings, three different animals. Only one of them is
// loud.
//
const std = @import("std");
const zml = @import("zml");

/// The vector of token 0. {.s, .d} -> {.d}, {.doc, .s, .d} -> {.doc, .d}
fn clsPool(hidden: zml.Tensor) zml.Tensor {
    return hidden.slice(.s, .single(0));
}

/// The mean over all the positions, padding included.
fn meanPool(hidden: zml.Tensor) zml.Tensor {
    return hidden.mean(.s).squeeze(.s);
}

/// The mean over the real tokens only.
/// `mask` has the shape of `hidden` without `.d`: 1 for a token, 0 for padding.
fn maskedMeanPool(hidden: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    return ???;
}

/// The cosine of the query {.d} with each document {.doc, .d}: {.doc}
fn cosine(query: zml.Tensor, docs: zml.Tensor) zml.Tensor {
    const d = zml.nn.normalizeL2(docs, 1e-12);
    return d.dot(query, .d);
}

const Advice = struct {
    cls: zml.Tensor,
    plain: zml.Tensor,
    masked: zml.Tensor,
    /// The animal of the model card: the best of `masked`.
    best: zml.Tensor,
};

fn advise(question: zml.Tensor, question_mask: zml.Tensor, signs: zml.Tensor, signs_mask: zml.Tensor) Advice {
    const masked = cosine(maskedMeanPool(question, question_mask), maskedMeanPool(signs, signs_mask));
    return .{
        .cls = cosine(clsPool(question), clsPool(signs)),
        .plain = cosine(meanPool(question), meanPool(signs)),
        .masked = masked,
        .best = masked.argMax(.doc).indices.squeeze(.doc),
    };
}

const animals = [_][]const u8{ "rabbit", "goose", "tortoise" };

//                               fluffy, loud
const question_hidden = [4][2]f32{
    .{ 1, 0 }, // [CLS]
    .{ 0, 1 }, // loud
    .{ 0.2, 0.8 }, // one
    .{ 3, -1 }, // (padding)
};
const question_tokens = [4]i32{ 1, 1, 1, 0 };

const signs_hidden = [3][4][2]f32{
    // "[CLS] soft, quiet, warm"
    .{ .{ 0.8, 0.6 }, .{ 1, 0 }, .{ 1, 0.1 }, .{ 0.9, 0 } },
    // "[CLS] honks"
    .{ .{ 0, 1 }, .{ 0.2, 1 }, .{ -3, -1 }, .{ -3, -1 } },
    // "[CLS] hard, silent"
    .{ .{ 1, 0 }, .{ -1, 0 }, .{ -0.9, -0.3 }, .{ 3, -1 } },
};
const signs_tokens = [3][4]i32{
    .{ 1, 1, 1, 1 },
    .{ 1, 1, 0, 0 },
    .{ 1, 1, 1, 0 },
};

fn printRow(stdout: *std.Io.Writer, pooling: []const u8, scores: [3]f32) !void {
    var best: usize = 0;
    for (scores, 0..) |score, i| {
        if (score > scores[best]) best = i;
    }
    try stdout.print("{s:<12}", .{pooling});
    for (scores) |score| try stdout.print(" {d:8.3}", .{score});
    try stdout.print("   -> the {s}\n", .{animals[best]});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const question: zml.Tensor = .init(.{ .s = 4, .d = 2 }, .f32);
    const question_mask: zml.Tensor = .init(.{ .s = 4 }, .i32);
    const signs: zml.Tensor = .init(.{ .doc = 3, .s = 4, .d = 2 }, .f32);
    const signs_mask: zml.Tensor = .init(.{ .doc = 3, .s = 4 }, .i32);

    var exe = try platform.compileFn(gpa, io, advise, .{ question, question_mask, signs, signs_mask }, .{});
    defer exe.deinit();

    var question_buffer: zml.Buffer = try .fromBytes(io, platform, question.shape(), std.mem.sliceAsBytes(&question_hidden));
    defer question_buffer.deinit();
    var signs_buffer: zml.Buffer = try .fromBytes(io, platform, signs.shape(), std.mem.sliceAsBytes(&signs_hidden));
    defer signs_buffer.deinit();
    // The two masks, as Buffers.
    ???;

    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);

    // The run. The four results go in `advice`: free them too.
    ???;

    try stdout.print("\"which one is loud?\"\n", .{});
    try stdout.print("{s:<12} {s:>8} {s:>8} {s:>8}\n", .{ "pooling", animals[0], animals[1], animals[2] });
    try printRow(stdout, "[CLS] token", try advice.cls.getValue([3]f32, io));
    try printRow(stdout, "plain mean", try advice.plain.getValue([3]f32, io));
    try printRow(stdout, "masked mean", try advice.masked.getValue([3]f32, io));
    const best = try advice.best.getValue(i32, io);
    try stdout.print("the card says mean pooling, {f} -> {f}: go and pet the {s}\n", .{ signs.shape(), advice.masked.shape(), animals[@intCast(best)] });
    try stdout.flush();
}
