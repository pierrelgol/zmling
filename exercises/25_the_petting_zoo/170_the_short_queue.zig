//
// Second enclosure: the barnyard, where the animals gossip like
// the llamas of chapter 15, with the causal mask of exercise
// 104 (nobody hears those who stand behind).
//
// Marge wants to run TWO queues through the barn in one call,
// as you ran parcels in exercise 119: a batch, {.b, .s, .d}.
// The first queue has three animals, the second only two. A
// tensor is a rectangle, so the short queue gets a PADDING
// token, and the tokenizer says where, with the attention mask
// of the last exercise ({.b, .s}, 1 for real, 0 for padding).
//
// On which side? Both exist:
//
//  - on the RIGHT, for encoders: "hen duck (pad)";
//  - on the LEFT, for anything that generates text in a batch:
//    "(pad) hen duck". The next token is predicted from the
//    LAST position, so every row must END with a real token.
//    (HuggingFace: `tokenizer.padding_side = "left"`.)
//
// The barn generates. Left it is.
//
// In the last enclosure, padding spoiled a mean. Here it is
// worse: a padded position is a KEY like any other, the real
// animals listen to it, and what it says is noise. Pooling
// cannot repair that afterwards. The padding must be masked
// INSIDE the attention, the way the future is: with a table of
// 0 and -inf added to the scores before the softmax.
//
//     causal mask        {.q, .k}      who stands behind whom,
//                                      the same for every queue
//     key-padding mask   {.b, .k}      which KEYS are padding,
//                                      the same for every listener
//
// A listener may hear a key when BOTH say so. With 0 and -inf
// that "and" is an addition (0 + 0 = 0, anything + -inf =
// -inf), once the two are `broad`ened to {.b, .q, .k}. The
// causal one is in the pantry since exercise 130:
//
//     zml.nn.causalAttnMask(.{ .q = n, .k = n }, .f32, null)
//
// The key-padding one is yours: the attention mask speaks of
// positions `.s`, the scores speak of keys `.k`, and a `cmp`
// with a `select` turns 1/0 into 0/-inf (exercise 104 again).
//
// Now look at the mask of the short queue, "(pad) hen duck":
//
//              (pad)  hen  duck
//     (pad)      x     x     x       x: may not listen
//     hen        x     .     x
//     duck       x     .     .
//
// The FIRST ROW IS ALL x. The padding, as a listener, hears
// nobody: herself she may not (she is padding), the others
// stand behind her. Nobody cares what a padding hears. But the
// softmax of that row is computed anyway, and
//
//     exp(-inf) = 0 for every key, the sum is 0, and 0 / 0
//
// is NaN. "So what, it is the row of the padding." So this:
// the barn gossips for TWO rounds, and in the second round the
// hen multiplies the padding's vector by her weight for it.
// Her weight is 0. The vector is NaN. And 0 * NaN is NaN: one
// rotten row, and after the next layer the whole queue is NaN,
// real animals included. In a model of 12 layers nobody ever
// finds where it started (well, you would: chapter 20).
//
// A fully masked row must give ZEROS. `Tensor.softmax` knows:
//
//     pub fn softmax(self: Tensor, axis_: anytype) Tensor {
//         const a = self.axis(axis_);
//         const max_val = self.max(a);
//         const row_mask = max_val.cmp(.GT, .scalar(-std.math.inf(f64), self.dtype()));
//
//         const exp_diff_max = self.sub(max_val).convert(.f32).exp();
//         const res = exp_diff_max.div(exp_diff_max.sum(a)).convert(self.dtype());
//
//         // If a row is full -inf return full 0 instead of full nan,
//         // this fix attention when mask hides a full row.
//         return row_mask.broad(self.shape()).select(res, .scalar(0, self.dtype()));
//     }
//
// A row whose maximum is -inf is a row of -inf: it is replaced
// by zeros. (`zml.nn.sdpa` calls this softmax. A softmax you
// write yourself from `exp` and `sum`, straight from the
// formula of exercise 044, does not have that last line.
// Dewey, the summer intern, wrote his own.)
//
// And how do you know the masks are right? The padding must be
// INVISIBLE: the hen and the duck must get exactly the numbers
// they get when their queue goes through the barn alone,
// without any padding. That is a test you can always write,
// for any model, and it needs the same function compiled for
// another shape (exercise 015: an Exe is compiled for ONE shape).
//
// To do:
//
//  1. `keyPadding`.
//  2. `weigh` is Dewey's softmax. Put the real one.
//  3. `gossip` only uses the padding mask: in Dewey's barn
//     everybody hears those who stand behind. Add the causal
//     mask. (The picture printed by `main` shows both.)
//  4. In `main`, the second Exe: `gossip` for one queue of two
//     animals. Its Buffers are given, the name `alone_exe` is
//     expected below.
//
const std = @import("std");
const zml = @import("zml");

const rounds = 2;

/// The attention mask {.b, .s} (i32, 1 real, 0 padding) as a mask over
/// the keys {.b, .k}, f32: 0 for a real key, -inf for a padded one.
fn keyPadding(attention_mask: zml.Tensor) zml.Tensor {
    return ???;
}

/// Scores {.b, .q, .k}, already masked -> weights, each row sums to 1.
/// Or to 0, when there is nobody to listen to.
fn weigh(scores: zml.Tensor) zml.Tensor {
    const e = scores.exp();
    return e.div(e.sum(.k));
}

/// One round: every animal becomes the blend of what she may hear.
fn round(x: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    const d: f32 = @floatFromInt(x.dim(.d));
    const q = x.rename(.{ .s = .q });
    const k = x.rename(.{ .s = .k });
    const scores = q.dot(k, .d).scale(1.0 / @sqrt(d));
    return weigh(scores.add(mask)).dot(k, .k).rename(.{ .q = .s });
}

const Barn = struct {
    /// {.b, .q, .k}: the mask that was used.
    mask: zml.Tensor,
    /// {.b, .s, .d}: the animals after the last round.
    heard: zml.Tensor,
};

/// x: {.b, .s, .d}, attention_mask: {.b, .s}
fn gossip(x: zml.Tensor, attention_mask: zml.Tensor) Barn {
    const n = x.dim(.s);
    const table: zml.Shape = .init(.{ .b = x.dim(.b), .q = n, .k = n }, .f32);
    const mask = keyPadding(attention_mask).broad(table);

    var heard = x;
    for (0..rounds) |_| heard = round(heard, mask);
    return .{ .mask = mask, .heard = heard };
}

const names = [2][3][]const u8{
    .{ "goat ", "pig  ", "cow  " },
    .{ "(pad)", "hen  ", "duck " },
};

const queues = [2][3][2]f32{
    .{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } },
    // What a padding token holds is nobody's business. Today: this.
    .{ .{ 7, -7 }, .{ 0, 2 }, .{ 2, 0 } },
};
const attention_masks = [2][3]i32{
    .{ 1, 1, 1 },
    .{ 0, 1, 1 },
};

// The short queue, alone: no padding at all.
const short_queue = [1][2][2]f32{.{ .{ 0, 2 }, .{ 2, 0 } }};
const short_mask = [1][2]i32{.{ 1, 1 }};

fn printAnimal(stdout: *std.Io.Writer, name: []const u8, row: [2]f32) !void {
    try stdout.print("  {s}", .{name});
    for (row) |r| {
        if (std.math.isNan(r)) try stdout.print("    nan", .{}) else try stdout.print(" {d:6.3}", .{r});
    }
    try stdout.print("\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The two queues, in one batch.
    const x: zml.Tensor = .init(.{ .b = 2, .s = 3, .d = 2 }, .f32);
    const attention_mask: zml.Tensor = .init(.{ .b = 2, .s = 3 }, .i32);

    var exe = try platform.compileFn(gpa, io, gossip, .{ x, attention_mask }, .{});
    defer exe.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&queues));
    defer x_buffer.deinit();
    var mask_buffer: zml.Buffer = try .fromBytes(io, platform, attention_mask.shape(), std.mem.sliceAsBytes(&attention_masks));
    defer mask_buffer.deinit();

    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);

    var barn: zml.Bufferized(Barn) = undefined;
    runner.run(io, .{ x_buffer, mask_buffer }, .{&barn}, .{ .wait = true });
    defer zml.Buffer.deinitAll(Barn, &barn);

    try stdout.print("the short queue, as the mask sees it (x: may not listen):\n", .{});
    const mask = try barn.mask.getValue([2][3][3]f32, io);
    for (names[1], mask[1]) |name, row| {
        try stdout.print("  {s}", .{name});
        for (row) |m| try stdout.print(" {s}", .{if (m == 0) "." else "x"});
        try stdout.print("\n", .{});
    }

    try stdout.print("after {d} rounds, in the batch {f}:\n", .{ rounds, barn.heard.shape() });
    const heard = try barn.heard.getValue([2][3][2]f32, io);
    for (names, heard) |queue_names, queue| {
        for (queue_names, queue) |name, row| try printAnimal(stdout, name, row);
    }

    // The short queue, alone.
    const alone_x: zml.Tensor = .init(.{ .b = 1, .s = 2, .d = 2 }, .f32);
    const alone_mask: zml.Tensor = .init(.{ .b = 1, .s = 2 }, .i32);

    // `gossip` again, for this shape.
    ???;

    var alone_x_buffer: zml.Buffer = try .fromBytes(io, platform, alone_x.shape(), std.mem.sliceAsBytes(&short_queue));
    defer alone_x_buffer.deinit();
    var alone_mask_buffer: zml.Buffer = try .fromBytes(io, platform, alone_mask.shape(), std.mem.sliceAsBytes(&short_mask));
    defer alone_mask_buffer.deinit();

    var alone_runner = try alone_exe.runner(gpa);
    defer alone_runner.deinit(gpa);

    var alone_barn: zml.Bufferized(Barn) = undefined;
    alone_runner.run(io, .{ alone_x_buffer, alone_mask_buffer }, .{&alone_barn}, .{ .wait = true });
    defer zml.Buffer.deinitAll(Barn, &alone_barn);

    try stdout.print("the short queue alone {f}:\n", .{alone_barn.heard.shape()});
    const alone = try alone_barn.heard.getValue([1][2][2]f32, io);
    var same = true;
    for (names[1][1..], alone[0], heard[1][1..]) |name, row, batched| {
        try printAnimal(stdout, name, row);
        for (row, batched) |a, b| {
            if (!(@abs(a - b) < 1e-5)) same = false;
        }
    }
    try stdout.print("the padding is invisible: {}\n", .{same});
    try stdout.flush();
}
