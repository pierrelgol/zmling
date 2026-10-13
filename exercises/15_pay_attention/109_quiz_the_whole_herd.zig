//
// Quiz! Time to assemble the real thing: one complete
// self-attention layer, as found (a few dozen times over) in
// every Llama, and every llama.
//
// You have all the parts. What a real layer adds is four Linear
// layers, because so far queries, keys and values fell from the
// sky, and in a model they are COMPUTED from the tokens:
//
//     q_proj   "given what I am, what do I want to hear?"
//     k_proj   "given what I am, what do I know about?"
//     v_proj   "given what I am, what do I have to say?"
//     o_proj   mixes what the heads heard into one vector
//
// All four take `.d` numbers and return `.d` numbers, without
// bias. The whole layer is one model struct, and its `forward`
// turns `x: {.s, .d}` into a tensor of the same shape:
//
//  1. Project x three times: q, k and v. Split each of them
//     into `self.heads` heads. (What is the name of the axis a
//     Linear hands you back?)
//
//  2. Spin the queries and the keys: RoPE, default options,
//     positions 0, 1, 2... Not the values.
//
//  3. Scaled dot product attention, with a causal mask: no
//     llama hears the ones behind her. Bring back your mask
//     from exercise 104 (there is a place for it below). The
//     sequence plays both roles, so it must be given both
//     names.
//
//  4. Glue the heads back together into one `.d` axis, and
//     call the sequence `.s` again.
//
//  5. Through `o_proj`. And then, the last ingredient, new but
//     tiny: the RESIDUAL. A layer does not replace a token by
//     what it heard, it ADDS what it heard to what the token
//     already was:
//
//         return x.add(heard)
//
//     That's why the output must have exactly the shape, and
//     the tags, of x. It is also what lets forty of these
//     layers be stacked: each one only nudges the tokens.
//
// Every one of these steps is a line or two that you have
// already written in this chapter or in chapter 13. Mind the
// tags at each step: `.dout`, `.d`, `.h`, `.hd`, `.s`, `.q`,
// `.k` all show up, each in its place. When ZML panics, the
// message names the shapes it got. Read them, the answer is
// usually right there.
//
// Then `main`, which is yours too. In the order you learned
// them:
//
//  - the model: a `SelfAttention` value whose Linears hold
//    placeholders, `{.dout = 4, .d = 4}` f32, no bias, with 2
//    heads (chapter 13). Four Linears, four placeholders: a
//    ghost can only be used once (exercise 15);
//  - a placeholder for the herd, `{.s = 4, .d = 4}` f32, and
//    the Exe of `forward`;
//  - the Buffers. The numbers are below, hand-made, so that you
//    can follow a llama through the layer with a pencil if you
//    feel brave. Four weights, gathered in a
//    `zml.Bufferized(SelfAttention)` (exercise 94: only the
//    Tensors are in there, and each Linear still wants a
//    `null` for its `quantization`, exercise 89), and the herd;
//  - THREE rounds of gossip. The output of a round has the
//    shape of its input, so nothing stops you from feeding it
//    back in: what the herd is after round one is what goes
//    into round two. One Exe, called three times, and after
//    each round a call to `printHerd` (rounds 1, 2 and 3).
//    That is three layers stacked (sharing their weights),
//    driven from `main`. Mind the Buffers: each round makes a
//    new one, and each of them has to be freed exactly once.
//
const std = @import("std");
const zml = @import("zml");

/// The causal mask {.q = len, .k = len}, f32: 0 where the listener may
/// hear the speaker, -inf elsewhere. Yours, from exercise 104.
fn causalMask(len: i64) zml.Tensor {
    return ???;
}

/// One layer of causal self-attention, with RoPE and a residual.
const SelfAttention = struct {
    // Four Linear layers, `q_proj`, `k_proj`, `v_proj`, `o_proj`,
    // and `heads`, a plain number.
    ???,

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: SelfAttention, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

/// The weights, {.dout = 4, .d = 4}.
/// "I want to hear about what I am": the identity.
const q_weight = [4][4]f32{
    .{ 1, 0, 0, 0 },
    .{ 0, 1, 0, 0 },
    .{ 0, 0, 1, 0 },
    .{ 0, 0, 0, 1 },
};

/// "I know about the opposite of what I am": the two halves, swapped.
const k_weight = [4][4]f32{
    .{ 0, 0, 1, 0 },
    .{ 0, 0, 0, 1 },
    .{ 1, 0, 0, 0 },
    .{ 0, 1, 0, 0 },
};

/// "What I say is what I am, only louder at the end."
const v_weight = [4][4]f32{
    .{ 1, 0, 0, 0 },
    .{ 0, 2, 0, 0 },
    .{ 0, 0, 3, 0 },
    .{ 0, 0, 0, 4 },
};

/// What was heard counts for half.
const o_weight = [4][4]f32{
    .{ 0.5, 0, 0, 0 },
    .{ 0, 0.5, 0, 0 },
    .{ 0, 0, 0.5, 0 },
    .{ 0, 0, 0, 0.5 },
};

/// Four llamas, four numbers each: {.s = 4, .d = 4}.
const herd = [4][4]f32{
    .{ 0, 0, 1, 0 }, // Tina
    .{ 1, 0, 0, 0 }, // Kuzco
    .{ 1, 0, 1, 0 }, // Paca
    .{ 0, 1, 0, 1 }, // Dolly
};

const names = [_][]const u8{ "Tina ", "Kuzco", "Paca ", "Dolly" };

/// To be called after each round, with the herd as it is by then.
fn printHerd(stdout: *std.Io.Writer, round: usize, shape: zml.Shape, rows: [4][4]f32) !void {
    try stdout.print("after {d} round(s) of gossip {f}:\n", .{ round, shape });
    for (names, rows) |name, row| {
        try stdout.print("  {s}", .{name});
        for (row) |r| try stdout.print(" {d:6.3}", .{r});
        try stdout.print("\n", .{});
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The model, the placeholder, the Exe, the Buffers, and three
    // rounds of gossip, with a `printHerd` after each.
    ???;

    try stdout.flush();
}
