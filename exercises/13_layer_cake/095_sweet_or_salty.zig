//
// Chef Babette has exactly one rule about fillings, and she has
// had it since 1987:
//
//     "Sweet, or salty. Never both. Never neither."
//
// Strawberry: sweet, fine. Olive: salty, fine. Plain: neither,
// rejected. Salted caramel: both, rejected, and please don't
// bring it up again, she has heard all the arguments.
//
// You are to build the network that enforces the rule. It takes
// two numbers per filling (sweet? salty? as 1 or 0) and outputs
// a verdict. Two inputs, a yes/no answer. One Linear ought to
// do it?
//
// It cannot, and this little problem is famous for it (the
// textbooks call it XOR). A single Linear draws one straight
// line through the plane and says yes on one side. But the two
// "yes" corners, (1,0) and (0,1), sit diagonally across from
// each other, with the two "no" corners on the other diagonal.
// No straight line separates them. Try it on a napkin.
//
// Two tiers with an oven in between can. This shape of network
// is the MLP (multi-layer perceptron), and it is the "feed
// forward" block of every transformer you'll meet:
//
//     hidden = relu(W1·x + b1)
//     logits = W2·hidden + b2
//
// Chef Babette worked the weights out by hand. Call the inputs
// s (sweet) and t (salty):
//
//     h0 = relu(s + t)          "how many flavors?"
//     h1 = relu(s + t - 1)      "is it more than one?"
//
//     reject = 0.5 + 2·h1       a shrug, or an outrage
//     serve  = 2·h0 - 4·h1
//
// Check the four cases, they fit on the napkin:
//
//                       s  t    h0  h1    reject  serve
//     plain             0  0     0   0      0.5     0
//     strawberry        1  0     1   0      0.5     2
//     olive             0  1     1   0      0.5     2
//     salted caramel    1  1     2   1      2.5     0
//
// `serve` wins exactly when there is one flavor. The relu is
// doing all the work: h1 stays at zero until the second flavor
// shows up, which a straight line could never do. (Without the
// relu, plain gives h1 = -1: a NEGATIVE amount of "more than
// one flavor", and the arithmetic happily serves it.)
//
// Two practical matters when stacking Linears:
//
//  1. THE RENAME. `Linear.forward` contracts `.d` and returns
//     `.dout`. The next Linear contracts `.d` again... and the
//     tensor it receives has no `.d` anymore. Between two
//     Linears you say that this output is the next input:
//
//         .rename(.{ .dout = .d })
//
//     You'll see that line all over ZML's model files. Forget
//     it, and `dot` panics while looking for a `.d` axis.
//
//  2. THE VERDICT. The last layer gives one score per class
//     (the "logits"). Scores are not an answer yet. Every
//     classifier you will ever meet finishes with the same two
//     moves, and you have known both since the reduction
//     kitchen. "The classes" are the axis the last Linear
//     produced, so you know its tag:
//
//      - `softmax` over the classes (exercise 044) turns the
//        scores into probabilities that add up to 1. With two
//        classes only the GAP between the scores matters:
//
//            gap 0.5 -> 62.2%    gap 1.5 -> 81.8%
//            gap 2.5 -> 92.4%    gap 5   -> 99.3%
//
//        A network that is "sure" is a network whose scores
//        are far apart. Nothing more mystical than that;
//      - `argMax` over the classes (exercise 042) returns
//        `.values` and `.indices`: the highest probability, and
//        WHERE it is. The index is the class, an i32. The value
//        is how sure the network is of it. The reduced axis is
//        kept with a size of 1, so `squeeze` it away.
//
// That makes TWO results per filling, so `forward` returns a
// struct of Tensors, and the Exe has two outputs. `exe.eval`
// only knows how to return one Buffer. You need the runner of
// exercise 083, and a `zml.Bufferized` of the result struct for
// it to fill.
//
// The Taster below was assembled in a hurry:
//
//  - its `forward` goes straight from one Linear into the next:
//    no rename, and no oven either. You will meet the panic
//    first. Fix it, and you get to see what a two-tier cake
//    without an oven is worth (exercise 090 warned you): look
//    at who gets served. Then bake it. No need for an `oven`
//    field this time, the formula says which one: call the
//    Tensor method;
//  - the end of `forward`, the two moves above, is not written;
//  - neither are the placeholders of the two Linears (each has
//    a weight AND a bias; the sizes are in the arrays below, and
//    you know the convention for the tags), the Exe, the
//    Buffers of the Taster, and the run.
//
// The printing at the bottom tells you which names it expects.
// And it prints how sure the Taster is. Chef Babette, it turns
// out, rejects plain with a shrug and salted caramel with her
// whole heart; and she could not tell you which of strawberry
// and olive she likes better. (Look at h0 and h1: neither could
// the network. It only ever COUNTS the flavors.)
//
const std = @import("std");
const zml = @import("zml");

const Taster = struct {
    hidden: zml.nn.Linear,
    verdict: zml.nn.Linear,

    pub const Verdict = struct {
        /// {.cake}, i32: the index of the most likely class.
        class: zml.Tensor,
        /// {.cake}, f32: the probability of that class.
        confidence: zml.Tensor,
    };

    /// fillings: {.cake, .d} -> for each cake, a class and a confidence.
    pub fn forward(self: Taster, fillings: zml.Tensor) Verdict {
        const hidden = self.hidden.forward(fillings, fillings.dtype());
        const logits = self.verdict.forward(hidden, hidden.dtype());
        // The two moves: probabilities, then the best one.
        return ???;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Two Linears: 2 flavors in, 2 hunches out; 2 hunches in, 2
    // classes out. Both with a bias.
    const taster: Taster = ???;
    const fillings: zml.Tensor = .init(.{ .cake = 4, .d = 2 }, .f32);

    // TODO: `exe`.

    // h0 = s + t, h1 = s + t - 1 (the relu is in `forward`)
    var w1: zml.Buffer = try .fromBytes(io, platform, taster.hidden.weight.shape(), std.mem.sliceAsBytes(&[2][2]f32{ .{ 1, 1 }, .{ 1, 1 } }));
    defer w1.deinit();
    var b1: zml.Buffer = try .fromBytes(io, platform, taster.hidden.bias.?.shape(), std.mem.sliceAsBytes(&[2]f32{ 0, -1 }));
    defer b1.deinit();

    // class 0, reject = 0.5 + 2·h1; class 1, serve = 2·h0 - 4·h1
    var w2: zml.Buffer = try .fromBytes(io, platform, taster.verdict.weight.shape(), std.mem.sliceAsBytes(&[2][2]f32{ .{ 0, 2 }, .{ 2, -4 } }));
    defer w2.deinit();
    var b2: zml.Buffer = try .fromBytes(io, platform, taster.verdict.bias.?.shape(), std.mem.sliceAsBytes(&[2]f32{ 0.5, 0 }));
    defer b2.deinit();

    const taster_buffers: zml.Bufferized(Taster) = ???;

    const names = [_][]const u8{ "plain", "strawberry", "olive", "salted caramel" };
    //                          sweet, salty
    const flavors = [4][2]f32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } };
    var fillings_buffer: zml.Buffer = try .fromBytes(io, platform, fillings.shape(), std.mem.sliceAsBytes(&flavors));
    defer fillings_buffer.deinit();

    // TODO: two outputs. A runner, and `verdicts`, a struct of
    // Buffers for it to fill. Both of those Buffers are yours to
    // free.

    try stdout.print("verdicts {f}, confidence {f}\n", .{ verdicts.class.shape(), verdicts.confidence.shape() });
    const classes = [_][]const u8{ "rejected", "served" };
    const confidences = try verdicts.confidence.getValue([4]f32, io);
    for (names, try verdicts.class.getValue([4]i32, io), confidences) |name, class, confidence| {
        try stdout.print("  {s:<14} -> {s:<8} ({d:.1}% sure)\n", .{ name, classes[@intCast(class)], confidence * 100 });
    }
    try stdout.flush();
}
