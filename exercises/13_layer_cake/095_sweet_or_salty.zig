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
//     reject = 0.5
//     serve  = h0 - 2·h1
//
// Check the four cases: `serve` is 0, 1, 1, 0. It beats the
// constant 0.5 of `reject` exactly when there is one flavor.
// The relu is doing all the work: h1 stays at zero until the
// second flavor shows up, which a straight line could never do.
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
//     (the "logits"). The answer is the index of the highest
//     one, and you know how to ask for that: `argMax(.dout)`
//     returns `.values` and `.indices`, the indices are i32,
//     and the reduced axis is kept with a size of 1, so
//     `squeeze` it away.
//
// The Taster below was assembled in a hurry. Its `forward`
// goes straight from one Linear into the next, and it has no
// oven at all. You will meet the panic first. Fix it, and you
// get to see what a two-tier cake without an oven is worth.
// (Exercise 090 warned you.)
//
const std = @import("std");
const zml = @import("zml");

const Taster = struct {
    hidden: zml.nn.Linear,
    verdict: zml.nn.Linear,

    /// fillings: {.cake, .d} -> the index of the best class for each cake: {.cake}, i32.
    pub fn forward(self: Taster, fillings: zml.Tensor) zml.Tensor {
        const hidden = self.hidden.forward(fillings, fillings.dtype());
        const logits = self.verdict.forward(hidden, hidden.dtype());
        return logits.argMax(.dout).indices.squeeze(.dout);
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

    const taster: Taster = .{
        .hidden = .init(.init(.{ .dout = 2, .d = 2 }, .f32), .init(.{ .dout = 2 }, .f32), .d),
        .verdict = .init(.init(.{ .dout = 2, .d = 2 }, .f32), .init(.{ .dout = 2 }, .f32), .d),
    };
    const fillings: zml.Tensor = .init(.{ .cake = 4, .d = 2 }, .f32);

    var exe = try platform.compile(gpa, io, taster, .forward, .{fillings}, .{});
    defer exe.deinit();

    // h0 = s + t, h1 = s + t - 1 (the relu is in `forward`)
    var w1: zml.Buffer = try .fromBytes(io, platform, taster.hidden.weight.shape(), std.mem.sliceAsBytes(&[2][2]f32{ .{ 1, 1 }, .{ 1, 1 } }));
    defer w1.deinit();
    var b1: zml.Buffer = try .fromBytes(io, platform, taster.hidden.bias.?.shape(), std.mem.sliceAsBytes(&[2]f32{ 0, -1 }));
    defer b1.deinit();

    // class 0, reject = 0.5; class 1, serve = h0 - 2·h1
    var w2: zml.Buffer = try .fromBytes(io, platform, taster.verdict.weight.shape(), std.mem.sliceAsBytes(&[2][2]f32{ .{ 0, 0 }, .{ 1, -2 } }));
    defer w2.deinit();
    var b2: zml.Buffer = try .fromBytes(io, platform, taster.verdict.bias.?.shape(), std.mem.sliceAsBytes(&[2]f32{ 0.5, 0 }));
    defer b2.deinit();

    const taster_buffers: zml.Bufferized(Taster) = .{
        .hidden = .{ .weight = w1, .bias = b1, .quantization = null },
        .verdict = .{ .weight = w2, .bias = b2, .quantization = null },
    };

    const names = [_][]const u8{ "plain", "strawberry", "olive", "salted caramel" };
    //                          sweet, salty
    const flavors = [4][2]f32{ .{ 0, 0 }, .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } };
    var fillings_buffer: zml.Buffer = try .fromBytes(io, platform, fillings.shape(), std.mem.sliceAsBytes(&flavors));
    defer fillings_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ taster_buffers, fillings_buffer });
    defer result.deinit();

    try stdout.print("verdicts {f}\n", .{result.shape()});
    const classes = [_][]const u8{ "rejected", "served" };
    for (names, try result.getValue([4]i32, io)) |name, class| {
        try stdout.print("  {s:<14} -> {s}\n", .{ name, classes[@intCast(class)] });
    }
    try stdout.flush();
}
