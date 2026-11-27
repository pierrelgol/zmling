//
// Brenda stamped the transcript. Next desk: the fitting room,
// where every graduate gets a gown (one size, it must fit
// everybody) and a cap. Tiny's two smallest pieces of clothing
// are the first two classes of `model.py`, and neither of them
// has a ready-made struct in ZML. No `transformers` class to
// copy the wiring from, either: this time the reference is the
// author's own forty lines.
//
// THE GOWN: RMSNorm. It brings every vector to the same size
// before a layer looks at it.
//
//     class RMSNorm(torch.nn.Module):
//         def __init__(self, dim: int, eps: float):
//             super().__init__()
//             self.eps = eps
//             self.weight = nn.Parameter(torch.ones(dim))
//
//         def _norm(self, x):
//             return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
//
//         def forward(self, x):
//             output = self._norm(x.float()).type_as(x)
//             return output * self.weight
//
// Compare with the LayerNorm of chapters 18 to 20: no mean is
// subtracted, and there is no bias. Divide by the root of the
// mean of the squares, multiply by a learned weight, done.
// ZML has the first half:
//
//     pub fn rmsNorm(x: Tensor, axis: anytype, eps: f32) Tensor
//
// Read its five lines in zml/nn/nn.zig and put them next to
// `_norm`: same formula, and NO WEIGHT. `zml.nn.rmsNorm` is
// `_norm`, not `forward`. The multiplication is yours, and the
// weight is a `{.d}` vector meeting a `{.s, .d}` tensor
// (chapter 5 knows what to do about that).
//
// And `eps`? The checkpoint has no such tensor and
// `config.json` no such key. Yesterday's fine print: it is a
// default of the `ModelArgs` class, `norm_eps: float = 1e-5`.
//
// THE CAP: the feed-forward. Not the `fc -> gelu -> proj` of
// GPT-2. Llama's has THREE matrices and is called SwiGLU:
//
//     class FeedForward(nn.Module):
//         def __init__(self, dim, hidden_dim, multiple_of, dropout):
//             ...
//             self.w1 = nn.Linear(dim, hidden_dim, bias=False)
//             self.w2 = nn.Linear(hidden_dim, dim, bias=False)
//             self.w3 = nn.Linear(dim, hidden_dim, bias=False)
//             self.dropout = nn.Dropout(dropout)
//
//         def forward(self, x):
//             return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
//
// Two matrices go UP from 64 to 172, side by side, on the same
// input. One of the two results goes through `silu` (in ZML:
// `Tensor.silu()`), the other does not, and they are
// multiplied: one is a gate for the other. `w2` brings the
// product back DOWN to 64. Dropout does nothing outside of
// training: a port drops it.
//
// The names are no help at all. `w1`, `w2`, `w3`: the one in
// the middle is the LAST one applied. And here is what makes
// this dangerous: `w1` and `w3` have the SAME shape. Swap
// them, or put the `silu` on the wrong one, and every shape
// still fits, the program runs, and the numbers are wrong.
//
// The rule: WHEN TWO WEIGHTS HAVE THE SAME SHAPE, THE TYPE
// SYSTEM IS BLIND. Tags cannot tell `w1` from `w3`. Only two
// things can: the line of the source, read character by
// character, and a number from PyTorch. So you test this
// layer alone, with the method of chapter 20 (exercises 135
// to 137): feed it what PyTorch fed it, compare with what
// PyTorch got.
//
// `activations.safetensors` has, for the prompt "Once upon a
// time" (5 tokens with the BOS) and for every layer N:
//
//     layers.N.in                    what enters the block
//     layers.N.attention_norm.out
//     layers.N.attention.q_rope
//     layers.N.attention.out
//     layers.N.ffn_norm.out
//     layers.N.feed_forward.out
//     layers.N.out                   what leaves the block
//
// One thing is not there: what ENTERS `ffn_norm`. A dump never
// has everything, and you do not need it to. The block is
//
//     h = x + self.attention.forward(self.attention_norm(x), ...)
//     out = h + self.feed_forward.forward(self.ffn_norm(h))
//
// so `h` is `layers.0.in` plus `layers.0.attention.out`: both
// are in the file. A missing activation is often one addition
// away from two that you have.
//
// To do:
//
//  1. `norm_eps`.
//  2. `rmsNorm`: the gown, weight included.
//  3. `FeedForward.forward` runs, and Tiny's cap is on
//     backwards: the program prints `false`. Read the Python
//     line again, slowly.
//  4. In `main`, the test of the gown is whole. The test of
//     the cap is missing: its input is `ffn_norm.out`, as
//     PyTorch computed it (NOT your own result of the first
//     test: one suspect at a time).
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// `ModelArgs.norm_eps`: in the source, and nowhere else.
const norm_eps: f32 = ???;

/// model.py, RMSNorm.forward. x: {.s, .d}, weight: {.d}
fn rmsNorm(x: zml.Tensor, weight: zml.Tensor) zml.Tensor {
    return ???;
}

/// A torch `nn.Linear(bias=False)`: its weight is {out, in}.
fn linear(view: zml.io.TensorStore.View, name: []const u8) zml.nn.Linear {
    return .init(view.withPrefix(name).createReplicatedTensor("weight", .{ .dout, .d }), null, .d);
}

const FeedForward = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(view: zml.io.TensorStore.View) FeedForward {
        return .{ .w1 = linear(view, "w1"), .w2 = linear(view, "w2"), .w3 = linear(view, "w3") };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: FeedForward, x: zml.Tensor) zml.Tensor {
        const gate = self.w3.forward(x, .f32).silu();
        const up = self.w1.forward(x, .f32);
        return self.w2.forward(gate.mul(up).rename(.{ .dout = .d }), .f32).rename(.{ .dout = .d });
    }
};

/// The two pieces of layer 0 that are tried on today.
const FittingRoom = struct {
    ffn_norm: zml.Tensor,
    feed_forward: FeedForward,

    pub fn init(view: zml.io.TensorStore.View) FittingRoom {
        return .{
            .ffn_norm = view.withPrefix("ffn_norm").createReplicatedTensor("weight", .{.d}),
            .feed_forward = .init(view.withPrefix("feed_forward")),
        };
    }

    /// What enters the block, what the attention added -> ffn_norm's output.
    pub fn gown(self: FittingRoom, x: zml.Tensor, attention_out: zml.Tensor) zml.Tensor {
        return rmsNorm(x.add(attention_out), self.ffn_norm);
    }

    /// ffn_norm's output -> the feed-forward's output.
    pub fn cap(self: FittingRoom, h: zml.Tensor) zml.Tensor {
        return self.feed_forward.forward(h);
    }
};

/// One tensor of a file, as a Buffer with the given tags.
fn reference(gpa: std.mem.Allocator, io: std.Io, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8, tags: anytype) !zml.Buffer {
    const shape = (store.view().getShape(key) orelse return error.NoSuchTensor).withTags(tags);
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);
    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);
    return .fromBytes(io, platform, shape, bytes);
}

/// Chapter 20's verdict: are the two close enough? (On a `false`,
/// `expectClose` prints its report on stderr: read it.)
fn matches(io: std.Io, want: zml.Buffer, got: zml.Buffer) bool {
    zml.testing.expectClose(io, want, got, .{}) catch return false;
    return true;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Tiny's weights.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "stories260K.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    // What PyTorch computed.
    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    const room: FittingRoom = .init(store.view().withPrefix("_orig_mod").withPrefix("layers").withLayer(0));
    try stdout.print("w1 {f}, w3 {f}, w2 {f}\n", .{
        room.feed_forward.w1.weight.shape(),
        room.feed_forward.w3.weight.shape(),
        room.feed_forward.w2.weight.shape(),
    });

    var weights = try zml.mem.bufferize(gpa, FittingRoom, &room);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(FittingRoom, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, FittingRoom, &room, &weights, &store, .{});
    try loader.await(io);

    // The gown.
    {
        var x = try reference(gpa, io, platform, &acts, "layers.0.in", .{ .s, .d });
        defer x.deinit();
        var attention_out = try reference(gpa, io, platform, &acts, "layers.0.attention.out", .{ .s, .d });
        defer attention_out.deinit();
        var want = try reference(gpa, io, platform, &acts, "layers.0.ffn_norm.out", .{ .s, .d });
        defer want.deinit();

        var exe = try platform.compile(gpa, io, room, .gown, .{ zml.Tensor.fromShape(x.shape()), zml.Tensor.fromShape(attention_out.shape()) }, .{});
        defer exe.deinit();
        var got = try exe.eval(gpa, io, .{ weights, x, attention_out });
        defer got.deinit();

        try stdout.print("the gown {f} matches PyTorch: {}\n", .{ got.shape(), matches(io, want, got) });
    }

    // The cap.
    {
        // `want`, an Exe for `room.cap`, and `got`.
        ???;

        try stdout.print("the cap {f} matches PyTorch: {}\n", .{ got.shape(), matches(io, want, got) });
    }
    try stdout.flush();
}
