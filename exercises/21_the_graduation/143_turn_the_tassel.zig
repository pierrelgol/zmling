//
// The rehearsal. At the end of the ceremony every graduate
// moves the tassel of the cap from one side to the other, and
// for a week the whole class has argued about WHICH side it
// starts on. Everybody is certain. Nobody has checked.
//
// Tiny turns too. She is a llama: before the scores of the
// attention are computed, her queries and her keys SPIN, each
// by an angle that depends on its position. RoPE, exercise
// 107. You wrote the formula there:
//
//     ( x cos a - y sin a ,  x sin a + y cos a )
//
// A rotation needs PAIRS of numbers, an x and a y. A head of
// Tiny is 8 numbers wide (`.hd = 8`), so 4 pairs, each
// turning at its own speed. Exercise 107 skipped one question
// with a "for the curious", because with `.hd = 2` it has
// only one answer: WHICH numbers form a pair?
//
// There are two schools, and both are everywhere:
//
//     neighbours        (x0, x1) (x2, x3) (x4, x5) (x6, x7)
//     the two halves    (x0, x4) (x1, x5) (x2, x6) (x3, x7)
//
// Neither is better. A model was trained with one of them and
// its weights make sense with that one only. In ZML it is a
// field of the options:
//
//     pub fn rope(x: Tensor, pos_idx: ?Tensor, opts: RopeOpts) Tensor
//
//     pub const RopeOpts = struct {
//         layout: Layout = .real_im_pass,
//         scaling: Scaling = .{ .default = .{} },
//         pub const Layout = enum { real_im_pass, real_pass_im_pass, interleaved };
//     };
//
// `.real_im_pass`: all the x's ("real" parts), then all the
// y's ("imaginary" parts): the two halves. It is what
// HuggingFace's `rotate_half` does, hence the default.
// `.interleaved`: x, y, x, y: the neighbours.
//
// Which one is Tiny? There is no config key for that, there
// never is. It is in the code:
//
//     def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
//         # reshape xq and xk to match the complex representation
//         xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
//         ...
//         # apply rotation using real numbers
//         xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
//         xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
//         ...
//         # flatten last two dimensions
//         xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
//
// `reshape(... + (-1, 2))` cuts the last axis in groups of 2,
// in order, and `unbind(-1)` takes the first of each group as
// `xq_r` and the second as `xq_i`. Decide which school that is.
//
// The speeds are in the source too, and they are the same
// kind of unchecked decision:
//
//     def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
//         freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
//
// `theta = 10000`: this time it IS what `RopeOpts.Scaling`
// does by default (`rope_theta: f32 = 10000`). Llama 3 changed
// it. You read the line, you compared, now you know.
//
// THE RULE OF THE DAY: AN OPTION WITH A DEFAULT IS A DECISION
// YOU HAVE NOT CHECKED YET. `.{}` compiles, and so does every
// wrong layout: the shapes are the same, only the numbers
// differ. Each default in a port is a question to the source
// of the ORIGINAL: what does the author do here?
//
// And then you prove it, against `layers.0.attention.q_rope`:
// the queries of layer 0 after the spin, {5, 8, 8} (token,
// head, numbers of the head).
//
// The previous graduate left a test in `main`. It compares
// the queries of the FIRST token only, it passes, and it has
// passed since the day it was written, whatever the layout.
// Think about what the angle of position 0 is. A test that
// cannot fail is not a test: the second rule of the day.
//
// To do:
//
//  1. `Spinner.forward`: the queries are `wq` applied to the
//     input, then cut in heads. `wq` gives a `.dout` axis of
//     64: split it in `.h` and `.hd` (exercise 106), and let
//     the number of heads decide the rest.
//  2. In `main`, `all_match`: every token, not just the first.
//     Run it: `true`, then `false`. There is your bug.
//  3. Now the layout.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// A torch `nn.Linear(bias=False)`: its weight is {out, in}.
fn linear(view: zml.io.TensorStore.View, name: []const u8) zml.nn.Linear {
    return .init(view.withPrefix(name).createReplicatedTensor("weight", .{ .dout, .d }), null, .d);
}

/// The first half of the attention of layer 0: the queries, spun.
const Spinner = struct {
    wq: zml.nn.Linear,
    heads: u32,

    pub fn init(view: zml.io.TensorStore.View, heads: u32) Spinner {
        return .{ .wq = linear(view.withPrefix("attention"), "wq"), .heads = heads };
    }

    /// h: {.s, .d}, what attention_norm gave -> {.s, .h, .hd}
    pub fn forward(self: Spinner, h: zml.Tensor) zml.Tensor {
        // The queries, one head at a time: {.s, .h, .hd}
        const q = ???;
        return zml.nn.rope(q, null, .{});
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

/// Are these numbers the same, give or take a thousandth?
fn close(want: []const f32, got: []const f32) bool {
    if (want.len != got.len) return false;
    for (want, got) |w, g| {
        if (@abs(w - g) > 1e-3) return false;
    }
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

    // `n_heads` of config.json.
    const spinner: Spinner = .init(store.view().withPrefix("_orig_mod").withPrefix("layers").withLayer(0), 8);

    var weights = try zml.mem.bufferize(gpa, Spinner, &spinner);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Spinner, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Spinner, &spinner, &weights, &store, .{});
    try loader.await(io);

    var h = try reference(gpa, io, platform, &acts, "layers.0.attention_norm.out", .{ .s, .d });
    defer h.deinit();
    var want = try reference(gpa, io, platform, &acts, "layers.0.attention.q_rope", .{ .s, .h, .hd });
    defer want.deinit();

    var exe = try platform.compile(gpa, io, spinner, .forward, .{zml.Tensor.fromShape(h.shape())}, .{});
    defer exe.deinit();
    var got = try exe.eval(gpa, io, .{ weights, h });
    defer got.deinit();

    // On the host, token after token: 8 heads of 8 numbers each.
    const want_slice = try want.toSliceAlloc(gpa, io);
    defer want_slice.free(gpa);
    const got_slice = try got.toSliceAlloc(gpa, io);
    defer got_slice.free(gpa);
    const want_numbers = want_slice.items(f32);
    const got_numbers = got_slice.items(f32);
    const per_token = 8 * 8;

    // The test of the previous graduate.
    const first_matches = close(want_numbers[0..per_token], got_numbers[0..per_token]);
    // The test.
    const all_match = ???;

    try stdout.print("the queries, after the turn: {f}\n", .{got.shape()});
    try stdout.print("  the first token matches PyTorch: {}\n", .{first_matches});
    try stdout.print("  all 5 tokens match PyTorch: {}\n", .{all_match});
    try stdout.flush();
}
