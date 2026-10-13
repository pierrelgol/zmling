//
// A confession from the front row. The herd of chapter 15 had
// one mouth per ear: every head of the attention had its own
// queries, its own keys, its own values. Tiny could not afford
// that. She has EIGHT heads that ask questions and only FOUR
// sets of notes to answer them from: all through school, her
// heads shared notes, two by two.
//
// You saw it on the transcript and maybe wondered:
//
//     wq   {64, 64}     8 heads of 8 numbers
//     wk   {32, 64}     4 heads of 8 numbers
//     wv   {32, 64}     4 heads of 8 numbers
//
// This is GROUPED-QUERY ATTENTION. Keys and values are what a
// model must REMEMBER for every token it has read (tomorrow's
// lesson has a lot to say about that), so halving them halves
// the memory, and the questions stay as varied as before.
// Llama 2 70B, Llama 3, Mistral, Gemma: all of them do it, and
// `n_kv_heads` (HuggingFace says `num_key_value_heads`) is
// the config key to look for.
//
// The whole `Attention.forward` of `model.py`, minus dropout
// and the batch axis:
//
//     xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
//     xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
//     xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
//     xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
//
//     # RoPE relative positional embeddings
//     xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
//
//     # grouped multiquery attention: expand out keys and values
//     xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
//     xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
//     ...
//     output = torch.nn.functional.scaled_dot_product_attention(
//         xq, xk, xv, attn_mask=None, ..., is_causal=True)
//
//     # restore time as batch dimension and concat heads
//     output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
//     output = self.wo(output)
//
// Everything but the middle is chapter 15: three Linears, a
// split in heads (with TWO different numbers of heads), the
// spin of yesterday on queries and keys, a causal mask, a
// merge, `wo`. The middle is `repeat_kv`:
//
//     def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
//         """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
//
// The 4 heads of keys become 8 by repetition, and the ORDER
// of the copies is the whole question:
//
//     repeat_interleave    A A B B C C D D     Tiny, and Llama
//     repeat (tile)        A B C D A B C D     somebody else
//
// With the first, query heads 0 and 1 read the notes of key
// head A. With the second, heads 0 and 4 do. Same shapes,
// different model. Sounds familiar?
//
// `zml.nn.sdpa` does not want the copies. From its source:
//
//     /// .h is allowed to differ from queries and keys as long as the key heads
//     /// can be repeated to match query heads.
//     ...
//     // This is the Llama convention though.
//     q = q.splitAxis(.h, .{ .h = k.dim(.h), .hq = .auto });
//
// It goes the other way around: instead of copying the keys,
// it GROUPS the queries. `.h = 8` becomes `.h = 4, .hq = 2`:
// query heads 0 and 1 are group 0, and meet key head 0. That
// is `repeat_interleave`, without a single copy. You give it
// 8 heads of queries and 4 heads of keys and values, as they
// come out of the Linears.
//
// The rule: WHEN ZML DOES SOMETHING FOR YOU, FIND OUT WHICH
// CONVENTION IT PICKED. Here the source says "the Llama
// convention", the author's docstring says
// `repeat_interleave`, they agree, and you still prove it
// with a number, because a model that tiles (or that groups
// the heads in some third way) would need the copies made by
// hand, in its own order, BEFORE `sdpa`.
//
// What `sdpa` wants, as in exercise 105: tags `.h`, `.q`, `.hd`
// on the queries, `.h`, `.k`, `.hd` on keys and values, and a
// mask if there is one:
//
//     zml.nn.sdpa(q, k, v, .{ .attn_mask = mask })
//     zml.nn.causalAttnMask(.{ .q = n, .k = n }, .f32, null)
//
// The result has the tags of the queries.
//
// To do:
//
//  1. `Attention.forward`. All of it, from the Python above.
//     Every op in it has been yours since chapter 15, and
//     the layout of the spin since yesterday.
//  2. Run it. ZML panics, and the message quotes three shapes.
//     Read the `.hd` of each: the numbers given to `init`
//     look innocent. They are not.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

/// A torch `nn.Linear(bias=False)`: its weight is {out, in}.
fn linear(view: zml.io.TensorStore.View, name: []const u8) zml.nn.Linear {
    return .init(view.withPrefix(name).createReplicatedTensor("weight", .{ .dout, .d }), null, .d);
}

const Attention = struct {
    wq: zml.nn.Linear,
    wk: zml.nn.Linear,
    wv: zml.nn.Linear,
    wo: zml.nn.Linear,
    heads: u32,
    kv_heads: u32,

    pub fn init(view: zml.io.TensorStore.View, heads: u32, kv_heads: u32) Attention {
        return .{
            .wq = linear(view, "wq"),
            .wk = linear(view, "wk"),
            .wv = linear(view, "wv"),
            .wo = linear(view, "wo"),
            .heads = heads,
            .kv_heads = kv_heads,
        };
    }

    /// h: {.s, .d}, what attention_norm gave -> {.s, .d}
    pub fn forward(self: Attention, h: zml.Tensor) zml.Tensor {
        return ???;
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

    // The heads that ask, and the heads that answer.
    const heads = 8;
    const kv_heads = 8;

    const view = store.view().withPrefix("_orig_mod").withPrefix("layers").withLayer(0).withPrefix("attention");
    const attention: Attention = .init(view, heads, kv_heads);

    var weights = try zml.mem.bufferize(gpa, Attention, &attention);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Attention, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Attention, &attention, &weights, &store, .{});
    try loader.await(io);

    var h = try reference(gpa, io, platform, &acts, "layers.0.attention_norm.out", .{ .s, .d });
    defer h.deinit();
    var want = try reference(gpa, io, platform, &acts, "layers.0.attention.out", .{ .s, .d });
    defer want.deinit();

    var exe = try platform.compile(gpa, io, attention, .forward, .{zml.Tensor.fromShape(h.shape())}, .{});
    defer exe.deinit();
    var got = try exe.eval(gpa, io, .{ weights, h });
    defer got.deinit();

    try stdout.print("{d} heads ask, {d} heads answer: {d} share each set of notes\n", .{ heads, kv_heads, heads / kv_heads });
    try stdout.print("the attention {f} matches PyTorch: {}\n", .{ got.shape(), matches(io, want, got) });
    try stdout.flush();
}
