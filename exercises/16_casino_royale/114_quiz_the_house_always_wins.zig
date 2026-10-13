//
// The last quiz. The House has heard about you, and the House
// wants a talking sign above the door: a machine that recites
// the motto of the casino, one word after the other, forever.
//
// You are going to build a language model. A real one, with
// everything a big one has, only smaller:
//
//     a vocabulary        8 tokens
//     a token embedding   each token becomes 3 numbers
//     a linear head       3 numbers become 8 logits, one per token
//     a sampler           the logits become ONE token
//     weights in a file   `house.safetensors`, next to this one
//
// and a loop around it, in `main`, which is how every language
// model on Earth writes text:
//
//     start with a token
//     repeat:
//         run the model on the last token: it picks the next one
//         print it
//         feed it back in
//
// One token per call of the Exe. The token that comes OUT of a
// call goes IN at the next one. That's all "generating text"
// is. (A big model looks at all the previous tokens, with the
// attention of chapter 15, and yours only looks at the last
// one. That makes it a "bigram" model. It also makes it fit in
// 48 floats.)
//
// The file holds two tensors:
//
//     embed.weight   {8, 3} f32    for the TokenEmbedding: tag it {.voc, .d}
//     head.weight    {8, 3} f32    for the Linear, no bias: tag it {.voc, .d}
//
// Why `.voc` for the output axis of the head, and not the
// usual `.dout`? Because what comes out of that Linear are the
// logits, and you know who is waiting for them, and which tag
// she insists on.
//
// Nobody trained these weights. `make_fixtures.py`, in this
// directory, computes them by hand, and explains the trick (8
// tokens, 3 bits, a dot product). Read it after you win: a
// model whose every weight you can explain is a rare animal.
//
// The model is a single token in, a single token out:
//
//     token: {} i32   ->   embedding {.d = 3}   ->   logits {.voc = 8}
//                     ->   next token: {} i32, and the Rng
//
// A scalar token works because `TokenEmbedding.forward` is a
// `gather`: the result has the shape of the indices, plus `.d`.
//
// The House wants the motto word for word, so the sampling is
// greedy: `topk = 1`. (Once it works, set `topk` to 3 in your
// copy and run it by hand a few times with different seeds.
// The House after a few drinks.)
//
// Everything from here on is yours. Shapes, ghosts, tags, a
// gather, a dot, an argMax, a model struct, a file of weights,
// a compiled Exe with a state threaded through it, and a loop.
// Sixteen chapters. You know all of it.
//
// Good luck. Not that luck has anything to do with it.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/16_casino_royale/house.safetensors";

/// The vocabulary: token id -> text.
const words = [_][]const u8{ ".", "again", "always", "house", "play", "please", "the", "wins" };

/// The id of "the", the first word of the motto.
const first_token: i32 = 6;
/// The id of ".", where a sentence ends.
const end_token: i32 = 0;

const Sign = struct {
    embed: zml.nn.TokenEmbedding,
    head: zml.nn.Linear,

    pub fn init(view: zml.io.TensorStore.View) Sign {
        return ???;
    }

    /// token: {} i32 -> the next token: {} i32, and the Rng.
    pub fn forward(self: Sign, token: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        return ???;
    }

    pub fn load(
        self: *const Sign,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Sign) {
        return ???;
    }

    pub fn unload(buffers: *zml.Bufferized(Sign)) void {
        buffers.embed.weight.deinit();
        buffers.head.weight.deinit();
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

    // The weights: header, key cabinet, model.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, weights_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const sign: Sign = ???;

    // The placeholders: one token (a scalar), and the dice.
    const token: zml.Tensor = ???;
    const rng: zml.Tensor.Rng = ???;

    var exe = ???;
    defer exe.deinit();

    var sign_buffers = try sign.load(gpa, io, platform, &store);
    defer Sign.unload(&sign_buffers);

    var rng_buffer = ???;
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);

    // The loop. `token_buffer` always holds the last token.
    var token_buffer: zml.Buffer = try .scalar(io, platform, first_token, .i32);
    defer token_buffer.deinit();

    try stdout.print("{s}", .{words[@intCast(first_token)]});

    // The sign would go on forever: stop at the end of the
    // sentence (or after 20 tokens, if the model has other ideas).
    for (0..20) |_| {
        // Run the model on `token_buffer`. The token that comes out
        // is the token that goes in next: free the old Buffer (it
        // was not donated, it is still yours) and keep the new one.
        // And mind the dice.
        ???

        const next_token = try token_buffer.getValue(i32, io);
        if (next_token == end_token) break;
        try stdout.print(" {s}", .{words[@intCast(next_token)]});
    }
    try stdout.print(".\n", .{});
    try stdout.flush();
}
