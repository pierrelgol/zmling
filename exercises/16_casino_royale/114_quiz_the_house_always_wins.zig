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
// The sign has two moods, and a mood is a
// `zml.nn.SamplingStrategy`. It is the third argument of
// `forward`: a plain Zig value, baked in at compile time. So,
// ONE model, ONE set of weights loaded once, and TWO Exes:
//
//     sober    the default strategy: greedy. The House wants
//              the motto word for word.
//     tipsy    `topk = 3`, at the default temperature. The
//              House after a few drinks.
//
// What `main` must print:
//
//  1. THE MOTTO. The sober sign starts from "the" and writes
//     until the token it picks is the "." (or for 20 tokens, if
//     the model has other ideas): the first word, then a space
//     and a word for each new token, then ".\n" at the end.
//
//  2. THE NIGHT SHIFT. The tipsy sign starts from "the" as
//     well, and writes `night_shift` tokens without ever
//     stopping. Its words are not printed (which ones come out
//     depends on your machine, remember exercise 110). They are
//     COUNTED: how many times was the token that came out the
//     one the motto wanted after the token that went in?
//     `script`, below, has the answer for every token. Then two
//     facts, with the two format strings you will find at the
//     bottom of `main`.
//
// Before you run the night shift, work out what to expect.
// `make_fixtures.py` explains that the right next word scores
// 3, and that the runners-up score 1. Top 3, softmax (you
// printed such odds in the previous exercise): how often
// should a tipsy sign stay on script?
//
// One state for the dice, seed 2024, threaded through every
// call of both Exes.
//
// Everything from here on is yours: the fields of the struct,
// its four methods, and `main`. Shapes, ghosts, tags, a gather,
// a dot, an argMax, a model struct, a file of weights, two
// compiled Exes with a state threaded through them, and a loop
// that feeds a model its own words. Sixteen chapters. You know
// all of it, and exercises 101 (the weights), 112 (the
// sampler) and 111 (the loop) are where to look when you don't.
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

/// The motto, as a table: `script[t]` is the token that follows
/// token `t`. After "the" (6) comes "house" (3). After "." (0), the
/// motto starts over.
const script = [8]i32{ 6, 0, 7, 2, 1, 4, 3, 5 };

/// How many tokens the tipsy sign writes.
const night_shift = 500;

const Sign = struct {
    // A token embedding and a linear head.
    ???,

    /// The model, with placeholders made from the file.
    pub fn init(view: zml.io.TensorStore.View) Sign {
        return ???;
    }

    /// token: {} i32 -> the next token: {} i32, and the Rng.
    pub fn forward(
        self: Sign,
        token: zml.Tensor,
        strategy: zml.nn.SamplingStrategy,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor.Rng } {
        return ???;
    }

    /// The weights, from the file to the device.
    pub fn load(
        self: *const Sign,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Sign) {
        return ???;
    }

    /// Frees what `load` returned.
    pub fn unload(buffers: *zml.Bufferized(Sign)) void {
        ???;
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

    // The file, the model, the placeholders, two Exes, the weights,
    // the dice. Then the motto, printed with these three lines (the
    // "." token itself is never printed as a word: the last line
    // stands for it). Then the night shift, counted.
    //
    //     try stdout.print("{s}", .{words[@intCast(first_token)]});
    //     try stdout.print(" {s}", .{words[@intCast(next_token)]});
    //     try stdout.print(".\n", .{});
    ???;

    // `on_script`: how many of the `night_shift` tokens followed the motto.
    try stdout.print("After a few drinks, the sign went off script: {}\n", .{on_script < night_shift});
    try stdout.print("It still got the next word right more often than not: {}\n", .{2 * on_script > night_shift});
    try stdout.flush();
}
