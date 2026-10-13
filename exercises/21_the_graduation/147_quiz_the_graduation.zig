//
// Graduation day. The folding chairs are full. Shifty came, in
// a shape nobody recognized. Madame Zelda brought the ghosts,
// who do not take up seats. Earl closed the diner. Phil came
// out of his burrow, saw his shadow, and stayed anyway. The
// Coach is crying already. The herd has the front row, and at
// the very back, standing, there is someone in a dark suit
// whom Monte calls "the House" and will not look at.
//
// The Dean taps the microphone. Tradition wants the smallest
// graduate to give the speech, and the smallest graduate is
// waiting behind the curtain, 260 000 parameters in a gown
// that fits. She is not ported yet. That is your diploma
// piece: Tiny, whole, from the blank part of this file.
//
// No new idea today. This is everything at once, and it is
// what a port IS, on any model, when nobody is there to hand
// you the pieces:
//
//     the paperwork      141   a prefix, a tied head, sizes from shapes
//     RMSNorm, SwiGLU    142   the weight of the norm, w1 w2 w3
//     RoPE               143   the layout: neighbours
//     the attention      144   8 heads ask, 4 answer
//     the model          145   blocks, layers, the BOS, the tokenizer
//     the cache          146   positions, prefill, decode
//
// and from the casino, the dice of exercises 112 and 113:
//
//     zml.nn.sampleTokens(logits, strategy, rng)
//         -> .{ next_token, new_rng }
//
// `logits` needs its `.voc` axis, a strategy is a plain Zig
// value baked into the Exe, and the Rng is a state that goes
// in and comes back out, like the cache. The author samples
// too, at the end of `generate` in `model.py`:
//
//     logits = logits / temperature
//     if top_k is not None:
//         v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
//         logits[logits < v[:, [-1]]] = -float('Inf')
//     probs = F.softmax(logits, dim=-1)
//     idx_next = torch.multinomial(probs, num_samples=1)
//
// Same recipe as `sampleTokens`: top-k, temperature, draw.
//
// THE PROGRAM. One model, one set of weights, one function
//
//     forward(tokens, pos, cache, strategy, rng)
//         -> .{ next_token, cache, rng }
//
// and three speeches, each of `story_len` tokens, each from
// the prompt "Once upon a time" behind a BOS, each with the
// cache: a prefill of the prompt, then one token per run.
// Each speech starts from an EMPTY cache and from the seed.
//
//  1. THE CEREMONY. Greedy: `greedy`, below. Tiny says what
//     PyTorch says, token for token (`reference_story`), and
//     it is printed, as text.
//
//  2. THE PHOTOGRAPH. `cold`: top 5, at a temperature of one
//     thousandth. Dice ARE rolled. But dividing by 0.001
//     multiplies every gap between two logits by a thousand,
//     and no roll of `sampleTokens` can jump that far: the
//     photographer asks for the speech again and gets the
//     same one, to the token. That is not luck, and you can
//     see why in zml/nn.zig and in `Rng.gumbel`.
//
//  3. THE PARTY. `warm`: top 5, temperature 1. After the
//     ceremony, Tiny improvises. Nobody can say what comes
//     out (it depends on the seed, and on your machine,
//     remember exercise 110), so it is NOT printed. Three
//     facts about it are, and they hold for any dice: it is
//     not the speech of the ceremony, it starts with the
//     prompt, and every token of it is one of Tiny's 512.
//
// That makes six Exes: a prefill and a decode per strategy.
// Write the telling of ONE speech as a function, and call it
// three times.
//
// Things that bite, in the order they will:
//
//  - the names in the file, and what is not in the file;
//  - `zml.nn.rmsNorm` has no weight, and the eps is 1e-5;
//  - which of w1, w2, w3 goes where;
//  - the layout of `rope`, and its positions;
//  - the order of the axes of what you file in the cache;
//  - a cache that does not start as zeros;
//  - the BOS;
//  - a `decoder` REMEMBERS the tokens of its previous call:
//    you print one text only, so one `decodeAlloc` is fine.
//    If you ever want a second one, `decoder.reset()` first.
//
// When the first line of output is wrong and you cannot see
// why: stop staring at the whole model. `activations.safetensors`
// is still here, with what PyTorch computed at every layer
// for this very prompt, and you know the method. Compare
// `tok_embeddings.out`, then `layers.0.out`, and walk towards
// the first number that differs. It works on every model
// there is. It is the one thing this course was about.
//
// You started with a rank and a dtype. You can now take a
// model that exists only as somebody's Python file and a
// checkpoint, read both, and make it run in Zig, on any
// machine ZML compiles for, and PROVE that it is the same
// model. Next: zml/examples/llm/ holds the grown-up versions
// of the file you are about to write. Go read them. They are
// big. They are not harder.
//
// The Dean is waving at you. Go get Tiny.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/21_the_graduation/";

const prompt = "Once upon a time";

/// Token 1: "beginning of sequence".
const bos: u32 = 1;

/// A whole speech, prompt included, in tokens.
const story_len = 106;

/// The seed of every speech.
const seed = 2024;

/// The three ways to pick a word.
const greedy: zml.nn.SamplingStrategy = .{};
const cold: zml.nn.SamplingStrategy = .{ .topk = 5, .temperature = 0.001 };
const warm: zml.nn.SamplingStrategy = .{ .topk = 5, .temperature = 1.0 };

/// The speech as PyTorch gives it: `make_fixtures.py` prints these.
const reference_story = [story_len]u32{
    1,   403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421, 395, 317, 426,
    338, 401, 396, 267, 337, 410, 408, 419, 292, 411, 322, 265, 282, 295, 433, 426,
    385, 328, 432, 358, 394, 261, 370, 432, 352, 266, 268, 388, 426, 338, 391, 266,
    267, 337, 335, 312, 432, 398, 312, 286, 267, 414, 270, 333, 415, 426, 13,  438,
    310, 439, 419, 357, 336, 432, 313, 438, 310, 432, 278, 316, 439, 419, 298, 414,
    267, 265, 282, 295, 433, 426, 436, 317, 286, 296, 418, 269, 279, 292, 416, 439,
    413, 409, 416, 327, 263, 415, 294, 267, 400, 426,
};

// The model, top to bottom, and whatever helps you tell a speech.
???

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The config, the weights, the tokenizer, the model. Then the three
    // speeches.
    ???;

    // `ceremony`, `photograph`, `party`: the three speeches, [story_len]u32.
    // `text`: the first one, decoded. `start`: the BOS and the prompt, []u32.
    // `config`: what config.json says. `in_vocabulary`: is every token of
    // `party` below `config.vocab_size`?
    try stdout.print("{s}\n", .{text.items});
    try stdout.print("The speech is PyTorch's, token for token: {}\n", .{std.mem.eql(u32, &ceremony, &reference_story)});
    try stdout.print("For the photograph, dice in hand, the same speech again: {}\n", .{std.mem.eql(u32, &photograph, &ceremony)});
    try stdout.print("At the party, Tiny told another story: {}\n", .{!std.mem.eql(u32, &party, &ceremony)});
    try stdout.print("It started with \"{s}\": {}\n", .{ prompt, std.mem.startsWith(u32, &party, start) });
    try stdout.print("Every word of it was one of her {d}: {}\n", .{ config.vocab_size, in_vocabulary });
    try stdout.print("Class dismissed.\n", .{});
    try stdout.flush();
}
