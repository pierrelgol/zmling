//
// The quiz. Story hour was a success, of a kind ("he was TOO
// hungry?" asked a girl in the front row, and nobody had an
// answer). Madame Plume has a request for tomorrow: a story
// about an owl. She has also taken away everything Pip typed.
//
// Port the model again, from the file to the story, in this
// almost empty file. What you have:
//
//  - `model.safetensors`, `config.json` and `tokenizer.json`,
//    in the directory `dir`;
//  - the prompt;
//  - `pytorch_story`: the token ids PyTorch wrote from that
//    prompt, greedily, the 9 of the prompt included. Your
//    story must be those 33 tokens, exactly;
//  - `readFile`, and the three prints at the end of `main`.
//
// What you do not have is a file of reference activations.
// Tonight the only judge is the story. If it comes out wrong,
// you know where the checks are: exercises 129 to 132, one per
// part, and nothing stops you from borrowing them.
//
// The model, for the record. Everything in this list was a
// surprise at some point of the chapter:
//
//     tokens         BPE, no special token added          (129)
//     embeddings     wte[token] + wpe[position]           (129)
//     8 blocks       x = x + attn(ln_1(x))                (131)
//                    x = x + mlp(ln_2(x))
//       attention    q, k, v without bias, out_proj with  (128)
//                    scores NOT divided by sqrt(hd)       (130)
//                    causal; "local" blocks hear the last
//                    `window_size` tokens only            (130)
//       mlp          c_fc, gelu_new, c_proj               (131)
//     ln_f           the last norm                        (132)
//     head           tied to wte, no tensor of its own    (132)
//     generation     one Exe for the whole page, scores
//                    read at position len - 1, greedy     (133)
//
// The names in the file, under "transformer.":
//
//     wte.weight   wpe.weight   ln_f.{weight,bias}
//     h.N.ln_1.{weight,bias}             h.N.ln_2.{weight,bias}
//     h.N.attn.attention.{q,k,v}_proj.weight
//     h.N.attn.attention.out_proj.{weight,bias}
//     h.N.mlp.c_fc.{weight,bias}         h.N.mlp.c_proj.{weight,bias}
//
// and the keys of `config.json` worth a field in your
// `Config`: `num_layers`, `num_heads`, `window_size`,
// `layer_norm_epsilon`, `attention_layers`.
//
// The story is 33 tokens long, shorter than any window. A port
// without the local window would write the same owl. Write
// the window anyway: a port is not finished when its test
// passes, it is finished when it does what the original does.
// That is the difference between you and Pip.
//
// (Pip, for what it's worth, has spent the day reading
// `modeling_gpt_neo.py` from top to bottom, and says it is
// "actually quite short".)
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

const prompt = "The old owl wanted to read a story to";

/// What PyTorch wrote from that prompt, greedily (make_fixtures.py).
const pytorch_story = [_]u32{
    464, 1468, 39610, 2227, 284,   1100, 257, 1621,  284,  262,  39610,
    13,  679,  1965,  262,  39610, 611,  339, 714,   1100, 262,  1621,
    13,  383,  39610, 531,  3763,  290,  262, 39610, 373,  3772, 13,
};

/// The length of the page: the story stops when it is full.
const max_len = pytorch_story.len;

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Everything. For the prints: `prompt_ids` are the tokens of the
    // prompt, `story` is the full page, a `[max_len]u32`, and `text`
    // is what the decoder makes of it.
    ???;

    try stdout.print("{d} tokens in, {d} tokens out\n", .{ prompt_ids.len, story.len });
    try stdout.print("{s}\n", .{text.items});
    try stdout.print("the same story as PyTorch: {}\n", .{std.mem.eql(u32, &story, &pytorch_story)});
    try stdout.flush();
}
