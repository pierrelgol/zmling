//
// Opening day of the Summit. The delegates mumble, the
// microphones crackle, and Madame Vogel wants Bertie in the
// booth, filling in the word nobody caught. The trainee has
// been moved to the cloakroom. The file below is, apart from
// the sentences, empty.
//
// Port Bertie. All of him, from the four files to a word:
//
//     config.json         -> Config (and the defaults it lacks)
//     tokenizer.json      -> ids, with [CLS] and [SEP]
//     model.safetensors   -> Embeddings, Layers, Head
//     a sentence          -> the word behind its [MASK]
//
// What the file of weights contains, with `L` for each layer
// number the config announces (`W` is a `weight`, `W+b` a
// `weight` and a `bias`):
//
//     bert.embeddings.word_embeddings          W   {30522,128}
//     bert.embeddings.position_embeddings      W   {512,128}
//     bert.embeddings.token_type_embeddings    W   {2,128}
//     bert.embeddings.LayerNorm                W+b
//     bert.encoder.layer.L.attention.self.query      W+b
//     bert.encoder.layer.L.attention.self.key        W+b
//     bert.encoder.layer.L.attention.self.value      W+b
//     bert.encoder.layer.L.attention.output.dense    W+b
//     bert.encoder.layer.L.attention.output.LayerNorm  W+b
//     bert.encoder.layer.L.intermediate.dense        W+b {512,128}
//     bert.encoder.layer.L.output.dense              W+b {128,512}
//     bert.encoder.layer.L.output.LayerNorm          W+b
//     cls.predictions.transform.dense          W+b
//     cls.predictions.transform.LayerNorm      W+b
//     cls.predictions.bias                         {30522}
//
// (plus `bert.pooler` and `cls.seq_relationship`, which this
// job does not need.)
//
// And the whole computation, as the Python does it, dropouts
// removed:
//
//     x = word[ids] + position[0, 1, 2...] + token_type[0]
//     x = LayerNorm(x)
//     for each layer:
//         a = attention(x)            # heads, biases, no mask
//         x = LayerNorm(dense(a) + x)
//         m = dense(gelu(dense(x)))   # 128 -> 512 -> 128
//         x = LayerNorm(m + x)
//     h = x[position of the MASK]
//     h = LayerNorm(gelu(dense(h)))
//     scores = h . word + bias        # one per word of the glossary
//
// The eight rules, since nobody is here to repeat them:
//
//  1. No number of the architecture is typed in the port.
//  2. The config is the json file plus the defaults of the
//     config class (`layer_norm_eps` is 1e-12).
//  3. Same ids as the reference, special tokens included.
//  4. The default arguments of the reference are part of the
//     model.
//  5. One module at a time, each compared with the reference.
//  6. A name is not a definition (which GELU?).
//  7. What the reference builds in a loop over the config, you
//     build in a loop over the config.
//  8. A weight that is missing from the file is tied to
//     another one.
//
// One thing is new. The sentences don't have the same number
// of tokens, and a compiled Exe has ONE input shape. With the
// llamas, a short text could be padded up to a fixed length:
// their mask hides what comes after a token. Bertie has no
// mask. Every token listens to every other, padding included,
// so padding him changes his answer. (Real BERT ports pass an
// attention mask to hide the padding. Bertie's port does
// something simpler.) Compile one Exe per sentence: the
// weights are loaded once and serve them all, and he is small
// enough to compile in a blink.
//
// There is no recording for these sentences. What you have is
// the word PyTorch answered for each, in `sentences`. If
// yours differ, go back to rule 5: the recording of exercise
// 126 is still in the directory, and so are your six
// exercises.
//
// His answers, by the way, are his own. He was hired for his
// size.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/18_lost_in_translation/";

/// What the delegates said, and the word PyTorch's Bertie heard
/// behind each [MASK].
const sentences = [_]struct { text: []const u8, pytorch: []const u8 }{
    .{ .text = "He was born in New [MASK] City.", .pytorch = "york" },
    .{ .text = "He opened the [MASK] and went inside.", .pytorch = "door" },
    .{ .text = "The prime [MASK] of Japan.", .pytorch = "minister" },
    .{ .text = "The interpreter speaks English and [MASK].", .pytorch = "english" },
};

// `Config`, the helpers, `Embeddings`, `Layer`, `Encoder`, `Head`,
// `Bert`, and whatever puts [CLS] and [SEP] around a sentence.
???

/// The whole file, as bytes. The caller frees them.
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

    // The config, the weights, the model, the glossary: everything
    // that is done once.
    ???;

    var agreed: usize = 0;
    for (sentences) |sentence| {
        // `word`: Bertie's guess for the [MASK] of `sentence.text`.
        ???;

        if (std.mem.eql(u8, word, sentence.pytorch)) agreed += 1;
        try stdout.print("{s}\n  Bertie: {s}, PyTorch: {s}\n", .{ sentence.text, word, sentence.pytorch });
    }
    try stdout.print("They agree on {d} sentences out of {d}.\n", .{ agreed, sentences.len });
    try stdout.flush();
}
