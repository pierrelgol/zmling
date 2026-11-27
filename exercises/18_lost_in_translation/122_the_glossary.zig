//
// Bertie does not read letters. He reads numbers: one per
// "token", and which piece of text gets which number is
// written in his glossary, tokenizer.json. A model and its
// tokenizer are ONE thing: the row 2605 of his word table was
// trained to mean "france", and it means nothing else. Give
// him the ids of another glossary and he will translate, with
// perfect confidence, a sentence nobody said.
//
// That is why the port starts here, before any tensor. In
// Python the whole business is one line:
//
//     ids = tokenizer.encode("the capital of france is [MASK].").ids
//
// ZML reads the same file:
//
//     var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(gpa, io, path);
//     defer tokenizer.deinit();
//
// (`path` is made absolute with `realPathFileAlloc`, like for
// the weights.) The tokenizer itself only answers questions about
// single tokens:
//
//     tokenizer.tokenId("[MASK]")   ?u32          text -> id
//     tokenizer.decode(2605)        ?[]const u8   id -> text
//
// Both return null when there is no such thing. Whole texts go
// through two helpers, which you create, use and deinit:
//
//     var encoder = try tokenizer.encoder();
//     defer encoder.deinit();
//     const ids: []u32 = try encoder.encodeAlloc(gpa, text);
//     defer gpa.free(ids);
//
//     var decoder = try tokenizer.decoder();
//     defer decoder.deinit();
//     var text = try decoder.decodeAlloc(gpa, ids);   // std.ArrayList(u8)
//     defer text.deinit(gpa);
//     ... text.items ...
//
// The encoder does everything the glossary says: it lowercases
// ("The" and "the" are one token for Bertie), splits the
// punctuation off, and cuts a word it doesn't know into pieces
// it does know. The pieces that continue a word start with
// "##". The decoder glues them back.
//
// A decoder remembers that it is in the middle of a text: the
// second text you give it comes out with a space in front.
// Make a new decoder for a new text. (There is a
// `decoder.reset()`. At the commit this course is pinned to,
// calling it does not compile.)
//
// Now the trap of this exercise, and of every port. What
// Python's `encode` returns for our sentence is not seven ids
// but NINE:
//
//     [CLS] the capital of france is [MASK] . [SEP]
//
// BERT was trained on sentences wrapped in two "special"
// tokens, and the Python tokenizer adds them in a last step
// (its "post-processor"). ZML's encoder does not run that
// step. Nothing fails if you forget them: seven ids are a
// perfectly good input, Bertie answers, and the answer is
// wrong, because he never saw a sentence that did not start
// with [CLS].
//
// PORTING RULE 3: before anything else, print the ids of a
// test sentence in the reference, record them, and make your
// port produce the SAME ids. Special tokens (BERT's [CLS] and
// [SEP], the "beginning of text" of the llamas) are your job.
// A wrong id is the cheapest bug to catch here, and the most
// expensive one to catch later, when all you see is a model
// that is "a bit off".
//
// The reference ids of our sentence are in the fourth file,
// activations.safetensors, under the name `input_ids`. It is a
// safetensors file like any other; `bufferFromStore`, below,
// reads one tensor of a store into a Buffer. You will use it
// in every exercise from now on.
//
// And never type an id either. Which number [MASK] has is
// written in the glossary: ask.
//
// To do:
//
//  1. `encodeForBert` runs, and returns what the encoder
//     returns. Make it return what Python returns.
//
//  2. The id of "[MASK]", and where it is in the sentence
//     (`std.mem.indexOfScalar(u32, haystack, needle)` returns
//     a `?usize`).
//
//  3. `tangled`, the ids of the second sentence, without the
//     special tokens this time, and `text`, the same ids
//     turned back into a text.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/18_lost_in_translation/";

/// The ids of `text` the way Bertie was trained to read them:
/// [CLS], the tokens of the text, [SEP]. The caller frees them.
fn encodeForBert(
    gpa: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    encoder: *zml.tokenizer.Tokenizer.Encoder,
    text: []const u8,
) ![]u32 {
    _ = tokenizer;
    return encoder.encodeAlloc(gpa, text);
}

/// One tensor of a file of weights (or of activations), as a Buffer.
fn bufferFromStore(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);

    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);

    return .fromBytes(io, platform, shape, bytes);
}

fn sameIds(ours: []const u32, theirs: []const i32) bool {
    if (ours.len != theirs.len) return false;
    for (ours, theirs) |a, b| if (a != b) return false;
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

    const tokenizer_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "tokenizer.json", gpa);
    defer gpa.free(tokenizer_path);
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(gpa, io, tokenizer_path);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const sentence = "The capital of France is [MASK].";
    const ids = try encodeForBert(gpa, &tokenizer, &encoder, sentence);
    defer gpa.free(ids);

    try stdout.print("{s}\n ", .{sentence});
    for (ids) |id| try stdout.print(" {s}", .{tokenizer.decode(id).?});
    try stdout.print("\n  ours:    {any}\n", .{ids});

    // What Python's tokenizer made of the same sentence.
    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    var reference_buffer = try bufferFromStore(gpa, io, platform, &acts, "input_ids");
    defer reference_buffer.deinit();
    const reference = try reference_buffer.toSliceAlloc(gpa, io);
    defer reference.free(gpa);

    try stdout.print("  PyTorch: {any}\n", .{reference.items(i32)});
    try stdout.print("  same ids: {}\n", .{sameIds(ids, reference.items(i32))});

    const mask_id = ???;
    const mask_position = ???;
    try stdout.print("  [MASK] is token {d}, at position {d}\n", .{ mask_id, mask_position });

    // A word that is not in the glossary.
    const tangled_sentence = "Unintelligible, said Bertie.";
    ???;

    try stdout.print("{s}\n ", .{tangled_sentence});
    for (tangled) |id| try stdout.print(" {s}", .{tokenizer.decode(id).?});
    try stdout.print("\n  and back: {s}\n", .{text.items});
    try stdout.flush();
}
