//
// Neo cannot read. Nobody in this library can, strictly
// speaking: a model takes numbers. So the first thing a port
// needs is the path from a sentence to the first tensor of
// floats, and it has three steps. Each of them is a place
// where "it's a GPT-2, I know how it goes" can be wrong.
//
// STEP 1: TEXT TO TOKEN IDS. The repository ships
// `tokenizer.json`, and ZML reads that format:
//
//     var tokenizer: zml.tokenizer.Tokenizer = try .fromBytes(gpa, bytes);
//     defer tokenizer.deinit();
//
//     var encoder = try tokenizer.encoder();
//     defer encoder.deinit();
//     const ids: []u32 = try encoder.encodeAlloc(gpa, "Once upon");
//     defer gpa.free(ids);
//
//     tokenizer.decode(id)   -> ?[]const u8   the piece behind one id
//
// Neo's tokenizer is a BPE ("byte pair encoding"): it cuts the
// text into the longest pieces it has a number for. Print the
// pieces and you will see a strange letter:
//
//     [Once][Ġupon][Ġa][Ġtime]
//
// `Ġ` is how this family of tokenizers writes "a space comes
// first". " dragon" and "dragon" are two different tokens.
//
// What the encoder does NOT do: add special tokens. Some
// models want a "beginning of sequence" token in front of
// every text, some want one at the end, some want nothing, and
// `config.json` listing a `bos_token_id` (Neo's does: 50256)
// does not tell you which. Pip's last port was a llama, which
// wants one. So Pip added one.
//
// The rule: never guess what the model was fed. ASK the
// original. `make_fixtures.py` ran the Python tokenizer on the
// same sentence and saved its ids. If yours differ, nothing
// after this point can match.
//
// STEP 2: IDS TO VECTORS. A table with one row of 64 numbers
// per token, `transformer.wte.weight`, and a gather. That is
// `zml.nn.TokenEmbedding`, exercise 093, unchanged.
//
// STEP 3: WHERE AM I? Exercise 107 began with an embarrassing
// fact: attention does not know who stands where. The llamas
// fixed it by spinning (RoPE). Neo is older, and does what
// GPT-2 does: a SECOND table, `transformer.wpe.weight`, with
// one row of 64 numbers per POSITION. Row 0 for the first
// token of the text, row 1 for the second... The rows were
// learned, like everything else, and they are simply added:
//
//     position_ids = torch.arange(inputs_embeds.shape[1])
//     position_embeds = self.wpe(position_ids)
//     hidden_states = inputs_embeds + position_embeds
//
// `arange` then a lookup: rows 0, 1, 2... n-1, in order. You
// do not need a gather for that. It is the first n rows of the
// table: a `slice` (exercise 054) along the position axis,
// whose result must then be called `.s` to be added to the
// tokens.
//
// Two things follow from a table of positions. The table has
// `max_position_embeddings` rows (2048 here) and not one more:
// that is the longest text Neo can read, a hard limit that a
// spinning llama does not have. And a port must use the SAME
// numbering as the original: position 0 first, no offset.
// (Some models start at 2. Really. Read the code.)
//
// HOW DO YOU KNOW IT IS RIGHT? This is the method of the whole
// chapter, and of every port you will ever write. The script
// ran the real PyTorch model on
//
//     "Once upon a time, there was a little dragon"
//
// and saved what every module received and returned in
// `activations.safetensors`, under the module's own name:
//
//     input_ids               {10} i32      what the tokenizer said
//     transformer.wte.out     {10,64} f32   the token vectors
//     transformer.wpe.out     {10,64} f32   the position vectors
//     transformer.drop.in     {10,64} f32   their sum
//
// (`drop` is a dropout layer, which does nothing outside of
// training. Its input is the cheapest place to catch the sum.)
//
// You compute the same thing, and compare:
//
//     zml.testing.expectClose(io, want, got, .{})  -> !void
//
// It takes two Buffers, returns an error when they are not the
// same up to rounding (by default: 1e-3, or 1% of the value,
// for 99.9% of the numbers), and on failure prints a small
// report on stderr: the worst error, the average error. Read
// that report when a check says false: an error of 1e-2 and an
// error of 40 are not the same kind of bug.
//
// The helpers `bufferFromStore` (one tensor of a file, as a
// Buffer) and `matches` (expectClose, as a bool) are given,
// and so is `readFile`. The plumbing of `main` is given too,
// this once.
//
// To do:
//
//  1. `Embeddings.init`: the two tables. The view stands on
//     "transformer.". Tag the tokens table `.{ .voc, .d }` and
//     the positions table `.{ .pos, .d }`.
//
//  2. `tokens`, `positions` and `forward`.
//
//  3. In `main`, the program runs and says "false" four times.
//     One line is responsible for all four.
//
//  4. In `main`, the third check has no Exe and no run.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/19_once_upon_a_time/";

const prompt = "Once upon a time, there was a little dragon";

/// `bos_token_id` in config.json.
const bos_token_id: u32 = 50256;

const Embeddings = struct {
    wte: zml.nn.TokenEmbedding,
    wpe: zml.Tensor,

    /// `view` must stand on "transformer.".
    pub fn init(view: zml.io.TensorStore.View) Embeddings {
        return ???;
    }

    /// ids: {.s} i32 -> one vector per token: {.s, .d}
    pub fn tokens(self: Embeddings, ids: zml.Tensor) zml.Tensor {
        return self.wte.forward(ids);
    }

    /// ids: {.s} i32 -> one vector per position: {.s, .d}
    pub fn positions(self: Embeddings, ids: zml.Tensor) zml.Tensor {
        return ???;
    }

    /// ids: {.s} i32 -> what the first block receives: {.s, .d}
    pub fn forward(self: Embeddings, ids: zml.Tensor) zml.Tensor {
        return ???;
    }
};

fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// The tensor `key` of a safetensors file, as a Buffer.
fn bufferFromStore(gpa: std.mem.Allocator, io: std.Io, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);
    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);
    return .fromBytes(io, platform, shape, bytes);
}

/// Are the two Buffers the same, up to rounding? When they are
/// not, `expectClose` says how far apart on stderr.
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

    // Two files: the weights, and what PyTorch computed with them.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const reference_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(reference_path);
    var reference_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, reference_path);
    defer reference_registry.deinit();
    var reference: zml.io.TensorStore = .fromRegistry(gpa, &reference_registry, &.{platform.replicated_mesh});
    defer reference.deinit();

    // The text, as token ids.
    const tokenizer_bytes = try readFile(gpa, io, dir ++ "tokenizer.json");
    defer gpa.free(tokenizer_bytes);
    var tokenizer: zml.tokenizer.Tokenizer = try .fromBytes(gpa, tokenizer_bytes);
    defer tokenizer.deinit();
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    const encoded = try encoder.encodeAlloc(gpa, prompt);
    defer gpa.free(encoded);
    // Every text begins with the beginning-of-sequence token.
    try ids.append(gpa, bos_token_id);
    try ids.appendSlice(gpa, encoded);

    try stdout.print("\"{s}\"\n", .{prompt});
    try stdout.print("{d} tokens: {any}\n", .{ ids.items.len, ids.items });
    try stdout.print("pieces: ", .{});
    for (ids.items) |id| try stdout.print("[{s}]", .{tokenizer.decode(id).?});
    try stdout.print("\n", .{});

    // (A token id is a u32 for the tokenizer and an i32 for the
    // model. Same four bytes.)
    const ids_tensor: zml.Tensor = .init(.{ .s = ids.items.len }, .i32);
    var ids_buffer: zml.Buffer = try .fromBytes(io, platform, ids_tensor.shape(), std.mem.sliceAsBytes(ids.items));
    defer ids_buffer.deinit();

    {
        var want = try bufferFromStore(gpa, io, platform, &reference, "input_ids");
        defer want.deinit();
        try stdout.print("same ids as PyTorch: {}\n", .{matches(io, want, ids_buffer)});
    }

    // The model and its weights.
    const model: Embeddings = .init(store.view().withPrefix("transformer"));
    try stdout.print("the position table: {f}\n", .{model.wpe.shape()});

    var weights = try zml.mem.bufferize(gpa, Embeddings, &model);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Embeddings, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Embeddings, &model, &weights, &store, .{});
    try loader.await(io);

    {
        var exe = try platform.compile(gpa, io, model, .tokens, .{ids_tensor}, .{});
        defer exe.deinit();
        var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, ids_buffer });
        defer got.deinit();
        var want = try bufferFromStore(gpa, io, platform, &reference, "transformer.wte.out");
        defer want.deinit();
        try stdout.print("token vectors match PyTorch: {}\n", .{matches(io, want, got)});
    }

    {
        var exe = try platform.compile(gpa, io, model, .positions, .{ids_tensor}, .{});
        defer exe.deinit();
        var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, ids_buffer });
        defer got.deinit();
        var want = try bufferFromStore(gpa, io, platform, &reference, "transformer.wpe.out");
        defer want.deinit();
        try stdout.print("position vectors match PyTorch: {}\n", .{matches(io, want, got)});
    }

    {
        // `got`: the model's `forward` on the ids. `want`: the
        // reference tensor "transformer.drop.in".
        ???;
        try stdout.print("what block 0 receives matches PyTorch: {}\n", .{matches(io, want, got)});
    }
    try stdout.flush();
}
