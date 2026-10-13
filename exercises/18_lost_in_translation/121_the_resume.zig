//
// Welcome to the interpreters' booth of the Summit of Very Small
// Nations. Madame Vogel runs it, and she has just hired someone
// from another agency: Bertie. He is small (two layers of
// thought), he is real (people know him as
// `prajjwal1/bert-tiny`, a genuine pretrained BERT), and until
// last week he worked in PyTorch. Your job, for this whole
// chapter: get him working here, in ZML. That is called PORTING
// a model.
//
// Bertie arrived with three files, like every model you will
// ever download:
//
//     config.json         his resume: how he is built
//     tokenizer.json      his glossary: text <-> token ids
//     model.safetensors   his memory: the weights
//
// and one that Madame Vogel asked his old agency for:
//
//     activations.safetensors   what he thought, step by step,
//                               on one sentence, over there
//
// Start with the resume. Here it is, whole:
//
//     {"hidden_size": 128, "hidden_act": "gelu",
//      "initializer_range": 0.02, "vocab_size": 30522,
//      "hidden_dropout_prob": 0.1, "num_attention_heads": 2,
//      "type_vocab_size": 2, "max_position_embeddings": 512,
//      "num_hidden_layers": 2, "intermediate_size": 512,
//      "attention_probs_dropout_prob": 0.1}
//
// Why bother, when you could type `128` and `2` in your code?
// Because there is not one BERT, there are hundreds: tiny,
// mini, base, large, all the same code with different numbers.
// The Python reads the numbers from this file:
//
//     self.query = nn.Linear(config.hidden_size, self.all_head_size)
//
// and so must you. PORTING RULE 1: no number of the
// architecture is typed in the port. They all come from the
// config, and a port written that way runs every sibling of
// the model for free.
//
// Zig's standard library parses JSON straight into a struct:
//
//     const parsed = try std.json.parseFromSlice(
//         Config, gpa, bytes, .{ .ignore_unknown_fields = true });
//     defer parsed.deinit();
//     const config: Config = parsed.value;
//
// A field of the struct is filled from the key of the same
// name. And there are two ways for this to go wrong, both
// with a Zig error, both on purpose:
//
//  - a key of the file that is not in your struct is
//    `error.UnknownField`, unless you pass
//    `.ignore_unknown_fields = true`. You want to pass it:
//    half of a config.json is about training (the dropouts,
//    `initializer_range`), and a port only runs the model.
//
//  - a field of your struct that is not in the file is
//    `error.MissingField`, unless the field has a default
//    value. Look for "layer_norm_eps" in Bertie's resume. It is
//    not there, and the model needs it: every LayerNorm
//    divides by sqrt(variance + eps).
//
// So where is it? In the second half of every HuggingFace
// config: the config CLASS of the reference implementation.
// `configuration_bert.py` begins like this:
//
//     class BertConfig(PreTrainedConfig):
//         vocab_size: int = 30522
//         hidden_size: int = 768
//         num_hidden_layers: int = 12
//         ...
//         hidden_act: str = "gelu"
//         ...
//         layer_norm_eps: float = 1e-12
//
// A config.json only has to say where the model DIFFERS from
// these defaults (old ones say more, new ones say less). A key
// you can't find in the file has its value there, and nowhere
// else. Not in ZML: `zml.nn.LayerNorm` has its own default eps,
// 1e-5, which is PyTorch's, and not BERT's.
//
// PORTING RULE 2: the config is the json file PLUS the defaults
// of the config class. Copy the defaults you need into your
// struct, and say where they come from.
//
// (In strings, std.json may hand you a slice of `bytes` itself
// instead of a copy: keep the file contents alive as long as
// the config.)
//
// One more habit. A resume can lie, and a checkpoint cannot:
// the shapes in model.safetensors are what the model really
// is. Compare the two before you build anything. You know how
// to open a file of weights (exercises 97 to 99); `hasKey`
// only asks whether some name of the file STARTS with what you
// give it, which makes it a fine way to ask "is there a layer
// number 2?".
//
// The trainee who started this read the resume diagonally:
//
//  1. Three facts printed in `main` have no field in `Config`.
//
//  2. Run it, read the error, and make the parser more
//     tolerant. Run it again: another error. That one is fixed
//     in `Config`, not in `main`.
//
//  3. `headDim`: attention splits the hidden size evenly
//     between the heads (exercise 106).
//
//  4. In `main`, count the layers that are really in the file.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/18_lost_in_translation/";

const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    ???,
    layer_norm_eps: f32,

    /// How many numbers each attention head works with.
    pub fn headDim(self: Config) u32 {
        return ???;
    }
};

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

    const config_bytes = try readFile(gpa, io, dir ++ "config.json");
    defer gpa.free(config_bytes);

    const parsed = try std.json.parseFromSlice(Config, gpa, config_bytes, .{});
    defer parsed.deinit();
    const config: Config = parsed.value;

    try stdout.print("Bertie's resume:\n", .{});
    try stdout.print("  layers: {d}\n", .{config.num_hidden_layers});
    try stdout.print("  hidden size: {d}\n", .{config.hidden_size});
    try stdout.print("  heads: {d}, with {d} numbers each\n", .{ config.num_attention_heads, config.headDim() });
    try stdout.print("  intermediate size: {d}\n", .{config.intermediate_size});
    try stdout.print("  vocabulary: {d} tokens\n", .{config.vocab_size});
    try stdout.print("  longest sentence: {d} tokens\n", .{config.max_position_embeddings});
    try stdout.print("  activation: {s}\n", .{config.hidden_act});
    try stdout.print("  layer norm eps: {e}\n", .{config.layer_norm_eps});

    // Now the checkpoint: does it agree?
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();
    const view = store.view();

    const word_table = view.getShape("bert.embeddings.word_embeddings.weight").?;
    const expected: zml.Shape = .init(.{ config.vocab_size, config.hidden_size }, .f32);
    try stdout.print("The checkpoint:\n", .{});
    try stdout.print("  word table {f}, as announced: {}\n", .{ word_table, word_table.eql(expected) });

    // The layers are named bert.encoder.layer.0.(...),
    // bert.encoder.layer.1.(...), and so on.
    var layers: u32 = 0;
    ???;
    try stdout.print("  layers in the file: {d}, as announced: {}\n", .{ layers, layers == config.num_hidden_layers });
    try stdout.flush();
}
