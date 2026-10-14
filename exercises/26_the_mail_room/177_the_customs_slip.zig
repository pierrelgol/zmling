//
// Every parcel comes with a slip that says what is inside:
// `config.json`. You have read some since exercise 121, where
// you learned the two rules of the trade: pass
// `.ignore_unknown_fields = true` (half of a config is about
// training), and give a default to every field a config may
// leave out.
//
// Those configs were written by one script, for one model, and
// you had read them first. The mail room gets slips filled in
// by whichever version of `transformers` the sender had, for
// architectures that did not exist last year. Same form, three
// handwritings. Here are three slips for the same field:
//
//     "eos_token_id": 5            floor14
//     "eos_token_id": [6, 7]       overseas
//     "max_length": null           overseas, another field
//
// A NUMBER OR A LIST. Python does not care. Zig does: a field
// has one type. The honest type is a union, and std.json has
// no idea how to choose a tag by looking at the value: for a
// `union(enum)` it expects `{"int": 5}`, finds `5`, and
// returns `error.UnexpectedToken`. `zml.stdx.json` (file
// `stdx/json.zig`) has the missing piece:
//
//     const Answer = union(enum) {
//         number: i32,
//         numbers: []const i32,
//
//         const Helpers = zml.stdx.json.UnionHelpers(@This());
//         pub const jsonParse = Helpers.jsonParse;
//         pub const jsonParseFromValue = Helpers.jsonParseFromValue;
//         pub const jsonStringify = Helpers.jsonStringify;
//     };
//
// std.json looks for declarations with exactly these names in
// your type, and calls them instead of its own code. The
// helpers try the fields of the union in order and take the
// first whose Zig type fits the JSON value: an integer field
// for a JSON integer, a slice for an array, a bool, a float, a
// string, a struct for an object. So: one field per KIND of
// value. This is what the `Config` of
// `examples/llm/models/llama/model.zig` does for this very
// field. (`zml.stdx.json.Union(T)` is the older spelling: a
// wrapper struct with a `.value`. It is marked deprecated.)
//
// NULL. An optional with a default looks right and is not:
//
//     max_length: ?u32 = 20,
//
// A default is used when the key is ABSENT. Here the key is
// present, and its value is null: you get null, and the 20 is
// lost. Two tools, for two situations:
//
//     max_length: zml.stdx.json.NeverNull(u32, 20) = .{},
//
//   a struct with one field, `.value`, a plain u32. Absent
//   (thanks to the `= .{}`) or null, it is 20.
//
//     try zml.stdx.json.fillDefaultStructValues(T, &value);
//
//   for a struct you have already parsed: every field that is
//   null and has a default gets its default back. It compares
//   each defaulted field with null, so EVERY field with a
//   default must be optional, or it does not compile.
//   `generation_config.json`, the slip about sampling, is the
//   natural customer: it is mostly nulls.
//
// AN ENUM. std.json fills an enum field from a string, by tag
// name: `"torch_dtype": "float16"` becomes `.float16`. A name
// that is not a tag is `error.InvalidEnumTag`: good for a
// field with three known values, bad for `model_type`, where
// the unknown value is the interesting case. That one stays a
// string, and you convert it yourself:
//
//     std.meta.stringToEnum(ModelType, text)   ?ModelType
//
// which is how `detectModelType` in `examples/llm/models.zig`
// picks one of its four architectures, or says
// `error.UnknownModelType` and stops before anybody allocates
// 40 GB.
//
// READING IT. A program is handed a repository, not a file.
// `examples/llm/models/common.zig` opens the directory once
// and reads what it needs from there, without loading the
// whole text first: a `std.json.Reader` pulls tokens from the
// file as the parser asks for them.
//
//     var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
//     defer dir.close(io);
//
//     const file = try dir.openFile(io, "config.json", .{});
//     defer file.close(io);
//     var buffer: [256]u8 = undefined;
//     var file_reader = file.reader(io, &buffer);
//     var reader: std.json.Reader = .init(gpa, &file_reader.interface);
//     defer reader.deinit();
//     return try std.json.parseFromTokenSource(T, gpa, &reader, options);
//
// The result is the `std.json.Parsed(T)` you know: the value
// is in `.value`, and `deinit` frees its strings and slices.
//
// Gus: "Three parcels, three slips. I want one form, typed, and
// I want it to say the same thing whoever filled it in."
//
//  1. `parseJson`: the last three lines are yours, with the
//     option of exercise 121. Without it, the first slip is
//     `error.UnknownField`.
//  2. `EosTokens` cannot be parsed yet, and cannot answer
//     `contains`.
//  3. Two fields of `Config` are missing: `max_length` (20
//     when the slip says nothing, 20 when it says null) and
//     `torch_dtype`. The annex stores its weights in a third
//     way: look at its slip.
//  4. `modelType`.
//  5. The sampling slip prints two nulls. Gus wants numbers.
//
const std = @import("std");
const zml = @import("zml");

const json = zml.stdx.json;

const parcels = "zmling/26_the_mail_room/parcels/";

const stations = [_][]const u8{ "dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement" };

const ModelType = enum { chute, conveyor };

const EosTokens = union(enum) {
    int: u32,
    ints: []const u32,

    /// Does a route end at this station?
    pub fn contains(self: EosTokens, token: u32) bool {
        return ???;
    }
};

const Config = struct {
    model_type: []const u8,
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32 = 0,
    bos_token_id: u32,
    eos_token_id: EosTokens,
    tie_word_embeddings: bool = false,
    // max_length and torch_dtype go here.
};

/// The sampling slip. Everything in it may be null.
const Generation = struct {
    max_new_tokens: ?u32 = 8,
    top_k: ?u32 = 50,
    temperature: ?f32 = 1.0,
};

/// Reads `name` in the repository `repo` as a `T`.
fn parseJson(comptime T: type, gpa: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, name: []const u8) !std.json.Parsed(T) {
    const file = try repo.openFile(io, name, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    // A json reader on top of `file_reader`, and the parse.
    return ???;
}

fn modelType(config: Config) !ModelType {
    return ???;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    for ([_][]const u8{ "floor14", "overseas", "annex" }) |name| {
        var path_buffer: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buffer, "{s}{s}", .{ parcels, name });

        var repo = try std.Io.Dir.cwd().openDir(io, path, .{});
        defer repo.close(io);

        const parsed = try parseJson(Config, gpa, io, repo, "config.json");
        defer parsed.deinit();
        const config = parsed.value;

        try stdout.print("{s}\n", .{name});
        try stdout.print("  model type: {s}\n", .{@tagName(try modelType(config))});
        try stdout.print("  layers: {d}, tied head: {}, stored as {s}\n", .{
            config.num_hidden_layers,
            config.tie_word_embeddings,
            @tagName(config.torch_dtype),
        });
        try stdout.print("  a route starts at: {s}\n", .{stations[config.bos_token_id]});
        try stdout.print("  and ends at:", .{});
        for (stations, 0..) |station, id| {
            if (config.eos_token_id.contains(@intCast(id))) try stdout.print(" [{s}]", .{station});
        }
        try stdout.print("\n  or after {d} stations\n", .{config.max_length.value});
        try stdout.flush();
    }

    // A slip from the future.
    const tube = try std.json.parseFromSlice(Config, gpa,
        \\{ "model_type": "pneumatic_tube", "vocab_size": 8, "hidden_size": 8,
        \\  "bos_token_id": 0, "eos_token_id": 5 }
    , .{});
    defer tube.deinit();
    if (modelType(tube.value)) |model_type| {
        try stdout.print("{s}: we have that\n", .{@tagName(model_type)});
    } else |err| {
        try stdout.print("{s}: {}\n", .{ tube.value.model_type, err });
    }

    // The sampling slip of the overseas parcel.
    var repo = try std.Io.Dir.cwd().openDir(io, parcels ++ "overseas", .{});
    defer repo.close(io);
    const parsed = try parseJson(Generation, gpa, io, repo, "generation_config.json");
    defer parsed.deinit();
    const generation = parsed.value;
    try stdout.print("sampling: max_new_tokens {?d}, top_k {?d}, temperature {?d:.1}\n", .{
        generation.max_new_tokens,
        generation.top_k,
        generation.temperature,
    });
    try stdout.flush();
}
