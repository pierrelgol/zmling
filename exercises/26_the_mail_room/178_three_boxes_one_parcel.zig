//
// The overseas parcel arrives in three boxes, with a sheet of
// paper taped on the first one. Priya signs for it, opens the
// biggest box, and reports: "three tensors, boss."
//
// Gus does not look up. "There are five."
//
// Welcome to the SHARDED checkpoint. One `model.safetensors`
// is fine for the gym's 968 bytes. A real model weighs 16, 80,
// 400 GB, and nobody wants one file that size: a download that
// fails at 97% starts over, and a file cannot be fetched by
// eight connections as easily as eight files can. So the
// libraries cut a checkpoint in pieces of a few GB:
//
//     model-00001-of-00003.safetensors
//     model-00002-of-00003.safetensors
//     model-00003-of-00003.safetensors
//     model.safetensors.index.json
//
// Each piece is a complete, ordinary safetensors file
// (exercise 097): its own header, its own offsets. A tensor is
// never cut in two. And nothing in a piece says what the other
// pieces hold. That is the job of the sheet of paper, the
// INDEX, a small JSON file:
//
//     {
//       "metadata": { "total_size": 528 },
//       "weight_map": {
//         "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
//         "model.layers.0.fc.weight":  "model-00002-of-00003.safetensors",
//         ...
//       }
//     }
//
// `weight_map`: for every tensor, the file it is in.
// `total_size`: the bytes of all tensors together (no headers).
//
// In Zig, a JSON object whose KEYS are data, not field names,
// is a `std.json.ArrayHashMap(T)`: a struct with one field,
// `.map`, and on it `.keys()`, `.values()` (two slices, same
// order as the file) and `.count()`.
//
// Now the ZML side, all of it in `zml/core/safetensors.zig`:
//
//     zml.safetensors.resolveModelRepo(io, path)   !std.Io.Dir
//         `path` is a directory: opens it. `path` ends in
//         `.safetensors` or `.safetensors.index.json`: opens
//         the directory that file is in. Errors: FileNotFound,
//         InvalidPath. The Dir is yours to `close(io)`.
//
//     zml.safetensors.resolveModelEntrypoint(io, repo)
//         !struct { zml.safetensors.FileType, std.Io.File }
//         Looks in the directory for
//         `model.safetensors.index.json`, then for
//         `model.safetensors`, in that order (and for a few
//         other spellings: `consolidated.*`, `*.full_index.json`),
//         and returns WHAT it found, `.index` or
//         `.safetensors`, and the open file. Neither:
//         `error.FileNotFound`. The File is yours to close too.
//
//     zml.safetensors.TensorRegistry.fromRepo(gpa, io, repo)
//         The two together: finds the entry point; if it is an
//         index, opens every file named in `weight_map` and
//         reads its header into ONE registry. This is the call
//         `examples/llm/main.zig` makes.
//
// After that, the pieces are gone from sight. The registry has
// every name, and each entry remembers where its bytes are:
//
//     const entry = registry.tensors.get(name).?;   // or .values()
//     entry.shape       zml.Shape, dims and dtype
//     entry.file_uri    the absolute path of ITS piece
//     entry.offset      where its bytes start in that piece
//     registry.totalBytes()    bytes of all tensors
//
// Store, View, Loader: nothing changes. They ask the entry.
//
// And `fromPath`, the one you have used since exercise 098? It
// is the same machine with one more door:
//
//  - given a directory, it behaves like `fromRepo`;
//  - given a file (an absolute path, as in 098), it
//    reads THAT FILE. An index: the whole parcel. One piece:
//    one piece, and no complaint, because a piece is a valid
//    safetensors file. That was Priya.
//
// Nothing fails until a model asks for a tensor from another
// box, and then the message, `Checkpoint has no tensor named
// lm_head.weight`, blames the checkpoint. So: hand ZML the
// directory, and count before you build.
//
// To do:
//
//  1. `shardName`, `tensorsIn`: the index, read by hand. The
//     numbers in a piece's name are padded to five digits
//     (`{d:0>5}` in a format string).
//  2. `summary`: repo, entry point, registry. Everything it
//     opens, it closes.
//  3. `missing`: how many of these names the registry lacks.
//  4. The overseas registry is still Priya's: three tensors
//     on the list, and a conveyor could not be built from it.
//  5. `countLayers` is exercise 141's, word for word, and it
//     finds no layer at all. Look at the names on the list.
//
const std = @import("std");
const zml = @import("zml");

const View = zml.io.TensorStore.View;

const parcels = "zmling/26_the_mail_room/parcels/";

const Index = struct {
    metadata: struct { total_size: u64 },
    weight_map: std.json.ArrayHashMap([]const u8),
};

/// What a two-layer conveyor asks of a checkpoint.
const conveyor_needs = [_][]const u8{
    "model.embed_tokens.weight",
    "model.layers.0.fc.weight",
    "model.layers.1.fc.weight",
    "lm_head.weight",
};

/// "model-00002-of-00003.safetensors" for (2, 3).
fn shardName(buffer: []u8, number: usize, total: usize) ![]const u8 {
    return ???;
}

/// How many tensors the index puts in the file `shard`.
fn tensorsIn(index: Index, shard: []const u8) usize {
    return ???;
}

/// Prints "<kind> entry point, <n> tensors, <n> bytes" for the
/// repository at `path`.
fn summary(gpa: std.mem.Allocator, io: std.Io, stdout: *std.Io.Writer, path: []const u8) !void {
    // The repository, `entry_point` (a FileType), and `registry`.
    ???

    try stdout.print("{s} entry point, {d} tensors, {d} bytes\n", .{
        @tagName(entry_point),
        registry.tensors.count(),
        registry.totalBytes(),
    });
}

fn countLayers(view: View) u32 {
    var n: u32 = 0;
    while (view.withPrefix("layers").withLayer(n).hasKey("")) n += 1;
    return n;
}

fn missing(registry: *const zml.safetensors.TensorRegistry, names: []const []const u8) usize {
    return ???;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    var repo = try zml.safetensors.resolveModelRepo(io, parcels ++ "overseas");
    defer repo.close(io);

    // The sheet of paper, by hand.
    {
        const file = try repo.openFile(io, "model.safetensors.index.json", .{});
        defer file.close(io);
        var file_buffer: [256]u8 = undefined;
        var file_reader = file.reader(io, &file_buffer);
        var reader: std.json.Reader = .init(gpa, &file_reader.interface);
        defer reader.deinit();
        const parsed = try std.json.parseFromTokenSource(Index, gpa, &reader, .{});
        defer parsed.deinit();
        const index = parsed.value;

        try stdout.print("the index of overseas, by hand:\n", .{});
        try stdout.print("  {d} tensors, {d} bytes\n", .{ index.weight_map.map.count(), index.metadata.total_size });
        for (1..4) |number| {
            var name_buffer: [64]u8 = undefined;
            const shard = try shardName(&name_buffer, number, 3);
            try stdout.print("  {s}: {d}\n", .{ shard, tensorsIn(index, shard) });
        }
    }

    // Every parcel on the shelf, and one thing that is not a parcel.
    try stdout.print("\n", .{});
    for ([_][]const u8{ "floor14", "overseas", "annex", "mop_closet" }) |name| {
        var path_buffer: [128]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buffer, "{s}{s}", .{ parcels, name });
        try stdout.print("{s}: ", .{name});
        summary(gpa, io, stdout, path) catch |err| try stdout.print("{}\n", .{err});
    }
    // A directory that exists, with no model in it.
    try stdout.print("the shelf itself: ", .{});
    summary(gpa, io, stdout, parcels) catch |err| try stdout.print("{}\n", .{err});
    try stdout.flush();

    // Priya: "the big box is number 2, so I opened that one."
    const box = try repo.realPathFileAlloc(io, "model-00002-of-00003.safetensors", gpa);
    defer gpa.free(box);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, box);
    defer registry.deinit();

    try stdout.print("\noverseas, tensor by tensor:\n", .{});
    for (registry.tensors.values()) |entry| {
        try stdout.print("  {s} {f} in {s}\n", .{ entry.name, entry.shape, std.fs.path.basename(entry.file_uri) });
    }

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    try stdout.print("layers: {d}\n", .{countLayers(store.view())});
    try stdout.print("missing for a conveyor: {d}\n", .{missing(&registry, &conveyor_needs)});
    try stdout.flush();
}
