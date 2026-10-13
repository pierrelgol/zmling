//
// You opened a locker by hand: eight bytes, some JSON, a seek.
// Good for the soul, like carrying your own plates. Now meet the
// person who has done it ten thousand times: Denise.
//
// Denise runs the front desk. She has never lifted anything
// heavier than the logbook, and that is the point: she knows
// WHAT is in every locker and WHERE, without ever touching a
// single plate. In ZML, Denise is spelled
//
//     zml.safetensors.TensorRegistry
//
// A registry is the header of a safetensors file, parsed. Names,
// shapes, dtypes, positions in the file: the METADATA. Not one
// number of any tensor is read. Building the registry of a
// 70 GB model takes milliseconds and a few kilobytes of RAM.
// (Which lets a program check that a model is the one it expects
// BEFORE it spends five minutes loading it.)
//
//     var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
//     defer registry.deinit();
//
// What you can ask her, straight from zml/safetensors.zig:
//
//     registry.tensors            a std.StringArrayHashMap: name -> Tensor,
//                                 in the order of the file
//     registry.tensors.count()    how many
//     registry.tensors.get(name)  ?Tensor: one entry, if it exists
//     registry.iterator()         all of them, one by one
//     registry.totalBytes()       the size of all the data, in bytes
//     registry.metadata           the "__metadata__" of the file
//
// Careful: the "Tensor" in there is NOT our ghost `zml.Tensor`.
// It is `zml.safetensors.Tensor`, a plain logbook entry:
//
//     pub const Tensor = struct {
//         file_uri: []const u8,   // which file it lives in
//         name: []const u8,
//         shape: Shape,           // a zml.Shape! dims + dtype
//         offset: u64,            // where its bytes start in that file
//
//         pub fn byteSize(self: Tensor) u64
//     };
//
// (One registry can cover SEVERAL files. Big models are split
// in pieces of a few GB, plus a `model.safetensors.index.json`
// saying which tensor is in which piece. Hence `file_uri`.)
//
// The iterator is the one of Zig's hash maps:
//
//     var it = registry.iterator();
//     while (it.next()) |entry| {
//         entry.key_ptr.*      // the name, a []const u8
//         entry.value_ptr.*    // the logbook entry
//     }
//
// ONE QUIRK, and you will meet it within the minute.
//
// `fromPath` wants an ABSOLUTE path. With a relative one it
// opens the directory part of the path, then looks for the
// whole path again INSIDE that directory, and reports a
// `FileNotFound` about a file that is sitting right there.
// (An absolute path means the same thing wherever you stand,
// so it survives the detour.) Zig will make one for you:
//
//     const path = try std.Io.Dir.cwd().realPathFileAlloc(io, relative_path, gpa);
//     defer gpa.free(path);
//
// "Real path" also means that symbolic links are followed. The
// runfiles tree of Bazel is made of links, so what comes back is
// the true address of the locker: the file next to this one, in
// your checkout. It contains your home directory, which is why
// we won't print it.
//
// Denise opens the logbook. "Read it back to me. Slowly. And
// find the real address first, the new kid never does."
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, weights_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, weights_path);
    defer registry.deinit();

    try stdout.print("{d} tensors, {d} bytes of weights.\n\n", .{
        registry.tensors.count(),
        // The size of all the tensor data. Denise knows the total.
        ???,
    });

    try stdout.print("name                 shape       bytes   at\n", .{});
    var it = registry.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const locker = entry.value_ptr.*;
        // A Shape is too proud to be padded by `print`, so we
        // format it into a small buffer first.
        var shape_buffer: [32]u8 = undefined;
        const shape = try std.fmt.bufPrint(&shape_buffer, "{f}", .{locker.shape});
        try stdout.print("{s:<20} {s:<11} {d:>5}   {d}\n", .{ name, shape, locker.byteSize(), locker.offset });
    }

    // In exercise 097 you computed where the plates start. Does
    // Denise agree?
    const plates = registry.tensors.get("plates").?;
    try stdout.print("\nThe plates start at byte {d} of the file.\n", .{plates.offset});
    try stdout.flush();
}
