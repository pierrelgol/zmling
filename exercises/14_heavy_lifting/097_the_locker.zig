//
// Chapter 14: Heavy Lifting
//
// Chef Babette wrote you a kind reference ("he never dropped a
// cake that mattered"), and so here you are at the IRON TENSOR
// GYM, home of Coach Magnus, who communicates exclusively in
// capital letters, and of Denise, who runs the front desk and
// therefore everything else.
//
// A gym is a place where people come to lift WEIGHTS. So is a
// neural network. In the bakery you typed every weight yourself,
// as Zig arrays, a dozen numbers at a time. A real model has a
// few BILLION of them, and nobody types those: they come out of
// training, they get saved in a file, and your program's first
// job is to carry them from the disk to the device. That's the
// heavy lifting, and that's this chapter.
//
// At the Iron Tensor, weights are kept in lockers. The lockers
// are files, and their format is called SAFETENSORS (the same
// one you download from Hugging Face). It is "safe" because it
// is too simple to hide anything in. The whole specification:
//
//     +----------+----------------------+---------------------+
//     | 8 bytes  | N bytes              | everything else     |
//     | N        | header (JSON text)   | raw tensor data     |
//     +----------+----------------------+---------------------+
//
//  1. Eight bytes: one unsigned 64-bit integer, LITTLE-ENDIAN.
//     It is N, the length of the header.
//
//  2. N bytes of JSON. For each tensor, its name, and:
//
//         "plates":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}
//
//     `data_offsets` is [begin, end): where the bytes of this
//     tensor sit, counted FROM THE END OF THE HEADER, not from
//     the start of the file. One special key, "__metadata__",
//     holds free-form text about the file instead of a tensor.
//
//  3. The numbers of all tensors, back to back, exactly as they
//     will sit in memory. No compression, no encoding.
//
// That design is the whole point. To know everything ABOUT a
// model (names, shapes, dtypes) you read a few kilobytes at the
// front. And when you do want the numbers, each tensor is one
// seek and one read, straight into its final resting place.
//
// Before ZML does it for us, let's open a locker with our bare
// hands. No `zml` in this exercise: only `std`.
//
// The locker is `gym.safetensors`, 968 bytes, and it sits next
// to this file. (`make_fixtures.py`, same directory, is the
// script that wrote it: thirty lines of Python, worth a look.)
//
// How does the program find it? Bazel runs every exercise from
// a private directory, its "runfiles" tree, which mirrors the
// ZML workspace and contains only what the BUILD.bazel file of
// this chapter declared as `data`. In that tree, this chapter
// is `zmling/14_heavy_lifting/`. Hence the path below: it is
// relative to the directory the program is started in.
//
// The two Zig calls you need, on a `std.Io.File`:
//
//     file.readPositionalAll(io, buffer, offset)
//         fills `buffer` with the bytes found at `offset`
//
//     std.mem.readInt(u64, &eight_bytes, .little)    (or .big)
//         turns 8 bytes into an integer
//
// Denise hands you the key to locker number one. "Somebody put
// the plates away in there. Tell me what we've got. And while
// you're in there, read me the combination of locker 3, I've
// lost the sticky note again."
//
// Four steps, each standing on the previous one:
//
//  1. the length of the header, from the first eight bytes;
//  2. the header itself. It starts right after those eight
//     bytes. It gets printed, so that you can READ it: the next
//     steps are written in there;
//  3. `dataOffset`: where a `data_offsets` position really is,
//     counted from the start of the file. Then the plates;
//  4. the combination, all yours: find its entry in the header
//     you printed, declare the right array for its dtype and
//     shape, and read it the way the plates were read.
//
const std = @import("std");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

/// Where byte `begin` of the tensor data sits, counted from the
/// start of the file.
fn dataOffset(header_length: usize, begin: usize) usize {
    return ???;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const file = try std.Io.Dir.cwd().openFile(io, weights_path, .{});
    defer file.close(io);

    // 1. The first eight bytes are the length of the header.
    // TODO: `header_length`, a usize. Read the eight bytes, then
    // turn them into an integer (`@intCast` makes a usize of a u64).

    // 2. The header comes right after them.
    var header_bytes: [2048]u8 = undefined;
    const header = header_bytes[0..header_length];
    _ = try file.readPositionalAll(io, header, ???);

    try stdout.print("The header is {d} bytes of JSON:\n", .{header.len});
    // (Writers pad the header with spaces. Nobody needs to see those.)
    try stdout.print("{s}\n", .{std.mem.trimEnd(u8, header, " ")});

    // 3. The header says: "plates" is 4 F32, data_offsets [0, 16].
    //    Read the 16 bytes straight into four floats: the file holds
    //    exactly the bytes an [4]f32 is made of. (On a little-endian
    //    machine, which is to say: on yours.)
    var plates: [4]f32 = undefined;
    _ = try file.readPositionalAll(io, std.mem.asBytes(&plates), 0);

    try stdout.print("\nThe plates, in kg: {any}\n", .{plates});

    // 4. The combination of locker 3 is in here too.
    // TODO: `combination`, read from the file.

    try stdout.print("The combination of locker 3: {any}\n", .{combination});
    try stdout.flush();
}
