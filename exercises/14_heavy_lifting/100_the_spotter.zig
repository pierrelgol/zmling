//
// You hold keys: placeholder Tensors that know which entry of
// the file they stand for. Time to actually LIFT something.
//
// Recall the two faces of a model, from the bakery:
//
//     Model                    structs full of zml.Tensor   (ghosts)
//     zml.Bufferized(Model)    the same structs, with every
//                              Tensor replaced by a zml.Buffer
//
// The first is what you compile. The second is what you run it
// with. Back then you filled the Bufferized by hand, one
// `Buffer.fromBytes` per field. Today the file does it, in two
// moves.
//
// MOVE ONE: set up the empty bar.
//
//     var buffers = try zml.mem.bufferize(gpa, Model, &model);
//
// `bufferize` walks through your model and returns the matching
// `zml.Bufferized(Model)`, where every Buffer has the right
// shape and NOTHING ELSE. No memory on the device, no numbers.
// A bar with no plates on it. Do not read these Buffers, and do
// not `deinit` them either: there is nothing in there yet.
// (The allocator is only used if your model contains slices,
// which a model with a variable number of layers would.)
//
// MOVE TWO: call the spotter.
//
//     var loader: zml.io.Loader = try .init(gpa, platform, .default);
//     defer loader.deinit();
//
//     try loader.load(io, Model, &model, &buffers, &store, .{});
//     try loader.await(io);
//
// Three calls, three jobs:
//
//  - `Loader.init` prepares the transfer machinery for a
//    platform. `.default` means: one tensor at a time, small
//    chunks. (The options are `parallelism`, `dma_chunks` and
//    `dma_chunk_size`; ZML's LLM example turns them up to load
//    several tensors at once, straight from disk to GPU.)
//
//  - `loader.load` walks through `model` and `buffers` side by
//    side. For each Tensor it asks the store "which entry is
//    this?" (by the id that `createReplicatedTensor` wrote
//    down), and STARTS a task that streams those bytes from the
//    file into device memory, and puts the result in the
//    matching Buffer. It returns at once. The last argument can
//    carry a progress bar; `.{}` is "no thanks".
//
//  - `loader.await` waits until every task is done. Before it
//    returns, your Buffers may be half there. After it, they
//    are complete, and they are YOURS: `deinit` each of them
//    when you're done, like any Buffer.
//
// Note what never happened: the weights were never a Zig array,
// never a `zml.Slice`, never "in your program" at all. File to
// device. For a 70 GB model on a machine with 32 GB of RAM,
// that is the difference between running and not running.
//
// And note what `load` needs to find its way: the SAME `model`
// value you got the keys with (it carries the ids), and the
// SAME `store` (it has the book).
//
// The rack below is not much of a model. It has no `forward`,
// it does nothing. But it has two Tensors with entries in the
// file, and that is all the Loader cares about.
//
// Coach Magnus points at the bar, then at you. "LOAD IT."
//
//  1. THE KEYS. Write `Rack.init`: the plates are the entry
//     "plates", tagged `.plate`; the combination is the entry
//     "locker_combination", tagged `.turn`. And in `main`, the
//     registry and the key cabinet: exercises 098 and 099, from
//     memory.
//
//  2. THE LIFT. Write `Rack.load`, the two moves above. Then
//     call it, and free what it gives you when you are done:
//     two Buffers, both yours.
//
//  3. THE POINT. A Buffer that came from a file is a Buffer like
//     any other, and the placeholder that came from the store is
//     a placeholder like any other. So: write `onTheBar` (the
//     diner and the reduction kitchen have all you need),
//     compile it for the plates of the rack, and run it on the
//     plates you loaded. Coach puts TWO of each plate on a bar
//     that weighs 20 kg by itself. He wants ONE number.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

const Rack = struct {
    plates: zml.Tensor,
    combination: zml.Tensor,

    pub fn init(view: zml.io.TensorStore.View) Rack {
        return ???;
    }

    /// Brings every tensor of the rack from the file to the device.
    /// The caller owns the Buffers.
    pub fn load(
        self: *const Rack,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Rack) {
        return ???;
    }
};

/// plates: {.plate}, in kg. The weight of the bar (20 kg) with two
/// of each plate on it: one number, no axis left.
fn onTheBar(plates: zml.Tensor) zml.Tensor {
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

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, weights_path, gpa);
    defer gpa.free(path);

    // TODO: `registry`, the logbook of the file at `path`, and
    // `store`, the key cabinet on top of it.

    const rack: Rack = .init(store.view());

    // TODO: `rack_buffers`, the rack, loaded. Free what you own.

    // Real Buffers, on the device. Bring them back to look at them.
    try stdout.print("plates      {f}: {any}\n", .{
        rack_buffers.plates.shape(),
        try rack_buffers.plates.getValue([4]f32, io),
    });
    try stdout.print("combination {f}: {any}\n", .{
        rack_buffers.combination.shape(),
        try rack_buffers.combination.getValue([3]i32, io),
    });

    // TODO: `onTheBar`, compiled for the plates of the rack, and
    // run on the plates you loaded. The result goes in `total`.

    try stdout.print("on the bar  {f}: {d} kg\n", .{ total.shape(), try total.getValue(f32, io) });
    try stdout.flush();
}
