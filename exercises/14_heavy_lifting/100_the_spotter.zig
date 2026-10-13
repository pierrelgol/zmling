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
// Write the body of `load`. Everything you need is above, and
// the rest of the program already knows what to do with the
// result.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

const Rack = struct {
    plates: zml.Tensor,
    combination: zml.Tensor,

    pub fn init(view: zml.io.TensorStore.View) Rack {
        return .{
            .plates = view.createReplicatedTensor("plates", .{.plate}),
            .combination = view.createReplicatedTensor("locker_combination", .{.turn}),
        };
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

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const rack: Rack = .init(store.view());

    var rack_buffers: zml.Bufferized(Rack) = try rack.load(gpa, io, platform, &store);
    defer rack_buffers.plates.deinit();
    defer rack_buffers.combination.deinit();

    // Real Buffers, on the device. Bring them back to look at them.
    try stdout.print("plates      {f}: {any}\n", .{
        rack_buffers.plates.shape(),
        try rack_buffers.plates.getValue([4]f32, io),
    });
    try stdout.print("combination {f}: {any}\n", .{
        rack_buffers.combination.shape(),
        try rack_buffers.combination.getValue([3]i32, io),
    });
    try stdout.flush();
}
