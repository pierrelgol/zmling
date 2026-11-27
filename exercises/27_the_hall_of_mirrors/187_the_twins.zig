//
// At the end of the corridor stand the Twins: two mirrors of
// the same size, in the same gilt frame, bought the same day.
// Nobody can tell them apart. The removal men can: each has a
// number chalked on its back.
//
// So does every Tensor. Open `zml/zml/core/tensor.zig`:
//
//     pub const Tensor = struct {
//         pub const Id = enum(u64) { _ };
//         id: Tensor.Id,
//         _shape: Shape,
//         _value: ?*const mlir.Value = null,
//         ...
//
// A placeholder is a shape and a NUMBER, taken from a counter
// the day it is made: by `Tensor.init`, by `Tensor.fromShape`,
// by a `createTensor` of a store view. (The result of an
// operation gets one too.) Two placeholders of the same shape
// are two different tensors, and the `id` is how ZML knows.
//
// Three machines are keyed on it.
//
// THE COMPILER. While it declares the inputs of a program
// (`createBlockArguments`, exercise 184), it files each of
// them under its id. Later, when `forward` uses `self.glass`,
// the id is how it finds which input that was. And if the walk
// meets the same id TWICE, you get the panic of exercise 015:
//
//     Tensor with id ... has already been used once as an
//     argument
//
// which you can now read properly: not "the same variable",
// the same NUMBER, wherever the two copies sit in your struct.
//
// THE STORE. `view.createTensor("weight", ...)` does two
// things: it makes a placeholder with the shape found in the
// file, and it writes in the store "id 7 is `bar.0.weight`".
// You can ask:
//
//     store.getSourcesById(t.id)    ?TensorStore.Binding
//
// A Binding has `.tensors`, a slice of the entries of the file
// it comes from: one, for everything you have loaded so far,
// and `binding.tensors[0].name` is the key.
//
// THE LOADER. `loader.load(io, Model, &model, &buffers, ...)`
// walks the model and its Bufferized twin side by side, and
// for each pair asks the store what to pour: BY THE ID OF THE
// TENSOR. And when the store has never heard of that id, the
// Loader says nothing and moves on. The crate stays empty,
// and you find out at the first `eval`:
//
//     Argument 3 has 0 shards but executable expects 4
//
// ("Shards" are the pieces of a Buffer, one per device. A
// Buffer with none holds no numbers.)
//
// So, what keeps an id and what does not?
//
//   - COPYING a Tensor copies its id. `const b = a;` is the
//     same tensor, and so is `a.withTags(...)` or `a.rename(...)`:
//     they return a copy with another label on the axes.
//   - `Tensor.fromShape(a.shape())` is a NEW tensor that looks
//     like `a`. The store has never heard of it.
//   - Asking the view for the same key twice gives two ids,
//     both bound: two tensors, and the file is read twice.
//
// And that is the whole secret of TIED WEIGHTS (exercise 126):
// the same Tensor used at two places of `forward` is ONE input
// of the Exe, loaded once. But it must sit in ONE field of the
// model. Put a copy in a second field and the walk meets its
// id twice.
//
// The hall bought its glass second-hand, from a gym that
// closed down: the smoothie bar of chapter 14, `bar.0`,
// `bar.1`, `bar.2` in `gym.safetensors`, each a 2x2 `weight`
// and a `bias`. Ottoline hangs `bar.0` at the entrance and
// `bar.1` in the middle. The way out is THE SAME MIRROR as the
// way in, seen again: tied. `bar.2` stays in its crate.
//
//  1. Open the store: a registry from `path`, then a store.
//     Exercise 099.
//
//  2. `source`: the key of the file a tensor is bound to, or
//     "nothing".
//
//  3. `mirror` wanted its frame tagged `.dout`, and went about
//     it in a way that looks harmless.
//
//  4. `Hall` holds the entrance twice. One field must go, and
//     `forward` must do the tying.
//
//  5. The Loader: exercise 100. `weights` is waiting.
//
// `main` prints who is bound to what BEFORE it compiles
// anything, and flushes: when a panic comes, look up.
//
// A visitor { 3, 1 } comes out as { 13, 1 }.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

const Mirror = struct {
    glass: zml.Tensor,
    frame: zml.Tensor,

    pub fn forward(self: Mirror, x: zml.Tensor) zml.Tensor {
        return self.glass.dot(x, .d).add(self.frame).rename(.{ .dout = .d });
    }
};

/// `view` stands on one blender of the bar.
fn mirror(view: zml.io.TensorStore.View) Mirror {
    const frame = view.createReplicatedTensor("bias", null);
    return .{
        .glass = view.createReplicatedTensor("weight", .{ .dout, .d }),
        .frame = .fromShape(frame.shape().withTags(.{.dout})),
    };
}

const Hall = struct {
    entrance: Mirror,
    middle: Mirror,
    way_out: Mirror,

    pub fn init(view: zml.io.TensorStore.View) Hall {
        const entrance = mirror(view.withLayer(0));
        return .{
            .entrance = entrance,
            .middle = mirror(view.withLayer(1)),
            .way_out = entrance,
        };
    }

    /// In by the entrance, through the middle, out by the entrance.
    pub fn forward(self: Hall, visitor: zml.Tensor) zml.Tensor {
        return self.way_out.forward(self.middle.forward(self.entrance.forward(visitor)));
    }
};

/// The key of the file `t` is bound to, or "nothing".
fn source(store: *const zml.io.TensorStore, t: zml.Tensor) []const u8 {
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

    // `store`: a registry first, then the store itself.
    ???;

    const view = store.view().withPrefix("bar");

    // The Twins: the same key, asked for twice.
    const left = view.createReplicatedTensor("2.weight", .{ .dout, .d });
    const right = view.createReplicatedTensor("2.weight", .{ .dout, .d });
    const copy = left;
    const lookalike: zml.Tensor = .fromShape(left.shape());

    try stdout.print("The Twins have the same shape: {}, the same id: {}\n", .{ left.shape().eql(right.shape()), left.id == right.id });
    try stdout.print("  left <- {s}, right <- {s}\n", .{ source(&store, left), source(&store, right) });
    try stdout.print("A copy of the left one has its id: {}, and <- {s}\n", .{ copy.id == left.id, source(&store, copy) });
    try stdout.print("A lookalike has its id: {}, and <- {s}\n", .{ lookalike.id == left.id, source(&store, lookalike) });

    const hall: Hall = .init(view);

    try stdout.print("The hall:\n", .{});
    try stdout.print("  entrance.glass <- {s}\n", .{source(&store, hall.entrance.glass)});
    try stdout.print("  entrance.frame <- {s}\n", .{source(&store, hall.entrance.frame)});
    try stdout.print("  middle.glass <- {s}\n", .{source(&store, hall.middle.glass)});
    try stdout.print("  middle.frame <- {s}\n", .{source(&store, hall.middle.frame)});
    try stdout.flush();

    const visitor: zml.Tensor = .init(.{ .d = 2 }, .f32);
    var exe = try platform.compile(gpa, io, hall, .forward, .{visitor}, .{});
    defer exe.deinit();
    try stdout.print("Three mirrors on the way, {d} slots.\n", .{exe.input_shapes.len});

    var weights = try zml.mem.bufferize(gpa, Hall, &hall);
    defer zml.Buffer.deinitAll(Hall, &weights);

    // Fill `weights` from the store.
    ???;

    var guest: zml.Buffer = try .fromBytes(io, platform, visitor.shape(), std.mem.sliceAsBytes(&[_]f32{ 3, 1 }));
    defer guest.deinit();

    var out: zml.Buffer = try exe.eval(gpa, io, .{ weights, guest });
    defer out.deinit();
    try stdout.print("A visitor {{ 3, 1 }} comes out as {any}\n", .{try out.getValue([2]f32, io)});
    try stdout.flush();
}
