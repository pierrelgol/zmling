//
// Denise knows what is in every locker. Knowing is not lifting.
// To get a weight out of a locker and into a model, the model
// needs a KEY: something that says "this Tensor of mine is THAT
// entry of the file".
//
// Remember what a weight is, from the model's point of view. In
// the bakery you wrote
//
//     const weight: zml.Tensor = .init(.{ .dout = 2, .d = 3 }, .f32);
//
// a placeholder, a ghost: a shape, a dtype, and a promise that a
// Buffer will show up later. You copied the dims from the recipe
// card yourself. But the file KNOWS the shape and the dtype of
// its tensors. Typing them again is a chance to get them wrong.
//
// So ZML has a key cabinet:
//
//     var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
//     defer store.deinit();
//
// A TensorStore sits on top of a registry and hands out
// placeholders cut to fit the entries of the file. (The third
// argument lists the "meshes" the weights may be spread over,
// for machines with several GPUs. We have one CPU: every weight
// lives on it, whole. That's the `replicated_mesh` of the
// platform, and that's the last you'll hear of it.)
//
// You don't use the store directly. You ask it for a VIEW:
//
//     const view: zml.io.TensorStore.View = store.view();
//
// and the view cuts the keys:
//
//     view.createReplicatedTensor("plates", .{.plate})  -> zml.Tensor
//
// This returns an ordinary placeholder Tensor, with the dims and
// the dtype READ FROM THE FILE, and the tags given by YOU (a
// safetensors file has no tags: names for axes are a ZML thing).
// And the store writes in its book: "the Tensor with this id is
// the entry `plates`". Nothing is loaded. It's still a ghost.
// But it's a ghost with an address.
//
// Two ways to get thrown out of the key cabinet:
//
//  - Ask for a name the file does not have:
//
//        panic: Checkpoint has no tensor named coach.0.bias
//
//    ("Checkpoint" is what ML people call a file of weights.)
//
//  - Give a number of tags that is not the rank of the entry:
//
//        panic: tensor coach.1.bias from store has shape {3,f32},
//               but `createTensor` got only 2 tags: {dout,d}
//
//    ("Only" is ZML being polite. It says that for too many
//    tags as well.)
//
//    One tag per axis, or `null` for no tags at all.
//
// Now, names. Look at the logbook of exercise 098 again:
//
//     coach.0.weight   coach.1.weight   coach.1.bias   bar.0.weight ...
//
// The dots are a convention (PyTorch's): they spell the path to
// a tensor through the nested structs of a model. The coach has
// layers 0 and 1, each layer has a weight. A View can stand
// anywhere on that path. It is a store plus a PREFIX:
//
//     view.withPrefix("coach")    a view where names start with "coach."
//     view.withLayer(1)           same, with a number: "1."
//     view.prefix()               ?[]const u8: where this view stands
//
// They chain, and they return new views (a view is a small
// value: copy it, pass it around):
//
//     view.withPrefix("coach").withLayer(1).createReplicatedTensor("bias", .{.dout})
//         // the entry "coach.1.bias"
//
// And three ways to look before you leap:
//
//     view.getShape("weight")   ?zml.Shape: null if there is no such entry
//     view.hasKey("bias")       is there an entry whose name STARTS like this?
//     view.count()              how many entries are under this view
//
// Mind `hasKey`: it matches the beginning of a name, so it
// answers for whole groups too. `hasKey("coach")` is true.
//
// Why bother? Because not every layer has a bias. A model
// should ask before it grabs. That's what the `?` in
// `bias: ?zml.Tensor` of `zml.nn.Linear` has been waiting for.
//
// Coach Magnus wants the keys to his two layers. "WEIGHT. BIAS.
// BOTH LAYERS. GO." You have looked at the logbook; he has not.
// This panics twice. Read each message: it names its culprit.
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

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, weights_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const view = store.view();
    try stdout.print("{d} entries in the locker, {d} of them under \"coach\".\n", .{
        view.count(),
        view.withPrefix("coach").count(),
    });

    // A loose tensor, at the root of the file.
    const plates = view.createReplicatedTensor("plates", .{ .plate, .kg });
    try stdout.print("plates: {f}\n", .{plates});

    for (0..2) |i| {
        // Stand in front of layer i of the coach.
        const layer = view.withPrefix("coach").withLayer(i);
        try stdout.print("\nlayer \"{s}\"\n", .{layer.prefix().?});

        // A weight is a matrix: outputs first, then inputs.
        const weight = layer.createReplicatedTensor("weight", .{ .dout, .d });
        try stdout.print("  weight: {f}\n", .{weight});

        // Not every layer has a bias. Only cut a key if there is
        // a lock for it.
        const bias: ?zml.Tensor = layer.createReplicatedTensor("bias", .{.dout});
        try stdout.print("  bias:   {?f}\n", .{bias});
    }
    try stdout.flush();
}
