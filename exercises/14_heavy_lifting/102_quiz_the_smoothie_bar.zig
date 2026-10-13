//
// Quiz! After the workout, the smoothie bar. (The smoothie bar
// is where the Iron Tensor makes its money. The lifting is a
// pretext.)
//
// Behind the counter stands the BLENDTRON, a machine with three
// blenders in a row. You tell it how many cups of banana and of
// spinach you want; it answers with two numbers: the price, in
// dollars, and how good the thing will taste, from 0 up.
//
// Nobody at the gym knows HOW it decides. Denise only knows
// that its weights are in the same locker as everything else,
// under "bar":
//
//     bar.0.weight  bar.0.bias     2 numbers in, 2 numbers out
//     bar.1.weight  bar.1.bias     2 in, 2 out
//     bar.2.weight  bar.2.bias     2 in, 2 out
//
// and the manual, which is a napkin, says:
//
//     blender 0, relu, blender 1, relu, blender 2. NO relu at
//     the end. Do not put your hand in blender 1.
//
// That's a model nobody understands, with weights in a file,
// and a note about the architecture. Welcome to machine
// learning! You have everything you need to run it:
//
//     TensorRegistry.fromPath     the header of the file
//     TensorStore.fromRegistry    the key cabinet, and its `view()`
//     withPrefix / withLayer      where a view stands
//     createReplicatedTensor      a placeholder with an address
//     zml.mem.bufferize           the empty bar
//     zml.io.Loader               init, load, await, deinit
//     platform.compile            shapes are enough
//     exe.eval                    weights first, then the input
//
// This time the model is yours to wire: `init`, `forward` and
// `load` are empty, and so are a few lines of `main`. The
// `linear` helper of the previous exercise is still here (every
// blender has a bias, so it has an easy day).
//
// Three customers are waiting. One of them ordered nothing at
// all, and will be charged for it.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

/// A Linear from the "weight" and (if there is one) the "bias"
/// found under `view`.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        if (view.hasKey("bias")) view.createReplicatedTensor("bias", .{.dout}) else null,
        .d,
    );
}

const Blendtron = struct {
    chop: zml.nn.Linear,
    blend: zml.nn.Linear,
    pour: zml.nn.Linear,

    /// `view` stands on the machine: its blenders are "0", "1" and "2".
    pub fn init(view: zml.io.TensorStore.View) Blendtron {
        return ???;
    }

    /// cups: {.order, .d} -> {.order, .dout} price and taste
    pub fn forward(self: Blendtron, cups: zml.Tensor) zml.Tensor {
        return ???;
    }

    /// Brings the six tensors of the machine from the file to the
    /// device. The caller owns the Buffers (see `unload`).
    pub fn load(
        self: *const Blendtron,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Blendtron) {
        return ???;
    }

    pub fn unload(buffers: *zml.Bufferized(Blendtron)) void {
        inline for (.{ &buffers.chop, &buffers.blend, &buffers.pour }) |blender| {
            blender.weight.deinit();
            if (blender.bias) |*bias| bias.deinit();
        }
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

    // The logbook, then the key cabinet.
    var registry: zml.safetensors.TensorRegistry = ???;
    defer registry.deinit();

    var store: zml.io.TensorStore = ???;
    defer store.deinit();

    // The machine. Mind where its view stands.
    const blendtron: Blendtron = .init(???);

    // Three orders: cups of banana, cups of spinach.
    const cups: zml.Tensor = .init(.{ .order = 3, .d = 2 }, .f32);

    var exe = ???;
    defer exe.deinit();

    var blendtron_buffers = try blendtron.load(gpa, io, platform, &store);
    defer Blendtron.unload(&blendtron_buffers);

    const orders = [3][2]f32{
        .{ 2, 1 }, // the Classic
        .{ 1, 3 }, // the Regret
        .{ 0, 0 }, // "just a glass, please"
    };
    var cups_buffer: zml.Buffer = try .fromBytes(io, platform, cups.shape(), std.mem.sliceAsBytes(&orders));
    defer cups_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, ???);
    defer result.deinit();

    try stdout.print("receipt {f}\n", .{result.shape()});
    for (orders, try result.getValue([3][2]f32, io)) |order, receipt| {
        try stdout.print("  {d} banana, {d} spinach: ${d:.2}, taste {d}\n", .{ order[0], order[1], receipt[0], receipt[1] });
    }
    try stdout.flush();
}
