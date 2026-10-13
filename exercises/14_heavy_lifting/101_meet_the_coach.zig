//
// Registry, store, view, keys, loader. You have met the whole
// staff. Now put them to work on a model that DOES something.
//
// Every model in ZML's examples (go read
// zml/examples/mnist/mnist.zig after this, you'll recognize all
// of it) follows the same convention:
//
//     const Model = struct {
//         some_layer: Layer,
//         ...
//
//         pub fn init(view: zml.io.TensorStore.View) Model
//         pub fn forward(self: Model, x: zml.Tensor) zml.Tensor
//     };
//
// `init` takes a VIEW and builds the model out of it: every
// Tensor comes from `view.createReplicatedTensor(...)`. And a
// model made of smaller parts hands each part a NARROWER view:
//
//     pub fn init(view: zml.io.TensorStore.View) Model {
//         return .{
//             .encoder = .init(view.withPrefix("encoder")),
//             .decoder = .init(view.withPrefix("decoder")),
//         };
//     }
//
// It's Russian dolls. Each `init` only knows the names of its
// own fields ("weight", "bias") and the names of its children.
// Nobody ever spells out "decoder.layers.11.mlp.fc2.weight":
// the prefix builds up on the way down. The struct tree of the
// model mirrors the dotted names of the file, and that is how
// a 500-tensor checkpoint gets wired without 500 strings.
//
// The complete life of a model that comes from a file:
//
//     registry   <- TensorRegistry.fromPath(path)       read the header
//     store      <- TensorStore.fromRegistry(&registry)
//     model      <- Model.init(store.view())            ghosts with addresses
//     exe        <- platform.compile(model, .forward)   needs shapes only
//     buffers    <- bufferize + Loader                  the heavy lifting
//     result     <- exe.eval(.{ buffers, input })
//
// Look at the order of the last three. Compiling comes BEFORE
// loading, and that's allowed: compiling needs shapes, not
// numbers, and shapes are in the header. (Real programs do both
// at the same time, on two threads. Compiling a big model and
// loading 70 GB each take a while.)
//
// Our model is Coach Magnus himself, or the part of him that
// fits in 14 floats. He looks at three lifts (squat, bench
// press, deadlift, as multiples of the lifter's body weight)
// and delivers one of three verdicts. Two Linear layers:
//
//     coach.0   3 lifts -> 2 hunches (no bias), then relu
//                   hunch 0: squat + bench + deadlift    "TOTAL"
//                   hunch 1: bench - squat               "SKIPPED LEGS"
//     coach.1   2 hunches -> 3 verdict scores (with a bias)
//
// and the highest score wins. `zml.nn.Linear` has no `init`
// taking a view (it is older than the convention), so we give
// it one: the `linear` function below.
//
// Two reminders from the bakery, because they bite here too:
//
//  - a Linear consumes `.d` and produces `.dout`. To stack a
//    second one, the output must be renamed:
//    `.rename(.{ .dout = .d })`;
//  - the Bufferized of a Linear has `weight`, `bias` (optional,
//    like in the model), and a `quantization` to ignore.
//
// Coach Magnus is ready to judge. The apprentice who wired him
// up has never seen the logbook, and it shows: three mistakes.
// One doesn't compile. One panics. And the last one runs just
// fine, and tells a beginner that his 0.4 bench press is LIGHT
// WEIGHT, BABY. Flattery is not coaching.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

/// A Linear from the "weight" and (if there is one) the "bias"
/// found under `view`.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        null,
        .d,
    );
}

const Coach = struct {
    hunches: zml.nn.Linear,
    verdict: zml.nn.Linear,

    /// `view` must stand on the coach: its layers are "0" and "1".
    pub fn init(view: zml.io.TensorStore.View) Coach {
        return .{
            .hunches = linear(view.withLayer(0)),
            .verdict = linear(view.withLayer(1)),
        };
    }

    /// lifts: {.lifter, .d} -> {.lifter} index of the verdict
    pub fn forward(self: Coach, lifts: zml.Tensor) zml.Tensor {
        const hunches = self.hunches.forward(lifts, .f32).relu().rename(.{ .dout = .d });
        const scores = self.verdict.forward(hunches, .f32);
        return scores.argMax(.dout).indices.squeeze(.dout);
    }

    pub fn load(
        self: *const Coach,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Coach) {
        var buffers = try zml.mem.bufferize(gpa, Coach, self);

        var loader: zml.io.Loader = try .init(gpa, platform, .default);
        defer loader.deinit();

        try loader.load(io, Coach, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unload(buffers: *zml.Bufferized(Coach)) void {
        buffers.hunches.weight.deinit();
        buffers.hunches.bias.deinit();
        buffers.verdict.weight.deinit();
        buffers.verdict.bias.deinit();
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

    // The coach is one of several things kept in this locker.
    const coach: Coach = .init(store.view());

    // Three lifters, three lifts each.
    const lifts: zml.Tensor = .init(.{ .lifter = 3, .d = 3 }, .f32);

    // Shapes are enough to compile...
    var exe = try platform.compile(gpa, io, coach, .forward, .{lifts}, .{});
    defer exe.deinit();

    // ...numbers are needed to run.
    var coach_buffers = try coach.load(gpa, io, platform, &store);
    defer Coach.unload(&coach_buffers);

    const names = [_][]const u8{ "Olga", "Chad", "You" };
    var lifts_buffer: zml.Buffer = try .fromBytes(io, platform, lifts.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        //  squat, bench, deadlift
        .{ 2.0, 1.5, 2.5 }, // Olga
        .{ 0.5, 1.5, 1.0 }, // Chad
        .{ 0.5, 0.4, 0.6 }, // You (it's your first week)
    }));
    defer lifts_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ coach_buffers, lifts_buffer });
    defer result.deinit();

    const verdicts = [_][]const u8{
        "DO YOU EVEN LIFT?",
        "LIGHT WEIGHT, BABY!",
        "NEVER SKIP LEG DAY.",
    };
    for (names, try result.getValue([3]i32, io)) |name, verdict| {
        try stdout.print("{s:<5} {s}\n", .{ name, verdicts[@intCast(verdict)] });
    }
    try stdout.flush();
}
