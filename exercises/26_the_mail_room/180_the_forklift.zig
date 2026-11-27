//
// Since the gym you have moved weights with the same two lines:
//
//     var loader: zml.io.Loader = try .init(gpa, platform, .default);
//     try loader.load(io, Model, &model, &weights, &store, .{});
//
// `.default` and `.{}`. Two arguments you never opened. Gus has
// a name for `.default`: "the hand truck". Here it is, from
// `zml/core/io.zig`:
//
//     pub const Opts = struct {
//         pub const default: Opts = .{
//             .parallelism = 1,
//             .dma_chunks = 2,
//             .dma_chunk_size = 4096,
//         };
//         parallelism: usize,
//         dma_chunks: usize,
//         dma_chunk_size: usize,
//     };
//
//  - `parallelism`: how many tensors are read at the same time.
//    `load` starts one task per Tensor of your model; this is
//    the number allowed to run together. One: a queue.
//  - `dma_chunks`, `dma_chunk_size`: the bytes of a tensor do
//    not jump from the file to the device. They travel through
//    staging buffers, in memory the device can take from
//    directly. Each device gets a pool of at most `dma_chunks`
//    buffers of `dma_chunk_size` bytes, made when first needed.
//
// One tensor at a time through two 4 KiB buffers: right for
// 968 bytes. What every model of `examples/llm` asks for:
//
//     .{ .parallelism = 16, .dma_chunks = 32, .dma_chunk_size = 256 * zml.MiB }
//
// sixteen tensors at once, through up to 8 GiB of staging.
// (`zml.KiB`, `zml.MiB`, `zml.GiB` are plain numbers.) These
// are the three knobs you turn when a 40 GB model takes four
// minutes to load from a disk that could do it in one. There
// is no `.fast`: no field has a default, you say all three.
//
// The other argument, the last one of `load`:
//
//     pub const LoadOpts = struct {
//         progress: ?*std.Progress.Node = null,
//     };
//
// `std.Progress` is Zig's progress display, the one the
// compiler uses: a tree of named tasks drawn on the terminal,
// on stderr. Give `load` a node and each tensor shows up under
// it while its bytes move.
//
//     var progress = std.Progress.start(io, .{ .root_name = "overseas" });
//     defer progress.end();
//     try loader.load(io, Model, &model, &weights, &store, .{ .progress = &progress });
//
// ONE `start` per program: a second one panics. (The zmling
// runner hides stderr, and stderr is not a terminal there, so
// nothing is drawn. Run the binary by hand to see it.)
//
// Then three facts about the Loader that the hand truck hid.
//
// `load` DOES NOT LOAD. It schedules, and returns. The bytes
// are on the device after
//
//     try loader.await(io);
//
// and not before. You have always written the two lines
// together; now you know why.
//
// IT COUNTS. `loader.bytes_loaded` is an atomic counter of
// what really moved; read it with `.load(.monotonic)`. Compare
// it with `registry.totalBytes()`: the Loader only moves what
// a Tensor of your model CLAIMED through the store. A tensor
// of the file that no field asked for stays in its box, and
// nobody says a word. Less on the device than in the boxes is
// a question you want to ask yourself before your users do.
//
// COMPILE FIRST, LOAD SECOND. `examples/llm/main.zig` says why
// in one comment: "Load buffers after the model compilation to
// be sure to give enough room to the autotune." On a GPU,
// compiling may try several implementations of an operation
// on the device itself, and that needs device memory which 40
// GB of weights would be sitting on. There is a plainer reason
// too: compiling needs shapes, not numbers, and it finds your
// shape mistakes in seconds. On this CPU the order changes
// nothing you can see. Keep the habit.
//
// And the parcel itself. Its slip says `"torch_dtype":
// "float16"`, and the registry agrees: `{8,8,f16}`. Half
// floats (chapter 22) are how most checkpoints are stored:
// half the disk, half the download. The store cuts f16
// placeholders, the Loader moves f16 bytes, and nothing
// converts them for you. Now read `zml.nn.Linear.forward`
// once more:
//
//     const y = x.dot(self.weight.convert(x.dtype()), self.tag).convert(out_dtype);
//
// The weight is converted to the dtype of X, the product is
// computed in the dtype of X, and only the RESULT becomes
// `out_dtype`. Feed it an f16 activation, and the arithmetic is
// f16 whatever you ask for at the end. Exercise 148 gave you
// the rule for the ticks: an f16 has 10 mantissa bits, so from
// 2048 to 4096 it counts in steps of 2. How a file is STORED
// and how a model COMPUTES are two decisions, and the second
// one is yours: one `convert` on the first activation, and
// every Linear after it follows.
//
// Gus: "Overseas parcel. Use the forklift. And find out why
// everything from overseas ends up returned to sender."
//
// The conveyor is a chute with layers in between:
//
//     x = table[station]
//     for each layer:  x = relu(fc(x))      fc: a Linear
//     next = argmax(lm_head(x))
//
//  1. `Conveyor.init`. Each layer is under `model.layers.N.`,
//     and the number of layers is on the slip. (`Layer.init`
//     is done: Priya wrote it before lunch.)
//  2. The Loader, with the options of the label (`Label` is
//     exercise 176's, `ByteSize` included), then `load` with
//     the progress node, then the wait.
//  3. Run it. The parcel is returned to sender. Compare the
//     two byte counts the program prints: somebody decided
//     that a tensor was not needed. A layer may or may not
//     have a bias: `maybeCreateTensor`, exercise 128.
//  4. Still returned to sender. Now it is the arithmetic.
//
const std = @import("std");
const zml = @import("zml");

const flags = zml.stdx.flags;
const View = zml.io.TensorStore.View;

const parcels = "zmling/26_the_mail_room/parcels/";

const stations = [_][]const u8{ "dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement" };

const Label = struct {
    parallelism: usize = 1,
    chunks: usize = 2,
    chunk_size: flags.ByteSize = .{ .value = 4, .unit = .kib },
    positional: struct { repo: []const u8 },
};

const Config = struct {
    num_hidden_layers: u32 = 0,
    bos_token_id: u32,
    eos_token_id: []const u32,
    torch_dtype: enum { float32, float16, bfloat16 } = .float32,
};

fn parseJson(comptime T: type, gpa: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, name: []const u8) !std.json.Parsed(T) {
    const file = try repo.openFile(io, name, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(gpa, &file_reader.interface);
    defer reader.deinit();
    return try std.json.parseFromTokenSource(T, gpa, &reader, .{ .ignore_unknown_fields = true });
}

const Layer = struct {
    fc: zml.nn.Linear,

    pub fn init(view: View) Layer {
        const fc = view.withPrefix("fc");
        return .{ .fc = .init(
            fc.createReplicatedTensor("weight", .{ .dout, .d }),
            null,
            .d,
        ) };
    }

    /// x: {.d} -> {.d}
    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        return self.fc.forward(x, x.dtype()).relu().rename(.{ .dout = .d });
    }
};

const Conveyor = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []Layer,
    lm_head: zml.nn.Linear,

    pub fn init(gpa: std.mem.Allocator, view: View, config: Config) !Conveyor {
        return ???;
    }

    pub fn deinit(self: Conveyor, gpa: std.mem.Allocator) void {
        gpa.free(self.layers);
    }

    /// station: a rank-0 i32. Returns the next station, a rank-0 i32.
    pub fn forward(self: Conveyor, station: zml.Tensor) zml.Tensor {
        var x = self.embed_tokens.forward(station);
        for (self.layers) |layer| x = layer.forward(x);
        const scores = self.lm_head.forward(x, x.dtype());
        return scores.argMax(.dout).indices.squeeze(.dout);
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const label = flags.parse(.{ .vector = &.{
        "mailroom",
        "--parallelism=4",
        "--chunks=4",
        "--chunk-size=64KiB",
        parcels ++ "overseas",
    } }, Label);
    try stdout.print("forklift: {d} at a time, {d} buffers of {d} bytes\n", .{
        label.parallelism,
        label.chunks,
        label.chunk_size.bytes(),
    });

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    var repo = try zml.safetensors.resolveModelRepo(io, label.positional.repo);
    defer repo.close(io);

    const parsed = try parseJson(Config, gpa, io, repo, "config.json");
    defer parsed.deinit();
    const config = parsed.value;

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(gpa, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const model: Conveyor = try .init(gpa, store.view(), config);
    defer model.deinit(gpa);
    try stdout.print("the slip says {s}, the table is {f}\n", .{ @tagName(config.torch_dtype), model.embed_tokens.weight });
    try stdout.flush();

    // Compile first...
    var exe = try platform.compile(gpa, io, model, .forward, .{zml.Tensor.init(.{}, .i32)}, .{});
    defer exe.deinit();

    // ...load second.
    var progress = std.Progress.start(io, .{ .root_name = "overseas" });
    defer progress.end();

    var weights = try zml.mem.bufferize(gpa, Conveyor, &model);
    defer gpa.free(weights.layers);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Conveyor, &weights);

    // The forklift: `loader`, with the options of the label, at work
    // under `progress`.
    ???

    try stdout.print("in the boxes: {d} bytes\n", .{registry.totalBytes()});
    try stdout.print("on the device: {d} bytes\n", .{loader.bytes_loaded.load(.monotonic)});
    try stdout.flush();

    var station: u32 = config.bos_token_id;
    var count: usize = 1;
    try stdout.print("route: {s}", .{stations[station]});
    while (std.mem.findScalar(u32, config.eos_token_id, station) == null and count < 8) : (count += 1) {
        var here: zml.Buffer = try .scalar(io, platform, station, .i32);
        defer here.deinit();
        var next = try exe.eval(gpa, io, .{ weights, here });
        defer next.deinit();
        station = @intCast(try next.getValue(i32, io));
        try stdout.print(" > {s}", .{stations[station]});
    }
    try stdout.print("\n", .{});
    try stdout.flush();
}
