//
// M. Nist is hired, on probation. And at five o'clock the ferry
// docks, and the parcels start coming in faster than Basile
// can carry them to the counter one by one.
//
// So far, one call of the Exe read one parcel. Every call has
// a fixed cost (arguments in, result out, a trip to the device
// and back), and the layers of M. Nist are matrix products,
// which a processor does much faster in one big piece than in
// two hundred small ones. Every real model is run on a BATCH:
// several inputs stacked along one more axis, by convention
// the first, and in ZML by convention tagged `.b`.
//
// PyTorch code is always written for a batch. The `forward` of
// M. Nist, as it really is in Python:
//
//     def forward(self, x):          # x: [batch, 28, 28]
//         x = x.flatten(1)           # [batch, 784]
//         x = F.relu(self.fc1(x))    # [batch, 500]
//         return self.fc2(x)         # [batch, 10]
//
//     digits = net(x).argmax(dim=1)  # [batch]
//
// Look at the two numbers in there: `flatten(1)` means "merge
// everything FROM AXIS 1 on", and `dim=1` means "the scores
// are on axis 1". Both numbers are there to step over the
// batch axis. PyTorch counts axes, so every line has to know
// how many there are in front of the one it wants. Half of
// the porting bugs you will ever write come from such a
// number: a `dim=-1` read as "the last one" when your tensor
// has its axes in another order, a `transpose(1, 2)` ported
// to a tensor that has no batch axis.
//
// ZML code NAMES its axes instead, and a named axis does not
// move when another one appears in front:
//
//     x.merge(.{ .d = .{ .h, .w } })   {.b, .h, .w} -> {.b, .d}
//                                      and {.h, .w} -> {.d}
//     linear.forward(x, .f32)          contracts `.d`, keeps
//                                      whatever else is there
//     scores.argMax(.dout)             along `.dout`, for each
//                                      of the others
//
// The fourth rule of porting follows:
//
//     TRANSLATE EVERY AXIS NUMBER OF THE PYTHON INTO A TAG,
//     ONCE, WHERE THE TENSOR IS BORN. THEN NEVER COUNT AGAIN.
//
// A `forward` written that way works for one parcel or two
// hundred without a change. Basile's does not. He wrote it for
// one image, with `flatten()`, which merges EVERY axis, the
// batch included, into one parcel of 156800 pixels, and with
// an `argMax(0)`. It is not his fault: he copied ZML's MNIST
// example again, and that example only ever reads one image.
//
// Two more facts about batches:
//
//  - Shapes are static (chapter 02). The batch size is part
//    of the compiled program: an Exe compiled for 200 parcels
//    takes 200 parcels. For the last, incomplete batch of a
//    real dataset, you pad it, or you compile a second Exe.
//
//  - The register is already a batch. An IDX file is the
//    images back to back, row after row: exactly the bytes of
//    a {.b, .h, .w} tensor. The first 200 parcels are the
//    first 200 * 784 bytes after the header. No copy, no loop.
//
// One more thing. Until now you compared M. Nist with PyTorch.
// The reference file also has `labels`: the digit that is
// REALLY written on each parcel. Those are two different
// questions, and you must never mix them up:
//
//     like PyTorch      is the PORT right? It must be all of
//                       them. That is your job.
//     like the label    is the MODEL right? It never is all of
//                       them, and that is not your problem.
//
// A port that agrees with the labels more often than PyTorch
// does is still a port with a bug.
//
// To do:
//
//  - in `main`: the placeholder, the Exe, the Buffer, the run,
//    the digits back on the host, and the two counts;
//  - then run it, BEFORE you touch `forward`, and read what
//    ZML thinks of a parcel of 156800 pixels;
//  - `Net.forward` takes a batch. Two lines of it count axes.
//    The first line this program prints is the shape of the
//    result: as long as it does not say `b=200`, one of them
//    is still counting.
//
const std = @import("std");
const zml = @import("zml");

const checkpoint_path = "zmling/17_customs_office/mnist.safetensors";
const register_path = "zmling/17_customs_office/t10k-images.idx3-ubyte";
const reference_path = "zmling/17_customs_office/activations.safetensors";

/// How many parcels go through in one call.
const batch = 200;

/// An IDX file of images, already in memory.
const Register = struct {
    count: usize,
    rows: usize,
    cols: usize,
    /// Everything after the header.
    pixels: []const u8,

    /// Reads the header of `bytes` (the whole file), and checks it:
    /// `error.NotAnImageFile` if the magic number is not 2051,
    /// `error.TruncatedFile` if the pixels are not all there.
    pub fn parse(bytes: []const u8) !Register {
        if (bytes.len < 16 or std.mem.readInt(u32, bytes[0..4], .big) != 2051) return error.NotAnImageFile;
        const register: Register = .{
            .count = std.mem.readInt(u32, bytes[4..8], .big),
            .rows = std.mem.readInt(u32, bytes[8..12], .big),
            .cols = std.mem.readInt(u32, bytes[12..16], .big),
            .pixels = bytes[16..],
        };
        if (register.pixels.len != register.count * register.rows * register.cols) return error.TruncatedFile;
        return register;
    }

    /// The `rows * cols` pixels of image `i`.
    pub fn image(self: Register, i: usize) []const u8 {
        const size = self.rows * self.cols;
        return self.pixels[i * size ..][0..size];
    }
};

fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        view.createReplicatedTensor("bias", .{.dout}),
        .d,
    );
}

const Net = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    pub fn init(view: zml.io.TensorStore.View) Net {
        return .{
            .fc1 = linear(view.withPrefix("fc1")),
            .fc2 = linear(view.withPrefix("fc2")),
        };
    }

    /// images: {.b, .h = 28, .w = 28} u8 -> the digits: {.b} i32
    pub fn forward(self: Net, images: zml.Tensor) zml.Tensor {
        const pixels = images.convert(.f32).divByConst(255).flatten().withTags(.{.d});
        const hidden = self.fc1.forward(pixels, .f32).relu().rename(.{ .dout = .d });
        const scores = self.fc2.forward(hidden, .f32);
        return scores.argMax(0).indices.squeeze(0);
    }

    pub fn load(
        self: *const Net,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Net) {
        var buffers = try zml.mem.bufferize(gpa, Net, self);

        var loader: zml.io.Loader = try .init(gpa, platform, .default);
        defer loader.deinit();

        try loader.load(io, Net, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unload(buffers: *zml.Bufferized(Net)) void {
        buffers.fc1.weight.deinit();
        if (buffers.fc1.bias) |*bias| bias.deinit();
        buffers.fc2.weight.deinit();
        if (buffers.fc2.bias) |*bias| bias.deinit();
    }
};

/// The whole content of a file. The caller frees it.
fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// The numbers of one tensor of a file, in host memory. The caller
/// frees them with `slice.free(gpa)`.
fn reference(gpa: std.mem.Allocator, io: std.Io, store: *zml.io.TensorStore, key: []const u8) !zml.Slice {
    const shape = store.view().getShape(key) orelse return error.NoSuchTensor;
    const slice: zml.Slice = try .alloc(gpa, shape);
    errdefer slice.free(gpa);

    var read_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &read_buffer);
    defer reader.deinit();
    try reader.interface.readSliceAll(slice.data());
    return slice;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const register_bytes = try readFile(gpa, io, register_path);
    defer gpa.free(register_bytes);

    const register: Register = try .parse(register_bytes);

    // What PyTorch read, and what is really written.
    const reference_file = try std.Io.Dir.cwd().realPathFileAlloc(io, reference_path, gpa);
    defer gpa.free(reference_file);

    var reference_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, reference_file);
    defer reference_registry.deinit();

    var reference_store: zml.io.TensorStore = .fromRegistry(gpa, &reference_registry, &.{platform.replicated_mesh});
    defer reference_store.deinit();

    const pytorch = try reference(gpa, io, &reference_store, "digits");
    defer pytorch.free(gpa);
    const labels = try reference(gpa, io, &reference_store, "labels");
    defer labels.free(gpa);

    // M. Nist.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, checkpoint_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const net: Net = .init(store.view());

    // `batch` parcels at once.
    const images: zml.Tensor = ???;

    // TODO: `exe`.

    var net_buffers = try net.load(gpa, io, platform, &store);
    defer Net.unload(&net_buffers);

    // The first `batch` parcels of the register are the first
    // `batch * 784` bytes of its pixels: nothing to rearrange.
    // TODO: `images_buffer`.

    var result: zml.Buffer = ???;
    defer result.deinit();

    // TODO: `digits`, a zml.Slice: the result, on the host.

    try stdout.print("{d} parcels in one call: {f}\n", .{ batch, result.shape() });

    // For each parcel: is the digit the one PyTorch read? Is it the
    // one on the label? Every parcel that M. Nist misreads gets a
    // line:
    //
    //     try stdout.print("parcel {d}: M. Nist reads a {d}, the label says {d}\n", .{ parcel, digit, label });
    var like_pytorch: usize = 0;
    var like_label: usize = 0;
    ???;
    try stdout.print("Like PyTorch: {d} / {d}\n", .{ like_pytorch, batch });
    try stdout.print("Like the label: {d} / {d}\n", .{ like_label, batch });
    try stdout.flush();
}
