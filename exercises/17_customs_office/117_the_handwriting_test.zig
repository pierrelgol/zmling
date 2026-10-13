//
// The form is filled in, M. Nist has weights, and a blank page
// looks like a 5 to him. Time for the real test. In the back
// room of the customs office sits the REGISTER: ten thousand
// parcels, each with one handwritten digit on its label,
// photographed in 28 x 28 pixels. One file:
//
//     t10k-images.idx3-ubyte        7.8 MB
//
// To run M. Nist on a parcel, you must give him what PyTorch
// gave him. Not "an image". The SAME NUMBERS. Here is what
// happened in Python between the file and the first layer:
//
//     image = images[i]                   # [28, 28] uint8, 0..255
//     x = torch.from_numpy(image).float() # the same, as f32
//     x = x / 255                         # 0..1
//     x = x.flatten()                     # [784]
//     x = F.relu(net.fc1(x))              # [500]
//     scores = net.fc2(x)                 # [10]
//     digit = scores.argmax()
//
// Lines 2 to 4 are called PREPROCESSING. Second rule of
// porting:
//
//     PREPROCESSING IS PART OF THE MODEL. PORT IT TOO.
//
// It is the part nobody writes down. It is not in the
// checkpoint: no tensor is named "divide_by_255". It is not in
// the `nn.Module` either. It hides in the training script, in
// a `transforms.Compose([...])`, in a `preprocessor_config.json`
// next to the weights, or in a tokenizer (chapter 18). And a
// model fed with numbers 255 times too big does not panic. It
// answers. For M. Nist we had to MEASURE which preprocessing
// he was trained with (the three usual suspects for images are
// raw 0..255, divided by 255, and divided by 255 then shifted
// and scaled by a mean and a deviation): `make_fixtures.py`
// tells the story. Divided by 255 it is.
//
// In ZML the preprocessing goes inside `forward`, so it is
// compiled with the rest and nobody can forget it:
//
//     image.convert(.f32)                {.h, .w} u8 -> f32
//          .divByConst(255)              exercise 018
//          .merge(.{ .d = .{ .h, .w } }) exercise 049: the two
//                                        axes become one, `.d`
//
// then the two layers (you saw them last exercise: a Linear
// consumes `.d`, produces `.dout`, and the second one wants a
// `.d` again), then the winner, as in exercise 101:
// `argMax(.dout)`, its `.indices`, without the leftover axis.
//
// Now the file. The register is in the IDX format, which is
// as old as the digits in it, and as simple:
//
//     bytes  0..4     the magic number
//     bytes  4..8     how many images
//     bytes  8..12    rows per image
//     bytes 12..16    columns per image
//     bytes 16..      the pixels, one byte each, image after
//                     image, row after row
//
// The four numbers of the header are unsigned 32-bit integers,
// BIG-ENDIAN (safetensors, in exercise 097, was little-endian:
// always check). Zig reads one like this:
//
//     std.mem.readInt(u32, bytes[0..4], .big)
//
// The magic number of a file of u8 images is 2051. A reader
// that does not check it will happily "read" any file at all.
// And a reader that forgets the 16 bytes of the header gets
// every image shifted by 16 pixels: it runs, and the digits
// come out wrong. So before any model sees a pixel, LOOK at
// one image. `draw`, below, prints a parcel with `#` and `.`.
// If you cannot read the digit yourself, don't ask M. Nist.
//
// Odile pulls eight parcels from the register. What PyTorch
// read on them is written below, mistakes included: on parcel
// 1522 there is a 7, and PyTorch's M. Nist says 9. Your port
// must say 9. A port that is "better" than the original is a
// port with a bug that happens to help today.
//
// To write:
//
//  - `Register.parse` and `Register.image`;
//  - `Net.forward`, preprocessing included;
//  - in `main`: the Exe, and for each parcel the Buffer and
//    the run.
//
const std = @import("std");
const zml = @import("zml");

const checkpoint_path = "zmling/17_customs_office/mnist.safetensors";
const register_path = "zmling/17_customs_office/t10k-images.idx3-ubyte";

/// The parcels Odile pulled, and what PyTorch read on each.
const pulled = [_]usize{ 0, 1, 2, 3, 4, 115, 674, 1522 };
const pytorch = [_]i32{ 7, 2, 1, 0, 4, 4, 5, 9 };

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
        return ???;
    }

    /// The `rows * cols` pixels of image `i`.
    pub fn image(self: Register, i: usize) []const u8 {
        return ???;
    }
};

/// Prints an image: `#` where there is ink, `.` where there is none.
fn draw(stdout: *std.Io.Writer, pixels: []const u8, cols: usize) !void {
    for (pixels, 0..) |pixel, i| {
        try stdout.writeByte(if (pixel > 127) '#' else '.');
        if (i % cols == cols - 1) try stdout.writeByte('\n');
    }
}

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

    /// image: {.h = 28, .w = 28} u8, as found in the register
    /// -> the digit: {} i32
    pub fn forward(self: Net, image: zml.Tensor) zml.Tensor {
        return ???;
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The register: look before you compute.
    const register_bytes = try readFile(gpa, io, register_path);
    defer gpa.free(register_bytes);

    const register: Register = try .parse(register_bytes);
    try stdout.print("The register: {d} parcels of {d} x {d} pixels\n", .{ register.count, register.rows, register.cols });
    try stdout.print("Parcel 0:\n", .{});
    try draw(stdout, register.image(0), register.cols);

    // M. Nist, as in exercise 116.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, checkpoint_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const net: Net = .init(store.view());

    // One parcel, as it is in the file: bytes.
    const image: zml.Tensor = .init(.{ .h = 28, .w = 28 }, .u8);

    // TODO: `exe`.

    var net_buffers = try net.load(gpa, io, platform, &store);
    defer Net.unload(&net_buffers);

    var same: usize = 0;
    for (pulled, pytorch) |parcel, expected| {
        // TODO: `image_buffer`, the pixels of this parcel.

        var result: zml.Buffer = ???;
        defer result.deinit();

        const digit = try result.getValue(i32, io);
        if (digit == expected) same += 1;
        try stdout.print("parcel {d:>4}: M. Nist reads a {d}\n", .{ parcel, digit });
    }
    try stdout.print("Same as PyTorch: {d} / {d}\n", .{ same, pulled.len });
    try stdout.flush();
}
