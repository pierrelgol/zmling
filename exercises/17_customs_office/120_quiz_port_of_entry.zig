//
// Monday morning. Odile is on holiday, for the first time in
// thirty years. On your desk: the suitcase, the register, the
// reference file, and a note in her handwriting.
//
//     "A new M. Nist is at the counter. Same suitcase, he says.
//      I believe nothing. Do the whole procedure, from the
//      inventory to the verdict. Basile is not to touch the
//      keyboard.        O."
//
// This is the quiz, and it is a whole port, the way you will
// do the next ones alone. The procedure, in the order you
// learned it:
//
//  1. INSPECT (115). Open the checkpoint and count: how many
//     tensors, how many parameters. Before any model code.
//
//  2. DECLARE (116). The struct, its `init`, its `load`, its
//     `unload`. The papers of M. Nist, once more:
//
//         class Net(nn.Module):
//             def __init__(self):
//                 super().__init__()
//                 self.fc1 = nn.Linear(784, 500)
//                 self.fc2 = nn.Linear(500, 10)
//
//             def forward(self, x):          # [batch, 28, 28]
//                 x = x.flatten(1)           # [batch, 784]
//                 x = F.relu(self.fc1(x))    # [batch, 500]
//                 return self.fc2(x)         # [batch, 10]
//
//     and what was done to the images before `forward`:
//
//         x = torch.from_numpy(images).float() / 255
//
//  3. RUN (117, 119). The register is an IDX file: check its
//     header, then feed it to the model by batches of `batch`
//     parcels, `batches` times, WITH ONE EXE. This time
//     `forward` returns the SCORES, {.b, .dout = 10} f32, and
//     not the digits: you cannot verify an argMax.
//
//  4. VERIFY (118). The reference file has the `logits` of
//     the first 200 parcels: your first batch, exactly. One
//     `expectClose`, default options, on the whole batch. If
//     it fails, the report on stderr is your friend.
//
//  5. And only then, the digits. Bring each batch back on the
//     host (`toSliceAlloc`), and find the winner of each
//     parcel there: the scores of parcel `i` of the batch are
//     ten floats in a row,
//
//         scores.items(f32)[i * 10 ..][0..10]
//
//     and `std.mem.indexOfMax(f32, row)` is the index of the
//     biggest. Count how many digits are the ones PyTorch
//     read (`digits` in the reference file), and how many are
//     the ones really written (`labels`). Parcel `i` of batch
//     number `n` is parcel `n * batch + i` of the register.
//
// `readFile` and `reference` are given: they have not changed
// since exercise 118. The five lines to print are at the
// bottom of `main`, and they tell you the names of the things
// you must compute. Everything else is yours.
//
// When it prints what it should, you have ported a model:
// you opened a checkpoint nobody documented, rebuilt it from
// its names, fed it real data the way it was trained, and
// proved, number against number, that it is the same model.
// The next four chapters do exactly this, on bigger suitcases.
//
const std = @import("std");
const zml = @import("zml");

const checkpoint_path = "zmling/17_customs_office/mnist.safetensors";
const register_path = "zmling/17_customs_office/t10k-images.idx3-ubyte";
const reference_path = "zmling/17_customs_office/activations.safetensors";

/// How many parcels go through in one call, and how many calls.
const batch = 200;
const batches = 5;

/// An IDX file of images, already in memory.
const Register = struct {
    // How many images, their size, and the pixels.
    ???,

    /// Reads the header of `bytes` (the whole file), and checks it.
    pub fn parse(bytes: []const u8) !Register {
        return ???;
    }
};

const Net = struct {
    // The two layers.
    ???,

    /// The model, with placeholders made from the checkpoint.
    pub fn init(view: zml.io.TensorStore.View) Net {
        return ???;
    }

    /// images: {.b, .h = 28, .w = 28} u8 -> the scores: {.b, .dout = 10} f32
    pub fn forward(self: Net, images: zml.Tensor) zml.Tensor {
        return ???;
    }

    /// The weights, from the file to the device.
    pub fn load(
        self: *const Net,
        gpa: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Net) {
        return ???;
    }

    /// Frees what `load` returned.
    pub fn unload(buffers: *zml.Bufferized(Net)) void {
        ???;
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

    // The checkpoint and its inventory, the model, the Exe, the
    // weights, the register, the reference, and the batches.
    //
    // The lines below want: `registry`, `parameters`, `register`,
    // `scores_like_pytorch` (a bool), `like_pytorch` and `like_label`
    // (two counts).
    ???;

    try stdout.print("In the suitcase: {d} tensors, {d} parameters\n", .{ registry.tensors.count(), parameters });
    try stdout.print("The register: {d} parcels of {d} x {d} pixels\n", .{ register.count, register.rows, register.cols });
    try stdout.print("Scores like PyTorch on the first {d} parcels: {}\n", .{ batch, scores_like_pytorch });
    try stdout.print("Digits like PyTorch: {d} / {d}\n", .{ like_pytorch, batch * batches });
    try stdout.print("Digits like the label: {d} / {d}\n", .{ like_label, batch * batches });
    try stdout.flush();
}
