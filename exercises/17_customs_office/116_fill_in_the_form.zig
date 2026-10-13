//
// Four tensors, 397510 numbers, and two matrices that say
// "784 in, 500 out" and "500 in, 10 out". Odile makes a phone
// call. An hour later the fax machine prints the papers of
// M. Nist, the Python class he was trained as:
//
//     class Net(nn.Module):
//         def __init__(self):
//             super().__init__()
//             self.fc1 = nn.Linear(784, 500)
//             self.fc2 = nn.Linear(500, 10)
//
//         def forward(self, x):         # x: 784 pixels
//             x = F.relu(self.fc1(x))   # 500 numbers
//             return self.fc2(x)        # 10 scores, one per digit
//
// (Full disclosure: these weights come from ZML's own MNIST
// example, and nobody kept the training script. This class was
// written back from the four names, the way you could have
// done it after exercise 115. It is the right one: PyTorch
// loads the file into it with `strict=True`, which fails on
// any missing, extra or misshapen tensor. `make_fixtures.py`,
// in this directory, does that.)
//
// Customs has a form for everything, and porting a model is
// filling in form ZML-116, "Declaration of a Model". It has
// three boxes, and you know them from exercise 101:
//
//  1. THE STRUCT: one field per thing that has weights. Two
//     `nn.Linear` in Python, two `zml.nn.Linear` in Zig.
//
//  2. `init`: takes a view of the store and builds every
//     Tensor out of it, each part from a narrower view:
//
//         view.withPrefix("fc1")     "fc1." goes in front of
//                                    every name asked below
//         view.createReplicatedTensor("weight", tags)
//                                    a placeholder with the
//                                    shape and dtype found in
//                                    the FILE, and your tags
//
//  3. `load`: `zml.mem.bufferize` and a `zml.io.Loader`. You
//     wrote it in exercise 100. Write it again.
//
// Three facts make this form easy to get wrong.
//
// THE KEY STRING IS THE ONLY LINK. Between the Zig struct and
// the file there is nothing but the strings you pass to
// `withPrefix` and `createReplicatedTensor`. The names of your
// Zig fields are yours: call them `fc1` and `fc2` because the
// next reader will have the Python source open, not because
// ZML looks at them. It doesn't. A string, on the other hand,
// must be the name in the file, to the letter, and when it
// isn't, ZML panics with the full name it looked for:
//
//     panic: Checkpoint has no tensor named encoder.wieght
//
// Read that name, then read the inventory of exercise 115
// again. The fix is always in one of the two.
//
// A SHAPE COMES WITH NO LABELS. The file says {500, 784}. It
// does not say which of the two is the input. PyTorch's
// convention does: a `nn.Linear` weight is {out, in}, so that
// it can compute `x @ weight.T`. In ZML you never transpose.
// You TAG, and the tags decide which axis gets contracted:
//
//     zml.nn.Linear.init(weight, bias, .d)
//     linear.forward(x, .f32)
//
// contracts the axis `.d` of `x` with the axis `.d` of the
// weight, and what is left of the weight is the output. So
// the weight is tagged in the order of the file, with `.d` on
// the INPUT side, and the bias goes with the output:
//
//     weight    {.dout, .d}        out first, as PyTorch stores it
//     bias      {.dout}
//
// THE SWAP IS ONLY LOUD WHEN YOU ARE LUCKY. Tag the first
// weight {.d, .dout} and ZML tries to contract 784 pixels with
// an axis of 500:
//
//     panic: dotGeneral expects contracting dimensions to be
//     equal, got 0 and 0 in Tensor({d=784,f32}) and
//     Tensor({d=500,dout=784,f32})
//
// (The "0 and 0" are the POSITIONS of the two `.d` axes, not
// their sizes. The sizes are in the shapes: 784 against 500.)
// But a SQUARE weight, {500, 500}, would take the swap without
// a word and compute with the transposed matrix. Wrong
// numbers, no message. Models are full of square matrices.
// Remember this one: the tags are not decoration.
//
// Basile filled in the helper `linear` before the fax arrived,
// from memory. It has two mistakes, and each one is a panic
// worth reading. The rest of the form is blank:
//
//  - the fields, `init` and `load`;
//  - in `main`: the model, the Exe, the run.
//
// `forward` is given, this once (it is the Python `forward`,
// line for line, and next exercise it is yours). To see the
// whole thing work, you run a BLANK form through it: 784
// pixels, all zero. PyTorch says M. Nist sees a 5 in it.
//
const std = @import("std");
const zml = @import("zml");

const checkpoint_path = "zmling/17_customs_office/mnist.safetensors";

/// A `nn.Linear` with a bias, from the "weight" and the "bias"
/// found under `view`.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weights", .{ .d, .dout }),
        view.createReplicatedTensor("bias", .{.dout}),
        .d,
    );
}

const Net = struct {
    // One field per layer of the Python class.
    ???,

    /// `view` must stand at the root of the checkpoint.
    pub fn init(view: zml.io.TensorStore.View) Net {
        return ???;
    }

    /// pixels: {.d = 784} f32 -> scores: {.dout = 10} f32
    pub fn forward(self: Net, pixels: zml.Tensor) zml.Tensor {
        const hidden = self.fc1.forward(pixels, .f32).relu().rename(.{ .dout = .d });
        return self.fc2.forward(hidden, .f32);
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
        buffers.fc1.weight.deinit();
        if (buffers.fc1.bias) |*bias| bias.deinit();
        buffers.fc2.weight.deinit();
        if (buffers.fc2.bias) |*bias| bias.deinit();
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

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, checkpoint_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    // The form, filled in.
    const net: Net = ???;

    // What the file gave, and what you called it.
    try stdout.print("fc1.weight {f}\n", .{net.fc1.weight.shape()});
    try stdout.print("fc1.bias   {f}\n", .{net.fc1.bias.?.shape()});
    try stdout.print("fc2.weight {f}\n", .{net.fc2.weight.shape()});
    try stdout.print("fc2.bias   {f}\n", .{net.fc2.bias.?.shape()});

    // One image, already flat: 784 pixels.
    const pixels: zml.Tensor = .init(.{ .d = 784 }, .f32);

    // TODO: `exe`.

    var net_buffers = try net.load(gpa, io, platform, &store);
    defer Net.unload(&net_buffers);

    // A blank form: every pixel is 0.
    const blank: [784]f32 = @splat(0);
    var blank_buffer: zml.Buffer = try .fromBytes(io, platform, pixels.shape(), std.mem.sliceAsBytes(&blank));
    defer blank_buffer.deinit();

    var result: zml.Buffer = ???;
    defer result.deinit();

    // The digit with the highest score wins.
    const scores = try result.getValue([10]f32, io);
    try stdout.print("{f} in, {f} out\n", .{ pixels.shape(), result.shape() });
    try stdout.print("In a blank form, M. Nist sees a {d}\n", .{std.mem.indexOfMax(f32, &scores)});
    try stdout.flush();
}
