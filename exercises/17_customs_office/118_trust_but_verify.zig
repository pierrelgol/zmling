//
// Eight parcels, eight digits, all of them the ones PyTorch
// read. Basile has already stamped the form: PORTED.
//
// Odile takes the stamp away from him. Third rule of porting,
// and the one this whole chapter was written for:
//
//     A PORT THAT RUNS PROVES NOTHING. COMPARE THE NUMBERS.
//
// A digit is the end of a long computation, and an `argMax`
// forgives almost everything: scale all ten scores by 255,
// and the biggest is still the biggest. A language model
// forgives even more: with a small bug it still writes
// English, only a little worse, and nobody can tell "a little
// worse" from "it's a small model" by reading the output. The
// only test that means something is done BEFORE the argMax:
// same input, same numbers, at every step, as the original.
//
// So the original must be run once, in Python, and its
// numbers kept. That is what `make_fixtures.py` did for you:
// `activations.safetensors`, next to this file, holds what
// PyTorch computed on the parcels of the register:
//
//     hidden    {8, 500} f32      after the relu of fc1,
//                                 parcels 0 to 7
//     logits    {200, 10} f32     the ten scores, parcels
//                                 0 to 199 ("logits" is the
//                                 usual name for raw scores)
//     digits    {10000} u8        the digit PyTorch reads
//     labels    {10000} u8        the digit that is really
//                                 written (next exercise)
//
// When you port a model yourself, writing that script is the
// FIRST thing you do: twenty lines of Python that run the
// original on one input and save a tensor after each stage.
//
// A reference file is a safetensors like any other: a
// registry, a store. But you do not want these tensors on the
// device as weights. You want their numbers, here, to look at.
// `reference`, below, reads one tensor into a `zml.Slice`
// (numbers in host memory, with a shape; exercise 013 gave
// you one with `toSliceAlloc`):
//
//     store.view().getShape(key)          ?zml.Shape
//     zml.Slice.alloc(gpa, shape)         room for it
//     store.view().getReader(key, io, &b) a reader on its bytes
//     reader.interface.readSliceAll(slice.data())
//
// And the judge:
//
//     try zml.testing.expectClose(io, want, got, opts);
//
// `want` and `got` are each a Buffer or a Slice, of the same
// shape and dtype (tags are not compared). `opts` is a
// `zml.testing.CompareOpts`, with these defaults:
//
//     .absolute_tolerance = 1e-3
//     .relative_tolerance = 1e-2
//     .minimum_close_fraction = 0.999
//
// Two numbers are CLOSE when
//
//     |want - got| <= absolute + relative * max(|want|, |got|)
//
// and the comparison passes when the fraction of close numbers
// reaches `minimum_close_fraction` (and nothing is NaN). Two
// honest computations of the same thing never give the same
// bits: PyTorch and ZML do not add 784 products in the same
// order. Here they differ by about 0.000004. Hence a tolerance,
// and hence the danger: a tolerance is also a way to pass.
//
// When it fails, `expectClose` returns
// `error.TestUnexpectedResult` and prints a report on stderr.
// This is the one you are about to get, for parcel 0:
//
//     nan_or_inf: false
//     max_absolute_error: 28.528177
//     mean_absolute_error: 12.563228
//     rmse: 15.679107
//     close_fraction: 0.1
//     p50_absolute_error: 11.806733
//     p90_absolute_error: 28.377861
//     ...
//
// Learn to read it, it tells you what KIND of bug you have:
//
//  - errors around 1e-6, 1e-5: no bug. Rounding.
//  - errors around 1e-3, 1e-2, on most numbers: the right
//    computation, done slightly differently. A dtype (f16
//    where PyTorch had f32), an epsilon, an approximated
//    activation (chapter 18 has one).
//  - errors as big as the numbers themselves, like here: a
//    wrong computation. A missing or extra operation, the
//    wrong weight, a swapped axis.
//  - `close_fraction`: 0.1 of ten scores is ONE score. One is
//    right, nine are wrong. Which one is right, do you think,
//    given that the digit came out right?
//
// And when the end is wrong, do not stare at the code. BISECT:
// compare half-way. If `hidden` matches and `logits` does
// not, the bug sits between the two, and with a model of
// forty layers that is how you find it in six runs.
//
// Basile's program is below, and it runs, and it prints 8 / 8
// everywhere he looked. Three things are wrong with it:
//
//  1. his `opts`. He "relaxed them a little, because it kept
//     failing". Work out what a relative tolerance of 1 lets
//     through, and give the defaults back;
//  2. he never checked the hidden layer. Do it: a second Exe,
//     on `.hidden`, and a count like the one of the scores;
//  3. and then the bug itself, somewhere in the model. The
//     two counts will tell you where to look. (Basile says he
//     copied that line from `zml/examples/mnist/mnist.zig`.
//     He did. Go and see. That program finds the right digit
//     on 9826 parcels of the register, and PyTorch on 9831.)
//
const std = @import("std");
const zml = @import("zml");

const checkpoint_path = "zmling/17_customs_office/mnist.safetensors";
const register_path = "zmling/17_customs_office/t10k-images.idx3-ubyte";
const reference_path = "zmling/17_customs_office/activations.safetensors";

/// How close is close.
const opts: zml.testing.CompareOpts = .{ .relative_tolerance = 1 };

/// How many parcels are checked (the reference has 8 hidden layers).
const parcels = 8;

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

    /// image: {.h = 28, .w = 28} u8 -> {.d = 500} f32, what the
    /// second layer receives.
    pub fn hidden(self: Net, image: zml.Tensor) zml.Tensor {
        const pixels = image.convert(.f32).divByConst(255).merge(.{ .d = .{ .h, .w } });
        return self.fc1.forward(pixels, .f32).relu().rename(.{ .dout = .d });
    }

    /// image: {.h = 28, .w = 28} u8 -> the scores: {.dout = 10} f32
    pub fn forward(self: Net, image: zml.Tensor) zml.Tensor {
        return self.fc2.forward(self.hidden(image), .f32).relu();
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

/// Row `i` of a reference of rank 2: {n, d} -> {d}. No copy: it
/// is a window on the same memory.
fn row(slice: zml.Slice, i: usize) zml.Slice {
    return slice.subSlice(0, @intCast(i), 1).squeeze(0);
}

/// Is `got` what PyTorch computed? When it is not, `expectClose`
/// prints its report on stderr.
fn isClose(io: std.Io, want: zml.Slice, got: zml.Buffer) bool {
    zml.testing.expectClose(io, want, got, opts) catch return false;
    return true;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The register. (The header was checked in exercise 117: the
    // pixels of parcel `i` start at byte 16 + i * 784.)
    const register_bytes = try readFile(gpa, io, register_path);
    defer gpa.free(register_bytes);

    // What PyTorch computed.
    const reference_file = try std.Io.Dir.cwd().realPathFileAlloc(io, reference_path, gpa);
    defer gpa.free(reference_file);

    var reference_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, reference_file);
    defer reference_registry.deinit();

    var reference_store: zml.io.TensorStore = .fromRegistry(gpa, &reference_registry, &.{platform.replicated_mesh});
    defer reference_store.deinit();

    const want_hidden = try reference(gpa, io, &reference_store, "hidden");
    defer want_hidden.free(gpa);
    const want_scores = try reference(gpa, io, &reference_store, "logits");
    defer want_scores.free(gpa);
    const want_digits = try reference(gpa, io, &reference_store, "digits");
    defer want_digits.free(gpa);

    // M. Nist.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, checkpoint_path, gpa);
    defer gpa.free(path);

    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const net: Net = .init(store.view());

    const image: zml.Tensor = .init(.{ .h = 28, .w = 28 }, .u8);

    var exe = try platform.compile(gpa, io, net, .forward, .{image}, .{});
    defer exe.deinit();

    // TODO: `hidden_exe`, the same model, stopped half-way.

    var net_buffers = try net.load(gpa, io, platform, &store);
    defer Net.unload(&net_buffers);

    var same_digits: usize = 0;
    const same_hidden: usize = 0;
    var same_scores: usize = 0;
    for (0..parcels) |parcel| {
        var image_buffer: zml.Buffer = try .fromBytes(io, platform, image.shape(), register_bytes[16 + parcel * 784 ..][0..784]);
        defer image_buffer.deinit();

        var scores: zml.Buffer = try exe.eval(gpa, io, .{ net_buffers, image_buffer });
        defer scores.deinit();

        const values = try scores.getValue([10]f32, io);
        if (std.mem.indexOfMax(f32, &values) == want_digits.items(u8)[parcel]) same_digits += 1;

        if (isClose(io, row(want_scores, parcel), scores)) same_scores += 1;

        // TODO: the same question, half-way through the model:
        // `hidden`, and `same_hidden`.
    }

    try stdout.print("Tolerances: absolute {d}, relative {d}, close fraction {d}\n", .{
        opts.absolute_tolerance,
        opts.relative_tolerance,
        opts.minimum_close_fraction,
    });
    try stdout.print("Digits like PyTorch:       {d} / {d}\n", .{ same_digits, parcels });
    try stdout.print("Hidden layer like PyTorch: {d} / {d}\n", .{ same_hidden, parcels });
    try stdout.print("Scores like PyTorch:       {d} / {d}\n", .{ same_scores, parcels });
    try stdout.flush();
}
