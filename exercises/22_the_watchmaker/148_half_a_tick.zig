//
// Welcome to Tick & Daughter, watchmakers since 1904. Mr. Tick
// builds movements. Tock, his apprentice, sweeps the floor and
// asks questions. Today's question: "Why does the cheap watch
// say 70144 when I set it to 70000?"
//
// Every float is a watch movement with three parts:
//
//     sign | exponent | mantissa
//
// The mantissa holds the significant digits (in binary), the
// exponent says where the point goes. More mantissa bits: finer
// ticks. More exponent bits: a wider range before the hands
// fly off (inf) or stop moving (0).
//
//     dtype   bits   exponent   mantissa   decimal digits
//     .f32     32       8          23         about 7
//     .f16     16       5          10         about 3
//     .bf16    16       8           7         about 2
//
// .f16 and .bf16 cost the same two bytes and spend them
// differently. .f16 ("half") keeps a decent mantissa and a
// tiny range: its largest number is 65504. .bf16 ("brain
// float") is an .f32 with the 16 lowest bits sawn off: the
// full range of .f32, and ticks 65536 times coarser.
//
// Why you care: most language models are published in .bf16
// or .f16. Half the bytes on disk, half the memory on the
// GPU, twice as many numbers through the same memory bus.
// The checkpoint of a big Llama is .bf16; the PyTorch you
// compare against (chapter 20) often ran in .f32. Before you
// call a difference a bug, you must know what the movement
// itself can and cannot hold.
//
// `Tensor.convert(dtype)` (exercise 21) changes the movement
// and ROUNDS each value to the nearest one the new dtype has.
// Going up (.bf16 to .f32) is exact. Going down loses what
// does not fit, for good: converting back up gives the
// rounded value, not the original.
//
// The spacing between two neighbours is not constant. With m
// mantissa bits, numbers in [2^e, 2^(e+1)) are 2^(e-m) apart.
// For .bf16 (m = 7), between 65536 and 131072 the ticks are
// 512 apart: 70000 does not exist, 70144 does.
//
// One more thing, so that nobody feels too proud: .f32 has 24
// significant bits. 16777217 is 2^24 + 1. It does not exist
// in .f32 either. Zig rounded it when it compiled `readings`
// below. (Write it without the `.0` and Zig refuses: "type
// 'f32' cannot represent integer value". Zig is that kind of
// watchmaker. A float literal, it rounds without a word.)
//
// To do:
//
//  1. Write `through`: the same values, as .f32, after a trip
//     through `dtype`.
//  2. Write `weigh`: the bytes a tensor of `shape` takes when
//     stored as `dtype`. Shapes carry their dtype (exercise
//     3), and you have met `withDtype` and `byteSize`.
//  3. In `main`, the .f16 line is yours: compile, run, read.
//     The .bf16 one is given as a model.
//
const std = @import("std");
const zml = @import("zml");

/// What `x` is after being stored as `dtype` and read back.
fn through(x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
    return ???;
}

fn throughBf16(x: zml.Tensor) zml.Tensor {
    return through(x, .bf16);
}

fn throughF16(x: zml.Tensor) zml.Tensor {
    return through(x, .f16);
}

/// The bytes needed to store a tensor of `shape` as `dtype`.
fn weigh(shape: zml.Shape, dtype: zml.DataType) usize {
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

    const readings = [_]f32{ 0.1, 3.14159265, 1000.001, 70000, 16777217.0 };
    const x: zml.Tensor = .init(.{ .reading = readings.len }, .f32);
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&readings));
    defer x_buffer.deinit();

    try stdout.print("f32: ", .{});
    for (readings) |v| try stdout.print(" {d:.4}", .{v});
    try stdout.print("\n", .{});

    {
        var exe = try platform.compileFn(gpa, io, throughBf16, .{x}, .{});
        defer exe.deinit();
        var out = try exe.eval(gpa, io, .{x_buffer});
        defer out.deinit();
        const slice = try out.toSliceAlloc(gpa, io);
        defer slice.free(gpa);
        try stdout.print("bf16:", .{});
        for (slice.items(f32)) |v| try stdout.print(" {d:.4}", .{v});
        try stdout.print("\n", .{});
    }
    // The .f16 line goes here, printed as "f16: " (two spaces).

    // A 4096 x 4096 weight matrix: one layer of a mid-size model.
    const layer: zml.Shape = .init(.{ .dout = 4096, .d = 4096 }, .f32);
    try stdout.print("One layer in f32:  {d} bytes\n", .{weigh(layer, .f32)});
    try stdout.print("One layer in bf16: {d} bytes\n", .{weigh(layer, .bf16)});
    try stdout.print("One layer in f16:  {d} bytes\n", .{weigh(layer, .f16)});
    try stdout.flush();
}
