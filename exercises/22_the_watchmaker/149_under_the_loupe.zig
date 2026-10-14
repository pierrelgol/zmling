//
// Mr. Tick hands Tock a loupe. "Stop guessing what is inside.
// Open it."
//
// There are two ways to turn an .f32 into a .u32, and they
// have nothing in common:
//
//     x.convert(.u32)    the VALUE is kept, the bits change.
//                        1.0 becomes the integer 1.
//     x.bitCast(.u32)    the BITS are kept, the value changes.
//                        1.0 becomes 0x3f800000 = 1065353216.
//
// `convert` is arithmetic. `bitCast` is the loupe: the same
// 32 bits, read as another type. Nothing is computed.
//
//     pub fn bitCast(self: Tensor, dt: DataType) Tensor
//
// When both types have the same size, the shape is unchanged.
// When the target is SMALLER, each element becomes several,
// and ZML appends an axis tagged `.bitcast` to hold them:
//
//     {reading=4, f32}.bitCast(.u8)  ->  {reading=4, bitcast=4, u8}
//
// The bytes come out in memory order, and your CPU is little
// endian: lowest byte first. 1.0 is 0x3f800000, its bytes are
// 00 00 80 3f. This is exactly what sits in a safetensors
// file (exercise 97), and why a file read with the wrong
// dtype gives numbers that look random but never crash.
//
// When the target is LARGER, the last axis is eaten: its
// elements must add up to exactly one element of the target.
//
// With the bits as integers, the three parts of the movement
// come apart with integer tools:
//
//     a.shiftRightLogical(n)     a >> n, zeros come in
//     a.logical(.AND, mask)      a & mask (also .OR, .XOR)
//
// Both take TENSORS on the right: a rank-0 `Tensor.scalar`
// (exercise 22) of the same dtype broadcasts.
//
//     .f32:   s eeeeeeee mmmmmmmmmmmmmmmmmmmmmmm
//     bit:   31 30....23 22...................0
//
// The value is (-1)^s * 1.m * 2^(e - 127). The 127 is the
// "bias": the exponent is stored shifted so that it is never
// negative. 1.0 is 1.0 * 2^0: e = 127, m = 0.
//
// When does a port need this? Reading packed weights (4-bit
// quantized models pack two values per byte), checking that a
// suspicious number is really a NaN and which one, hashing a
// tensor, or proving that two results are the same to the
// last bit instead of "close".
//
// To do:
//
//  1. Tock wrote `asInteger`. It answers, and the answer is
//     not what the customer asked for. Read the two lines of
//     output it produces before you touch it.
//  2. Write `anatomy`: sign, exponent (as stored, biased) and
//     mantissa of each reading, stacked on a new first axis
//     `.part`, in that order. `Tensor.stack` is exercise 57.
//  3. Write `bytesOf`.
//
const std = @import("std");
const zml = @import("zml");

/// The 32 bits of each reading, as an integer.
fn asInteger(x: zml.Tensor) zml.Tensor {
    return x.convert(.u32);
}

/// {reading} f32 -> {part=3, reading} u32: sign, exponent, mantissa.
fn anatomy(x: zml.Tensor) zml.Tensor {
    return ???;
}

/// {reading} f32 -> {reading, bitcast=4} u8
fn bytesOf(x: zml.Tensor) zml.Tensor {
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

    const readings = [_]f32{ 1.0, 0.1, -2.5, 0.15625 };
    const x: zml.Tensor = .init(.{ .reading = readings.len }, .f32);
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&readings));
    defer x_buffer.deinit();

    {
        var exe = try platform.compileFn(gpa, io, asInteger, .{x}, .{});
        defer exe.deinit();
        var out = try exe.eval(gpa, io, .{x_buffer});
        defer out.deinit();
        const slice = try out.toSliceAlloc(gpa, io);
        defer slice.free(gpa);
        try stdout.print("bits:", .{});
        for (slice.items(u32)) |v| try stdout.print(" {x:0>8}", .{v});
        try stdout.print("\n", .{});
    }
    {
        var exe = try platform.compileFn(gpa, io, anatomy, .{x}, .{});
        defer exe.deinit();
        var out = try exe.eval(gpa, io, .{x_buffer});
        defer out.deinit();
        const slice = try out.toSliceAlloc(gpa, io);
        defer slice.free(gpa);
        const parts = slice.items(u32);
        try stdout.print("{f}\n", .{out.shape()});
        for (readings, 0..) |v, i| {
            const sign = parts[i];
            const exponent = parts[readings.len + i];
            const mantissa = parts[2 * readings.len + i];
            try stdout.print("{d}: sign {d}, exponent {d} (2^{d}), mantissa {x:0>6}\n", .{
                v, sign, exponent, @as(i32, @intCast(exponent)) - 127, mantissa,
            });
        }
    }
    {
        var exe = try platform.compileFn(gpa, io, bytesOf, .{x}, .{});
        defer exe.deinit();
        var out = try exe.eval(gpa, io, .{x_buffer});
        defer out.deinit();
        const slice = try out.toSliceAlloc(gpa, io);
        defer slice.free(gpa);
        try stdout.print("{f}\n", .{out.shape()});
        try stdout.print("1.0 on disk:", .{});
        for (slice.items(u8)[0..4]) |v| try stdout.print(" {x:0>2}", .{v});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
