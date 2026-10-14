//
// A customer wants the grand watch in a cheaper movement.
// "Will it still keep time?" Mr. Tick never guesses. He has
// a trick: run the EXPENSIVE movement with a file held
// against its gears, so that it ticks as coarsely as the
// cheap one would. No new parts, and you see at once which
// gear cannot take it.
//
//     pub fn reducePrecision(self: Tensor,
//                            exponent_bits: i32,
//                            mantissa_bits: i32) Tensor
//
// The tensor KEEPS ITS DTYPE. Each value is rounded to what
// a float with that many exponent and mantissa bits could
// hold, overflow to inf and underflow to 0 included. With
// the table of exercise 148:
//
//     x.reducePrecision(5, 10)   x as an .f16 would see it
//     x.reducePrecision(8, 7)    x as a .bf16 would see it
//
// and for ordinary values it is bit for bit what
// `x.convert(.f16).convert(.f32)` gives. Why bother, then?
// Because the dtype does not
// change, you can drop it in the MIDDLE of an .f32 model,
// after one layer and not the others, without touching a
// weight, a placeholder or a Buffer. It answers "which layer
// of this model cannot live in .f16?" in one line per try.
// (It is also how you make `expectClose` compare a cheap run
// with an expensive one: it refuses two different dtypes
// with `left.dtype (.f32) != right.dtype (.bf16)`.)
//
// The file is not a perfect copy, and you should know where
// it lies. A real .f16 has SUBNORMALS: below its smallest
// ordinary number (about 6.1e-5) it does not drop to zero at
// once, it keeps going down to 6e-8 with fewer and fewer
// mantissa bits. `reducePrecision` has no subnormals: below
// 6.1e-5, it says 0. So the file is slightly PESSIMISTIC for
// tiny values. A layer that survives the file survives the
// real movement; the reverse is not guaranteed.
//
// Today's sensitive gear is the one you wrote in exercise 92
// and again in chapter 21: the RMS norm.
//
//     rms_norm(x) = x / sqrt(mean(x^2) + eps)
//
// Look at x^2 with small activations, say 1e-5. Squared:
// 1e-10. The smallest positive .f16, subnormals included, is
// about 6e-8. The square is ZERO, in the real movement too. So is eps = 1e-8, a value some models
// really use. The mean is 0, 0 + 0 = 0, 1 / sqrt(0) = inf,
// and x * inf is inf. In .f32, the same line is fine.
//
// Now you know why `zml.nn.rmsNorm` starts with
// `var x = x_.convert(.f32); // upcast to improve precision`
// and ends with `.convert(x_.dtype())`. Squares, sums and
// reciprocals are computed in the big movement; only the
// result goes back to the small one. The values of a model
// fit in .f16. Their SQUARES do not.
//
// To do:
//
//  1. Write `asSeenBy`: `reducePrecision` with the right
//     numbers for .f16 and .bf16, and `x` untouched for
//     .f32. Any other dtype: `@panic`.
//  2. `cheapNorm` is the customer's request, taken
//     literally: every step in .f16. It is given.
//  3. Write `carefulNorm`: takes .f16, returns .f16, and is
//     the formula above computed where it is safe. No
//     calling `zml.nn.rmsNorm`: it is what you are compared
//     with.
//
const std = @import("std");
const zml = @import("zml");

const eps = 1e-8;

/// `x`, same dtype, rounded as `dtype` would store it.
fn asSeenBy(x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
    return ???;
}

/// x: {.d} f16 -> {.d} f16, every step in .f16.
fn cheapNorm(x: zml.Tensor) zml.Tensor {
    const mean_square = x.mul(x).mean(.d);
    return x.mul(mean_square.addConstant(eps).rsqrt());
}

/// x: {.d} f16 -> {.d} f16
fn carefulNorm(x: zml.Tensor) zml.Tensor {
    return ???;
}

fn countBad(x: zml.Tensor) zml.Tensor {
    return x.isFinite().not().convert(.i32).flatten().sum(0).squeeze(0);
}

const Bench = struct {
    by_file: zml.Tensor,
    by_convert: zml.Tensor,
    bf16_by_file: zml.Tensor,
    bf16_by_convert: zml.Tensor,
    tiny_by_file: zml.Tensor,
    tiny_by_convert: zml.Tensor,
    square_seen_by_f16: zml.Tensor,
    eps_seen_by_f16: zml.Tensor,
    cheap_bad: zml.Tensor,
    careful: zml.Tensor,
    library: zml.Tensor,
};

/// readings: ordinary values. x: {.d} f32, small activations.
fn bench(readings: zml.Tensor, x: zml.Tensor) Bench {
    const half = x.convert(.f16);
    return .{
        .by_file = asSeenBy(readings, .f16),
        .by_convert = readings.convert(.f16).convert(.f32),
        .bf16_by_file = asSeenBy(readings, .bf16),
        .bf16_by_convert = readings.convert(.bf16).convert(.f32),
        .tiny_by_file = asSeenBy(x, .f16),
        .tiny_by_convert = half.convert(.f32),
        .square_seen_by_f16 = asSeenBy(x.mul(x), .f16),
        .eps_seen_by_f16 = asSeenBy(zml.Tensor.scalar(eps, .f32), .f16),
        .cheap_bad = countBad(cheapNorm(half)),
        .careful = carefulNorm(half).convert(.f32),
        .library = zml.nn.rmsNorm(half, .d, eps).convert(.f32),
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const activations = [_]f32{ 1e-5, -2e-5, 3e-5, 1e-5 };
    const x: zml.Tensor = .init(.{ .d = activations.len }, .f32);
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&activations));
    defer x_buffer.deinit();

    const reading_values = [_]f32{ 0.1, 3.14159265, 1000.001, 70000 };
    const readings: zml.Tensor = .init(.{ .reading = reading_values.len }, .f32);
    var readings_buffer: zml.Buffer = try .fromBytes(io, platform, readings.shape(), std.mem.sliceAsBytes(&reading_values));
    defer readings_buffer.deinit();

    var exe = try platform.compileFn(gpa, io, bench, .{ readings, x }, .{});
    defer exe.deinit();
    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);
    var out: zml.Bufferized(Bench) = undefined;
    runner.run(io, .{ readings_buffer, x_buffer }, &out, .{ .wait = true });
    defer zml.Buffer.deinitAll(Bench, &out);

    const by_file = try out.by_file.toSliceAlloc(gpa, io);
    defer by_file.free(gpa);
    const by_convert = try out.by_convert.toSliceAlloc(gpa, io);
    defer by_convert.free(gpa);
    try stdout.print("The file is an f16, bit for bit: {}\n", .{std.mem.eql(u8, by_file.constData(), by_convert.constData())});

    const bf16_by_file = try out.bf16_by_file.toSliceAlloc(gpa, io);
    defer bf16_by_file.free(gpa);
    const bf16_by_convert = try out.bf16_by_convert.toSliceAlloc(gpa, io);
    defer bf16_by_convert.free(gpa);
    try stdout.print("The file is a bf16, bit for bit: {}\n", .{std.mem.eql(u8, bf16_by_file.constData(), bf16_by_convert.constData())});

    const tiny_by_file = try out.tiny_by_file.toSliceAlloc(gpa, io);
    defer tiny_by_file.free(gpa);
    const tiny_by_convert = try out.tiny_by_convert.toSliceAlloc(gpa, io);
    defer tiny_by_convert.free(gpa);
    try stdout.print("1e-5, seen by the file: {d}\n", .{tiny_by_file.items(f32)[0]});
    try stdout.print("1e-5, seen by a real f16: {e:.3}\n", .{tiny_by_convert.items(f32)[0]});

    const squares = try out.square_seen_by_f16.toSliceAlloc(gpa, io);
    defer squares.free(gpa);
    try stdout.print("The squares, seen by an f16: {any}\n", .{squares.items(f32)});
    try stdout.print("eps, seen by an f16: {d}\n", .{try out.eps_seen_by_f16.getValue(f32, io)});
    try stdout.print("The cheap norm: {d} values out of 4 are not finite\n", .{try out.cheap_bad.getValue(i32, io)});

    const careful = try out.careful.toSliceAlloc(gpa, io);
    defer careful.free(gpa);
    const library = try out.library.toSliceAlloc(gpa, io);
    defer library.free(gpa);
    try stdout.print("The careful norm:", .{});
    for (careful.items(f32)) |v| try stdout.print(" {d:.3}", .{v});
    try stdout.print("\n", .{});
    try stdout.print("Same as zml.nn.rmsNorm, bit for bit: {}\n", .{std.mem.eql(u8, careful.constData(), library.constData())});
    try stdout.flush();
}
