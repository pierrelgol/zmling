//
// "Order up!"
//
// You've been chaining methods since the Manor without anybody
// making a fuss about it. Time to make a fuss, because this is
// how ALL ZML code is written, and it has one trap.
//
// Every operation takes Tensors and returns a new Tensor, so the
// next operation can be called on the result directly:
//
//     x.sub(mean).mul(gain).tanh()
//
// A chain is an assembly line. It reads left to right, in the
// order things happen: take x, subtract the mean, multiply by
// the gain, squash. Compare with the math notation, which you
// read inside-out: tanh((x - mean) * gain).
//
// THE TRAP: methods have no operator precedence. In math (and in
// Zig) a + b * c means a + (b * c), because multiplication binds
// tighter. A chain doesn't know or care:
//
//     a.add(b).mul(c)        is (a + b) * c
//     a.add(b.mul(c))        is  a + (b * c)
//
// The parentheses of the formula become the NESTING of the calls.
// Whatever must be computed first, on its own, goes inside an
// argument.
//
// A few habits from ZML's own source code:
//
//   - Name the intermediate results when a chain gets long, or
//     when a value is needed twice. Using a tensor twice costs
//     nothing: it is one node in the graph, computed once.
//
//         const scaled_x_cube = x.mul(x).mul(x).scale(0.044715);
//
//   - Ordinary Zig math on ordinary Zig numbers happens while
//     tracing and ends up baked in as a constant. This line is
//     from `gelu`: the square root is computed by Zig, once.
//
//         const beta = std.math.sqrt(2.0 / std.math.pi);
//         const tanh_ = x.add(scaled_x_cube).scale(beta).tanh();
//
//   - A recipe is a plain Zig function, so it can call other
//     plain Zig functions that take and return Tensors. That is
//     all a "layer" is, in the end.
//
//   - Where a Tensor is expected, Zig lets you drop the type name
//     of a constructor: `x.maximum(.scalar(3, .f32))` is
//     `x.maximum(zml.Tensor.scalar(3, .f32))`.
//
// Gus wants a "doneness score" for the four burners, and it is
// two classic formulas glued together. Both show up, in one form
// or another, in every modern neural network.
//
//   1. Standardize: how far is each temperature from the mean,
//      in units of standard deviation?
//
//          z = (temps - mean) / sqrt(variance + epsilon)
//
//      (`epsilon` is a tiny number that keeps the division from
//      blowing up if the variance happens to be zero.)
//
//   2. Softplus, one more smooth cousin of relu:
//
//          softplus(z) = log(1 + e^z)
//
// Earl offered to type in the first one. Earl reads formulas
// left to right, like a menu.
//
const std = @import("std");
const zml = @import("zml");

const epsilon = 1e-5;

/// (temps - mean) / sqrt(variance + epsilon)
fn standardize(temps: zml.Tensor, mean: zml.Tensor, variance: zml.Tensor) zml.Tensor {
    return temps.sub(mean).div(variance).addConstant(epsilon).sqrt();
}

/// log(1 + e^z)
fn softplus(z: zml.Tensor) zml.Tensor {
    return ???;
}

fn doneness(temps: zml.Tensor, mean: zml.Tensor, variance: zml.Tensor) zml.Tensor {
    return softplus(standardize(temps, mean, variance));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const temps: zml.Tensor = .init(.{ .burners = 4 }, .f32);
    const mean: zml.Tensor = .init(.{}, .f32);
    const variance: zml.Tensor = .init(.{}, .f32);

    var exe = try platform.compileFn(gpa, io, doneness, .{ temps, mean, variance }, .{});
    defer exe.deinit();

    var temps_buffer: zml.Buffer = try .fromBytes(io, platform, temps.shape(), std.mem.sliceAsBytes(&[_]f32{ 300, 350, 400, 450 }));
    defer temps_buffer.deinit();
    var mean_buffer: zml.Buffer = try .scalar(io, platform, 375, .f32);
    defer mean_buffer.deinit();
    // A standard deviation of 50 degrees.
    var variance_buffer: zml.Buffer = try .scalar(io, platform, 2500, .f32);
    defer variance_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ temps_buffer, mean_buffer, variance_buffer });
    defer result.deinit();

    try stdout.print("Doneness:", .{});
    for (try result.getValue([4]f32, io)) |score| try stdout.print(" {d:.3}", .{score});
    try stdout.print("\n", .{});
    try stdout.flush();
}
