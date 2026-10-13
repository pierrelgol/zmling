//
// Quiz! Madame Zelda is away for the evening and a client is at
// the door: an American ghost hunter with a very expensive
// thermometer.
//
// Ghosts cause cold spots. The Manor's instruments measure them
// in degrees Celsius; the client wants Fahrenheit:
//
//     fahrenheit = celsius * 1.8 + 32
//
// AND the client pays by the degree of "spectral intensity", a
// number supplied fresh for every reading, by which the Celsius
// values are multiplied first:
//
//     fahrenheit = (celsius * intensity) * 1.8 + 32
//
// So: 1.8 and 32 are laws of nature and may be baked into the
// Exe. `intensity` changes between runs and must be an input.
//
// You have seen everything you need:
//
//     a.mul(b)            element-wise product (scalar tensors
//                         combine with any shape)
//     a.scale(1.8)        multiply by a Zig number, baked in
//     a.addConstant(32)   add a Zig number, baked in
//
//     zml.Tensor.init / zml.Buffer.fromBytes / zml.Buffer.scalar
//     platform.compileFn / exe.eval / buffer.getValue
//
// This is the whole life cycle of a ZML program. Everything
// from here on is "more interesting things to put in the
// ritual". Write it from memory if you can!
//
const std = @import("std");
const zml = @import("zml");

fn toFahrenheit(celsius: zml.Tensor, intensity: zml.Tensor) zml.Tensor {
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

    // Readings from the cellar, the library, the attic, and
    // that one corner of the kitchen.
    const readings = [_]f32{ -40, 0, 10, 37 };

    // Placeholders: four f32 readings tagged .spots, and one f32 scalar.
    const celsius: zml.Tensor = ???;
    const intensity: zml.Tensor = ???;

    var exe = try platform.compileFn(gpa, io, toFahrenheit, ???, .{});
    defer exe.deinit();

    var celsius_buffer: zml.Buffer = try .fromBytes(io, platform, celsius.shape(), std.mem.sliceAsBytes(&readings));
    defer celsius_buffer.deinit();

    for ([_]f32{ 1, 2.5 }) |i| {
        var intensity_buffer: zml.Buffer = ???;
        defer intensity_buffer.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, ???);
        defer result.deinit();

        const fahrenheit = ???;

        try stdout.print("intensity {d}:", .{i});
        for (fahrenheit) |f| try stdout.print(" {d:.1}", .{f});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
