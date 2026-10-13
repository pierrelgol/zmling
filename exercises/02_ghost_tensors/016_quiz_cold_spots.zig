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
// That is part 1: one Exe, run twice, for a calm night
// (intensity 1) and a wild one (intensity 2.5).
//
// Part 2. The client then asks how much the wild night moved
// each reading: wild minus calm, in Fahrenheit. That is a second
// ritual, `shift`, and a second séance, and it is cheaper than
// it sounds:
//
//   - Its two inputs look exactly like the OUTPUT of the first
//     Exe. Don't spell that shape out again: the first Exe knows
//     it (exercise 010), and a placeholder can be summoned from a
//     Shape (exercise 009). Two inputs, so two summonings (015).
//
//   - Its bodies already exist. A Buffer that came out of one
//     `eval` can walk straight into another.
//
// You have seen everything you need:
//
//     a.mul(b)            element-wise product (scalar tensors
//                         combine with any shape)
//     a.sub(b)            element-wise difference
//     a.scale(1.8)        multiply by a Zig number, baked in
//     a.addConstant(32)   add a Zig number, baked in
//
//     zml.Tensor.init / zml.Tensor.fromShape
//     zml.Buffer.fromBytes / zml.Buffer.scalar
//     platform.compileFn / exe.output_shapes / exe.eval
//     buffer.getValue / buffer.toSliceAlloc
//
// This is the whole life cycle of a ZML program. Everything
// from here on is "more interesting things to put in the
// ritual". The names, the data and the print statements are
// given; the rest is yours. Write it from memory if you can!
//
const std = @import("std");
const zml = @import("zml");

/// (celsius * intensity) * 1.8 + 32
fn toFahrenheit(celsius: zml.Tensor, intensity: zml.Tensor) zml.Tensor {
    return ???;
}

/// wild - calm
fn shift(calm: zml.Tensor, wild: zml.Tensor) zml.Tensor {
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

    // PART 1
    //
    // Placeholders: four f32 readings tagged .spots, and one f32 scalar.
    const celsius: zml.Tensor = ???;
    const intensity: zml.Tensor = ???;

    // The seance.
    var exe = ???;
    defer exe.deinit();

    // The readings, as a body.
    var celsius_buffer: zml.Buffer = ???;
    defer celsius_buffer.deinit();

    // The two intensities, as bodies: 1 and 2.5.
    var calm_intensity: zml.Buffer = ???;
    defer calm_intensity.deinit();
    var wild_intensity: zml.Buffer = ???;
    defer wild_intensity.deinit();

    // The same Exe, run once for each intensity.
    var calm: zml.Buffer = ???;
    defer calm.deinit();
    var wild: zml.Buffer = ???;
    defer wild.deinit();

    // Both results, back in the land of the living.
    const calm_fahrenheit: [4]f32 = ???;
    const wild_fahrenheit: [4]f32 = ???;

    try stdout.print("intensity 1:", .{});
    for (calm_fahrenheit) |f| try stdout.print(" {d:.1}", .{f});
    try stdout.print("\nintensity 2.5:", .{});
    for (wild_fahrenheit) |f| try stdout.print(" {d:.1}", .{f});
    try stdout.print("\n", .{});

    // PART 2
    //
    // All yours, defers included: two placeholders, a second Exe
    // compiled with `shift`, and its result for the two Buffers
    // you already hold. The lines below expect that result to be
    // a zml.Buffer called `shifted`.

    // This one is on the house: the result, as a Slice.
    const photo: zml.Slice = try shifted.toSliceAlloc(gpa, io);
    defer photo.free(gpa);

    try stdout.print("The wild night moved {f} by:", .{photo.shape});
    for (photo.constItems(f32)) |f| try stdout.print(" {d:.1}", .{f});
    try stdout.print("\n", .{});
    try stdout.flush();
}
