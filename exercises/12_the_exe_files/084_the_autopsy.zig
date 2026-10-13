//
// "What we have on the table, Molder, is not a function."
//
// Up to now, every tensor your code used came in as a separate
// argument. Fine for `add(a, b)`. But a neural network has
// WEIGHTS: tensors of learned numbers, dozens or thousands of
// them, that belong together and never change between calls.
// Nobody wants a function with 200 parameters.
//
// So a model, in ZML, is a plain Zig struct. Its fields are
// Tensors. Its computation is a method, by convention called
// `forward`:
//
//     const Detector = struct {
//         weight: zml.Tensor,
//         bias: zml.Tensor,
//
//         pub fn forward(self: Detector, x: zml.Tensor) zml.Tensor {
//             return self.weight.dot(x, .d).add(self.bias);
//         }
//     };
//
// No base class, no registration, no magic. `self` is simply the
// first argument, and you know what ZML does with a struct
// argument: it finds the Tensors inside, in field order. The
// weights are INPUTS of the Exe, like any other.
//
// The model value itself is made of placeholders:
//
//     const model: Detector = .{
//         .weight = .init(.{ .d_out = 2, .d = 3 }, .f32),
//         .bias = .init(.{ .d_out = 2 }, .f32),
//     };
//
// To compile a method, there is a small helper next to
// `compileFn`:
//
//     var exe = try platform.compile(gpa, io, model, .forward, .{x}, .{});
//
// The model, the NAME of the method as an enum literal, then a
// tuple with the remaining arguments (everything after `self`).
// It is exactly `compileFn(gpa, io, Detector.forward, .{ model, x }, .{})`
// with less typing. (Declare the method `pub`.)
//
// So this Exe has three inputs: weight, bias, x. At run time it
// wants three Buffers. Two of them are the model's. For those,
// you need a struct shaped like `Detector`, but holding Buffers
// instead of Tensors. You have met its type already:
//
//     zml.Bufferized(Detector)
//
// It is computed by ZML at compile time from your struct: same
// field names, every `zml.Tensor` replaced by a `zml.Buffer`.
// You build one like any struct:
//
//     var weights: zml.Bufferized(Detector) = .{
//         .weight = try .fromBytes(io, platform, model.weight.shape(), ...),
//         .bias = try .fromBytes(io, platform, model.bias.shape(), ...),
//     };
//     defer weights.weight.deinit();
//     defer weights.bias.deinit();
//
// (Real models read these numbers from a file. That's chapter
// 14. Here, they are small enough to type.)
//
// And then it goes where `self` went: FIRST.
//
//     args.set(.{ weights, x_buffer });
//
// `set` digs the Buffers out of the struct, in field order, the
// same way the compiler dug out the Tensors. Tensors in, Buffers
// in, same walk, same order. The whole design rests on that
// symmetry.
//
// The body on the table: a detector with two output scores,
// "alien" and "weather balloon", computed from three
// measurements of a sighting (glow, hum, speed):
//
//     score = weight . x + bias
//
// The compile call is missing something, and whoever wrote the
// call at the bottom got a panic for it. Its message talks about
// "argument 0". Count the slots.
//
const std = @import("std");
const zml = @import("zml");

const Detector = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn forward(self: Detector, x: zml.Tensor) zml.Tensor {
        return self.weight.dot(x, .d).add(self.bias);
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

    // The model, as placeholders.
    const model: Detector = .{
        .weight = .init(.{ .d_out = 2, .d = 3 }, .f32),
        .bias = .init(.{ .d_out = 2 }, .f32),
    };
    const x: zml.Tensor = .init(.{ .d = 3 }, .f32);

    var exe = try platform.compile(gpa, io, model, ???, .{x}, .{});
    defer exe.deinit();

    try stdout.print("The Exe has {d} inputs: {f}, {f} and {f}.\n", .{
        exe.input_shapes.len,
        exe.input_shapes[0],
        exe.input_shapes[1],
        exe.input_shapes[2],
    });

    // The model, as numbers.
    //                            glow hum speed
    const weight_data = [2][3]f32{
        .{ 1, 2, 0 }, // alien
        .{ 0, -1, 3 }, // weather balloon
    };
    const bias_data = [2]f32{ 0.5, -1 };

    var weights: zml.Bufferized(Detector) = .{
        .weight = try .fromBytes(io, platform, model.weight.shape(), std.mem.sliceAsBytes(&weight_data)),
        .bias = try .fromBytes(io, platform, model.bias.shape(), std.mem.sliceAsBytes(&bias_data)),
    };
    defer weights.weight.deinit();
    defer weights.bias.deinit();

    // A sighting over Lake Okobogee: some glow, a little hum, fast.
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[_]f32{ 2, 1, 4 }));
    defer x_buffer.deinit();

    var args = try exe.args(gpa);
    defer args.deinit(gpa);
    var results = try exe.results(gpa);
    defer results.deinit(gpa);

    args.set(.{ x_buffer, weights });
    exe.call(io, args, &results, .{ .wait = true });

    var scores = results.get(zml.Buffer);
    defer scores.deinit();

    const s = try scores.getValue([2]f32, io);
    try stdout.print("alien: {d}, weather balloon: {d}\n", .{ s[0], s[1] });
    try stdout.flush();
}
