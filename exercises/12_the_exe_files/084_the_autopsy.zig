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
//             ...operations on self.weight, self.bias and x...
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
// The name `forward` is a habit, not a keyword. A struct may
// have several methods, and `.verdict` in that call compiles the
// method `verdict` just as well. Methods may call each other:
// like any helper, the callee is stamped into the caller's
// graph.
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
// The autopsy, step by step:
//
// 1. Write `forward`. You wrote this layer in exercise 067, and
//    this one is simpler: one sighting, no batch.
//
// 2. Write `verdict`: the INDEX of the higher of the two scores,
//    an i32 scalar (0 is alien, 1 is weather balloon). It calls
//    `forward`; finding where the maximum sits is from chapter
//    06.
//
// 3. The compile call for `forward` is missing something. The
//    one for `verdict` is missing entirely.
//
// 4. The model, as numbers: build the `zml.Bufferized(Detector)`
//    from the two arrays.
//
// 5. Whoever wrote the call at the bottom got a panic for it.
//    Its message talks about "argument 0". Count the slots. Then
//    ask for the verdict: one output, so `eval` will do.
//
const std = @import("std");
const zml = @import("zml");

const Detector = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn forward(self: Detector, x: zml.Tensor) zml.Tensor {
        return ???;
    }

    /// The index of the higher score: 0 alien, 1 weather balloon.
    pub fn verdict(self: Detector, x: zml.Tensor) zml.Tensor {
        return ???;
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
    var verdict_exe = ???;
    defer verdict_exe.deinit();

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

    var weights: zml.Bufferized(Detector) = ???;
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

    var verdict: zml.Buffer = ???;
    defer verdict.deinit();

    const names = [_][]const u8{ "alien", "weather balloon" };
    try stdout.print("The detector's verdict: {s}.\n", .{names[@intCast(try verdict.getValue(i32, io))]});
    try stdout.flush();
}
