//
// Back at the hut, Ingrid's shelf: forty winters of plaster
// casts. "A cast is not a foot," she says. "It is the shape of
// a foot, and a label saying where I found it. Take a cast to
// another valley and it tells you nothing."
//
// She is describing `zml.Tensor`. Here is all of it, from
// `zml/zml/core/tensor.zig` (minus one flag):
//
//     pub const Tensor = struct {
//         id: Tensor.Id,
//         _shape: Shape,
//         _value: ?*const mlir.Value = null,
//         ...
//     };
//
// A number, a shape, and an optional pointer. No data, ever.
// There are exactly two ways to get one.
//
// 1. `Tensor.init(shape, dtype)`, the placeholder:
//
//        return .{ .id = nextTensorId(), ._shape = shape_ };
//
//    A fresh id and a shape. `_value` stays null. It asks
//    nothing from any Compiler, and that is why you have always
//    been allowed to write it in `main`.
//
// 2. Any operation. Every one of them ends the same way:
//
//        const op = dialects.stablehlo.add(...)
//            .appendTo(currentBlock());
//        return _result(shape, op.result(0));
//
//    One line is added to the program that the current
//    Compiler is writing, and the Tensor you get back POINTS at
//    the result of that line. `_value` is set. The memory it
//    points to belongs to the Compiler.
//
// When an op needs its operands, it calls `value()` on each:
//
//     pub fn value(self: Tensor) *const mlir.Value {
//         const scope = Compiler.current().currentScope();
//         return scope.id_to_argument.get(self.id) orelse
//             self._value orelse
//             @panic("Something went really wrong, tensor is
//                     not an argument nor has an mlir.Value");
//     }
//
// Read it slowly, it is the heart of the chapter.
//
//   - First, "is this id one of the ARGUMENTS of the program I
//     am writing?" `createBlockArguments` filled that table
//     just before your function was called: one entry per
//     Tensor found in the args of the compile call.
//   - If not, "does it carry a pointer?" Then it is the result
//     of an op.
//   - Neither: the panic. A placeholder nobody declared.
//
// Three things follow, and each one bites somebody once.
//
//   A placeholder is only a label. It means something in a
//   compile if, and only if, it was in the args of THAT compile
//   call. The good news: nothing is consumed. The same
//   placeholder can be an argument of as many compiles as you
//   like.
//
//   Inside `forward`, an argument STILL has `_value == null`.
//   It is found by its id, not by a pointer. (Remember this
//   line. Exercise 193 is built on it.)
//
//   The result of an op lives as long as its Compiler, and
//   `compileInternal` says `defer compiler.deinit()`. Keep such
//   a Tensor in a global "to save work", use it in the next
//   compile, and `value()` finds a pointer and hands it over,
//   happily. It points into a program that was freed. No
//   message: a crash somewhere inside MLIR if you are lucky.
//   A Tensor never outlives the call that traced it.
//
// (Within ONE trace, reuse a Tensor as often as you like: it is
// a small struct, copies are free, and all of them point at the
// same line of the program.)
//
// Today's casts. Each print has a depth, {.print}, and the wind
// has blown some snow in since, one number for the whole trail.
// The morning reading is `depth + wind`. In the evening the
// snow has settled to half: `depth / 2 + wind`. Ingrid, saving
// plaster, made the `wind` placeholder once, left it on the
// shelf (a global), and `evening` helps itself.
//
// To do:
//
//  1. `morning` writes three facts in the notebook, at trace
//     time: does each of its arguments carry a pointer, and
//     does its result? `_value` is an optional: compare it
//     with `null`.
//  2. Run it. Read the panic and say which of the three
//     questions of `value()` got which answer. Then give
//     `evening` what it is missing. The shelf can go.
//  3. In `main`: the second Exe and its run. No new
//     placeholders: the two you have will do.
//
const std = @import("std");
const zml = @import("zml");

/// Written by `morning`, at trace time.
var notebook: struct {
    depth_has_value: bool = false,
    wind_has_value: bool = false,
    reading_has_value: bool = false,
} = .{};

/// depth: {.print}, wind: {} -> {.print}
fn morning(depth: zml.Tensor, wind: zml.Tensor) zml.Tensor {
    const reading = depth.add(wind);

    notebook.depth_has_value = ???;
    notebook.wind_has_value = ???;
    notebook.reading_has_value = ???;

    return reading;
}

/// Ingrid's shelf: the `wind` placeholder, put there by `main`.
var shelf: zml.Tensor = undefined;

/// depth: {.print} -> {.print}
fn evening(depth: zml.Tensor) zml.Tensor {
    return depth.scale(0.5).add(shelf);
}

const depth_data = [4]f32{ 12, 30, 8, 20 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const depth: zml.Tensor = .init(.{ .print = 4 }, .f32);
    const wind: zml.Tensor = .init(.{}, .f32);
    shelf = wind;

    try stdout.print("in main, the placeholder carries a pointer: {}\n", .{depth._value != null});

    var morning_exe = try platform.compileFn(gpa, io, morning, .{ depth, wind }, .{});
    defer morning_exe.deinit();

    try stdout.print("while tracing: depth {}, wind {}, their sum {}\n", .{
        notebook.depth_has_value,
        notebook.wind_has_value,
        notebook.reading_has_value,
    });

    var evening_exe = try platform.compileFn(gpa, io, evening, .{depth}, .{});
    defer evening_exe.deinit();

    try stdout.print("two placeholders, two compiles: {d} + {d} inputs\n", .{ morning_exe.input_shapes.len, evening_exe.input_shapes.len });

    var depth_buffer: zml.Buffer = try .fromBytes(io, platform, depth.shape(), std.mem.sliceAsBytes(&depth_data));
    defer depth_buffer.deinit();
    var wind_buffer: zml.Buffer = try .scalar(io, platform, 3, .f32);
    defer wind_buffer.deinit();

    var morning_reading: zml.Buffer = try morning_exe.eval(gpa, io, .{ depth_buffer, wind_buffer });
    defer morning_reading.deinit();
    try stdout.print("morning: {any}\n", .{try morning_reading.getValue([4]f32, io)});

    var evening_reading: zml.Buffer = ???;
    defer evening_reading.deinit();
    try stdout.print("evening: {any}\n", .{try evening_reading.getValue([4]f32, io)});

    try stdout.flush();
}
