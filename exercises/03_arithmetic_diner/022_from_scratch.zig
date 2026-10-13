//
// "Everything here is made from scratch." -- the menu
//
// Until now, every tensor in a recipe came from outside: a
// placeholder, filled by a Buffer. But some tensors are known
// entirely when you write the code. The number 3. A row of
// zeros. "0, 1, 2, 3, ..." Those don't need to come through the
// door; the kitchen can make them.
//
//     zml.Tensor.scalar(3, .f32)
//         A rank 0 constant. The value is any Zig number, the
//         dtype is yours to choose. This is what `addConstant`
//         and friends call for you.
//
//     zml.Tensor.constant(.{ .f32 = 1.5 })
//         The same thing, one floor down: it takes a
//         `zml.DataType.Value`, a tagged union with one field per
//         dtype. `some_dtype.zero()` and `some_dtype.one()` make
//         such a Value, handy when the dtype is only known at
//         run time: `Tensor.constant(x.dtype().zero())`.
//
//     zml.Tensor.zeroes(shape)
//         A tensor of the given `zml.Shape`, full of zeros.
//
//     zml.Tensor.arange(.{ .start = 2, .end = 26, .step = 4 }, .i32)
//         A rank 1 tensor counting from `start` (default 0) up to
//         but NOT including `end`, by `step` (default 1). This
//         one is { 2, 6, 10, 14, 18, 22 }. Python's `range`.
//
//     zml.Tensor.iota(shape, axis)
//         A tensor of the given shape where every element holds
//         its own INDEX along `axis`. For a shape {rows=2,cols=3}:
//
//             iota(shape, .rows)       iota(shape, .cols)
//               0 0 0                    0 1 2
//               1 1 1                    0 1 2
//
//         Careful: the result is always an INTEGER tensor (.i32,
//         or .i64 for absurdly long axes), whatever the dtype of
//         the shape you gave. They're indices, after all.
//
// (There is no `ones`. You rarely need a whole tensor of ones
// when a single `Tensor.scalar(1, dtype)` and the scalar rule
// will do.)
//
// Now the fine print, and you know it already. These are
// CONSTRUCTORS, but they are not placeholders: each one records
// an instruction in the graph ("make a constant here"). So they
// follow rule number one of Graymoor Manor: only inside a compile
// call. `Tensor.scalar(3, .f32)` in main() dies with the same
// "attempt to use null value" in `Compiler.current` as any ghost
// touched in the hallway.
//
// One nice consequence: a recipe may take NO input at all.
//
//     fn countdown() zml.Tensor { ... }
//
//     var exe = try platform.compileFn(gpa, io, countdown, .{}, .{});
//     var result = try exe.eval(gpa, io, .{});
//
// The diner has two jobs for you.
//
// 1. Tonight's order tickets are numbered 101 to 105. Print them.
//
// 2. Two tables of three seats each; `bills` is what each seat
//    ordered. Gus charges for the walk: seat number s (0, 1, 2:
//    seat 0 is next to the kitchen) pays s * 0.5 dollars extra.
//    And there is a house minimum: nobody leaves having paid less
//    than 3 dollars.
//
// The dishwasher had the right idea and made three mistakes.
// You've met two of them before; the third one doesn't crash, it
// just overcharges the wrong people.
//
// He also never got around to cooking the tickets: part 1 of
// main() prints a result nobody computed. A recipe without
// input, compiled and run. The sample above is most of it.
//
const std = @import("std");
const zml = @import("zml");

/// Ticket numbers 101, 102, 103, 104, 105, as .i32.
fn tickets() zml.Tensor {
    return zml.Tensor.arange(.{ .start = ???, .end = ??? }, .i32);
}

/// bills + 0.5 * seat number, and at least 3.
fn walkingFee(bills: zml.Tensor) zml.Tensor {
    const seat_number = zml.Tensor.iota(bills.shape(), .tables);
    const house_minimum = zml.Tensor.scalar(3, .f32);
    return bills.add(seat_number.scale(0.5)).maximum(house_minimum);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // 1. Compile `tickets` and run it, cleaning up as you go. No
    //    input: an empty tuple of placeholders, and later an
    //    empty tuple of Buffers.

    try stdout.print("Tickets: {any}\n", .{try tickets_result.getValue([5]i32, io)});

    // 2.
    const bills: zml.Tensor = .init(.{ .tables = 2, .seats = 3 }, .f32);

    var fee_exe = try platform.compileFn(gpa, io, walkingFee, .{bills}, .{});
    defer fee_exe.deinit();

    var bills_buffer: zml.Buffer = try .fromBytes(io, platform, bills.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 2, 4.5, 1 },
        .{ 10, 0.5, 6 },
    }));
    defer bills_buffer.deinit();

    var fee_result: zml.Buffer = try fee_exe.eval(gpa, io, .{bills_buffer});
    defer fee_result.deinit();

    try stdout.print("Each seat pays: {any}\n", .{try fee_result.getValue([2][3]f32, io)});
    try stdout.flush();
}
