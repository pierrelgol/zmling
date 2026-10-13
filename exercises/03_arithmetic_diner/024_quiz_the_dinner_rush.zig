//
// Quiz! It's Friday, 6 pm, the bowling league just walked in, and
// Flo has handed you the register. "You know the rules, hon."
//
// Four tables. For each one you get:
//
//     plates   how many plates they ordered        whole numbers
//     price    dollars per plate                   floats
//     mood     how the table feels about Gus's     floats, any
//              cooking                             sign
//
// The rules, in order:
//
//   1. The bill is plates times price.
//
//   2. Dinner rush coupon: 10 dollars off. But a bill never goes
//      below zero. Flo does not pay people to eat.
//
//   3. The tip is a percentage of that: it grows the bill by a
//      factor of
//
//          1 + 0.5 * sigmoid(mood)
//
//      so between nothing (furious) and 50% (delighted).
//
//   4. The total is rounded UP to whole dollars, and it is an
//      integer from here on. The register has no cents key.
//
//   5. The walk: table number t pays t dollars extra. Tables are
//      numbered 1, 2, 3, 4 and nobody hands you that list. Make
//      it from scratch.
//
// Everything you need is in exercises 017 to 023: element-wise
// operations and their constant forms, unary math, activations,
// `convert`, tensors made in the graph, and chains that nest in
// the right places. Mind the dtypes: main() below is already
// correct and tells you what goes in and what must come out.
//
const std = @import("std");
const zml = @import("zml");

fn register(plates: zml.Tensor, price: zml.Tensor, mood: zml.Tensor) zml.Tensor {
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

    const plates: zml.Tensor = .init(.{ .tables = 4 }, .i32);
    const price: zml.Tensor = .init(.{ .tables = 4 }, .f32);
    const mood: zml.Tensor = .init(.{ .tables = 4 }, .f32);

    var exe = try platform.compileFn(gpa, io, register, .{ plates, price, mood }, .{});
    defer exe.deinit();

    var plates_buffer: zml.Buffer = try .fromBytes(io, platform, plates.shape(), std.mem.sliceAsBytes(&[_]i32{ 2, 1, 6, 4 }));
    defer plates_buffer.deinit();
    var price_buffer: zml.Buffer = try .fromBytes(io, platform, price.shape(), std.mem.sliceAsBytes(&[_]f32{ 7.5, 6, 3.25, 5 }));
    defer price_buffer.deinit();
    var mood_buffer: zml.Buffer = try .fromBytes(io, platform, mood.shape(), std.mem.sliceAsBytes(&[_]f32{ 0, 3, -2, 2 }));
    defer mood_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ plates_buffer, price_buffer, mood_buffer });
    defer result.deinit();

    try stdout.print("Ring up {f}: {any}\n", .{
        exe.output_shapes[0],
        try result.getValue([4]i32, io),
    });
    try stdout.flush();
}
