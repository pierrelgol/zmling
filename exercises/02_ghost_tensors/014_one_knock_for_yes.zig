//
// "Spirit, if you are with us, knock once."
//
// Not every input of a model is a big array. A learning rate, a
// temperature, a position in a sequence, a threshold: single
// numbers. Scalars. The lonely sock returns.
//
// As a placeholder:
//
//     const volume: zml.Tensor = .init(.{}, .f32);     // rank 0
//
// As a body:
//
//     var volume_buffer: zml.Buffer = try .scalar(io, platform, 0.5, .f32);
//     defer volume_buffer.deinit();
//
// And back:
//
//     const v: f32 = try some_buffer.getValue(f32, io);
//
// Why make such a number a tensor INPUT at all? You could write
// `knocks.scale(0.5)` in the ritual instead. The answer is what
// happens during the séance. ZML runs your function once and
// records the graph; an ordinary Zig value used in there is
// "baked in": it becomes a constant inside the compiled Exe,
// forever. An Exe compiled with 0.5 will multiply by 0.5 until
// the sun burns out. If a number must be able to change from
// one run to the next WITHOUT recompiling (and compiling a real
// model takes seconds to minutes), it has to walk in through
// the door, as a Buffer.
//
// Here is the promised rule about scalars, the only automatic
// broadcasting ZML does on plain (untagged) tensors:
//
//     a rank 0 tensor combines with a tensor of ANY shape.
//
// `knocks.mul(volume)` multiplies every element of `knocks` by
// the one value of `volume`. But a {1} tensor is NOT a scalar,
// remember? `mul` of a {3,f32} and a {1,f32} is a shape
// mismatch, and ZML tells you so while tracing: a panic during
// `compileFn`, before any data is involved, naming the
// operation and both shapes. That's the kind of early error we
// were promised.
//
// Tonight's spirit knocks, and the apprentice controls how loud.
//
const std = @import("std");
const zml = @import("zml");

/// Every knock, `volume` times as loud.
fn amplify(knocks: zml.Tensor, volume: zml.Tensor) zml.Tensor {
    return knocks.mul(volume);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const knocks: zml.Tensor = .init(.{ .knocks = 3 }, .f32);
    // The volume is ONE number. No box around it.
    const volume: zml.Tensor = .init(.{1}, .f32);

    var exe = try platform.compileFn(gpa, io, amplify, .{ knocks, volume }, .{});
    defer exe.deinit();

    var knocks_buffer: zml.Buffer = try .fromBytes(io, platform, knocks.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3 }));
    defer knocks_buffer.deinit();

    // Same Exe, three different volumes, no recompilation.
    for ([_]f32{ 0.5, 1, 10 }) |v| {
        var volume_buffer: zml.Buffer = try .scalar(io, platform, v, .f32);
        defer volume_buffer.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ knocks_buffer, volume_buffer });
        defer result.deinit();

        try stdout.print("volume {d}: {any}\n", .{ v, try result.getValue([3]f32, io) });
    }
    try stdout.flush();
}
