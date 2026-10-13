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
// Here is the promised rule about scalars, the one piece of
// automatic stretching ("broadcasting") you can always count on:
//
//     a rank 0 tensor combines with a tensor of ANY shape.
//
// (There is some fine print for other shapes. It gets its own
// chapter, "Broadcast News".)
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
// The compiler will stop at the blanks first. Once they are
// filled in, the panic described above is waiting for you.
//
// Then comes a second, tiny séance: a whisper goes in, its echo
// comes out. One scalar each way. The ritual and the placeholder
// are there; every other step is yours to write. The first
// séance, right above it, is your model.
//
const std = @import("std");
const zml = @import("zml");

/// Every knock, `volume` times as loud.
fn amplify(knocks: zml.Tensor, volume: zml.Tensor) zml.Tensor {
    return knocks.mul(volume);
}

/// The echo of a whisper is the whisper, squared.
fn echo(whisper: zml.Tensor) zml.Tensor {
    return whisper.mul(whisper);
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
        // This round's volume, `v`, as a body.
        var volume_buffer: zml.Buffer = ???;
        defer volume_buffer.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ knocks_buffer, volume_buffer });
        defer result.deinit();

        try stdout.print("volume {d}: {any}\n", .{ v, try result.getValue([3]f32, io) });
    }

    // The second seance. One rank 0 placeholder...
    const whisper: zml.Tensor = .init(.{}, .f32);

    // ...compiled with the `echo` ritual...
    var echo_exe = ???;
    defer echo_exe.deinit();

    // ...a body holding the number 3, as an f32...
    var whisper_buffer: zml.Buffer = ???;
    defer whisper_buffer.deinit();

    // ...run through the Exe...
    var echo_buffer: zml.Buffer = ???;
    defer echo_buffer.deinit();

    // ...and brought back as a plain Zig number.
    const loudness: f32 = ???;

    try stdout.print("A whisper of 3 echoes back as {d}.\n", .{loudness});
    try stdout.flush();
}
