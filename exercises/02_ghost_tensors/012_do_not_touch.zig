//
// RULE NUMBER ONE of Graymoor Manor, embroidered on a cushion in
// the hallway:
//
//     DO NOT TOUCH THE GHOSTS OUTSIDE OF A SEANCE.
//
// Every tensor operation (`add`, `mul`, and the hundred others
// you'll learn) records itself into "the graph currently being
// compiled". Outside of `compileFn` there is no such graph.
// There is nowhere to write the recipe. So this:
//
//     pub fn main(...) !void {
//         const a: zml.Tensor = .init(.{3}, .f32);
//         const b: zml.Tensor = .init(.{3}, .f32);
//         const c = a.add(b);       // <-- touching ghosts in the hallway
//     }
//
// compiles perfectly well as Zig (it's just a method call!) and
// then blows up when it runs. Run this exercise as-is and look
// at the panic. The top of the stack trace says something like
//
//     ... Compiler.zig ... in current
//     ... tensor.zig   ... in add
//
// "The current compiler" is the thing that doesn't exist. Learn
// to recognize this one: sooner or later everybody calls a
// tensor operation from the wrong place, and the message doesn't
// exactly spell it out.
//
// This goes for operations that CREATE tensors from nothing too
// (`zml.Tensor.scalar(1, .f32)` and friends, coming up soon).
// The ONLY tensor constructors allowed in the hallway are the
// placeholders: `Tensor.init` and `Tensor.fromShape`.
//
// So the mental model is:
//
//     main()                      the hallway. Zig values, Shapes,
//                                 placeholders, Buffers.
//       '-- compileFn(ritual)     the seance. ZML calls your
//             '-- ritual()        function once; tensor operations
//                                 are legal in here, and in anything
//                                 it calls.
//
// The apprentice wanted the sum of two chills, minus a third.
// (`a.sub(b)` is the element-wise difference.) Please move the
// ghost-touching to where it belongs.
//
const std = @import("std");
const zml = @import("zml");

/// cold + colder - warm
fn haunt(cold: zml.Tensor, colder: zml.Tensor, warm: zml.Tensor) zml.Tensor {
    // TODO: this is where the ghost-touching should happen.
    _ = colder;
    _ = warm;
    return cold;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const cold: zml.Tensor = .init(.{3}, .f32);
    const colder: zml.Tensor = .init(.{3}, .f32);
    const warm: zml.Tensor = .init(.{3}, .f32);

    // The apprentice's attempt. In the hallway!
    const chill = cold.add(colder).sub(warm);
    _ = chill;

    var exe = try platform.compileFn(gpa, io, haunt, .{ cold, colder, warm }, .{});
    defer exe.deinit();

    var cold_buffer: zml.Buffer = try .fromBytes(io, platform, cold.shape(), std.mem.sliceAsBytes(&[_]f32{ 10, 20, 30 }));
    defer cold_buffer.deinit();
    var colder_buffer: zml.Buffer = try .fromBytes(io, platform, colder.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3 }));
    defer colder_buffer.deinit();
    var warm_buffer: zml.Buffer = try .fromBytes(io, platform, warm.shape(), std.mem.sliceAsBytes(&[_]f32{ 4, 4, 4 }));
    defer warm_buffer.deinit();

    // One Buffer per placeholder, in the same order.
    var result: zml.Buffer = try exe.eval(gpa, io, .{ cold_buffer, colder_buffer, warm_buffer });
    defer result.deinit();

    try stdout.print("A chill runs down your spine: {any}\n", .{try result.getValue([3]f32, io)});
    try stdout.flush();
}
