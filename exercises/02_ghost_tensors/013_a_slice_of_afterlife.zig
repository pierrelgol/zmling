//
// So far, bodies came from a Zig array and results went back
// into a Zig array. That only works when shapes are known when
// you write the program. Real tensors are big, and their sizes
// often come from a file or the command line.
//
// The third member of the family handles that:
//
//     zml.Tensor   ghost      shape + dtype           (compile time)
//     zml.Buffer   body       shape + dtype + numbers, on the device
//     zml.Slice    photograph shape + dtype + numbers, in YOUR memory
//
// A `zml.Slice` is to tensors what `[]T` is to arrays: a view on
// ordinary host memory (bytes you can read and write), together
// with a Shape saying how to interpret it.
//
// A Slice can borrow memory you already have:
//
//     const s: zml.Slice = .init(shape, std.mem.sliceAsBytes(&my_array));
//
// or own freshly allocated memory:
//
//     const s: zml.Slice = try .alloc(gpa, shape);
//     defer s.free(gpa);
//
// To get at the elements, tell it which Zig type they are:
//
//     for (s.items(f32)) |*x| x.* = 0;      // []f32
//     const view = s.constItems(f32);       // []const f32
//
// And it converts from and to Buffers:
//
//     var body: zml.Buffer = try .fromSlice(io, platform, s);    // host -> device
//
//     const out: zml.Slice = try body.toSliceAlloc(gpa, io);     // device -> host
//     defer out.free(gpa);
//
// (`Buffer.fromBytes`, from the previous exercises, is the same
// thing as `fromSlice` minus the Slice.)
//
// Now a WARNING, because Madame Zelda's cat is about to
// demonstrate it. `items(T)` trusts you. It does not check that
// T matches the dtype of the Slice; it just reinterprets the
// bytes. Ask an .f32 slice for `items(i32)` and you get a
// perfectly valid []i32 pointing at float bits. Write the
// integer 3 through it and the float that comes out the other
// end is 0.000000000000000000000000000000000000000000004.
//
// The apprentice photographed five ghosts and wrote down their
// ages, in years, as integers. The séance (it makes everybody
// one year older) wants floats.
//
// Before you can enjoy that, two lines are yours to write: the
// séance itself (exercise 010 showed you how, and from now on
// you'll be asked for it more and more often), and the trip back
// to host memory at the very end. Then run it, admire the
// result, and fix the paperwork: the loop that writes the ages
// down. (Zig will have a word with you about the cast in there
// once the element type is right. Listen to it.)
//
const std = @import("std");
const zml = @import("zml");

/// Another year in the attic.
fn birthday(ages: zml.Tensor) zml.Tensor {
    // A Zig number is turned into a tensor constant for us.
    return ages.addConstant(1);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const ages: zml.Tensor = .init(.{ .ghosts = 5 }, .f32);

    // Hold the seance: compile `birthday` for the placeholder
    // above.
    var exe = ???;
    defer exe.deinit();

    // Host memory for 5 floats.
    const photo: zml.Slice = try .alloc(gpa, ages.shape());
    defer photo.free(gpa);

    // Ghost number i is (i + 1) centuries old.
    for (photo.items(i32), 1..) |*age, i| {
        age.* = @intCast(i * 100);
    }

    var ages_buffer: zml.Buffer = try .fromSlice(io, platform, photo);
    defer ages_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ages_buffer});
    defer result.deinit();

    // Copy the result back to host memory, as a Slice we own.
    const older: zml.Slice = ???;
    defer older.free(gpa);

    try stdout.print("Result is a {f} slice. Happy birthday:", .{older.shape});
    for (older.constItems(f32)) |age| try stdout.print(" {d}", .{age});
    try stdout.print("\n", .{});
    try stdout.flush();
}
