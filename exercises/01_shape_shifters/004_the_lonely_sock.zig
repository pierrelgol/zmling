//
// Behind every washing machine Shifty & Sons has ever moved,
// there was exactly one sock.
//
// A single value, with no axes at all, is called a SCALAR. Its
// shape has rank 0:
//
//     const sock = zml.Shape.init(.{}, .f32);
//
//     sock.rank()   -> 0
//     sock.count()  -> 1     (an empty product is 1!)
//     sock.dims()   -> { }
//
// There's also a shortcut that says what it means:
//
//     const sock = zml.Shape.scalar(.f32);
//
// Now, be careful. This is NOT the same thing as:
//
//     const boxed_sock = zml.Shape.init(.{1}, .f32);
//
//     boxed_sock.rank()   -> 1
//     boxed_sock.count()  -> 1
//
// That's one sock inside a box: a vector that happens to have
// length one. And `.{ 1, 1 }` is a sock in a box in a box. All
// three hold one number and weigh the same, and yet ZML treats
// them as different shapes. The difference is not pedantry:
//
//   - scalars are the ONLY tensors ZML will silently stretch to
//     fit any other shape (so "x + 1" works for any x),
//   - plenty of functions want a scalar and will refuse a
//     box-with-one-thing-in-it. A loss, a temperature, a loop
//     counter, a "should I stop?" flag: all scalars.
//
// You will meet both rules again soon. For now, the nephew filed
// the lonely sock as "a box of socks", and the customer is
// confused about where the box went.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // One sock. No box. Rank 0.
    const sock = zml.Shape.init(.{1}, .u8);

    // One sock, in a box, in a crate, in a container. Rank 3.
    const paranoid_sock = zml.Shape.init(.{}, .u8);

    try stdout.print("sock:          rank {d}, count {d}, dims {any}\n", .{ sock.rank(), sock.count(), sock.dims() });
    try stdout.print("paranoid sock: rank {d}, count {d}, dims {any}\n", .{ paranoid_sock.rank(), paranoid_sock.count(), paranoid_sock.dims() });

    // Shapes compare with `eql`: same dims and same dtype.
    try stdout.print("Same thing? {}\n", .{sock.eql(paranoid_sock)});
    try stdout.print("Is the sock a scalar? {}\n", .{sock.eql(zml.Shape.scalar(.u8))});
    try stdout.flush();
}
