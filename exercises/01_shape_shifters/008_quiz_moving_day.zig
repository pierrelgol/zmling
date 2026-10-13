//
// Quiz time! Your probation period at Shifty & Sons ends today,
// with a real customer: a photographer moving her archive.
//
// The archive is a batch of pictures. In machine learning, images
// are almost always stored as a rank 4 tensor, and the two usual
// layouts are named after their axes:
//
//     NHWC: (number of images, height, width, channels)
//     NCHW: (number of images, channels, height, width)
//
// Different libraries prefer different layouts, half of all
// computer vision bugs are a mix-up between the two, and this is
// precisely the mess that tagged axes were invented to end.
//
// The customer has 32 photos of 480 (height) by 640 (width)
// pixels, with 3 color channels, one byte per channel, stored as
// NHWC. Here is what needs to happen:
//
//   1. Write up the archive, with the tags .n, .h, .w and .c.
//
//   2. Her new studio's software wants NCHW. Reorder the axes.
//
//   3. Neural networks eat floats, not bytes. The studio converts
//      everything to 32-bit floats. How heavy is the archive now?
//
//   4. The preview wall shows one tiny grayscale thumbnail per
//      photo: no channel axis at all, 48 by 64.
//
//   5. For the filing cabinet, each thumbnail is flattened into a
//      single row of pixels: rank 2, tagged .n and .pixels. Don't
//      compute 48 * 64 yourself.
//
//   6. The invoice has one line per photo: rank 1, tagged .n.
//      Don't type 32 again, ask the cabinet for the size of its
//      .n axis. Amounts are stored as whatever DataType matches
//      Zig's `u16`. Don't type that one either: compute it.
//
//   7. The grand total is a single 64-bit float. One number, no
//      box around it.
//
// No hand-holding this time: the print statements are given, and
// every line with a number above it is yours to write. Everything
// you need is in exercises 001 to 007.
//
const std = @import("std");
const zml = @import("zml");

const MiB = 1024 * 1024;

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // 1.
    const archive = ???;
    try stdout.print("archive:    {f}, {d} MiB\n", .{ archive, archive.byteSize() / MiB });

    // 2.
    const studio = ???;
    try stdout.print("studio:     {f}\n", .{studio});

    // 3.
    const floats = ???;
    try stdout.print("floats:     {f}, {d} MiB\n", .{ floats, floats.byteSize() / MiB });

    // 4.
    const thumbnails = ???;
    try stdout.print("thumbnails: {f}\n", .{thumbnails});

    // 5.
    const cabinet = ???;
    try stdout.print("cabinet:    {f}, {d} pixels per photo\n", .{ cabinet, cabinet.dim(-1) });

    // 6.
    const invoice = ???;
    try stdout.print("invoice:    {f}, {d} bytes\n", .{ invoice, invoice.byteSize() });

    // 7.
    const total = ???;
    try stdout.print("total:      {f}, rank {d}, {d} bytes\n", .{ total, total.rank(), total.byteSize() });
    try stdout.flush();
}
