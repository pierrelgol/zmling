//
// Remember the photographer from your last day at Shifty & Sons?
// A batch of images is a rank 4 tensor, and the world never
// agreed on the order of its axes:
//
//     NHWC   (number of images, height, width, channels)
//     NCHW   (number of images, channels, height, width)
//
// Code written with axis NUMBERS picks a side without saying so.
// `x.dim(1)` is the height in one layout and the channel count
// in the other. Nothing fails when you get it wrong. An image
// with 3 channels and a height of 2 is perfectly plausible to a
// computer, and your results are simply, quietly, off.
//
// Here is everything this chapter was building up to:
//
//     Code written with TAGS doesn't have a layout.
//
// `x.dim(.h)` is the height. In NHWC. In NCHW. In whatever layout
// somebody invents next year. A function that only ever names
// its axes works on any of them, unchanged. And if it is handed
// something with no `.h` at all, it stops with a panic that
// prints the full shape, instead of computing nonsense.
//
// One more thing about `dim`, which makes it more useful than it
// looks. It returns an ordinary Zig integer (an i64), at tracing
// time. You can do ordinary Zig arithmetic on it and hand the
// result to any of the constant forms from the diner:
//
//     x.divByConst(x.dim(.h) * x.dim(.w))
//
// The product is computed by Zig, once, while the function is
// traced, and gets baked into the Exe. This is the normal way to
// write "divide by the number of pixels" in ZML.
//
// And since the very same Zig function can be compiled several
// times for different input shapes, one function gives you one
// Exe per layout.
//
// Zigville International just bought a second X-ray machine.
// The old one produces NHWC scans. The new one, from another
// vendor, produces NCHW. (Of course it does.) Both scans go
// through `dilute`, which divides every pixel by the number of
// pixels in one image, height times width; never mind why,
// Brenda says it's "for the average".
//
// Kevin wrote `dilute`, and tested it. On the old machine.
//
// And this morning a third one arrived: a handheld scanner, for
// the bags that don't fit on the belt. One image at a time, no
// colors: its scans are just 2 rows of 5 pixels, tagged like the
// others. `dilute` must not notice the difference. Describe its
// layout in main().
//
// The testing helper, `firstPixel`, was Kevin's too, and it lost
// both ends on the way to your desk. Sizes are not known when you
// write this function (they come with `layout`), so a Zig array
// won't do: it is the photograph of exercise 013, both ways.
// Host memory for one scan with every pixel set to 10 going in,
// and the first pixel of the result coming out.
//
const std = @import("std");
const zml = @import("zml");

/// Each pixel divided by the number of pixels per image (h * w).
fn dilute(scan: zml.Tensor) zml.Tensor {
    return scan.divByConst(scan.dim(1) * scan.dim(2));
}

/// Compiles `dilute` for the given layout, runs it on a scan
/// where every pixel is 10, and returns the first pixel of the
/// result.
fn firstPixel(gpa: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, layout: zml.Shape) !f32 {
    const scan: zml.Tensor = .fromShape(layout);

    var exe = try platform.compileFn(gpa, io, dilute, .{scan}, .{});
    defer exe.deinit();

    // One scan in host memory, every pixel is 10.

    var scan_buffer: zml.Buffer = try .fromSlice(io, platform, pixels);
    defer scan_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{scan_buffer});
    defer result.deinit();

    // The result, back in host memory; its first pixel.
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

    // One image of 2 by 5 pixels, 3 channels. Tiny bags.
    const old_machine = zml.Shape.init(.{ .n = 1, .h = 2, .w = 5, .c = 3 }, .f32);
    const new_machine = zml.Shape.init(.{ .n = 1, .c = 3, .h = 2, .w = 5 }, .f32);
    // 2 rows of 5 pixels, and nothing else.
    const handheld: zml.Shape = ???;

    // 10 pixels per image, every pixel is 10: all must print 1.
    try stdout.print("Old machine {f}: a pixel of 10 becomes {d:.3}\n", .{
        old_machine,
        try firstPixel(gpa, io, platform, old_machine),
    });
    try stdout.print("New machine {f}: a pixel of 10 becomes {d:.3}\n", .{
        new_machine,
        try firstPixel(gpa, io, platform, new_machine),
    });
    try stdout.print("Handheld    {f}: a pixel of 10 becomes {d:.3}\n", .{
        handheld,
        try firstPixel(gpa, io, platform, handheld),
    });
    try stdout.flush();
}
