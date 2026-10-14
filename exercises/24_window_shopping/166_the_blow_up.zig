//
// The Grand Bazar wants a poster. Monsieur Flou has sent a
// thumbnail of 2 pixels by 2 (he charges by the pixel) and the
// poster is 4 by 4. Somebody has to invent twelve pixels.
//
// Making an image bigger is called UPSAMPLING, and networks do
// it all the time: everything that draws (image generators,
// the decoder half of a segmentation model) starts small and
// grows. Making it smaller happens before the network even
// starts: a vision model wants its 224 x 224 pixels, and your
// phone does not take photos of 224 x 224.
//
// There are two honest ways to invent a pixel.
//
// NEAREST NEIGHBOUR: copy the closest pixel you know. Blocky,
// cheap, and it never creates a value that was not there (which
// matters when the "pixels" are labels: halfway between "cat"
// and "dog" there is no "cog").
//
//     pub fn upsample(input: Tensor, opts: struct {
//         mode: UpsampleMode,            // .nearest, the only one
//         scale_factor: []const f64,
//     }) Tensor
//
// It lives in `zml.nn`, and it comes from the same family as
// `conv2d`: it expects (batch, channels, ...) with 1 to 3 axes
// to scale after them (rank 3, 4 or 5, or it panics with a
// clear message), and it does not read tags. You give FACTORS:
// one for all the axes, `&.{2}`, or one per scaled axis,
// `&.{ 2, 3 }`.
//
// Two traps, both silent:
//
//  - an NHWC photo gets its width and its colours "scaled",
//    like it got them pooled in exercise 165;
//  - the result comes back WITHOUT the tags of the axes it
//    scaled: `{b=1,c=1,4,4}`. The next op that asks for `.h`
//    will not find it. Put them back (exercise 25).
//
// BILINEAR: draw a straight line between the two pixels you
// know and read the new one off the line. In 2D, do it along
// the height, then along the width. Smooth, and the default of
// nearly every preprocessing pipeline.
//
//     pub fn resizeBilinear(image: Tensor, resized_axes: anytype,
//                           opt: ResizeOpts) Tensor
//
// Also in `zml.nn`, and from another family altogether:
//
//     zml.nn.resizeBilinear(photo, .{ .h = 4, .w = 4 }, .{})
//
// It finds the axes by TAG, so the layout does not matter, and
// you may resize one axis only. And the numbers are the new
// SIZES, not factors. Ask a 2 x 2 image for `.{ .h = 2, .w =
// 2 }` and you get your image back, untouched. It also shrinks:
// any size will do. (Write the sizes as plain numbers: the
// i64 that comes out of a `dim()` is refused at compile time,
// with an error deep inside ZML.)
//
// How it picks its points matters the day you port a model.
// Output pixel i reads the input at position
//
//     i * in / out
//
// For 2 pixels blown up to 4: positions 0, 0.5, 1, 1.5. The
// first two fall on or between the known pixels. The last one
// is past the end, and repeats the last pixel. PyTorch's
// `F.interpolate(mode="bilinear")` reads at
// (i + 0.5) * in / out - 0.5 instead, which centres things
// differently and gives other numbers at the borders. Neither
// is wrong. A port that mixes them is.
//
// (There is a `zml.nn.resizeBicubic` too, which fits curves
// instead of lines. At the commit zmling pins, it does not
// work on a (b, c, h, w) image: it trips an assertion inside
// ZML. Now you know it is not you.)
//
// And a warning about SHRINKING with `resizeBilinear`. With
// the rule above, halving an image reads positions 0, 2, 4...
// exactly on existing pixels: it keeps one pixel out of two
// and throws the others away without looking at them. The
// wallpaper of the Grand Bazar has stripes one pixel wide, and
// you are about to make them disappear. The cure is to average
// before you drop: the average pooling of exercise 165. Bring
// it back.
//
// To do:
//
//  - `blowUp`: the poster comes out without its tags, and
//    the first function to ask for `.h` will panic;
//  - `smooth`: it runs, and the poster looks a lot like the
//    thumbnail;
//  - `blowUpPhoto`: nearest neighbour on an NHWC image;
//  - `average`: from exercise 165, with a stride of 2;
//  - `main`: the last line.
//
const std = @import("std");
const zml = @import("zml");

/// Twice as big, nearest neighbour.
/// thumb: {.b, .c, .h, .w} -> {.b, .c, .h, .w}
fn blowUp(thumb: zml.Tensor) zml.Tensor {
    return zml.nn.upsample(thumb, .{ .mode = .nearest, .scale_factor = &.{2} });
}

/// Twice as big, bilinear: a poster of 4 x 4.
/// thumb: {.b, .c, .h = 2, .w = 2} -> {.b, .c, .h = 4, .w = 4}
fn smooth(thumb: zml.Tensor) zml.Tensor {
    return zml.nn.resizeBilinear(thumb, .{ .h = 2, .w = 2 }, .{});
}

/// Twice as big, nearest neighbour, for a photo that comes
/// channels last. photo: {.b, .h, .w, .c} -> {.b, .h, .w, .c}
fn blowUpPhoto(photo: zml.Tensor) zml.Tensor {
    return ???;
}

/// Half the size, by keeping one pixel out of two.
/// wallpaper: {.b, .c, .h = 4, .w = 4} -> {.b, .c, .h = 2, .w = 2}
fn shrink(wallpaper: zml.Tensor) zml.Tensor {
    return zml.nn.resizeBilinear(wallpaper, .{ .h = 2, .w = 2 }, .{});
}

fn plus(args: zml.ops.ReduceArgs) [1]zml.Tensor {
    return .{args.left.add(args.right)};
}

/// Half the size, by averaging every 2 x 2 block.
/// wallpaper: {.b, .c, .h, .w}
fn average(wallpaper: zml.Tensor) zml.Tensor {
    return ???;
}

/// Prints `name`, the shape, and the pixels, one row per line.
fn show(stdout: *std.Io.Writer, gpa: std.mem.Allocator, io: std.Io, name: []const u8, image: zml.Buffer, width: usize) !void {
    const pixels = try image.toSliceAlloc(gpa, io);
    defer pixels.free(gpa);

    try stdout.print("{s} {f}\n", .{ name, image.shape() });
    for (pixels.items(f32), 0..) |pixel, i| {
        try stdout.print(" {d:>3}", .{pixel});
        if ((i + 1) % width == 0) try stdout.print("\n", .{});
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The same four pixels, in the two layouts. With one
    // channel, the bytes are the same too.
    const thumb: zml.Tensor = .init(.{ .b = 1, .c = 1, .h = 2, .w = 2 }, .f32);
    const photo: zml.Tensor = .init(.{ .b = 1, .h = 2, .w = 2, .c = 1 }, .f32);
    const four_pixels = [2][2]f32{ .{ 0, 4 }, .{ 8, 12 } };

    var thumb_buffer: zml.Buffer = try .fromBytes(io, platform, thumb.shape(), std.mem.sliceAsBytes(&four_pixels));
    defer thumb_buffer.deinit();
    var photo_buffer: zml.Buffer = try .fromBytes(io, platform, photo.shape(), std.mem.sliceAsBytes(&four_pixels));
    defer photo_buffer.deinit();

    // Stripes, one pixel wide.
    const wallpaper: zml.Tensor = .init(.{ .b = 1, .c = 1, .h = 4, .w = 4 }, .f32);
    const stripes: [4][4]f32 = @splat(.{ 0, 8, 0, 8 });
    var wallpaper_buffer: zml.Buffer = try .fromBytes(io, platform, wallpaper.shape(), std.mem.sliceAsBytes(&stripes));
    defer wallpaper_buffer.deinit();

    var blow_up_exe = try platform.compileFn(gpa, io, blowUp, .{thumb}, .{});
    defer blow_up_exe.deinit();
    var blown_up = try blow_up_exe.eval(gpa, io, .{thumb_buffer});
    defer blown_up.deinit();
    try show(stdout, gpa, io, "nearest", blown_up, 4);

    var photo_exe = try platform.compileFn(gpa, io, blowUpPhoto, .{photo}, .{});
    defer photo_exe.deinit();
    var blown_up_photo = try photo_exe.eval(gpa, io, .{photo_buffer});
    defer blown_up_photo.deinit();
    try show(stdout, gpa, io, "nearest, the photo", blown_up_photo, 4);

    var smooth_exe = try platform.compileFn(gpa, io, smooth, .{thumb}, .{});
    defer smooth_exe.deinit();
    var smoothed = try smooth_exe.eval(gpa, io, .{thumb_buffer});
    defer smoothed.deinit();
    try show(stdout, gpa, io, "bilinear", smoothed, 4);

    var shrink_exe = try platform.compileFn(gpa, io, shrink, .{wallpaper}, .{});
    defer shrink_exe.deinit();
    var shrunk = try shrink_exe.eval(gpa, io, .{wallpaper_buffer});
    defer shrunk.deinit();
    try show(stdout, gpa, io, "the stripes, shrunk", shrunk, 2);

    // The stripes again, through `average` this time. Show them
    // under the name "the stripes, averaged", 2 pixels per row.
    ???;

    try stdout.flush();
}
