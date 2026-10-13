//
// Quick: in a shape `.{ 12, 5, 30 }`, which axis is the shelves?
//
// Exactly. Nobody knows. Not you, not the nephew, and certainly
// not you in six months. Unlabeled axes are THE classic source of
// bugs in machine learning code: everything compiles, everything
// runs, and the result is silently garbage because somebody
// summed over the batch instead of over the features.
//
// ZML's answer is to let you write a name on each axis. Instead
// of a tuple, pass a struct literal:
//
//     const fridge = zml.Shape.init(.{ .shelves = 5, .eggs = 12 }, .u8);
//
// These names are called TAGS, and from now on wherever ZML asks
// for an axis, you may give its tag instead of its number:
//
//     fridge.dim(.eggs)    -> 12     same as fridge.dim(1)
//     fridge.axis(.eggs)   -> 1      tag  -> axis number
//     fridge.tag(1)        -> "eggs" axis -> tag (a C string)
//
// Tags cost nothing when your model runs: they only exist while
// ZML is building the computation. They show up when you print a
// shape, too. `{f}` is how Zig 0.17 invokes a type's `format`
// method:
//
//     try stdout.print("{f}", .{fridge});   -> {shelves=5,eggs=12,u8}
//
// We'll spend a whole chapter on tags later (they are ZML's best
// feature, honestly). For now you only need to know they exist,
// and what happens when you get one wrong. Because the nephew
// did. Run this, and read the panic message carefully: it is the
// kind of message you'll be decoding for the rest of this course.
//
// (Notice it's a runtime panic and not a compile error? A Shape
// is an ordinary runtime value. ZML can't know at Zig's compile
// time which tags it will carry.)
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const fridge = zml.Shape.init(.{ .shelves = 5, .eggs = 12 }, .u8);

    const eggs_per_shelf = fridge.dim(.egg);

    // And which axis NUMBER is the shelves one? Ask the shape,
    // using the tag.
    const shelves_axis = fridge.axis(???);

    try stdout.print("{f}: {d} eggs per shelf, shelves are axis {d}, axis 1 is called '{s}'.\n", .{
        fridge,
        eggs_per_shelf,
        shelves_axis,
        fridge.tag(1),
    });
    try stdout.flush();
}
