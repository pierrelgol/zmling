//
// "Which axis was it again?"
//
// ZML asks you for an axis all the time: which one to measure,
// which one to rename, and soon which one to sum over, slice,
// sort or glue along. Wherever it asks, it accepts the same
// three kinds of answer. Take
//
//     x = {days=7, 2, bags=5, cm=3, f32}
//
// 1. A TAG.                   x.dim(.bags)   -> 5
//                             x.axis(.bags)  -> 2
//
//    The good answer. It says what you mean, and it stays right
//    when the tensor gains an axis or changes layout.
//
// 2. A NUMBER.                x.dim(2)       -> 5
//                             x.dim(-1)      -> 3   (the last one)
//                             x.axis(-1)     -> 3
//
//    Counted from 0, or from the end with negative numbers. Works
//    on any tensor, tagged or not, and says nothing to the reader.
//
// 3. A POSITIONAL LITERAL.    x.dim(._2)     -> 5
//
//    An underscore followed by one digit, `._0` to `._7`, is not
//    a real tag: it means "the axis at this position". `x.dim(._2)`
//    is `x.dim(2)` in a costume.
//
// Why would anyone need the costume? Because of Zig syntax. Many
// ZML functions take a struct literal whose FIELD NAMES are
// axes: you saw `rename(.{ .old = .new })`. A field name cannot
// be the integer 2. It can be `._2`:
//
//     x.rename(.{ ._1 = .sides })
//  -> {days=7, sides=2, bags=5, cm=3, f32}
//
// which is how you label ONE untagged axis and keep the benefit
// of `rename`'s duplicate check.
//
// Don't mix up these two:
//
//     ._       "no tag". You can give it (withTags), never ask
//              for it.
//     ._0      "the axis at position 0", tagged or not.
//
// And `axis` is the translator between the worlds: tag, negative
// number or positional literal in, plain position out. When some
// piece of code insists on a number, hand it `x.axis(.bags)`
// instead of typing the 2 yourself.
//
// (Small print. Give ZML a tag that isn't there, and you get the
// helpful panic of exercise 005. Give it a positional literal
// past the end, like `._5` on a rank 4 tensor, and you get a raw
// "index out of bounds" from deep inside: ZML doesn't check
// those. Another reason to prefer real tags.)
//
// Brenda needs the paperwork for the week of X-ray scans filled
// in. Kevin left one blank, which for once was the right call.
// He should have left two more.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const week = zml.Tensor.init(.{ 7, 2, 5, 3 }, .f32).withTags(.{ .days, ._, .bags, .cm });
    try stdout.print("{f}\n", .{week});

    // How many bags? Ask by tag.
    try stdout.print("Bags per scan: {d}\n", .{week.dim(???)});

    // At which position is the .cm axis? And which position is
    // "second from the end"? Let ZML do the counting.
    try stdout.print("cm is axis {d}; second from the end is axis {d}\n", .{ week.axis(.cm), week.axis(-2) });

    // The size of the axis nobody named, by positional literal.
    try stdout.print("The mystery axis has size {d}\n", .{week.dim(._2)});

    // Mystery solved: the machine scans every bag from two
    // sides. Label that one axis .sides, with `rename`.
    const solved = week.rename(.{ ._ = .sides });
    try stdout.print("{f}: fully tagged? {}\n", .{ solved, solved.shape().isFullyTagged() });

    // Total number of measurements in a week, from tags only.
    const total = solved.dim(.days) * solved.dim(.sides) * solved.dim(.bags) * solved.dim(.cm);
    try stdout.print("{d} measurements, and count() agrees: {}\n", .{ total, total == solved.count() });
    try stdout.flush();
}
