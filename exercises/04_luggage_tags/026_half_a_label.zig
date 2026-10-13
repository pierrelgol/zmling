//
// Some luggage arrives at the office and you honestly don't know
// what all of its axes are. And often you shouldn't care.
//
// Take a function that weighs bags. Its input might be `{bags}`
// today, `{flights, bags}` tomorrow, `{days, flights, bags}` next
// week: whatever comes in front, the LAST axis is the bags, and
// that is the only one the function looks at. Code like this is
// everywhere in machine learning; the anonymous leading axes are
// called "batch dimensions".
//
// So there is a second label maker, for the lazy and the wise:
//
//     x.withPartialTags(.{ .a, .b })
//
// It tags the LAST axes of `x`, as many as you give tags, and
// leaves the ones in front alone:
//
//     {7,2,5,3,f32}  .withPartialTags(.{ .bags, .cm })
//  -> {7,2,bags=5,cm=3,f32}
//
// Give it more tags than there are axes and it panics: "Too many
// tags for shape ...". (This time the message means it.)
//
// For a hole anywhere else, `withTags` and the "no tag" tag:
//
//     {7,2,5,3,f32}  .withTags(.{ .days, ._, .bags, .cm })
//  -> {days=7,2,bags=5,cm=3,f32}
//
// What can you do with an untagged axis? Everything, by number:
// `x.dim(1)`. What you can't do is ask for it by "name":
//
//     x.dim(._)
//
//     panic: The unknown tag ._ can't be used to fetch axis in
//            {7,2,bags=5,cm=3,f32}
//
// Which is fair. "Bring me the bag with no label" is how this
// office got to its current state.
//
// And when you need to know what you're holding, ask its Shape:
//
//     x.shape().isFullyTagged()   -> bool: a tag on every axis?
//     x.shape().hasTag(.bags)     -> ?u3: the axis number of that
//                                    tag, or null if there is none
//     x.shape().tag(1)            -> the tag of an axis, as a C
//                                    string; "_" if untagged
//
// `hasTag` is the polite version of `axis`: it returns null where
// `axis` would panic. Use it when a tag MAY be there.
//
// The X-ray machine upstairs sends its measurements down as a
// rank 4 tensor. The manual survived a flood. What is still
// readable: the last two axes are the bags, then the three
// measurements of each (length, width, height, in centimeters).
// The rest is anybody's guess.
//
// That is the job of `labelScan`, and nobody wrote it: one
// function that labels what the manual says, on ANY scan, however
// many axes come in front.
//
// Kevin filled in the rest of the paperwork from memory. His
// memory has two holes in it, and each comes with a panic you
// have just read about.
//
// To finish, three more deliveries go through the same
// `labelScan`: a single scan, a pair, and a month of weeks. For
// each one the report says where the bags ended up. Kevin typed
// the axis number he remembered from upstairs. The report should
// ask the scan instead, by name (exercise 025).
//
const std = @import("std");
const zml = @import("zml");

/// Labels the last two axes of any scan: .bags, then .cm.
fn labelScan(scan: zml.Tensor) zml.Tensor {
    return ???;
}
pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const xray: zml.Tensor = .init(.{ 7, 2, 5, 3 }, .f32);

    // Label what the manual still says.
    const scans = labelScan(xray);
    try stdout.print("{f}: fully tagged? {}\n", .{ scans, scans.shape().isFullyTagged() });

    // Is there a .bags axis? A .days axis?
    try stdout.print("bags: axis {?d}, days: axis {?d}\n", .{
        scans.shape().hasTag(.bags),
        scans.shape().hasTag(.days),
    });

    // Brenda remembers: the machine keeps one week of scans, so
    // the first axis is .days. Nobody knows about the second one.
    // (She suspects it's "left and right". Of what? Exactly.)
    const week = scans.withTags(.{ .days, .bags, .cm });
    try stdout.print("{f}: fully tagged? {}\n", .{ week, week.shape().isFullyTagged() });

    try stdout.print("days: axis {?d}, and the mystery axis has {d} of something called '{s}'.\n", .{
        week.shape().hasTag(.days),
        // The size of the axis that has no name.
        week.dim(._),
        week.shape().tag(1),
    });

    // Same function, whatever comes in front. (`inline for`: the
    // three tuples don't have the same length, so each turn of
    // the loop is compiled separately, like in exercise 020.)
    inline for (.{ .{ 5, 3 }, .{ 2, 5, 3 }, .{ 4, 7, 2, 5, 3 } }) |dims| {
        const labeled = labelScan(zml.Tensor.init(dims, .f32));
        try stdout.print("{f}: bags on axis {d}\n", .{ labeled, 2 });
    }
    try stdout.flush();
}
