//
// Chapter 4: Luggage Tags
//
// The diner let you go after the dinner rush (nobody had told
// you about the bowling league's discount). So here you are, at
// the Lost & Found office of Zigville International Airport,
// Terminal Z, basement level, between the boiler room and a
// vending machine that only takes exact change.
//
// Your supervisor is Brenda. Thirty-one years in baggage, and
// she owns a label maker that she does not lend. The other
// employee is Kevin, the intern, who believes he can remember
// what is in every suitcase "by position".
//
// Way back in exercise 005 you put names on the axes of a Shape,
// and were promised a whole chapter on those TAGS. This is it.
// Quick recap, on a tensor this time:
//
//     const t: zml.Tensor = .init(.{ .carousels = 3, .bags = 4 }, .f32);
//
//     t.dim(.bags)      -> 4
//     t.axis(.bags)     -> 1
//
// That works when YOU create the tensor. But much of the luggage
// in a ZML program is not yours: weights loaded from a file,
// inputs described by somebody else's config, the result of some
// function written before tags were fashionable. All of that
// shows up at the office untagged: `{3,4,f32}`. Three what? Four
// what?
//
// The label maker:
//
//     const tagged = t.withTags(.{ .carousels, .bags });
//
// `withTags` takes a tuple with one tag per axis, in order, and
// returns a tensor that is the very same tensor (same values,
// same dims, same dtype) wearing new labels. The original is
// left alone, as always.
//
// Three things to know about it.
//
// 1. It wants EXACTLY one tag per axis. Otherwise it panics, and
//    the message says "Not enough tags for shape ..., got ..."
//    even when the trouble is that you gave too many. Count the
//    axes of the shape and the tags of the tuple yourself.
//
// 2. To leave an axis unlabeled, use the tag `._`: a single
//    underscore is ZML's "no tag". `t.withTags(.{ ._, .bags })`.
//
// 3. Labeling is NOT an operation. Remember rule number one of
//    Graymoor Manor, "do not touch the ghosts outside of a
//    seance"? Tags are an exception, because they aren't part of
//    the ghost. They live in the Shape, the Shape is a plain Zig
//    value, and `withTags` only swaps that value. Nothing is
//    recorded in any graph. You may call it anywhere: on a
//    placeholder in main(), or inside a recipe on its arguments.
//    And it costs nothing at run time. The compiled program never
//    hears about tags.
//
// (A Shape has the same method, `shape.withTags(...)`.)
//
// This morning's delivery: a tensor of bag weights, three
// carousels of four bags each. Kevin labeled it. Kevin labeled it
// with enthusiasm.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // As delivered. Yes, we are in main(), and yes, this is fine:
    // a placeholder, and nothing below records an operation.
    const delivery: zml.Tensor = .init(.{ 3, 4 }, .f32);
    try stdout.print("Delivered: {f}\n", .{delivery});

    // Carousels first, then bags. The weights are in kilograms,
    // but a unit is not an axis.
    const weights = delivery.withTags(.{ .carousels, .bags, .kg });
    try stdout.print("Labeled:   {f}\n", .{weights});

    // Use the labels, not the positions.
    try stdout.print("{d} carousels, {d} bags on each. Bags are axis {d}.\n", .{
        weights.dim(.carousels),
        weights.dim(???),
        weights.axis(.bags),
    });

    // The original didn't change.
    try stdout.print("The delivery slip still says {f}.\n", .{delivery.shape()});
    try stdout.flush();
}
