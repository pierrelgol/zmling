//
// Loading the truck is a game of Tetris, and Old Shifty has been
// playing since 1984. He rearranges the load a dozen times before
// he's happy.
//
// A Shape has a whole family of methods for this. Here they are,
// applied to `s = {10,11,12,f32}`:
//
//     s.setDim(1, 99)          -> {10,99,12,f32}     change one size
//     s.withDtype(.i8)         -> {10,11,12,i8}      change the dtype
//     s.append(.{4})           -> {10,11,12,4,f32}   add axes at the end
//     s.insert(0, .{4})        -> {4,10,11,12,f32}   add axes BEFORE an axis
//     s.insert(.last, .{4})    -> {10,11,12,4,f32}   (`.last` = at the end)
//     s.remove(1)              -> {10,12,f32}        delete an axis
//     s.transpose(.{ 2, 0, 1 })-> {12,10,11,f32}     reorder axes
//
// `transpose` takes a permutation: for each position of the NEW
// shape, which OLD axis goes there. Above: new axis 0 is old axis
// 2, new axis 1 is old axis 0, new axis 2 is old axis 1.
//
// Wherever an axis is expected, negative numbers and tags work
// too: `s.remove(-1)`, `s.setDim(.width, 3)`. And `append` and
// `insert` take the same things `Shape.init` does, so you can add
// tagged axes: `s.append(.{ .extra = 4 })`.
//
// Now the important part. Read it twice.
//
//     A Shape is a VALUE. None of these methods modify it.
//     Each one returns a NEW Shape.
//
// It's a small struct that lives on the stack and gets copied
// around like an integer. There is no allocator, nothing to free,
// and no way for some function you call to change your shape
// behind your back. (The same will be true of tensors.)
//
// The nephew, a recovering Java programmer, expected the truck to
// be mutated in place. Zig, bless its heart, refuses to let a
// return value be silently thrown away.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // 3 rows of 5 boxes.
    var truck = zml.Shape.init(.{ .rows = 3, .boxes = 5 }, .u8);

    // A fourth row fits.
    truck.setDim(.rows, 4);

    // Turns out we can stack them two high: add a tagged axis of
    // size 2 called .layers at the end.
    truck = truck.???(.{ .layers = 2 });

    // Shifty wants the layers first, then rows, then boxes.
    // (Yes, a tuple of tags works as a permutation.)
    truck = truck.transpose(.{ .rows, .boxes, .layers });

    try stdout.print("Loaded: {f}, {d} boxes in total.\n", .{ truck, truck.count() });

    // On the highway, boxes fly off the back. All of them. That
    // axis is gone; and what remains is counted in 64-bit signed
    // regrets.
    const aftermath = truck.remove(.rows).withDtype(.i64);

    try stdout.print("Aftermath: {f}.\n", .{aftermath});
    try stdout.flush();
}
