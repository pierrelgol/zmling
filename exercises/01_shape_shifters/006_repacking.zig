//
// Mrs. Dimitrova owns 120 plates. They arrived in 10 boxes of 12:
//
//     const plates = zml.Shape.init(.{ 10, 12 }, .u16);
//
// but the new cupboard only fits stacks of 8. Same plates, new
// arrangement. That operation is called a RESHAPE, and it has
// exactly one rule:
//
//     the total number of elements must not change.
//
//     plates.reshape(.{ 15, 8 })      -> {15,8,u16}    120 = 120, fine
//     plates.reshape(.{ 2, 6, 10 })   -> {2,6,10,u16}  fine too
//     plates.reshape(.{ 100, 2 })     -> panic! 200 != 120
//
// The rank may change freely. The dtype is carried over.
//
// Doing the division by hand gets old quickly, so you may leave
// (at most) ONE dimension for ZML to figure out. Write `.auto`,
// or `-1` if you miss numpy:
//
//     plates.reshape(.{ .auto, 8 })   -> {15,8,u16}
//     plates.reshape(.{ -1 })         -> {120,u16}    "flatten it"
//
// If the numbers don't divide evenly, ZML panics rather than
// smashing a plate to make it fit.
//
// `reshape` accepts everything `Shape.init` accepts, so the new
// axes can carry tags (exercise 005), and `.auto` still works:
//
//     plates.reshape(.{ .stacks = .auto, .plates = 8 })
//                                     -> {stacks=15,plates=8,u16}
//
// By the way, reshaping the Shape of a real tensor never moves
// any data around: the numbers stay in the same order in memory
// (row by row, last axis fastest) and only the bookkeeping
// changes. Keep that in the back of your head for the Origami
// chapter.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const plates = zml.Shape.init(.{ 10, 12 }, .u16);

    // Stacks of 8, as many as it takes. Let ZML do the division.
    const cupboard = plates.reshape(.{ ???, 8 });

    // The dishwasher takes racks of 6. The nephew insists it's
    // 7. It is not. (Run it once with the 7 to see what ZML has
    // to say about that.)
    const dishwasher = plates.reshape(.{ .auto, 7 });

    // For the insurance form: one long line of plates, rank 1.
    const insurance = plates.reshape(.{ 10, 12 });

    try stdout.print("cupboard {f}, dishwasher {f}, insurance {f}\n", .{ cupboard, dishwasher, insurance });

    // Moving day: everything goes into 5 crates. Write the
    // reshape yourself: two tagged axes, .crates and .plates, and
    // let ZML work out how many plates end up in each crate.
    const crates = ???;

    try stdout.print("moving {f}: {d} plates per crate\n", .{ crates, crates.dim(.plates) });
    try stdout.flush();
}
