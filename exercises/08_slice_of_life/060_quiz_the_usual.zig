//
// Quiz! It's Thursday. The bell above the door rings, and Nonna
// Rosa says, without turning around: "Buonasera, Mr.
// Henderson."
//
// Mr. Henderson has refined his order over many years. It is
// called "the usual", it has never been written down, and Luigi
// is alone at the counter tonight. So you are going to write it
// down, as a function.
//
// On the counter is a tray of three pizzas of 8 slices each:
//
//     tray {pizzas=3,slices=8}
//
// Mr. Henderson points at one of them. Which one changes every
// week, so it is an INPUT of the Exe: a scalar `which`. Then:
//
//   1. Take pizza number `which` out of the tray.
//
//   2. Cut it in two equal halves.
//
//   3. From the first half he wants one slice out of two,
//      starting with the very first. From the second half he
//      wants everything but its last slice ("too close to the
//      crust", it isn't, nobody argues).
//
//   4. Put those two selections end to end: that is HIS BOX,
//      {slices=5}.
//
//   5. Nonna always adds a free sample for Mrs. Henderson: the
//      last 5 slices of the LAST pizza of the tray, whatever
//      pizza he chose. Also {slices=5}.
//
//   6. Both go in the delivery bag, his box first:
//      {boxes=2,slices=5}.
//
//   7. The bill. Tensorino's charges by the number written on
//      the slice (don't ask, it's a family tradition): the
//      total of each box of the bag, {boxes=2}.
//
// Four helpers, each one a brick for the next:
//
//     hisBox      steps 2 to 4, from ONE pizza
//     herSample   step 5, from the tray
//     theUsual    steps 1 and 6, calling the two above
//     bill        step 7, calling theUsual
//
// Everything you need was served in this chapter:
//
//     slice      .single   .dynSingle   start / end / step
//     slices     chunkExact / split     concatenate   stack
//
// except the total, which comes from the Reduction Kitchen.
//
// Remember what must be decided at compile time (every size)
// and what may wait for the data (where `which` points). Each
// Exe is compiled once and run for three different Thursdays.
//
// The slices are numbered so that you can check the result by
// eye: slice 5 of pizza 2 is 25.
//
// Luigi did write something tonight: the numbers on the slices
// and the prints. Everything between them is yours. Two
// placeholders (one of them a scalar i32), two Exes
// (`theUsual` and `bill`), the tray's Buffer, and, every
// Thursday, a Buffer for `which` (`zml.Buffer.scalar`, as in
// the last exercise), two runs and two read-backs. The prints
// tell you the names and the types they expect.
//
const std = @import("std");
const zml = @import("zml");

/// Steps 2 to 4. {slices=8} in, {slices=5} out.
fn hisBox(pizza: zml.Tensor) zml.Tensor {
    return ???;
}

/// Step 5. The whole tray in, {slices=5} out.
fn herSample(tray: zml.Tensor) zml.Tensor {
    return ???;
}

/// Steps 1 and 6: {boxes=2,slices=5}.
fn theUsual(tray: zml.Tensor, which: zml.Tensor) zml.Tensor {
    return ???;
}

/// Step 7: {boxes=2}.
fn bill(tray: zml.Tensor, which: zml.Tensor) zml.Tensor {
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

    var numbers: [3][8]i32 = undefined;
    for (&numbers, 0..) |*pizza, p| {
        for (pizza, 0..) |*slice, s| slice.* = @intCast(p * 10 + s);
    }

    // Before the loop: the placeholders, the two Exes (call the
    // one for `theUsual` `usual_exe`), and the tray's Buffer.
    ???;

    try stdout.print("The usual is a {f}.\n", .{usual_exe.output_shapes[0]});

    for ([_]i32{ 0, 1, 2 }) |thursday| {
        // Every Thursday: `bag` is a [2][5]i32 (the usual), and
        // `total` is a [2]i32 (the bill).
        ???;

        try stdout.print("pizza {d}: his box {any}, her sample {any}, bill {any}\n", .{ thursday, bag[0], bag[1], total });
    }
    try stdout.flush();
}
