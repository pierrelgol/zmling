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
// Everything you need was served in this chapter:
//
//     slice      .single   .dynSingle   start / end / step
//     slices     chunkExact / split     concatenate   stack
//
// Remember what must be decided at compile time (every size)
// and what may wait for the data (where `which` points). The
// Exe is compiled once and run for three different Thursdays.
//
// The slices are numbered so that you can check the result by
// eye: slice 5 of pizza 2 is 25.
//
const std = @import("std");
const zml = @import("zml");

fn theUsual(tray: zml.Tensor, which: zml.Tensor) zml.Tensor {
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

    const tray: zml.Tensor = .init(.{ .pizzas = 3, .slices = 8 }, .i32);
    const which: zml.Tensor = .init(.{}, .i32);

    var exe = try platform.compileFn(gpa, io, theUsual, .{ tray, which }, .{});
    defer exe.deinit();

    var numbers: [3][8]i32 = undefined;
    for (&numbers, 0..) |*pizza, p| {
        for (pizza, 0..) |*slice, s| slice.* = @intCast(p * 10 + s);
    }
    var tray_buffer: zml.Buffer = try .fromBytes(io, platform, tray.shape(), std.mem.sliceAsBytes(&numbers));
    defer tray_buffer.deinit();

    try stdout.print("The usual is a {f}.\n", .{exe.output_shapes[0]});

    for ([_]i32{ 0, 1, 2 }) |thursday| {
        var which_buffer: zml.Buffer = try .scalar(io, platform, thursday, .i32);
        defer which_buffer.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ tray_buffer, which_buffer });
        defer result.deinit();

        const bag = try result.getValue([2][5]i32, io);
        try stdout.print("pizza {d}: his box {any}, her sample {any}\n", .{ thursday, bag[0], bag[1] });
    }
    try stdout.flush();
}
