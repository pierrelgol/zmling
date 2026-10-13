//
// Not everything that walks along an axis boils it away. Take
// the butter. (In this kitchen, everything starts with butter.)
//
// The fridge log says how much butter went in or out during
// each hour of service: +8 for a delivery, -3 when the sauce
// station had a good hour. The total of the day is a `sum`. But
// the question that keeps a chef awake is another one: how much
// was left AFTER EACH HOUR? That is a running total:
//
//     pub fn cumulativeSum(self: Tensor, axis_: anytype) Tensor
//
// Element i of the result is the sum of elements 0 to i of the
// input, both included. The example in ZML's doc comment:
//
//     [0, 1, 0, 1, 0, 0, 1, 1].cumulativeSum(0)
//  -> [0, 1, 1, 2, 2, 2, 3, 4]
//
// This is the odd one out of the chapter. The output has the
// SAME shape as the input: nothing is reduced, no size 1 axis,
// nothing to squeeze. Look at the last element though: it is the
// plain `sum`. A cumulative sum is a reduction that shows its
// work.
//
// On a tensor of higher rank, it runs along the axis you name,
// separately for each position of the other axes. Give it
// {fridge=2,hour=8} and `.hour`, and you get a running total for
// each fridge.
//
// You'll meet it again. Turning "probability of each word" into
// "probability of this word or any more likely one" is a
// cumulative sum, and it is how language models decide which
// words are worth considering at all.
//
// While we are in the fridge: there are two more reductions on
// the menu, `any(axis)` and `all(axis)`. They answer "is at
// least one value non-zero?" and "are all the values non-zero?"
// with a .bool tensor. Sadly, in the version of ZML that zmling
// is pinned to, both are OFF the menu: calling either one is a
// compile error, and the error is inside zml/zml/tensor.zig, not
// in your code. (Try it, if you like reading other people's
// compile errors.) The kitchen makes do with what's in the pan:
// for values that are never negative, "any" is `max > 0` and
// "all" is `min > 0`. And "was the stock ever below zero?" is a
// question about the `min`, too.
//
// The fridge held 10 kg of butter when service began. Chef Odile
// wants the level after each of the 8 hours, and then the one
// number that tells her if the kitchen ever ran out.
//
// The commis before you (he works at a TV station now) confused
// a few things. The first one earns you a panic from `getValue`,
// which counts bytes and doesn't like what it finds. The second
// one is quieter: the program runs, and tells Chef Odile that
// everything went fine. It did not go fine.
//
const std = @import("std");
const zml = @import("zml");

const start_of_service = 10.0;

/// Butter in the fridge after each hour, in kg: {hour=8}.
fn stock(log: zml.Tensor) zml.Tensor {
    return log.sum(.hour).addConstant(start_of_service);
}

/// The lowest the stock ever got, as a true scalar.
fn lowest(log: zml.Tensor) zml.Tensor {
    // A ritual is a Zig function. It can call other functions.
    return stock(log).max(.hour).squeeze(.hour);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const log: zml.Tensor = .init(.{ .hour = 8 }, .f32);

    var stock_exe = try platform.compileFn(gpa, io, stock, .{log}, .{});
    defer stock_exe.deinit();

    var lowest_exe = try platform.compileFn(gpa, io, lowest, .{log}, .{});
    defer lowest_exe.deinit();

    // Deliveries came at 1 pm and at 5 pm.
    var log_buffer: zml.Buffer = try .fromBytes(io, platform, log.shape(), std.mem.sliceAsBytes(&[_]f32{ -3, -4.5, 8, -5, -6, -2, 10, -4 }));
    defer log_buffer.deinit();

    var levels: zml.Buffer = try stock_exe.eval(gpa, io, .{log_buffer});
    defer levels.deinit();

    var low: zml.Buffer = try lowest_exe.eval(gpa, io, .{log_buffer});
    defer low.deinit();

    try stdout.print("Butter:", .{});
    for (try levels.getValue([8]f32, io)) |kg| try stdout.print(" {d:.1}", .{kg});
    try stdout.print("\n", .{});

    const lowest_kg = try low.getValue(f32, io);
    try stdout.print("Lowest point: {d:.1} kg. {s}\n", .{
        lowest_kg,
        if (lowest_kg < 0) "We ran out. Nobody tell Chef Odile." else "Everything went fine.",
    });
    try stdout.flush();
}
