//
// A bool tensor is nice to look at, but its real job is to
// CHOOSE. Splitting the loot on the Off-By-One follows three
// articles of the ship's code, and each one is a ZML method.
//
// ARTICLE 1: no chest counts for less than 5 doubloons (the
// digging was honest work) nor for more than 50 (the Captain
// keeps the rest, "for the ship").
//
//     x.clamp(min, max)
//
// Every element below `min` becomes `min`, every element above
// `max` becomes `max`. Both bounds are TENSORS of the same dtype
// as `x`: scalars, or tensors of the same shape as `x` if every
// element has its own bounds.
//
//     loot.clamp(.scalar(5, .i32), .scalar(50, .i32))
//
// ARTICLE 2: a cursed chest pays nothing. It COSTS 13 doubloons,
// which is what the exorcist charges.
//
//     pred.select(on_true, on_false)
//
// This is the if/else of tensors, and note who it is a method
// OF: the bool tensor. Where `pred` is true the result takes
// the element of `on_true`, elsewhere the one of `on_false`.
// The two branches must have the same dtype, and each is either
// a scalar or has the shape of `pred`.
//
// Do not read it as a branch! Both sides are computed in full,
// for every element, and then one of the two is picked. There
// is no way for `select` to "skip" work, or to protect you
// from, say, a division by zero in the branch not taken.
//
// ARTICLE 3: a chest that hasn't been FOUND yet counts for 0.
//
//     x.mask(pred, value)
//
// is a shortcut for `pred.select(x, .scalar(value, x.dtype()))`,
// and half of the people who meet it read it backwards. So,
// slowly: where `pred` is TRUE, `x` is KEPT. Where it is false,
// you get `value`. The mask says what survives, not what is
// erased. (`value` is a plain Zig number here, turned into a
// constant for you.)
//
// This time two of the inputs are bool tensors that walk in
// through the door. Nothing special about it: the placeholder
// has dtype .bool, and the Buffer is made from an array of Zig
// `bool`.
//
// The cabin boy kept the ledger. He read `mask` backwards.
//
const std = @import("std");
const zml = @import("zml");

/// What each chest adds to the crew's purse.
fn payout(loot: zml.Tensor, cursed: zml.Tensor, found: zml.Tensor) zml.Tensor {
    // Article 1.
    const capped = loot.clamp(.scalar(5, .i32), .scalar(50, .i32));

    // Article 2: -13 where the chest is cursed, `capped` elsewhere.
    const paid = ???;

    // Article 3: chests that were not found count for 0.
    const lost = found.not();
    return paid.mask(lost, 0);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const loot: zml.Tensor = .init(.{ .chest = 6 }, .i32);
    const cursed: zml.Tensor = .init(.{ .chest = 6 }, .bool);
    const found: zml.Tensor = .init(.{ .chest = 6 }, .bool);

    var exe = try platform.compileFn(gpa, io, payout, .{ loot, cursed, found }, .{});
    defer exe.deinit();

    var loot_buffer: zml.Buffer = try .fromBytes(io, platform, loot.shape(), std.mem.sliceAsBytes(&[_]i32{ 30, 5, 80, 12, 80, 1 }));
    defer loot_buffer.deinit();
    var cursed_buffer: zml.Buffer = try .fromBytes(io, platform, cursed.shape(), std.mem.sliceAsBytes(&[_]bool{ false, false, true, false, false, true }));
    defer cursed_buffer.deinit();
    var found_buffer: zml.Buffer = try .fromBytes(io, platform, found.shape(), std.mem.sliceAsBytes(&[_]bool{ true, true, true, false, true, false }));
    defer found_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ loot_buffer, cursed_buffer, found_buffer });
    defer result.deinit();

    const ledger = try result.getValue([6]i32, io);
    var purse: i32 = 0;
    for (ledger) |doubloons| purse += doubloons;

    try stdout.print("The ledger: {any}\n", .{ledger});
    try stdout.print("The crew's purse: {d} doubloons.\n", .{purse});
    try stdout.flush();
}
