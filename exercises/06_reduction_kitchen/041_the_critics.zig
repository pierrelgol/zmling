//
// Sum has three siblings, and they all behave the same way: one
// axis in, that axis comes out with size 1.
//
//     pub fn mean(self: Tensor, axis_: anytype) Tensor
//     pub fn max(self: Tensor, axis_: anytype) Tensor
//     pub fn min(self: Tensor, axis_: anytype) Tensor
//
//     scores: {dish=2,critic=3}          7  8  3
//                                        9  4  5
//
//     scores.max(.critic)    {dish=2,critic=1}     8, 9
//     scores.min(.critic)    {dish=2,critic=1}     3, 4
//     scores.mean(.critic)   {dish=2,critic=1}     6, 6
//     scores.max(.dish)      {dish=1,critic=3}     9, 8, 5
//
// (Don't mix up `max(axis)` with the `maximum(other)` from
// Graymoor Manor. `a.maximum(b)` compares TWO tensors element by
// element and reduces nothing. `a.max(axis)` looks inside ONE
// tensor and reduces an axis.)
//
// There is no magic in `mean`. Here is its whole body, straight
// from zml/zml/tensor.zig:
//
//     return self.sum(axis_).divByConst(self.dim(axis_));
//
// The sum, divided by the count. Two things follow from reading
// those few words:
//
//   - The count is `self.dim(axis_)`, a plain Zig number known
//     while the graph is built: it gets baked into the Exe.
//     Shapes are fixed at compile time, remember? That's what
//     buys you this.
//
//   - There is no dtype promotion, here or anywhere. The mean
//     of an .i32 tensor is an .i32 tensor, by integer division:
//     the mean of 1, 2 and 5 is 2. If you want 2.67, `convert`
//     to a float type BEFORE reducing.
//
// And when the mean you need isn't the stock one, you now know
// how to cook your own: reduce, then divide by whatever count is
// right.
//
// Which is today's job. Five critics came for lunch, and scored
// three dishes: {dish=3,critic=5}. Chef Odile reads the scores
// and puts down her knife. Critic number 2 hates everything on
// principle. Critic number 3 gave 10 to everything, and is the
// chef's mother.
//
// "The best score and the worst score of each dish do not count.
// Like in figure skating. The mean of the others."
//
//     fair = (sum - best - worst) / (number of critics - 2)
//
// One number per dish, so skim off what's left of the critics.
//
const std = @import("std");
const zml = @import("zml");

/// The mean score of each dish, extremes not included: {dish=3}.
fn fairScore(scores: zml.Tensor) zml.Tensor {
    const total = scores.sum(.critic);

    // The best and the worst score of each dish.
    const best = ???;
    const worst = ???;

    // How many scores are left, once those two are out? Don't
    // write 3: ask the tensor.
    const judges = ???;

    return total.sub(best).sub(worst).divByConst(judges).squeeze(.critic);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const scores: zml.Tensor = .init(.{ .dish = 3, .critic = 5 }, .f32);

    var exe = try platform.compileFn(gpa, io, fairScore, .{scores}, .{});
    defer exe.deinit();

    var scores_buffer: zml.Buffer = try .fromBytes(io, platform, scores.shape(), std.mem.sliceAsBytes(&[3][5]f32{
        .{ 7, 8, 2, 10, 8 },
        .{ 9, 9, 1, 10, 9 },
        .{ 4, 5, 0, 10, 3 },
    }));
    defer scores_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{scores_buffer});
    defer result.deinit();

    const fair = try result.getValue([3]f32, io);

    try stdout.print("{f}\n", .{result.shape()});
    for ([_][]const u8{ "soup", "duck", "tart" }, fair) |dish, score| {
        try stdout.print("{s}: {d:.2}\n", .{ dish, score });
    }
    try stdout.flush();
}
