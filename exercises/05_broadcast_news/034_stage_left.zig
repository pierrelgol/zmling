//
// `broad` found the right axis because it could read the tags.
// Now meet Kevin from Sales. Kevin exports his numbers from a
// spreadsheet, and spreadsheets have never heard of tags:
//
//     viewers   {3,4}    3 shows, 4 weeks (thousands of viewers)
//     price     {3}      what an ad costs, per show
//     season    {4}      a multiplier per week (week 4 is the
//                        holidays: everything costs double)
//
// Ask `price.broad(viewers.shape())` and ZML answers, quite
// reasonably:
//
//     panic: Can't broadcast Tensor({3,f32}) to {3,4,f32}
//
// A nameless axis of size 3: is that the shows? Says who? With
// no tags to go by, you have to say WHERE the axes of the small
// tensor go. There are two shortcuts and one do-it-yourself kit.
//
//     pub fn broadcastLeft(self: Tensor, output_shape: Shape) Tensor
//
// The axes of `self` are the LAST axes of the output; the missing
// axes are added on the LEFT. This is the rule numpy and PyTorch
// always apply.
//
//     {4}.broadcastLeft({3,4})     a b c d
//                                  a b c d      each ROW is a copy
//                                  a b c d
//
//     pub fn broadcastRight(self: Tensor, output_shape: Shape) Tensor
//
// The axes of `self` are the FIRST axes of the output; the
// missing ones are added on the RIGHT.
//
//     {3}.broadcastRight({3,4})    a a a a
//                                  b b b b      each COLUMN is a copy
//                                  c c c c
//
// Both are thin wrappers around the primitive, which you can
// call yourself:
//
//     pub fn broadcast(self: Tensor, output_shape: Shape, axes_: []const i64) Tensor
//
// `axes_` has one entry per axis of `self`: entry i is the axis
// of the OUTPUT that axis i of `self` becomes. So:
//
//     {4}.broadcast({3,4}, &.{1})      same as broadcastLeft
//     {3}.broadcast({3,4}, &.{0})      same as broadcastRight
//     {5}.broadcast({2,5,7}, &.{1})    neither Left nor Right can
//                                      reach an axis in the middle
//
// In all three, each axis of `self` must have the size of the
// axis it lands on (or size 1, coming up next). If not, you get
// the longest panic message in ZML:
//
//     panic: broadcast expects shape axes to either be 1-sized or
//     to match the target size. got broadcast(Tensor({3,f32}),
//     {3,4,f32}, { 1 }), error on self axis 0 mapping to other
//     axis 1
//
// Read it once, calmly: it gives the tensor, the target, the
// mapping you asked for, and the pair of axes that disagree.
//
// One thing to remember: the result gets the tags of
// `output_shape`. Broadcast a tagged tensor to an untagged shape
// with these three and its tags are gone. That's why `broad` is
// the one to reach for whenever you do have tags.
//
// Kevin wants the ad revenue of each show for each week:
//
//     revenue = viewers * price * season + sponsor
//
// where `sponsor` {3} is a flat weekly amount per show (the
// Morning Zoo is "brought to you by Big Al's Mattress Barn").
// Doug, who doesn't trust shortcuts, asked that the sponsor be
// done with the primitive.
//
const std = @import("std");
const zml = @import("zml");

fn revenue(viewers: zml.Tensor, price: zml.Tensor, season: zml.Tensor, sponsor: zml.Tensor) zml.Tensor {
    const table = viewers.shape();

    // One price per show: shows are the first axis.
    const price_table = price.???(table);

    // One multiplier per week: weeks are the last axis.
    const season_table = season.???(table);

    // One sponsor per show, the do-it-yourself way.
    const sponsor_table = sponsor.broadcast(table, &.{???});

    return viewers.mul(price_table).mul(season_table).add(sponsor_table);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const viewers: zml.Tensor = .init(.{ 3, 4 }, .f32);
    const price: zml.Tensor = .init(.{3}, .f32);
    const season: zml.Tensor = .init(.{4}, .f32);
    const sponsor: zml.Tensor = .init(.{3}, .f32);

    var exe = try platform.compileFn(gpa, io, revenue, .{ viewers, price, season, sponsor }, .{});
    defer exe.deinit();

    const shows = [_][]const u8{ "Morning Zoo", "Noon News", "Late Movie" };

    var viewers_buffer: zml.Buffer = try .fromBytes(io, platform, viewers.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 10, 12, 11, 20 },
        .{ 30, 28, 31, 40 },
        .{ 5, 6, 5, 9 },
    }));
    defer viewers_buffer.deinit();

    var price_buffer: zml.Buffer = try .fromBytes(io, platform, price.shape(), std.mem.sliceAsBytes(&[_]f32{ 2, 3, 1 }));
    defer price_buffer.deinit();

    var season_buffer: zml.Buffer = try .fromBytes(io, platform, season.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 1, 1, 2 }));
    defer season_buffer.deinit();

    var sponsor_buffer: zml.Buffer = try .fromBytes(io, platform, sponsor.shape(), std.mem.sliceAsBytes(&[_]f32{ 100, 0, 50 }));
    defer sponsor_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ viewers_buffer, price_buffer, season_buffer, sponsor_buffer });
    defer result.deinit();

    const dollars = try result.getValue([3][4]f32, io);

    for (shows, dollars) |show, row| {
        try stdout.print("{s:<12}", .{show});
        for (row) |d| try stdout.print(" {d:>4}", .{d});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
