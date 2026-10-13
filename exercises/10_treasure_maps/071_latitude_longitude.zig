//
// One index picks along one axis. A treasure is at a latitude
// AND a longitude, and `gather` has two more tricks for that.
//
// TRICK 1: several axes at once.
//
//     sea.gather(.{ .lat = lats, .lon = lons }, .{})
//
// The struct now has two fields, and the index tensors go
// together like the two halves of a coordinate:
//
//     sea   {lat=4,lon=3}
//     lats  {x=3}        lons  {x=3}
//
//     result {x=3}:      result[x] = sea[lats[x], lons[x]]
//
// One element per X. The gathered axes (both of them) are
// replaced by the axes of the indices, exactly once. Give the
// index tensors the same shape.
//
// There is a condition: the axes have to be listed in the order
// they have in the tensor, and they have to be neighbours.
// `sea` is {lat,lon}, so it's .lat and then .lon. Get it wrong
// and ZML says so in its own words:
//
//     gather expects 'idx_axes' to be sequential.
//     But { 1, 0 } aren't sequential in Tensor({lat=4,lon=3,i32})
//
// ({ 1, 0 } are axis numbers: you asked for axis 1, then 0.)
//
// TRICK 2: one pick per row. You met batching in `dot`: an axis
// with the same name on both sides means "for each, separately".
// `gather` plays the same game. When the index tensor carries a
// tag that `sea` also has (and that is not being gathered), that
// axis is not replaced, it is walked in step:
//
//     sea       {lat=4,lon=3}
//     best_lon  {lat=4}
//
//     sea.gather(.{ .lon = best_lon }, .{})   ->  {lat=4}
//     result[lat] = sea[lat, best_lon[lat]]
//
// "For each latitude, the square at THAT latitude's favourite
// longitude." Four numbers. Now give `best_lon` any other name,
// say {watch=4}. It's no longer batching, it's an ordinary index
// list, and the ordinary rule applies: .lon is replaced by
// .watch, for every latitude. You get {lat=4,watch=4}: sixteen
// numbers, and no error, because that's a perfectly legal thing
// to ask for.
//
// The lookout has marked three X's, and he also knows the best
// longitude at each latitude. The cabin boy wrote `dig`, which
// panics, and one placeholder in `main`, which doesn't. Trick 2
// he left to you: `bestOfEachLatitude` is empty.
//
// And since the crew can only dig one hole at a time: which X
// first? `firstToDig` is the POSITION, along .x, of the X with
// the most gold under it (the kitchen again, exercise 42): one
// number, no axis. Write it, compile it, run it into a Buffer
// called `first`.
//
const std = @import("std");
const zml = @import("zml");

/// The gold under each X.
fn dig(sea: zml.Tensor, lats: zml.Tensor, lons: zml.Tensor) zml.Tensor {
    return sea.gather(.{ .lon = lons, .lat = lats }, .{});
}

/// For each latitude, the gold at the lookout's favourite longitude.
fn bestOfEachLatitude(sea: zml.Tensor, best_lon: zml.Tensor) zml.Tensor {
    return ???;
}

/// The position, along .x, of the richest X. Rank 0.
fn firstToDig(sea: zml.Tensor, lats: zml.Tensor, lons: zml.Tensor) zml.Tensor {
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

    const sea: zml.Tensor = .init(.{ .lat = 4, .lon = 3 }, .i32);
    const lats: zml.Tensor = .init(.{ .x = 3 }, .i32);
    const lons: zml.Tensor = .init(.{ .x = 3 }, .i32);
    // One longitude per latitude. The tag has to say so.
    const best_lon: zml.Tensor = .init(.{ .watch = 4 }, .i32);

    var dig_exe = try platform.compileFn(gpa, io, dig, .{ sea, lats, lons }, .{});
    defer dig_exe.deinit();
    var best_exe = try platform.compileFn(gpa, io, bestOfEachLatitude, .{ sea, best_lon }, .{});
    defer best_exe.deinit();

    // `firstToDig` needs an Exe as well.

    // Gold under each square of the sea.
    var sea_buffer: zml.Buffer = try .fromBytes(io, platform, sea.shape(), std.mem.sliceAsBytes(&[4][3]i32{
        .{ 0, 7, 0 },
        .{ 3, 0, 9 },
        .{ 0, 0, 4 },
        .{ 8, 1, 0 },
    }));
    defer sea_buffer.deinit();
    var lats_buffer: zml.Buffer = try .fromBytes(io, platform, lats.shape(), std.mem.sliceAsBytes(&[_]i32{ 3, 0, 1 }));
    defer lats_buffer.deinit();
    var lons_buffer: zml.Buffer = try .fromBytes(io, platform, lons.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2 }));
    defer lons_buffer.deinit();
    var best_lon_buffer: zml.Buffer = try .fromBytes(io, platform, best_lon.shape(), std.mem.sliceAsBytes(&[_]i32{ 1, 2, 2, 0 }));
    defer best_lon_buffer.deinit();

    var dug: zml.Buffer = try dig_exe.eval(gpa, io, .{ sea_buffer, lats_buffer, lons_buffer });
    defer dug.deinit();
    var best: zml.Buffer = try best_exe.eval(gpa, io, .{ sea_buffer, best_lon_buffer });
    defer best.deinit();

    // Run it into a Buffer called `first`.

    try stdout.print("under the X's {f}: {any}\n", .{ dug.shape(), try dug.getValue([3]i32, io) });
    try stdout.print("dig X number {d} first.\n", .{try first.getValue(i32, io)});
    try stdout.print("best of each latitude {f}:", .{best.shape()});
    try stdout.flush();

    try stdout.print(" {any}\n", .{try best.getValue([4]i32, io)});
    try stdout.flush();
}
