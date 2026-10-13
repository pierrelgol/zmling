//
// Chapter 5: Broadcast News
//
// Welcome to WZML Channel 5, "the news, eventually". You are the
// new graphics operator. The staff: Brock Sterling and Linda
// Okafor behind the anchor desk, Gail Storm at the weather map
// (it's her real name, she is tired of the question), and Doug,
// who operates the camera. THE camera. There is one.
//
// A TV station takes one small signal and stretches it over a
// whole city. That is called broadcasting, and tensors do it too:
//
//     BROADCASTING = repeating a small tensor along some axes
//     so that it lines up with a bigger one.
//
// You have met one case already: a rank 0 tensor combines with
// anything. `temps.mul(two)` multiplies every element by the one
// value of `two`. The scalar was "stretched" over all of `temps`.
//
// Now the everyday case. The national feed sends temperatures
// for 3 cities over 4 days, {city=3,day=4}. Gail has ONE
// correction per city, {city=3}, because she knows what the
// national feed doesn't (Springfield's sensor is bolted to a
// pizza oven's chimney). What we mean is clear enough:
//
//     feed                correction         forecast
//     20 21 19 22         -4                 16 17 15 18
//     15 14 16 15    +     1.5        =      16.5 15.5 ...
//     30 31 29 33          0                 30 31 29 33
//
// numpy and PyTorch would try to guess what you mean, by lining
// up the shapes from the right. (Here they would guess wrong and
// complain; had there been 3 days instead of 4, they would have
// guessed wrong and said nothing.) ZML doesn't guess. If the
// shapes differ, `add` stops the compilation, and you have to
// say that you want a broadcast, and to what:
//
//     pub fn broad(self: Tensor, other: Shape) Tensor
//
// `small.broad(shape)` returns a tensor of the given shape (the
// dtype stays the one of `small`), made of copies of `small`.
// The good part is HOW it decides which axis is which: when
// `small` has fewer axes than `shape`, it goes BY TAG.
//
//     {city=3}  .broad( {city=3,day=4} )   one value per city,
//                                          repeated for each day
//     {day=4}   .broad( {city=3,day=4} )   one value per day,
//                                          repeated for each city
//     {}        .broad( anything )         scalars still go
//                                          everywhere
//
// Every tag of `small` must exist in `shape`, with the same size
// (or with size 1, but that's for later). Otherwise:
//
//     panic: Can't broadcast Tensor({town=3,f32}) to {city=3,day=4,f32}
//
// "Copies" sounds expensive. It isn't: the broadcast is a note in
// the recipe ("pretend this is repeated"), and the compiler
// nearly always arranges for the copy never to exist in memory.
//
// Gail goes on air in two minutes. Run this first, and look at
// the panic: it names the operation and both shapes. Then say
// what you mean.
//
const std = @import("std");
const zml = @import("zml");

/// The national feed, corrected by somebody who has been outside.
fn forecast(feed: zml.Tensor, correction: zml.Tensor) zml.Tensor {
    return feed.add(correction);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const feed: zml.Tensor = .init(.{ .city = 3, .day = 4 }, .f32);
    const correction: zml.Tensor = .init(.{ .city = 3 }, .f32);

    var exe = try platform.compileFn(gpa, io, forecast, .{ feed, correction }, .{});
    defer exe.deinit();

    const cities = [_][]const u8{ "Springfield", "Shelbyville", "Ogdenville" };

    var feed_buffer: zml.Buffer = try .fromBytes(io, platform, feed.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 20, 21, 19, 22 },
        .{ 15, 14, 16, 15 },
        .{ 30, 31, 29, 33 },
    }));
    defer feed_buffer.deinit();

    var correction_buffer: zml.Buffer = try .fromBytes(io, platform, correction.shape(), std.mem.sliceAsBytes(&[_]f32{ -4, 1.5, 0 }));
    defer correction_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ feed_buffer, correction_buffer });
    defer result.deinit();

    const temps = try result.getValue([3][4]f32, io);

    try stdout.print("The forecast is a {f}.\n", .{result.shape()});
    for (cities, temps) |city, row| {
        try stdout.print("{s:<12}", .{city});
        for (row) |t| try stdout.print(" {d:.1}", .{t});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
