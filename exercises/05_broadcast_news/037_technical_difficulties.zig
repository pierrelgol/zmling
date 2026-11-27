//
// WE ARE EXPERIENCING TECHNICAL DIFFICULTIES. PLEASE STAND BY.
//
// Tags have saved you from every shape mistake so far. This
// exercise is about the one place in this chapter where they
// don't, so that you meet it here and not at 3 a.m. in a model
// that "trains fine but gives weird results".
//
// Open zml/zml/core/tensor.zig, find `pub fn broad`, and read the
// comment its authors left in the middle:
//
//     // TODO: broad is error prone because of this:
//     // it will happily broadcast .{ .a = 10, .b = 1 }
//     // to .{ .b = 10, .a = 5 }
//
// Here is what happens. `broad` has two ways of working:
//
//   - `self` has FEWER axes than the target: each axis of `self`
//     is sent to the axis of the target with the same TAG. Safe.
//     It even gets the order right if the target has the tags in
//     another order.
//
//   - `self` has the SAME rank as the target: the axes are paired
//     up by POSITION. First with first, second with second. The
//     tags are not looked at. At all. Each pair must only have
//     equal sizes, or size 1 on the side of `self`. And the
//     result is stamped with the tags of the target.
//
// So if you added a size 1 axis yourself, and put it on the wrong
// side, `broad` can't notice from the tags. Its last line of
// defense is the sizes: stretching 3 values over an axis of size
// 4 is refused ("Can't broadcast ..."). But when the sizes
// happen to agree... it happily broadcasts.
//
// Remember Gail's corrections from exercise 033, one per city?
// The forecast had four days then. Somebody has since "tidied"
// the code by making the column by hand:
//
//     correction.insertAxes(.city, .{.day})
//
// which is {day=1,city=3}: a ROW, not a column. With 3 cities and
// 4 days that was a panic, and it would have been fixed before
// lunch. But this is the long weekend forecast: 3 cities, THREE
// days. Everything compiles. Everything runs. The output has the
// right shape and the right tags. And Springfield's correction
// (-4, the pizza oven, remember) is being applied to FRIDAY, in
// every city:
//
//     what Gail meant            what is on air
//     16   17   15               16   22.5 19
//     16.5 15.5 17.5             11   15.5 16
//     30   31   29               26   32.5 29
//
// No panic will help you this time: the only symptom is wrong
// numbers on air. Find the line, and put the 1 where it belongs.
//
// Before you do, get yourself a witness. Below `forecast` there
// is a second ritual, `forecastByRule3`, with nothing in it. It
// computes the same forecast the way its name says: make the
// column, and let the fine print of exercise 036 do the
// stretching, with no broadcast function in sight. (That is
// habit number two in the list below, and the list explains why
// this version can't go wrong silently. The program compares the
// witness's numbers, not its methods: a `broad` in there would
// be your secret, and one more place for today's bug to hide.)
//
// `main` compares the two results on its last line, but the
// witness was never called to the stand: compiling it and
// running it is left to you, right next to the lines that do it
// for `forecast`. Run the program once with the witness in place
// and the bug still there: the last line says `false`. That is
// what a second opinion is for.
//
// Three habits that keep you out of this hole:
//
//   - Don't do `broad`'s job for it. Hand it the tensor with the
//     axes it really has ({city=3}) and let the tags route it.
//
//   - Or let the operator do the stretching (rule 3 of the fine
//     print): `feed.add(correction.insertAxes(.city, .{.day}))`
//     DOES panic, because rule 3 insists on the same tags in the
//     same order: "add expects tensor shapes to match, got
//     {city=3,day=3,f32} and {day=1,city=3,f32}".
//
//   - Test with sizes that are all different. A square table
//     hides this bug; a 3 by 4 one screams.
//
const std = @import("std");
const zml = @import("zml");

/// The national feed, corrected by somebody who has been outside.
fn forecast(feed: zml.Tensor, correction: zml.Tensor) zml.Tensor {
    const column = correction.insertAxes(.city, .{.day});

    return feed.add(column.broad(feed.shape()));
}

/// The same forecast, with no broadcast function at all.
fn forecastByRule3(feed: zml.Tensor, correction: zml.Tensor) zml.Tensor {
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

    const feed: zml.Tensor = .init(.{ .city = 3, .day = 3 }, .f32);
    const correction: zml.Tensor = .init(.{ .city = 3 }, .f32);

    var exe = try platform.compileFn(gpa, io, forecast, .{ feed, correction }, .{});
    defer exe.deinit();

    // The witness: `forecastByRule3`, compiled into `witness_exe`.

    const cities = [_][]const u8{ "Springfield", "Shelbyville", "Ogdenville" };

    var feed_buffer: zml.Buffer = try .fromBytes(io, platform, feed.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ 20, 21, 19 },
        .{ 15, 14, 16 },
        .{ 30, 31, 29 },
    }));
    defer feed_buffer.deinit();

    var correction_buffer: zml.Buffer = try .fromBytes(io, platform, correction.shape(), std.mem.sliceAsBytes(&[_]f32{ -4, 1.5, 0 }));
    defer correction_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ feed_buffer, correction_buffer });
    defer result.deinit();

    // Same Buffers, other Exe: the result is called `witness`.

    const temps = try result.getValue([3][3]f32, io);
    const witness_temps = try witness.getValue([3][3]f32, io);

    try stdout.print("{f}\n", .{result.shape()});
    try stdout.print("             Fri  Sat  Sun\n", .{});
    for (cities, temps) |city, row| {
        try stdout.print("{s:<12}", .{city});
        for (row) |t| try stdout.print(" {d:.1}", .{t});
        try stdout.print("\n", .{});
    }
    try stdout.print("The witness agrees: {}\n", .{std.meta.eql(temps, witness_temps)});
    try stdout.flush();
}
