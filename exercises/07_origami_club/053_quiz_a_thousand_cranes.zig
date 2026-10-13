//
// Quiz! Championship day. The gym smells of floor polish and
// fear. Across the room, the Northside Paper Cranes are doing
// synchronized finger stretches.
//
// The final event is the Relay: one big sheet is torn into
// small square PATCHES, and every patch is handed to a
// different folder. The judges give you the sheet,
//
//     sheet {rows=4,cols=4}        {  0,  1,   2,  3 },
//                                  {  4,  5,   6,  7 },
//
//                                  {  8,  9,  10, 11 },
//                                  { 12, 13,  14, 15 }
//
// and want the four 2x2 patches, one per row of the result:
//
//     {patches=4,squares=4}        {  0,  1,  4,  5 }    top left
//                                  {  2,  3,  6,  7 }    top right
//                                  {  8,  9, 12, 13 }    bottom left
//                                  { 10, 11, 14, 15 }    bottom right
//
// Then, because the Relay is run backwards for reasons lost to
// history, the patches are handed out LAST ONE FIRST.
//
// Timmy's proposal was `sheet.reshape(.{ .patches = 4,
// .squares = 4 })`. Right shape! And it returns the four rows
// of the sheet, which are not patches. You know why.
//
// Here is the plan Ingrid drew on the back of a napkin:
//
//     1. cut .rows into (which patch row, which row inside it)
//     2. cut .cols the same way        -> four axes
//     3. reorder the axes so that the two "which patch" axes
//        are together, and the two "inside the patch" axes too
//     4. glue each pair into one axis  -> .patches, .squares
//     5. flip the order of the patches
//
// Everything you need is in this chapter:
//
//     splitAxis   merge   transpose   contiguous   reverse
//
// The tags of the intermediate axes are yours to choose; only
// the final {patches=4,squares=4} is checked. There is more
// than one correct way to write it.
//
// (This is not a toy, by the way. Cutting an image into patches
// and laying each patch flat is the very first thing a Vision
// Transformer does to its input. You just have to do it with
// Coach breathing down your neck.)
//
const std = @import("std");
const zml = @import("zml");

fn relay(sheet: zml.Tensor) zml.Tensor {
    // Timmy's proposal. Replace it with Ingrid's napkin.
    return sheet.reshape(.{ .patches = 4, .squares = 4 });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const sheet: zml.Tensor = .init(.{ .rows = 4, .cols = 4 }, .i32);

    var exe = try platform.compileFn(gpa, io, relay, .{sheet}, .{});
    defer exe.deinit();

    var squares: [16]i32 = undefined;
    for (&squares, 0..) |*square, i| square.* = @intCast(i);

    var sheet_buffer: zml.Buffer = try .fromBytes(io, platform, sheet.shape(), std.mem.sliceAsBytes(&squares));
    defer sheet_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{sheet_buffer});
    defer result.deinit();

    const patches: zml.Slice = try result.toSliceAlloc(gpa, io);
    defer patches.free(gpa);

    try stdout.print("{f}\n", .{patches.shape});

    const width: usize = @intCast(patches.shape.dim(-1));
    var lines = std.mem.window(i32, patches.constItems(i32), width, width);
    while (lines.next()) |line| try stdout.print("  {any}\n", .{line});
    try stdout.flush();
}
