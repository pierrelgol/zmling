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
// And while the folders fold, the judges grade the sheet: every
// patch gets the AVERAGE of its four squares, and the averages
// are pinned on the wall in the same layout as the patches,
//
//     {rows=2,cols=2}, f32         {  2.5,  4.5 },
//                                  { 10.5, 12.5 }
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
//
// That is `patchify`, and the other two helpers start from it:
//
//     relay     patchify, then flip the order of the patches
//     pool      patchify, then the mean of every patch, then
//               fold the four means into {rows=2,cols=2}
//
// Everything you need for the cutting is in this chapter:
//
//     splitAxis   merge   transpose   contiguous   reverse
//     reshape
//
// and the averaging comes from earlier ones: `convert` (021),
// because an average of integers is not an integer, and `mean`
// (041). Think about what shape `mean` hands you before you
// reshape it.
//
// The tags of the intermediate axes are yours to choose; only
// the final shapes are checked. There is more than one correct
// way to write it.
//
// Three helpers to write, then. And `main` stops right after
// the numbers: the placeholder, the two Exes (`relay` and
// `pool`), the Buffer, the runs and the read-backs are yours.
// The prints at the bottom tell you which names they expect.
//
// (This is not a toy, by the way. Cutting an image into patches
// and laying each patch flat is the very first thing a Vision
// Transformer does to its input. You just have to do it with
// Coach breathing down your neck.)
//
const std = @import("std");
const zml = @import("zml");

/// The four 2x2 patches in reading order: {patches=4,squares=4}.
fn patchify(sheet: zml.Tensor) zml.Tensor {
    // Ingrid's napkin, steps 1 to 4.
    return ???;
}

/// The patches, last one first.
fn relay(sheet: zml.Tensor) zml.Tensor {
    return ???;
}

/// The average of every patch, laid out like the patches: {rows=2,cols=2}, f32.
fn pool(sheet: zml.Tensor) zml.Tensor {
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

    var squares: [16]i32 = undefined;
    for (&squares, 0..) |*square, i| square.* = @intCast(i);

    // From here to the prints, it's all yours. The sheet is a
    // {rows=4,cols=4} of i32, and the prints need:
    //
    //     patches    a zml.Slice, the result of `relay`
    //     pool_exe   the Exe compiled from `pool`
    //     pooled     a [2][2]f32, the result of `pool`

    try stdout.print("{f}\n", .{patches.shape});

    const width: usize = @intCast(patches.shape.dim(-1));
    var lines = std.mem.window(i32, patches.constItems(i32), width, width);
    while (lines.next()) |line| try stdout.print("  {any}\n", .{line});

    try stdout.print("pooled {f}\n", .{pool_exe.output_shapes[0]});
    for (pooled) |row| try stdout.print("  {d:.1} {d:.1}\n", .{ row[0], row[1] });
    try stdout.flush();
}
