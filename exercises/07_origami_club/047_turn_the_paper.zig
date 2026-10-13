//
// "Turn the paper over" is step two of every origami diagram
// ever printed, and the step where Timmy (age nine, the club's
// newest member) usually ends up holding a paper ball.
//
// `reshape` changed the creases and left the strip alone.
// `transpose` is the opposite kind of operation. It keeps every
// axis, with its size and its tag, and changes their ORDER:
//
//     sheet   {rows=2,cols=3}      { { 0, 1, 2 },
//                                    { 3, 4, 5 } }
//
//     sheet.transpose(.{ .cols, .rows })
//
//             {cols=3,rows=2}      { { 0, 3 },
//                                    { 1, 4 },
//                                    { 2, 5 } }
//
// What used to be a row is now a column. The element that lived
// at (row 1, col 0) now lives at (col 0, row 1), and it is
// still 3. But look at the strip: it was 0 1 2 3 4 5 and it is
// now 0 3 1 4 2 5. The numbers MOVED. Transposing is real work
// for the machine, unlike reshaping.
//
// You saw `transpose` on Shapes in chapter 1, and the Tensor
// version takes the same argument: a permutation that says, for
// each position of the NEW shape, which OLD axis goes there.
//
//     pile: {sheets=2,rows=3,cols=4}
//
//     pile.transpose(.{ 2, 0, 1 })                  by index
//     pile.transpose(.{ .cols, .sheets, .rows })    by tag, same thing
//                              -> {cols=4,sheets=2,rows=3}
//
// Read the index version slowly: "new axis 0 is old axis 2".
// It does NOT say "old axis 0 goes to position 2". Getting this
// backwards is a rite of passage, and with tags you can't get
// it wrong: you simply write the order you want to see.
//
// The permutation must mention every axis exactly once. Two
// conveniences on top of that:
//
//     pile.transpose(.{})               an EMPTY tuple reverses
//                                       all the axes:
//                                       {cols=4,rows=3,sheets=2}
//
//     pile.swapAxes(.sheets, .rows)     exchanges two axes and
//                                       leaves the others alone:
//                                       {rows=3,sheets=2,cols=4}
//
// (`swapAxes` takes tags or indices, negative ones included.)
//
// Tags travel with their axis. This is where they start paying
// rent: after three transposes nobody remembers what "axis 1"
// is, but `.rows` is still `.rows`.
//
// Today's drill has four moves. `turn` works on a small sheet,
// so you can see the numbers move. The other three work on a
// pile of paper, and we only look at the shape they produce
// (`exe.output_shapes`, no need to run anything).
//
// Timmy wrote `restack`, Ingrid wrote `flip`. The other two
// moves were left to you, and so was the middle of the small sheet's block in
// `main`: the Exe is compiled and the Buffer is ready, but
// nobody runs the one on the other or brings the answer home.
// You did both in exercise 013.
//
const std = @import("std");
const zml = @import("zml");

/// Rows become columns.
fn turn(sheet: zml.Tensor) zml.Tensor {
    return sheet.transpose(???);
}

/// Coach wants {rows, cols, sheets}. Timmy insisted on using
/// indices, "like a professional".
fn restack(pile: zml.Tensor) zml.Tensor {
    return pile.transpose(.{ 2, 0, 1 });
}

/// Exchange .sheets and .rows, don't touch .cols.
fn flip(pile: zml.Tensor) zml.Tensor {
    return pile.swapAxes(.sheets, .rows);
}

/// All the axes, backwards.
fn insideOut(pile: zml.Tensor) zml.Tensor {
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

    // The small sheet, with numbers on it.
    {
        const sheet: zml.Tensor = .init(.{ .rows = 2, .cols = 3 }, .i32);

        var exe = try platform.compileFn(gpa, io, turn, .{sheet}, .{});
        defer exe.deinit();

        var sheet_buffer: zml.Buffer = try .fromBytes(io, platform, sheet.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2, 3, 4, 5 }));
        defer sheet_buffer.deinit();

        // Run the Exe on the sheet, then copy the result to host
        // memory as a zml.Slice called `turned`.
        var result: zml.Buffer = ???;
        defer result.deinit();

        const turned: zml.Slice = ???;
        defer turned.free(gpa);

        try stdout.print("turn:      {f}\n", .{turned.shape});

        // Print the strip, one line per step of the first axis.
        const width: usize = @intCast(turned.shape.dim(-1));
        var lines = std.mem.window(i32, turned.constItems(i32), width, width);
        while (lines.next()) |line| try stdout.print("  {any}\n", .{line});
    }

    // The pile: shapes only.
    const pile: zml.Tensor = .init(.{ .sheets = 2, .rows = 3, .cols = 4 }, .i32);

    inline for (.{ restack, flip, insideOut }, .{ "restack:  ", "flip:     ", "insideOut:" }) |move, name| {
        var exe = try platform.compileFn(gpa, io, move, .{pile}, .{});
        defer exe.deinit();

        try stdout.print("{s} {f}\n", .{ name, exe.output_shapes[0] });
    }
    try stdout.flush();
}
