//
// Championship rules, article 7: "Every entry shall be mounted
// on a card leaving a clear margin on all sides." Last year the
// Northside Paper Cranes got a rival disqualified over a margin
// of 0.9 squares. Coach Brunhilde owns a ruler now. It has a
// name.
//
// Adding a border around a tensor is called PADDING:
//
//     x.pad(value, paddings)
//
// `value` is a Zig number, the thing to fill the margin with.
// ZML turns it into a constant of x's dtype. `paddings` says
// how much to add, axis by axis, with this little struct:
//
//     pub const Pad = struct {
//         low: i64 = 0,        // how many BEFORE index 0
//         high: i64 = 0,       // how many AFTER the last index
//         interior: i64 = 0,   // how many BETWEEN two neighbours
//     };
//
// It lives at `zml.Tensor.Pad`. Axes you don't mention are not
// padded. For x = [1, 2, 3], tagged .a:
//
//     x.pad(0, .{ .a = Pad{ .low = 2 } })              0 0 1 2 3
//     x.pad(0, .{ .a = Pad{ .high = 1 } })             1 2 3 0
//     x.pad(9, .{ .a = Pad{ .low = 1, .high = 1 } })   9 1 2 3 9
//     x.pad(0, .{ .a = Pad{ .interior = 1 } })         1 0 2 0 3
//
// The new size of an axis of size d is
//
//     low + d + (d - 1) * interior + high
//
// Two things in there deserve a second look. `interior` spreads
// the elements apart, which is how some upsampling layers are
// built. And `low` and `high` may be NEGATIVE, in which case
// they remove elements instead of adding them (a bit of
// scissors after all; don't tell Coach):
//
//     x.pad(0, .{ .a = Pad{ .low = -1 } })             2 3
//
// Where will you meet padding? Everywhere a batch must have one
// shape while the data doesn't: sentences of different lengths
// padded to the longest one, images padded before a
// convolution so the output keeps its size.
//
// And now the trap of the day. ZML's own doc comment says:
//
//     /// Usage: x.pad(0, .{ .a = .{ .low = 1, .high = 1 }});
//
// At the version pinned by zmling, that line does not compile.
// `paddings` is `anytype`, so Zig has no idea that the inner
// `.{ .low = 1, .high = 1 }` is supposed to be a `Pad`; it
// makes an anonymous struct, and ZML's code then fails to store
// it:
//
//     error: expected type 'tensor.Tensor.Pad', found '...struct...'
//
// The error names the cure: write the type yourself, as in the
// examples above. A local alias keeps it short:
//
//     const Pad = zml.Tensor.Pad;
//
// The task: Ingrid's entry is a sheet of 2 rows by 3 columns.
// Mount it on a card with a margin of exactly ONE blank square
// (a zero) on all four sides. The apprentice who wrote the
// mounting code copied the doc comment, and also seems to have
// run out of card on one side. The ruler is watching.
//
const std = @import("std");
const zml = @import("zml");

const Pad = zml.Tensor.Pad;

fn mount(sheet: zml.Tensor) zml.Tensor {
    return sheet.pad(0, .{
        .rows = .{ .low = 1, .high = 1 },
        .cols = .{ .low = 1 },
    });
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const sheet: zml.Tensor = .init(.{ .rows = 2, .cols = 3 }, .i32);

    var exe = try platform.compileFn(gpa, io, mount, .{sheet}, .{});
    defer exe.deinit();

    var sheet_buffer: zml.Buffer = try .fromBytes(io, platform, sheet.shape(), std.mem.sliceAsBytes(&[_]i32{ 1, 2, 3, 4, 5, 6 }));
    defer sheet_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{sheet_buffer});
    defer result.deinit();

    const card: zml.Slice = try result.toSliceAlloc(gpa, io);
    defer card.free(gpa);

    try stdout.print("Mounted on {f}\n", .{card.shape});

    const width: usize = @intCast(card.shape.dim(-1));
    var lines = std.mem.window(i32, card.constItems(i32), width, width);
    while (lines.next()) |line| try stdout.print("  {any}\n", .{line});
    try stdout.flush();
}
