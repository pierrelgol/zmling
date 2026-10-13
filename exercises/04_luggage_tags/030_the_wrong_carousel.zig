//
// Element-wise operations compare dims and dtype, and never look
// at tags. Fine. Let's see what that means when two tensors hold
// the same kind of data, with the same tags, in a DIFFERENT
// ORDER:
//
//     weights      {flights=2, bags=3}    a row per flight
//     surcharges   {bags=3, flights=2}    a row per bag
//
// A human sees it at once: same table, one of them lying on its
// side. ZML sees dims { 2, 3 } against dims { 3, 2 } and stops:
//
//     panic: add expects tensor shapes to match, got
//            {flights=2,bags=3,f32} and {bags=3,flights=2,f32}
//
// It will not turn the table for you. Element-wise operations
// never reorder axes on their own initiative; moving every
// number of a tensor around is real work, and ZML doesn't do
// real work behind your back.
//
// But look at what the tags did to that error message. Without
// them it would read "got {2,3,f32} and {3,2,f32}", and you'd be
// off counting axes on your fingers. With them, the diagnosis is
// printed right there: same names, wrong order.
//
// Two ways to react, one of them wrong.
//
// THE WRONG ONE: relabel until the panic goes away. Kevin's
// first idea was `surcharges.withTags(.{ .flights, .bags })`.
// Labels are stickers. Moving the stickers does not move the
// suitcases: the dims are still { 3, 2 }, and the numbers are
// still stored a row per bag. (Here it would only panic again.
// With a SQUARE table, 3 flights of 3 bags, the dims would match
// either way, nothing would panic, and every surcharge would
// land on the wrong bag. Silently. This is the classic bug that
// tags are meant to help you see: so when an axis order is
// wrong, don't paper over it.)
//
// THE RIGHT ONE: actually turn the table.
//
//     surcharges.transpose(.{ .flights, .bags })
//
// `transpose` takes the axes in the order you want them, like
// its Shape namesake in exercise 007, except that on a Tensor
// the numbers really move, at run time; it gets a proper
// introduction in a later chapter.
//
// Notice that you never say where the axes currently are. You
// say where you want them, by name.
//
// The office computes what each passenger owes: the weight of
// the bag (one dollar per kilogram, Brenda keeps it simple) plus
// a surcharge that depends on the bag and on the flight. The two
// tables come from different departments. Kevin added them
// anyway, and then "fixed" the panic. You know which way.
//
// Then the square table, so that you have seen it once. On
// Tuesdays there are three flights of three bags, the same two
// departments send the same two kinds of table, and the same
// recipe is compiled a second time for the new sizes. Nothing
// about `owed` changes. What main() lacks is the Tuesday
// plumbing, all of it chapter 02: the two placeholders (each
// department keeps its habits: read the Monday ones), the compile
// call, and the run. The Buffers are there.
//
// Careful with those placeholders. Nine numbers are nine numbers:
// tag the accounting table the way you would LIKE it to be, and
// everything compiles, everything runs, and six passengers pay
// somebody else's surcharge.
//
const std = @import("std");
const zml = @import("zml");

/// weights + surcharges, a row per flight.
fn owed(weights: zml.Tensor, surcharges: zml.Tensor) zml.Tensor {
    return weights.add(surcharges.withTags(.{ .flights, .bags }));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // From the check-in desk: a row per flight.
    const weights: zml.Tensor = .init(.{ .flights = 2, .bags = 3 }, .f32);
    // From accounting: a row per bag. Accounting will not change.
    const surcharges: zml.Tensor = .init(.{ .bags = 3, .flights = 2 }, .f32);

    var exe = try platform.compileFn(gpa, io, owed, .{ weights, surcharges }, .{});
    defer exe.deinit();

    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 10, 20, 30 },
        .{ 40, 50, 60 },
    }));
    defer weights_buffer.deinit();

    var surcharges_buffer: zml.Buffer = try .fromBytes(io, platform, surcharges.shape(), std.mem.sliceAsBytes(&[3][2]f32{
        .{ 1, 2 },
        .{ 3, 4 },
        .{ 5, 6 },
    }));
    defer surcharges_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ weights_buffer, surcharges_buffer });
    defer result.deinit();

    try stdout.print("Owed, as {f}: {any}\n", .{
        result.shape(),
        try result.getValue([2][3]f32, io),
    });

    // Tuesday. Check-in: a row per flight. Accounting: a row per
    // bag. Three of each.
    const square_weights: zml.Tensor = ???;
    const square_surcharges: zml.Tensor = ???;

    // The same recipe, compiled for Tuesday's shapes.

    var square_weights_buffer: zml.Buffer = try .fromBytes(io, platform, square_weights.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ 10, 20, 30 },
        .{ 40, 50, 60 },
        .{ 70, 80, 90 },
    }));
    defer square_weights_buffer.deinit();

    var square_surcharges_buffer: zml.Buffer = try .fromBytes(io, platform, square_surcharges.shape(), std.mem.sliceAsBytes(&[3][3]f32{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
        .{ 7, 8, 9 },
    }));
    defer square_surcharges_buffer.deinit();

    // Tuesday's run.
    var square_result: zml.Buffer = ???;
    defer square_result.deinit();

    try stdout.print("Tuesday, as {f}: {any}\n", .{
        square_result.shape(),
        try square_result.getValue([3][3]f32, io),
    });
    try stdout.flush();
}
