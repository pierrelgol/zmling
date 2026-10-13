//
// Quiz! Flight ZG404 ("Luggage Not Found") has landed, and so
// has an audit. Brenda is on her break. The break is 45 minutes.
// The label maker is on her desk, and what Brenda doesn't know
// can't hurt you.
//
// Two tables arrive at the office:
//
//     weights   from the check-in scales. No tags at all, just
//               {2,3,f32}. The delivery slip says: one row per
//               flight, one column per bag.
//
//     limits    from the airline, in the airline's own dialect
//               AND the airline's own layout: {items=3, planes=2},
//               one row per item. A plane is a flight, an item is
//               a bag, and the airline will not change.
//
// The audit wants the EXCESS BAGGAGE REPORT:
//
//   1. Speak one language. Inside the recipe, both tables must
//      end up as {flights=2, bags=3}: tags and layout. One needs
//      labels. The other needs new labels and a quarter turn;
//      that part is `toOfficeLayout`.
//
//   2. Trust nobody. Before computing anything, the recipe checks
//      that the two tables now agree on dims, dtype AND tags.
//      (The check is already written. It's the argument that is
//      missing.)
//
//   3. The excess of a bag is its weight minus its limit, and a
//      bag under the limit has an excess of zero, not a negative
//      one. Nobody gets a refund for packing light. That part is
//      `excess`.
//
//   4. Brenda's fairness rule: the excess of each bag is divided
//      by the number of bags on a flight. No axis numbers in the
//      recipe: the auditor reads code.
//
//   5. In main(), report under which tags the result comes out,
//      and at which position the .bags axis is.
//
// Three recipes to write, then. And main() has been audited
// already: the auditor took the placeholders, the airline's
// Buffer and the run as "evidence". You've written all of those
// before. The data and the prints are still there.
//
// Everything is in exercises 025 to 031, with a little help from
// the diner. No hints this time. The break is now 40 minutes.
//
const std = @import("std");
const zml = @import("zml");

// One row per flight, one column per bag.
const weights_data = [2][3]f32{
    .{ 20, 26, 23 },
    .{ 32, 18, 29 },
};

// One row per item, one column per plane: the first plane allows
// 23 kg per bag, the second one 20.
const limits_data = [3][2]f32{
    .{ 23, 20 },
    .{ 23, 20 },
    .{ 23, 20 },
};

/// The airline's table, in the office's tags and layout:
/// {flights, bags}.
fn toOfficeLayout(limits: zml.Tensor) zml.Tensor {
    return ???;
}

/// Weight minus limit, never negative.
fn excess(weights: zml.Tensor, limits: zml.Tensor) zml.Tensor {
    return ???;
}

fn excessReport(weights: zml.Tensor, limits: zml.Tensor) zml.Tensor {
    // 1.
    const w = ???;
    const l = ???;

    // 2.
    std.debug.assert(???);

    // 3. and 4.
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

    // The two tables, as they arrive at the office.
    const weights: zml.Tensor = ???;
    const limits: zml.Tensor = ???;

    var exe = try platform.compileFn(gpa, io, excessReport, .{ weights, limits }, .{});
    defer exe.deinit();

    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), std.mem.sliceAsBytes(&weights_data));
    defer weights_buffer.deinit();

    // The airline's limits.

    // The run.
    var result: zml.Buffer = ???;
    defer result.deinit();

    // 5.
    const report: zml.Shape = ???;
    try stdout.print("Report {f}, bags on axis {d}, fully tagged: {}\n", .{
        report,
        report.axis(???),
        report.isFullyTagged(),
    });
    try stdout.print("Excess: {any}\n", .{try result.getValue([2][3]f32, io)});
    try stdout.flush();
}
