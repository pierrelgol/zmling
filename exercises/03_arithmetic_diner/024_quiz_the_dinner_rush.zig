//
// Quiz! It's Friday, 6 pm, the bowling league just walked in, and
// Flo has handed you the register. "You know the rules, hon."
//
// Four tables. For each one you get:
//
//     plates   how many plates they ordered        whole numbers
//     price    dollars per plate                   floats
//     mood     how the table feels about Gus's     floats, any
//              cooking                             sign
//
// and one number for the whole room:
//
//     coupon   dollars off every bill              a float
//
// The rules, in order:
//
//   1. The bill is plates times price.
//
//   2. The coupon comes off. But a bill never goes below zero.
//      Flo does not pay people to eat.
//
//   3. The tip is a percentage of that: it grows the bill by a
//      factor of
//
//          1 + 0.5 * sigmoid(mood)
//
//      so between nothing (furious) and 50% (delighted).
//
//   4. The total is rounded UP to whole dollars, and it is an
//      integer from here on. The register has no cents key.
//
//   5. The walk: table number t pays t dollars extra. Tables are
//      numbered 1, 2, 3, 4 and nobody hands you that list. Make
//      it from scratch. (A list that comes out of `arange`
//      carries no tag. That's fine: the sign in the window says
//      same shape and same dtype, and four integers are four
//      integers.)
//
// On a napkin, table 1 during the rush: 2 plates at 7.50 is 15
// dollars, the coupon leaves 5. Their mood is 0, the sigmoid of 0
// is 0.5, so the factor is 1.25 and the bill 6.25. Rounded up: 7.
// Plus 1 for the walk: 8.
//
// Flo likes her recipes small, so there are three of them: `bill`
// is rules 1 and 2, `tipFactor` is rule 3, and `register` calls
// both and finishes the job. A recipe may call other recipes,
// exercise 023 said so.
//
// The coupon is 10 dollars during the dinner rush and nothing
// afterwards, and Flo is not going to recompile her register at
// 9 pm: it is an INPUT, one number, and the same Exe rings up the
// room twice. Madame Zelda's thermometer worked the same way
// (exercise 016).
//
// Everything you need is in exercises 017 to 023: element-wise
// operations and their constant forms, unary math, activations,
// `convert`, tensors made in the graph, and chains that nest in
// the right places. Mind the dtypes: the data and the prints
// below tell you what goes in and what must come out.
//
// Of main(), the dishwasher left the data, one Buffer and the
// prints. The prints expect an Exe called `exe` and, on each turn
// of the loop, the four totals in `totals`; every other name is
// yours to choose. He's on his break. He is always on his break.
//
const std = @import("std");
const zml = @import("zml");

const plates_data = [4]i32{ 2, 1, 6, 4 };
const price_data = [4]f32{ 7.5, 6, 3.25, 5 };
const mood_data = [4]f32{ 0, 3, -2, 2 };

/// Rules 1 and 2: plates * price - coupon, never below zero.
/// Dollars and cents.
fn bill(plates: zml.Tensor, price: zml.Tensor, coupon: zml.Tensor) zml.Tensor {
    return ???;
}

/// Rule 3: 1 + 0.5 * sigmoid(mood)
fn tipFactor(mood: zml.Tensor) zml.Tensor {
    return ???;
}

/// All five rules. Whole dollars, as integers.
fn register(plates: zml.Tensor, price: zml.Tensor, mood: zml.Tensor, coupon: zml.Tensor) zml.Tensor {
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

    // Four placeholders: three of four elements tagged .tables,
    // and the coupon.
    const plates: zml.Tensor = ???;
    const price: zml.Tensor = ???;
    const mood: zml.Tensor = ???;
    const coupon: zml.Tensor = ???;

    // The register, compiled once, as `exe`.

    try stdout.print("The register rings up {f}\n", .{exe.output_shapes[0]});

    var plates_buffer: zml.Buffer = try .fromBytes(io, platform, plates.shape(), std.mem.sliceAsBytes(&plates_data));
    defer plates_buffer.deinit();
    // The prices and the moods.

    // Dinner rush, then the rest of the evening.
    for ([_]f32{ 10, 0 }) |dollars_off| {
        // The coupon as a body, the run, and the four totals
        // back on the host. That is more than one line: write
        // them above this one, cleanup included, and end with
        // `totals`.
        const totals = ???;

        try stdout.print("coupon {d}: {any}\n", .{ dollars_off, totals });
    }
    try stdout.flush();
}
