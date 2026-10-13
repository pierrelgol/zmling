//
// Friday night at Tensorino's. The phone doesn't stop, and
// nobody is cutting pizzas apart any more: the problem tonight
// is putting them TOGETHER, and there are two ways to do that.
// Luigi mixes them up, so does everybody at first, and the
// difference is one word: does the result get a NEW axis?
//
// `concatenate` glues tensors end to end along an axis that
// already EXISTS. That axis gets longer; the rank stays the
// same.
//
//     a {slices=4}:  1 2 3 4         b {slices=4}:  5 6 7 8
//
//     zml.Tensor.concatenate(&.{ a, b }, .slices)
//
//         {slices=8}:  1 2 3 4 5 6 7 8
//
// `stack` piles tensors up along an axis that does NOT exist
// yet. The rank grows by one, and you must name the new axis.
//
//     zml.Tensor.stack(&.{ a, b }, 0, .boxes)
//
//         {boxes=2,slices=4}:  { 1, 2, 3, 4 },
//                              { 5, 6, 7, 8 }
//
// Two pizzas pushed together on one very long table, versus two
// pizza boxes on top of each other.
//
// The signatures:
//
//     pub fn concatenate(tensors: []const Tensor, axis_: anytype) Tensor
//     pub fn stack(tensors: []const Tensor, axis_: anytype, tag: anytype) Tensor
//
// Things to notice:
//
//   - Neither is a method. There's no "self": the tensors come
//     as a Zig slice, hence `zml.Tensor.concatenate(&.{ a, b,
//     c }, ...)`, with as many tensors as you like.
//
//   - For `concatenate`, the pieces may have DIFFERENT sizes
//     along the glued axis (a {slices=4} and a {slices=2} make
//     a {slices=6}), and must agree on all the other axes. The
//     result takes its tags from the first piece.
//
//   - For `stack`, all the pieces must have exactly the SAME
//     shape, tags included. Otherwise:
//
//         stack expects tensor shapes to match, got {slices=4,i32} and {slices=3,i32}
//
//   - `stack`'s `axis_` says where the new axis is inserted:
//     BEFORE that axis of the inputs. So `0` puts it first. To
//     put it after everything, the special value `.last`:
//
//         zml.Tensor.stack(&.{ a, b }, .last, .kinds)
//
//             {slices=4,kinds=2}:  { 1, 5 },
//                                  { 2, 6 },
//                                  { 3, 7 },
//                                  { 4, 8 }
//
//     Same eight numbers as before, in a different order in
//     memory: the two pizzas are now interleaved.
//
// There is no magic in `stack`. Its implementation gives every
// piece a new axis of size 1 (a reshape, free) and calls
// `concatenate` on that axis. A box is a tray of one.
//
// Four orders, for a margherita and a pepperoni of 4 slices
// each:
//
//   - `party`:    one long pizza of 8 slices, margherita first.
//   - `delivery`: two boxes on top of each other: {boxes=2,slices=4}.
//   - `tasting`:  for each slice position, the two kinds side by
//                 side: {slices=4,kinds=2}.
//   - `byHand`:   `delivery` again, WITHOUT calling `stack`. Do
//                 what the paragraph above says `stack` does.
//                 You added axes of size 1 in the newsroom
//                 (`insertAxes`, exercise 035), and a `reshape`
//                 from the last chapter would do the job as
//                 well.
//
// Nonna wrote `party`. Luigi took `delivery` and `tasting`
// over the phone, with the oven fan on. `byHand` is yours from
// top to bottom.
//
// One more thing. The pepperoni never left the kitchen: `main`
// has no placeholder and no Buffer for it. Its four slices
// hold 5, 6, 7 and 8.
//
const std = @import("std");
const zml = @import("zml");

fn party(margherita: zml.Tensor, pepperoni: zml.Tensor) zml.Tensor {
    return zml.Tensor.concatenate(&.{ margherita, pepperoni }, .slices);
}

fn delivery(margherita: zml.Tensor, pepperoni: zml.Tensor) zml.Tensor {
    return zml.Tensor.concatenate(&.{ margherita, pepperoni }, .slices);
}

fn tasting(margherita: zml.Tensor, pepperoni: zml.Tensor) zml.Tensor {
    return zml.Tensor.stack(&.{ margherita, pepperoni }, 0, .kinds);
}

/// Two boxes, no `stack`: {boxes=2,slices=4}.
fn byHand(margherita: zml.Tensor, pepperoni: zml.Tensor) zml.Tensor {
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

    const margherita: zml.Tensor = .init(.{ .slices = 4 }, .i32);

    var margherita_buffer: zml.Buffer = try .fromBytes(io, platform, margherita.shape(), std.mem.sliceAsBytes(&[_]i32{ 1, 2, 3, 4 }));
    defer margherita_buffer.deinit();
    // The pepperoni goes here: `pepperoni` and `pepperoni_buffer`.
    ???;

    inline for (
        .{ party, delivery, tasting, byHand },
        .{ "party:   ", "delivery:", "tasting: ", "byHand:  " },
    ) |order, name| {
        var exe = try platform.compileFn(gpa, io, order, .{ margherita, pepperoni }, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ margherita_buffer, pepperoni_buffer });
        defer result.deinit();

        const table: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer table.free(gpa);

        // The shape, then the numbers exactly as they sit in memory.
        try stdout.print("{s} {f} {any}\n", .{ name, table.shape, table.constItems(i32) });
    }
    try stdout.flush();
}
