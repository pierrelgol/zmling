//
// Saturday morning. The queue goes around the corner, and the
// till is run by a neural network, because of course it is.
//
// Here is its problem. A customer says "croissant". A Linear
// layer wants a vector of floats. "croissant" is not a vector
// of floats. It is barely a word.
//
// The solution has two steps, and it is the front door of every
// language model ever built:
//
//  1. Number everything. The menu has five items, so they get
//     the ids 0 to 4. (For text, the thing that hands out ids is
//     called a tokenizer, the ids are TOKENS, and the menu is
//     the VOCABULARY. Hence the names below.)
//
//  2. Keep a table with one row per id, and replace each id by
//     its row. The rows are vectors of floats. Done.
//
// The table is called an EMBEDDING, and step 2 is nothing but
// the `gather` of chapter 10 wearing an apron:
//
//     pub const TokenEmbedding = struct {
//         weight: Tensor,
//
//         pub fn forward(self: TokenEmbedding, idx: Tensor) Tensor {
//             // (asserts: idx is of an integer dtype,
//             //           weight is of rank 2)
//             return self.weight
//                 .withTags(.{ .voc, .d })
//                 .gather(.{ .voc = idx }, .{});
//         }
//     };
//
// (It is `TokenEmbedding`. There is no `nn.Embedding` in ZML,
// whatever your fingers remember from PyTorch.)
//
// Read the one statement of `forward`:
//
//  - the weight is `{voc, d}`: one row per menu item, `d`
//    numbers per row. `forward` tags it itself, so it doesn't
//    matter what you called its axes;
//
//  - `idx` holds the ids. Any shape, but an INTEGER dtype: you
//    cannot order item number 2.5;
//
//  - the result has the axes of `idx`, followed by `.d`. An
//    order of four items, `{s=4}`, comes back as `{s=4, d=3}`:
//    four rows of the table. (`.s` for "sequence": the usual
//    tag for "position in the list of tokens".)
//
// Where do the rows come from? They are learned, like any other
// weight, and training has a pleasant side effect: items that
// are used alike end up with rows that look alike. Here Chef
// Babette wrote them by hand, and gave them a meaning, which
// real embeddings never have the courtesy to do: how sweet, how
// buttery, how crunchy.
//
// An order comes in: "A croissant, two eclairs, and a macaron."
// The till handles it three times over:
//
//  1. FROM THE PANTRY: the TokenEmbedding, alone. Somebody
//     declared the order wrong (ZML will tell you about it, read
//     the panic), and nobody typed the order in.
//
//  2. BY HAND: write `lookUp`, the gather of exercise 070, with
//     no layer around it. The table arrives tagged `{voc, d}`.
//     Compile it and run it: the program checks that it returns
//     the very same rows.
//
//  3. THE WHOLE ORDER: a `Till` is a model with a menu inside.
//     Its `forward` looks the items up and answers with ONE
//     flavor vector for the whole order: the average of its
//     items (chapter 6: reduce the axis, then get rid of it).
//     Write it, and the Buffers of a Till: structs in structs,
//     as in chapter 12.
//
const std = @import("std");
const zml = @import("zml");

const menu_names = [_][]const u8{ "croissant", "baguette", "eclair", "macaron", "brownie" };

/// The embedding, by hand: the rows of `table` {voc, d} that
/// `ids` asks for. Returns the axes of `ids`, then `.d`.
fn lookUp(table: zml.Tensor, ids: zml.Tensor) zml.Tensor {
    return ???;
}

const Till = struct {
    menu: zml.nn.TokenEmbedding,

    /// order: {.s} item ids -> {.d}, the average flavor of the order.
    pub fn forward(self: Till, order: zml.Tensor) zml.Tensor {
        return ???;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const menu: zml.nn.TokenEmbedding = .{ .weight = .init(.{ .voc = 5, .d = 3 }, .f32) };

    // The order: four positions, one item id each.
    const order: zml.Tensor = .init(.{ .s = 4 }, .f32);

    var exe = try platform.compile(gpa, io, menu, .forward, .{order}, .{});
    defer exe.deinit();

    // TODO: `hand_exe`, the same lookup by hand: `lookUp` is a
    // function of the table and the ids.

    // The till: a model that has the menu as a field.
    const till: Till = .{ .menu = menu };
    // TODO: `till_exe`.

    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, menu.weight.shape(), std.mem.sliceAsBytes(&[5][3]f32{
        //  sweet, buttery, crunchy
        .{ 0.2, 0.9, 0.7 }, // 0: croissant
        .{ 0.0, 0.1, 0.9 }, // 1: baguette
        .{ 0.8, 0.6, 0.1 }, // 2: eclair
        .{ 0.9, 0.2, 0.5 }, // 3: macaron
        .{ 1.0, 0.7, 0.2 }, // 4: brownie
    }));
    defer weight_buffer.deinit();

    const menu_buffers: zml.Bufferized(zml.nn.TokenEmbedding) = .{ .weight = weight_buffer };
    const till_buffers: zml.Bufferized(Till) = ???;

    // "A croissant, two eclairs, and a macaron."
    const items = [4]i32{ ???, ???, ???, ??? };
    var order_buffer: zml.Buffer = try .fromBytes(io, platform, order.shape(), std.mem.sliceAsBytes(&items));
    defer order_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ menu_buffers, order_buffer });
    defer result.deinit();

    try stdout.print("order {f}\n", .{result.shape()});
    for (items, try result.getValue([4][3]f32, io)) |item, flavor| {
        try stdout.print("  {s:<9}  sweet {d:.1}  buttery {d:.1}  crunchy {d:.1}\n", .{ menu_names[@intCast(item)], flavor[0], flavor[1], flavor[2] });
    }

    var by_hand: zml.Buffer = try hand_exe.eval(gpa, io, .{ weight_buffer, order_buffer });
    defer by_hand.deinit();
    const same = std.meta.eql(try result.getValue([4][3]f32, io), try by_hand.getValue([4][3]f32, io));
    try stdout.print("By hand {f}: {s}\n", .{ by_hand.shape(), if (same) "the same twelve numbers." else "NOT the same numbers!" });

    var taste: zml.Buffer = try till_exe.eval(gpa, io, .{ till_buffers, order_buffer });
    defer taste.deinit();
    const flavor = try taste.getValue([3]f32, io);
    try stdout.print("The whole order {f}: sweet {d:.3}  buttery {d:.3}  crunchy {d:.3}\n", .{ taste.shape(), flavor[0], flavor[1], flavor[2] });
    try stdout.flush();
}
