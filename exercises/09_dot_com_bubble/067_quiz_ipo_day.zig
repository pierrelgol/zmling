//
// Quiz! March 2000. PetRocks.com goes public on Friday, the
// NASDAQ has never been higher, and absolutely nothing can go
// wrong. The bank wants a model that tells investors to BUY or
// to SELL. One layer wasn't impressive enough for the brochure,
// so there are two of them, with a ReLU in between:
//
//     hidden  = relu(W1 x + b1)
//     verdict = W2 hidden + b2
//
// The tensors:
//
//     x    {startup=3,features=2}   press releases, foosball tables
//     w1   {hidden=3,features=2}
//     b1   {hidden=3}
//     w2   {verdict=2,hidden=3}     verdict 0 is BUY, 1 is SELL
//     b2   {verdict=2}
//
// Look at the tags for a moment. The first layer consumes
// .features and produces .hidden; the second consumes .hidden
// and produces .verdict. A model written this way documents
// itself, and wiring the layers in the wrong order is a panic
// instead of a bad quarter.
//
// PART 1. `layer` is the valuation engine of the previous
// exercise with one difference: the axis to contract is an
// argument. Its signature is given, because of that last
// parameter: `comptime tag: @EnumLiteral()` is how a Zig
// function says "hand me something like .features". Inside, use
// `tag` wherever you would have written the name of the axis.
// Write it well. You will meet this function again.
//
// PART 2. Three functions that are not there at all. Each takes
// (x, w1, b1, w2, b2) and returns one Tensor:
//
//     analyst          {startup=3,verdict=2}
//         The two formulas above, by calling `layer` twice.
//
//     recommendation   {startup=3}, .i32
//         For each startup, the POSITION of its best score
//         along .verdict: 0 or 1.
//
//     confidence       {startup=3}
//         Turn each startup's two scores into probabilities
//         that add up to 1, and keep the larger one.
//
// The last two are chapter 6, and they start by calling
// `analyst`.
//
// PART 3. `main` has the data and the prints. The seance in
// between is yours, from memory: five placeholders, three Exes,
// five Buffers, three results. The prints expect the results to
// be called `scores`, `picks` and `sure`. Free what you make.
//
// Everything you need, old and new:
//
//     a.dot(b, .tag)        contract over a named axis
//     t.broad(shape)        broadcast by matching tags
//     a.add(b)              same shapes, please
//     t.relu()              negative numbers become 0
//     t.softmax(.tag)       t.max(.tag)       t.argMax(.tag)
//     t.squeeze(.tag)       for the souvenirs
//
// No more hints. Brenda has gone on holiday; Chad is watching
// over your shoulder and saying "disrupt" a lot.
//
const std = @import("std");
const zml = @import("zml");

/// One linear layer: contract `x` and `w` over `tag`, add the bias.
fn layer(x: zml.Tensor, w: zml.Tensor, bias: zml.Tensor, comptime tag: @EnumLiteral()) zml.Tensor {
    return ???;
}

// Part 2 goes here: `analyst`, `recommendation`, `confidence`.

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Press releases, foosball tables.
    const x_data = [3][2]f32{
        .{ 1, 0 }, // PetRocks.com
        .{ 6, 0 }, // Gravel.net
        .{ 0, 3 }, // Foosball.org
    };
    const w1_data = [3][2]f32{
        .{ 1, 1 },
        .{ 1, -1 },
        .{ -1, 2 },
    };
    const b1_data = [_]f32{ 0, 0, -1 };
    const w2_data = [2][3]f32{
        .{ 2, 0, 1 },
        .{ -1, 5, 0 },
    };
    // The bank underwrites the IPO. BUY starts 10 points ahead.
    const b2_data = [_]f32{ 10, 0 };

    // Part 3 goes here. It ends with three Buffers: `scores`,
    // `picks` and `sure`.
    ???;

    const names = [_][]const u8{ "PetRocks.com", "Gravel.net", "Foosball.org" };
    const verdicts = [_][]const u8{ "BUY", "SELL" };

    try stdout.print("scores {f}, picks {f}, sure {f}\n", .{ scores.shape(), picks.shape(), sure.shape() });
    for (try scores.getValue([3][2]f32, io), try picks.getValue([3]i32, io), try sure.getValue([3]f32, io), names) |score, pick, p, name| {
        try stdout.print("{s}: buy {d}, sell {d}. {s}! ({d:.2}% sure)\n", .{ name, score[0], score[1], verdicts[@intCast(pick)], p * 100 });
    }
    try stdout.flush();
}
