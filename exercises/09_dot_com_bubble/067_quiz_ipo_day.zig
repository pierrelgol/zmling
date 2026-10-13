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
// and what the bank expects back is {startup=3,verdict=2}: for
// each startup, a BUY score and a SELL score.
//
// Look at the tags for a moment. The first layer consumes
// .features and produces .hidden; the second consumes .hidden
// and produces .verdict. A model written this way documents
// itself, and wiring the layers in the wrong order is a panic
// instead of a bad quarter.
//
// Everything you need:
//
//     a.dot(b, .tag)        contract over a named axis
//     t.broad(shape)        broadcast by matching tags
//     a.add(b)              same shapes, please
//     t.relu()              negative numbers become 0
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

/// Two layers and a ReLU: {startup,features} -> {startup,verdict}.
fn analyst(x: zml.Tensor, w1: zml.Tensor, b1: zml.Tensor, w2: zml.Tensor, b2: zml.Tensor) zml.Tensor {
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

    const x: zml.Tensor = .init(.{ .startup = 3, .features = 2 }, .f32);
    const w1: zml.Tensor = ???;
    const b1: zml.Tensor = .init(.{ .hidden = 3 }, .f32);
    const w2: zml.Tensor = ???;
    const b2: zml.Tensor = .init(.{ .verdict = 2 }, .f32);

    var exe = try platform.compileFn(gpa, io, analyst, ???, .{});
    defer exe.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[3][2]f32{
        .{ 1, 0 },
        .{ 2, 1 },
        .{ 0, 3 },
    }));
    defer x_buffer.deinit();
    var w1_buffer: zml.Buffer = try .fromBytes(io, platform, w1.shape(), std.mem.sliceAsBytes(&[3][2]f32{
        .{ 1, 1 },
        .{ 1, -1 },
        .{ -1, 2 },
    }));
    defer w1_buffer.deinit();
    var b1_buffer: zml.Buffer = try .fromBytes(io, platform, b1.shape(), std.mem.sliceAsBytes(&[_]f32{ 0, 0, -1 }));
    defer b1_buffer.deinit();
    var w2_buffer: zml.Buffer = try .fromBytes(io, platform, w2.shape(), std.mem.sliceAsBytes(&[2][3]f32{
        .{ 2, 0, 1 },
        .{ -1, 3, 0 },
    }));
    defer w2_buffer.deinit();
    // The bank underwrites the IPO. BUY starts 10 points ahead.
    var b2_buffer: zml.Buffer = try .fromBytes(io, platform, b2.shape(), std.mem.sliceAsBytes(&[_]f32{ 10, 0 }));
    defer b2_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ x_buffer, w1_buffer, b1_buffer, w2_buffer, b2_buffer });
    defer result.deinit();

    try stdout.print("verdicts {f}\n", .{result.shape()});
    for (try result.getValue([3][2]f32, io), [_][]const u8{ "PetRocks.com", "Gravel.net", "Foosball.org" }) |scores, name| {
        try stdout.print("{s}: buy {d}, sell {d}. {s}!\n", .{ name, scores[0], scores[1], if (scores[0] > scores[1]) "BUY" else "SELL" });
    }
    try stdout.flush();
}
