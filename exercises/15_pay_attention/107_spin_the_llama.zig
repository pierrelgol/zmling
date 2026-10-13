//
// Here is something embarrassing about attention as we've built
// it: it has no idea who stands where.
//
// Look at the recipe again. A score is the dot product of one
// query and one key. Nothing in it says "the llama right in
// front of me" or "the llama five places away". Shuffle the
// herd and every llama hears exactly the same blend as before
// (the causal mask aside). For gossip, maybe. For language, a
// disaster: "llama bites dog" and "dog bites llama" are made
// of the same tokens.
//
// Position has to be injected somewhere. The trick modern LLMs
// use is called RoPE, for ROtary Position Embedding, and it
// is exactly as silly as it sounds: before the scores are
// computed, every llama SPINS. The llama at position p turns
// her query and her key by an angle of p × (some step).
//
// Why would that help? Two facts about rotations:
//
//  1. Rotating a vector does not change its length. Nothing
//     about "what the llama wants" is lost or amplified.
//
//  2. The dot product of two vectors depends on the angle
//     BETWEEN them. If the listener at position m has turned by
//     m steps, and the speaker at position n by n steps, the
//     angle between them has changed by (m - n) steps.
//
// So after the spin, a score depends on what the two llamas
// want and know, as before, AND on their distance m - n. Not
// on m, not on n: only on how far apart they stand. Position
// 3 listening to position 2 looks exactly like position 103
// listening to position 102. "Relative position", for free.
//
// In ZML:
//
//     pub fn rope(x: Tensor, pos_idx: ?Tensor, opts: RopeOpts) Tensor
//
// From its doc comment and source:
//
//  - `x` must have a `.hd` axis of EVEN size (a rotation needs
//    pairs of numbers), and that's the axis which gets rotated.
//    The result has the shape of x.
//
//  - with `pos_idx = null`, x also needs a `.s` axis, and the
//    positions are 0, 1, 2... along it.
//
//  - or you pass the positions yourself, as a tensor. That is
//    for when the sequence doesn't start at 0: a model that
//    writes one token at a time spins the newcomer alone, at
//    her own position.
//
//  - `opts` has defaults: `.{}` is fine. For the curious: the
//    numbers of a vector are rotated in pairs, each pair at
//    its own speed. `opts.layout` says how they pair up (by
//    default the first half of `.hd` holds the x coordinates
//    and the second half the y coordinates), `opts.scaling`
//    sets the speeds. With the defaults, the first pair turns
//    by 1 radian per position, and the following ones turn
//    slower and slower.
//
// It is applied to the queries and to the keys. Never to the
// values: they are what gets said, not who is talking to whom.
//
// With `.hd = 2` there is a single pair, turning by 1 radian
// per position. So if every llama starts out facing east,
// (1, 0), the llama at position p ends up at (cos p, sin p).
// The program shows three things:
//
//  - the herd after the spin, and the length of each vector;
//  - the scores between the llamas at positions 0, 1, 2, 3;
//  - the same scores with the herd at 100, 101, 102, 103.
//
// The two tables must be identical, and constant along each
// diagonal: same distance, same score. Right now, nothing
// compiles, and then the tables won't match.
//
const std = @import("std");
const zml = @import("zml");

/// The herd, each llama turned according to her place in the line.
fn spin(x: zml.Tensor) zml.Tensor {
    return zml.nn.rope(x, ???, .{});
}

/// The score of each listener for each speaker, {.q, .k}, when the
/// llamas stand at the given positions. (No scaling, no softmax.)
fn scoresAt(x: zml.Tensor, positions: zml.Tensor) zml.Tensor {
    const q = zml.nn.rope(x, positions, .{}).rename(.{ .s = .q });
    const k = x.rename(.{ .s = .k });
    return q.dot(k, .hd);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Four llamas in a line, two numbers each.
    const x: zml.Tensor = .init(.{ .s = 4, .hd = 2 }, .f32);
    const positions: zml.Tensor = .init(.{ .s = 4 }, .i32);

    // Everybody faces east.
    const east: [4][2]f32 = @splat(.{ 1, 0 });
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&east));
    defer x_buffer.deinit();

    {
        var exe = try platform.compileFn(gpa, io, spin, .{x}, .{});
        defer exe.deinit();

        var spun: zml.Buffer = try exe.eval(gpa, io, .{x_buffer});
        defer spun.deinit();

        try stdout.print("after the spin {f}:\n", .{spun.shape()});
        for (try spun.getValue([4][2]f32, io), 0..) |llama, p| {
            const length = @sqrt(llama[0] * llama[0] + llama[1] * llama[1]);
            try stdout.print("  position {d}: ({d:6.3}, {d:6.3})  length {d:.3}\n", .{ p, llama[0], llama[1], length });
        }
    }

    var exe = try platform.compileFn(gpa, io, scoresAt, .{ x, positions }, .{});
    defer exe.deinit();

    for ([_][4]i32{ .{ 0, 1, 2, 3 }, .{ 100, 101, 102, 103 } }) |line| {
        var positions_buffer: zml.Buffer = try .fromBytes(io, platform, positions.shape(), std.mem.sliceAsBytes(&line));
        defer positions_buffer.deinit();

        var scores: zml.Buffer = try exe.eval(gpa, io, .{ x_buffer, positions_buffer });
        defer scores.deinit();

        try stdout.print("scores at positions {any}:\n", .{line});
        for (try scores.getValue([4][4]f32, io)) |row| {
            try stdout.print(" ", .{});
            for (row) |s| try stdout.print(" {d:6.3}", .{s});
            try stdout.print("\n", .{});
        }
    }
    try stdout.flush();
}
