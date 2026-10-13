//
// Every treasure map says the same thing: X marks the spot.
// On a tensor, X is an index, and "go to the X and bring back
// what's there" is called GATHER.
//
// A `slice` cuts out ONE contiguous stretch: from here to
// there, in order. Even when its start is decided at run time,
// it is still a single stretch. The islands the Captain wants
// to visit are scattered all over the map, in no particular
// order, some of them twice; there may be a thousand of them,
// and they change on every run. Positions like that have to be
// DATA: a tensor of indices.
//
//     values.gather(.{ .a = idx }, .{})
//
// Read it as: "walk along axis .a of `values`, and pick the
// elements at the positions listed in `idx`". With a vector,
//
//     values  {a=6}:   { 30, 5, 80, 12, 80, 1 }
//     idx     {n=3}:   { 2, 0, 2 }
//
//     result  {n=3}:   { 80, 30, 80 }
//
//     result[n] = values[idx[n]]
//
// The shape rule: the gathered axis is REPLACED by the axes of
// the index tensor. All other axes of `values` stay where they
// are. So with a table,
//
//     table {word=50000,d=768}  gathered along .word
//     ids   {seq=12}
//                           ->  {seq=12,d=768}
//
// you get one whole ROW per index. Does that look familiar? It
// is how every language model starts: token ids go in, one
// vector per token comes out. An "embedding layer" is a `gather`
// and nothing else.
//
// The fine print:
//
//   - `idx` must be a tensor of integers. .i32 and .i64 are the
//     usual choices and any integer dtype works. Floats are
//     turned down by the compiler.
//   - Indices are allowed to repeat, in any order.
//   - The second argument holds options; `.{}` is fine.
//   - And the one that sinks ships: an index that is OUT OF
//     RANGE is not an error. No panic, no warning. On this
//     platform it is quietly pulled back to the nearest valid
//     position, and you receive a perfectly well formed tensor
//     with somebody else's data in it. (Other platforms may do
//     something else. Nobody promises anything.)
//
// The map of the archipelago has 5 islands and 3 clues for each
// (paces north, paces east, depth of the hole). The Captain
// dictates today's route; the cabin boy writes it down, number
// for number. Pirates count from one. Tensors don't, and you
// have read the fine print: nothing is going to complain. The
// Captain's words are in a comment in `main` and they stay as
// they are (nobody corrects the Captain). The numbers right
// under them are the cabin boy's, and those are yours to put
// right.
//
// The bosun has a question of his own: how much work is the
// whole trip? Paces north, paces east and feet of digging, each
// added up over the stops of the route: three numbers, {clue=3}.
// That is `readClues` and then a reduction over the axis that
// `gather` just created. Both functions are empty.
//
// `main` has the map and its Buffer, and the route's Buffer,
// which is asking for the shape of a placeholder that isn't
// there: `route`, one position per stop along an axis tagged
// .stop, four stops, and a dtype that `gather` accepts (look at
// the numbers its Buffer is made of). Then compile both
// functions and run them, into Buffers called `result` and
// `effort`.
//
const std = @import("std");
const zml = @import("zml");

/// The clues of each island on the route, in the order of the route.
fn readClues(map: zml.Tensor, route: zml.Tensor) zml.Tensor {
    return ???;
}

/// Each clue added up over the whole route: {clue}.
fn totalEffort(map: zml.Tensor, route: zml.Tensor) zml.Tensor {
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

    const map: zml.Tensor = .init(.{ .island = 5, .clue = 3 }, .i32);

    // The placeholder `route` goes here, and then an Exe for each
    // of the two functions.
    ???;

    var map_buffer: zml.Buffer = try .fromBytes(io, platform, map.shape(), std.mem.sliceAsBytes(&[5][3]i32{
        .{ 1, 2, 3 }, // Skull Island
        .{ 11, 12, 13 }, // Isla Segfault
        .{ 21, 22, 23 }, // Dead Man's Cay
        .{ 31, 32, 33 }, // Tortuga
        .{ 41, 42, 43 }, // Null Island
    }));
    defer map_buffer.deinit();

    // The Captain said: "The fifth island, then the first, the
    // third, and the second."
    var route_buffer: zml.Buffer = try .fromBytes(io, platform, route.shape(), std.mem.sliceAsBytes(&[_]i32{ 5, 1, 3, 2 }));
    defer route_buffer.deinit();

    // Run both: `result` for the clues, `effort` for the bosun.
    ???;

    try stdout.print("clues {f}:\n", .{result.shape()});
    for (try result.getValue([4][3]i32, io), 1..) |clues, stop| {
        try stdout.print("  stop {d}: {d} paces north, {d} paces east, dig {d} feet\n", .{ stop, clues[0], clues[1], clues[2] });
    }
    const all = try effort.getValue([3]i32, io);
    try stdout.print("The whole trip: {d} paces north, {d} paces east, {d} feet of digging.\n", .{ all[0], all[1], all[2] });
    try stdout.flush();
}
