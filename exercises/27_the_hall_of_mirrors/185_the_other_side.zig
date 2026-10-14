//
// Behind every mirror of the hall there is a second corridor,
// the one the public never sees: same turns, same length, and
// crates of real glass where the front has painted panels.
//
// You have used `zml.Bufferized(Model)` since exercise 084, and
// exercise 085 let slip how it is made:
//
//     pub fn Bufferized(comptime T: type) type {
//         return meta.MapRestrict(Tensor, Buffer).map(T);
//     }
//
// `MapRestrict(From, To).map(T)` is a function from a type to a
// type. It runs in the Zig compiler, it reads `T` with
// `@typeInfo`, and it BUILDS the answer field by field:
//
//   - `Tensor` becomes `Buffer`, `?Tensor` becomes `?Buffer`
//     (with `null` as default value), `[]Tensor` `[]Buffer`;
//   - a struct becomes a NEW struct holding the fields that
//     contain a Tensor somewhere, mapped the same way, in the
//     same order. The other fields are not there at all;
//   - a type with no Tensor in it becomes `void`;
//   - slices, arrays and optionals are mapped through.
//
// The new struct has the old field names and nothing else. In
// particular it has NO METHODS: declarations cannot be copied
// into a type built at compile time. `Bufferized(Mirror)` has a
// `glass`. It has no `forward`.
//
// That gives you the TYPE of the back corridor. The value is
// another matter, and since chapter 17 you have had it made:
//
//     var weights = try zml.mem.bufferize(gpa, Hall, &hall);
//
// Open `zml/zml/mem.zig` and read `bufferizeInner`: it is forty
// lines. It walks the model and its reflection side by side,
// and
//
//   - for each Tensor, writes a Buffer that knows its SHAPE
//     and nothing else: no memory on any device, no numbers.
//     An empty crate with a label on it;
//   - for each slice, ALLOCATES a slice of the same length.
//     That one is yours to free;
//   - skips what the reflection does not have.
//
// Then somebody fills the crates: `Loader.load` in real life.
// Today, you.
//
// `bufferizeInner` is also short because it does not handle
// everything `visit` walks through. Three shapes of model that
// compile to an Exe without a murmur and cannot be bufferized:
//
//   - `[]const Layer`. It wants to write into the slice it
//     just allocated, and the reflection of a const slice is a
//     const slice. Zig refuses, somewhere in `mem.zig`:
//
//         expected type '*T', found '*const T'
//
//     Declare the field `[]Layer`.
//
//   - `[N]Layer`, or `[N]Tensor`. There is no case for arrays.
//     It compiles, and at run time you get the least helpful
//     panic of the whole library:
//
//         reached unreachable code
//
//     with `bufferizeInner` in the stack trace. Use a slice,
//     or a struct with N named fields.
//
//   - `*const Layer`. Same panic. Hold the layer by value.
//
// (Exercise 086 had a `[]const Layer`, and got away with it:
// it built its Bufferized by hand, on the stack.)
//
// The hall this morning: two doors that scale the visitor, one
// number per measurement, and a corridor of two mirrors in
// between. Its glass comes on ONE strip of sixteen numbers,
// already cut in the order of the walk.
//
//  1. The hall cannot be bufferized, twice over. Fix the
//     struct, then `forward` and `main`, which use it. Keep
//     ONE field `doors`, before the corridor (the strip is
//     cut that way): a struct of its own, holding an `entry`
//     and an `exit`.
//
//  2. `Strip.pour` fills ONE crate: it makes a real Buffer of
//     the shape written on the label (`pane.shape()`), out of
//     the next numbers of the strip, and moves on.
//     `Buffer.fromBytes` is from chapter 02; here it may fail,
//     so `pour` may too.
//
//  3. Then pour the strip over the whole back corridor.
//
//  4. The visitor: `eval`, exercise 084.
//
//  5. Whoever wrote the cleanup had not read exercise 086.
//     The right call was in chapter 21, and frees the device
//     memory of every Buffer, slices included. The slice
//     itself is already taken care of.
//
const std = @import("std");
const zml = @import("zml");

const Mirror = struct {
    glass: zml.Tensor,
    frame: zml.Tensor,
    curved: bool,

    pub fn forward(self: Mirror, x: zml.Tensor) zml.Tensor {
        const y = self.glass.dot(x, .d).add(self.frame).rename(.{ .out = .d });
        return if (self.curved) y.relu() else y;
    }
};

const Sign = struct {
    text: []const u8,
    bulbs: u8,
};

const Hall = struct {
    sign: Sign,
    doors: [2]zml.Tensor,
    corridor: []const Mirror,

    pub fn forward(self: Hall, visitor: zml.Tensor) zml.Tensor {
        var v = visitor.mul(self.doors[0]);
        for (self.corridor) |mirror| v = mirror.forward(v);
        return v.mul(self.doors[1]);
    }
};

const Strip = struct {
    io: std.Io,
    platform: *const zml.Platform,
    numbers: []const f32,

    fn pour(self: *Strip, pane: *zml.Buffer) !void {
        return ???;
    }
};

/// The glass, in the order of the walk.
const glass = [16]f32{
    2, 1, // the entry door
    1, 10, // the exit door
    1, 1,
    1, -1, // first mirror
    0, 0, //  and its frame
    0, 1,
    1, 0, // second mirror
    -1, 0, //  and its frame
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    var corridor: [2]Mirror = undefined;
    for (&corridor, 0..) |*mirror, i| mirror.* = .{
        .glass = .init(.{ .out = 2, .d = 2 }, .f32),
        .frame = .init(.{ .out = 2 }, .f32),
        .curved = i == 1,
    };
    const hall: Hall = .{
        .sign = .{ .text = "MIRRORS", .bulbs = 40 },
        .doors = .{ .init(.{ .d = 2 }, .f32), .init(.{ .d = 2 }, .f32) },
        .corridor = &corridor,
    };
    const visitor: zml.Tensor = .init(.{ .d = 2 }, .f32);

    var exe = try platform.compile(gpa, io, hall, .forward, .{visitor}, .{});
    defer exe.deinit();

    try stdout.print("A Hall has {d} fields, its reflection has {d}.\n", .{
        @typeInfo(Hall).@"struct".field_names.len,
        @typeInfo(zml.Bufferized(Hall)).@"struct".field_names.len,
    });
    try stdout.print("The reflection of the sign is void: {}\n", .{zml.Bufferized(Sign) == void});
    try stdout.print("A Mirror can `forward`: {}. Its reflection: {}.\n", .{
        @hasDecl(Mirror, "forward"),
        @hasDecl(zml.Bufferized(Mirror), "forward"),
    });

    var weights = try zml.mem.bufferize(gpa, Hall, &hall);
    defer gpa.free(weights.corridor);

    // `count(T, &v)` is a visit that counts, and `first(T, v)`
    // returns the first T of the walk.
    try stdout.print("Fresh from `bufferize`: {d} crates, the first one says {f}, full: {}\n", .{
        zml.meta.count(zml.Buffer, &weights),
        zml.meta.first(zml.Buffer, weights).shape(),
        zml.meta.first(zml.Buffer, weights).numShards() != 0,
    });

    var strip: Strip = .{ .io = io, .platform = platform, .numbers = &glass };
    defer zml.Buffer.deinitAll(Hall, &weights);
    // Every crate of the back corridor, in order.
    ???;

    try stdout.print("After the pour: full: {}, {d} numbers left on the strip.\n", .{
        zml.meta.first(zml.Buffer, weights).numShards() != 0,
        strip.numbers.len,
    });

    var guest: zml.Buffer = try .fromBytes(io, platform, visitor.shape(), std.mem.sliceAsBytes(&[_]f32{ 3, 1 }));
    defer guest.deinit();

    // `out`: the guest goes through.
    ???;

    try stdout.print("A visitor {{ 3, 1 }} comes out as {any}\n", .{try out.getValue([2]f32, io)});
    try stdout.flush();
}
