//
// The corridors of the hall are one visitor wide. Whoever goes
// in first comes out first, and nobody overtakes.
//
// Yesterday's walk has one property that everything else in
// ZML leans on: it is the SAME walk every time. Three places
// use it, and they never talk to each other.
//
//   - At compile time, `createBlockArguments` visits the tuple
//     of arguments, `.{ model, x }` (the model FIRST: `self` is
//     the first parameter of `forward`). Each `Tensor` it
//     meets becomes the next input of the program: %arg0,
//     %arg1, ... That is where `exe.input_shapes` comes from.
//
//   - At run time, `exe.eval(gpa, io, .{ weights, x_buffer })`
//     hands the tuple to `Exe.Arguments.set`, in
//     `zml/zml/exe.zig`: a visit again, this time with a
//     `*const zml.Buffer` callback. The first Buffer it meets
//     goes into slot 0, the next into slot 1, ...
//
//   - `zml.Bufferized(Model)` is built from `Model` with its
//     fields in the same order, so that the second walk meets
//     its Buffers in the order in which the first met their
//     Tensors.
//
// Nothing else connects a Buffer to "its" Tensor. Not the name
// of the field: names do not exist in an Exe. Only the place
// in the queue.
//
//     FIELD ORDER IS ARGUMENT ORDER.
//
// Three things follow.
//
// One. Swap two fields of a model struct and you have changed
// the signature of its Exe. Nothing breaks, as long as the
// Bufferized twin is derived from the same struct: it swaps
// too.
//
// Two. `set` does not care what the Buffers are wrapped in. A
// `zml.Bufferized(Model)`, a struct you typed yourself, a bare
// tuple of Buffers, an array: it digs the Buffers out and
// counts. What you pass is a queue, whatever it looks like.
//
// Three. There is exactly one check, per slot:
//
//     Expected argument 2 to have shape {f32}, got {d=3,f32}
//
// The shape. A Buffer of the wrong shape in a slot is a panic,
// and the number in the message is a position in the walk.
// Two Buffers of the SAME shape in each other's slot is no
// panic at all. It is a model that computes something else.
// (Think of the two weights of a real attention block, both
// {.d, .d}.)
//
// This morning: a one-mirror hall. A visitor is three
// measurements (height, width, hat). The mirror stretches them,
// shifts them, and the tinted glass at the exit multiplies
// everything by ten:
//
//     visitor * stretch + shift, then * tint
//
// A visitor { 1, 2, 3 } must come out { 30, 40, 50 }.
//
//  1. Mr. Pym wants the queue on paper BEFORE anything is
//     compiled: `Queue.join` notes the shape of who arrives,
//     and somebody must walk him past the model and the
//     visitor, in that order, in ONE visit. (A pointer to a
//     tuple is a fine thing to visit: `&.{ a, b }`.)
//
//  2. Then the Exe, to check his list against. Exercise 084.
//
//  3. Gus has packed the three Buffers in a crate of his own
//     design. He is a tidy boy: alphabetical order.
//
//  4. And Ottoline does not see why she needs a crate at all.
//     She is right. She gets a panic all the same: read the
//     slot number, then read Mr. Pym's list.
//
const std = @import("std");
const zml = @import("zml");

const Hall = struct {
    stretch: zml.Tensor,
    shift: zml.Tensor,
    tint: zml.Tensor,
    rounds: u8,

    pub fn forward(self: Hall, visitor: zml.Tensor) zml.Tensor {
        var v = visitor;
        for (0..self.rounds) |_| v = v.mul(self.stretch).add(self.shift);
        return v.mul(self.tint);
    }
};

/// Gus's crate.
const Crate = struct {
    shift: zml.Buffer,
    stretch: zml.Buffer,
    tint: zml.Buffer,
};

const Queue = struct {
    shapes: [8]zml.Shape = undefined,
    len: usize = 0,

    fn join(self: *Queue, t: *const zml.Tensor) void {
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

    const hall: Hall = .{
        .stretch = .init(.{ .d = 3 }, .f32),
        .shift = .init(.{ .d = 3 }, .f32),
        .tint = .init(.{}, .f32),
        .rounds = 1,
    };
    const visitor: zml.Tensor = .init(.{ .d = 3 }, .f32);

    var queue: Queue = .{};
    // The queue, as the Compiler will see it.
    ???;

    try stdout.print("Mr. Pym's list:\n", .{});
    for (queue.shapes[0..queue.len], 0..) |shape, i| {
        try stdout.print("  slot {d}: {f}\n", .{ i, shape });
    }

    // `exe`.
    ???;

    // `a.eql(b)`: same dims, same tags, same dtype.
    var same = queue.len == exe.input_shapes.len;
    for (queue.shapes[0..queue.len], exe.input_shapes) |mine, theirs| {
        if (!mine.eql(theirs)) same = false;
    }
    try stdout.print("The Exe has {d} slots. Same list: {}\n", .{ exe.input_shapes.len, same });

    var stretch: zml.Buffer = try .fromBytes(io, platform, hall.stretch.shape(), std.mem.sliceAsBytes(&[_]f32{ 2, 2, 2 }));
    defer stretch.deinit();
    var shift: zml.Buffer = try .fromBytes(io, platform, hall.shift.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 0, -1 }));
    defer shift.deinit();
    var tint: zml.Buffer = try .scalar(io, platform, 10, .f32);
    defer tint.deinit();
    var guest: zml.Buffer = try .fromBytes(io, platform, visitor.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3 }));
    defer guest.deinit();

    // Field NAMES in an initializer may come in any order. It is
    // the declaration of `Crate` that decides who walks first.
    const crate: Crate = .{ .shift = shift, .stretch = stretch, .tint = tint };

    var out: zml.Buffer = try exe.eval(gpa, io, .{ crate, guest });
    defer out.deinit();
    try stdout.print("Gus's crate: {any}\n", .{try out.getValue([3]f32, io)});

    var again: zml.Buffer = try exe.eval(gpa, io, .{ guest, stretch, shift, tint });
    defer again.deinit();
    try stdout.print("No crate at all: {any}\n", .{try again.getValue([3]f32, io)});
    try stdout.flush();
}
