//
// Ahoy! Welcome aboard the Off-By-One, the least feared ship in
// the Caribbean. Captain Scurvy has a map, a shovel, and a
// parrot called Polly who knows exactly two words: "true" and
// "false".
//
// So far, your tensors could compute but they could not DECIDE.
// Is this chest worth digging up? Is that one cursed? Questions
// like these are asked with one method:
//
//     a.cmp(.GT, b)
//
// It compares `a` and `b` element by element and returns a
// tensor of the same shape with dtype .bool: one Polly answer
// per element. The first argument is the direction, one of
//
//     .EQ   equal                  .NE   not equal
//     .GT   greater than           .GE   greater or equal
//     .LT   less than              .LE   less or equal
//
// (Zig has no operator overloading, so there is no `a > b`.
// The names are the ones of the compiler underneath.)
//
// `cmp` has two rules, and it panics during tracing when they
// are broken:
//
//   - both sides have the same dtype. Comparing an .i32 tensor
//     to an .f32 one isn't "obviously fine", it's
//         cmp expects input tensors to be of the same type
//   - both sides have exactly the same shape, or one of them is
//     a scalar (rank 0), which is compared against everybody.
//     Nothing else is broadcast for you, not even an axis of
//     size 1: reach for `broad` first.
//
// And as with arithmetic, a plain Zig number isn't a tensor. To
// compare against the number 10, make it one:
//
//     doubloons.cmp(.GT, .scalar(10, .i32))
//
// Bool tensors can be combined with each other,
//
//     p.logical(.AND, q)       also .OR and .XOR
//     p.not()
//
// and when a bool tensor is the output of your Exe, you read it
// like any other, with Zig's own `bool`:
//
//     const answers = try result.getValue([6]bool, io);
//
// The Captain will only dig for a chest that holds MORE THAN 10
// doubloons AND carries NO curse at all. The cabin boy wrote the
// two questions for Polly. He's twelve, be nice: one of them
// panics, and he gave up half way through the other.
//
// The Captain can't read a row of trues and falses (he can't
// read), so he also wants ONE number: how many holes to dig.
// Nobody can add up Polly's answers as they are, but once a
// bool is CONVERTED to an .i32 (chapter 3), true is 1 and false
// is 0, and counting is a reduction (chapter 6). `holesToDig`
// is empty, and so is the place in `main` where it should be
// compiled and run: the last print wants a Buffer called
// `holes`.
//
const std = @import("std");
const zml = @import("zml");

/// For each chest: more than 10 doubloons, and a curse level of exactly 0?
fn worthDigging(doubloons: zml.Tensor, curses: zml.Tensor) zml.Tensor {
    const rich = doubloons.cmp(.GT, .scalar(10, .f32));
    const safe = curses.cmp(???, .scalar(0, .i32));
    return rich.logical(.AND, safe);
}

/// How many chests are worth digging up: one .i32, no axis.
fn holesToDig(doubloons: zml.Tensor, curses: zml.Tensor) zml.Tensor {
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

    const doubloons: zml.Tensor = .init(.{ .chest = 6 }, .i32);
    const curses: zml.Tensor = .init(.{ .chest = 6 }, .i32);

    var exe = try platform.compileFn(gpa, io, worthDigging, .{ doubloons, curses }, .{});
    defer exe.deinit();

    // The Captain's number needs an Exe as well.
    ???;

    var doubloons_buffer: zml.Buffer = try .fromBytes(io, platform, doubloons.shape(), std.mem.sliceAsBytes(&[_]i32{ 30, 5, 80, 12, 80, 1 }));
    defer doubloons_buffer.deinit();
    // 0 is no curse, 1 is "mild itching", 2 is "skeleton crew".
    var curses_buffer: zml.Buffer = try .fromBytes(io, platform, curses.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 0, 2, 0, 0, 1 }));
    defer curses_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ doubloons_buffer, curses_buffer });
    defer result.deinit();

    // Run it into a Buffer called `holes`.
    ???;

    try stdout.print("Polly answers with a {f}.\n", .{result.shape()});
    try stdout.print("Worth the shovel: {any}\n", .{try result.getValue([6]bool, io)});
    try stdout.print("Holes to dig: {d}. Polly will supervise.\n", .{try holes.getValue(i32, io)});
    try stdout.flush();
}
