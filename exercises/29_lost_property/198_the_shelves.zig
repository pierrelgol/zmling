//
// "Where IS the umbrella?" is a fair question in a lost property
// office, and it has two answers.
//
// ON THE HOST: in the memory of your Zig program, where a `for`
// loop can touch it. ON THE DEVICE: in memory that belongs to the
// runtime, where only a compiled Exe can touch it. Every number of
// your program is on one side or the other, and crossing costs a
// copy. You have crossed many times since chapter 02 without
// looking at the bridge. Today you look.
//
// THE HOST SIDE is `zml.Slice` (`zml/zml/slice.zig`): bytes, plus
// a Shape that says how to read them. It is to the host what a
// Buffer is to the device.
//
//     Slice.init(shape, bytes)       BORROWS bytes you already have
//                                    (an array, a file). Nothing to
//                                    free. Panics when the length is
//                                    not `shape.byteSize()`.
//     Slice.alloc(allocator, shape)  allocates them. Not zeroed.
//     slice.free(allocator)          for `alloc`ed ones only
//     slice.items(T) / constItems(T) the bytes as a `[]T`. Nobody
//                                    checks T against the dtype.
//     slice.subSlice(axis, start, size)
//                                    a window on the SAME bytes: no
//                                    copy, and nothing to free. The
//                                    axis is a NUMBER: a tag does
//                                    not compile here.
//
// A window of whole rows is one block of memory. A window on a
// column is not: its numbers are four apart. `isContiguous()` tells
// you, and `items`, `constItems` and `Buffer.fromSlice` refuse a
// window that is not (a bare `reached unreachable code`, from
// `slice.zig`).
//
// THE BRIDGE, in both directions:
//
//     Buffer.fromSlice(io, platform, slice)      host -> device
//     buffer.toSliceAlloc(allocator, io)         device -> host, in
//                                                a NEW Slice that you
//                                                free
//     buffer.toSlice(io, slice)                  device -> host, in a
//                                                Slice that is yours
//                                                already. No
//                                                allocation. Panics
//                                                unless the shapes
//                                                match.
//
// (`getValue(T, io)` is `toSlice` into a T on the stack.)
//
// Every one of them COPIES. After `fromSlice`, the Buffer and the
// Slice are two umbrellas of the same colour: write in the Slice
// and the Buffer knows nothing about it. That is the most common
// way to compute yesterday's answer with today's code.
//
// There is a third constructor, for a shelf that is ordered before
// anybody knows what goes on it:
//
//     Buffer.uninitialized(io, platform, shape, .{})
//
// Device memory, no contents: reading it gives whatever was there.
// ZML's weight loader uses it, and so do the Buffers of exercise
// 201.
//
// THE DEVICE SIDE has a surprise. Print `buffer.numShards()` on
// any Buffer you have made since chapter 02: 4. And
// `buffer.byteSize()`: four times what the shape says. A Buffer
// holds one handle PER DEVICE of the platform, and your "CPU" has
// four, because `zml/zml/platform.zig` asks for them:
//
//     pub const CreateOptions = struct {
//         ...
//         cpu: Cpu = .{ .device_count = 4 },
//
// That is the `.{}` you have passed to `Platform.init` since
// exercise 010.
// Four pretend devices let ZML's multi-GPU code (meshes, sharding)
// run on a laptop. A tensor that nobody asked to cut up is
// REPLICATED: every device gets a full copy, and every upload is
// done four times. For a laptop program that never shards, one
// device is plenty:
//
//     try .init(gpa, io, .cpu, .{ .cpu = .{ .device_count = 1 } })
//
// The count must be a power of two (3 is `error.InvalidDeviceCount`,
// with a line of log to say why) and at most
// `zml.Platform.MAX_NUM_DEVICES`, 32. Buffers and Exes belong to
// the platform that made them: an Exe compiled for 4 devices wants
// 4 handles per argument.
//
// The office keeps its inventory in a book: 3 shelves of 4 slots,
// the number of items in each slot.
//
// To do:
//
//  1. `emptySlots`: for each shelf, how many slots hold nothing.
//     You counted this way in Loopsville, and again in exercise
//     151: compare, convert, sum.
//
//  2. `office` is the platform of a small office: ONE device.
//
//  3. A cello arrives and is written in the book. Then Basile
//     counts the empty slots again and finds what he found before.
//     Read his three lines.
//
//  4. The 19:05 brings a delivery for the MIDDLE shelf. It is
//     added on the device, and the result is read straight back
//     into the book, in place, with `toSlice`. Basile's window is
//     the wrong size, and ZML says so.
//
const std = @import("std");
const zml = @import("zml");

const inventory = [3][4]i32{
    .{ 2, 0, 1, 0 },
    .{ 0, 0, 3, 0 },
    .{ 1, 1, 0, 4 },
};

const delivery = [1][4]i32{.{ 1, 1, 0, 2 }};

/// book: {shelf, slot} -> {shelf}, the number of empty slots.
fn emptySlots(book: zml.Tensor) zml.Tensor {
    return ???;
}

fn restock(shelf: zml.Tensor, new: zml.Tensor) zml.Tensor {
    return shelf.add(new);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const book_shape: zml.Shape = .init(.{ .shelf = 3, .slot = 4 }, .i32);
    const shelf_shape: zml.Shape = .init(.{ .shelf = 1, .slot = 4 }, .i32);

    // What you have been using all along.
    {
        const usual: *zml.Platform = try .init(gpa, io, .cpu, .{});
        defer usual.deinit(gpa, io);
        var copy: zml.Buffer = try .fromBytes(io, usual, book_shape, std.mem.sliceAsBytes(&inventory));
        defer copy.deinit();
        try stdout.print("the usual platform: {d} devices, {d} handles, {d} bytes for a book of {d}\n", .{
            usual.devices.len, copy.numShards(), copy.byteSize(), book_shape.byteSize(),
        });
    }

    const office: *zml.Platform = try .init(gpa, io, .cpu, ???);
    defer office.deinit(gpa, io);

    // The book, on the host. `alloc` gives garbage: fill it.
    const book: zml.Slice = try .alloc(gpa, book_shape);
    defer book.free(gpa);
    for (inventory, 0..) |row, shelf| @memcpy(book.items(i32)[shelf * 4 ..][0..4], &row);

    var book_buffer: zml.Buffer = try .fromSlice(io, office, book);
    defer book_buffer.deinit();
    try stdout.print("the office: {d} device, {d} handle, {d} bytes\n", .{
        office.devices.len, book_buffer.numShards(), book_buffer.byteSize(),
    });

    var count_exe = try office.compileFn(gpa, io, emptySlots, .{zml.Tensor.fromShape(book_shape)}, .{});
    defer count_exe.deinit();

    var before = try count_exe.eval(gpa, io, .{book_buffer});
    defer before.deinit();
    try stdout.print("empty slots per shelf: {any}\n", .{try before.getValue([3]i32, io)});

    // A cello, shelf 1, slot 1. Written on the host.
    book.items(i32)[1 * 4 + 1] = 1;
    try stdout.print("the cello: in the book {d}, on the device {d}\n", .{
        book.constItems(i32)[5],
        (try book_buffer.getValue([3][4]i32, io))[1][1],
    });

    var after = try count_exe.eval(gpa, io, .{book_buffer});
    defer after.deinit();
    try stdout.print("empty slots, cello shelved: {any}\n", .{try after.getValue([3]i32, io)});

    // Windows on the book. No copy, nothing to free.
    const middle = book.subSlice(0, 1, 1);
    const first_slots = book.subSlice(1, 0, 1);
    try stdout.print("the middle shelf {f}: one block of memory: {}\n", .{ middle.shape, middle.isContiguous() });
    try stdout.print("the first slot of each shelf {f}: one block of memory: {}\n", .{ first_slots.shape, first_slots.isContiguous() });

    // The delivery for the middle shelf: up, added, and back down
    // into the book itself.
    var restock_exe = try office.compileFn(gpa, io, restock, .{ zml.Tensor.fromShape(shelf_shape), zml.Tensor.fromShape(shelf_shape) }, .{});
    defer restock_exe.deinit();

    var shelf_buffer: zml.Buffer = try .fromSlice(io, office, middle);
    defer shelf_buffer.deinit();
    var delivery_buffer: zml.Buffer = try .fromSlice(io, office, .init(shelf_shape, std.mem.sliceAsBytes(&delivery)));
    defer delivery_buffer.deinit();

    var restocked = try restock_exe.eval(gpa, io, .{ shelf_buffer, delivery_buffer });
    defer restocked.deinit();
    try restocked.toSlice(io, book.subSlice(0, 1, 2));

    try stdout.print("the book, after the 19:05:\n", .{});
    for (0..3) |shelf| {
        try stdout.print("  shelf {d}: {any}\n", .{ shelf, book.subSlice(0, @intCast(shelf), 1).constItems(i32) });
    }

    // A shelf ordered in advance: memory, and nothing on it yet.
    var spare: zml.Buffer = try .uninitialized(io, office, shelf_shape, .{});
    defer spare.deinit();
    try stdout.print("a spare shelf {f}, contents: nobody knows\n", .{spare.shape()});

    try stdout.flush();
}
