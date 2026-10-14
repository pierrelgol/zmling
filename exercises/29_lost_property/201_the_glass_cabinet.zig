//
// Behind the counter there is a glass cabinet with the board in
// it: four windows, and over each one the number of the ticket
// being served. Odile changes the numbers by hand, all day. She
// does not take the board down, copy it, and hang a new one.
//
// You do. Every `Buffer.fromBytes` since chapter 02 asks PJRT for
// new device memory, copies the host bytes over, and leaves you a
// handle to free. For weights, once, that is right. For four
// numbers that change before every call (a position, a token, a
// handful of sampling settings: think of the decode loop of
// chapter 23), it is a new shelf per umbrella.
//
// To do better you need to know that "device memory" is not one
// thing. `zml/zml/platform.zig`:
//
//     pub const Memory = struct {
//         pub const Kind = enum {
//             default,
//             host_unpinned,
//             host_pinned,
//             device,
//         };
//
//     device          the accelerator's own memory (VRAM)
//     host_unpinned   ordinary RAM
//     host_pinned     RAM that the OS promises never to move or
//                     swap, so the accelerator can read and write
//                     it directly. It is on the host, AND an Exe
//                     can take it as an input.
//     default         whatever the device prefers: `device`
//
// Every device lists its memories: `device.addressable_memories`,
// with `memory.kind()` for PJRT's name of each, and
// `device.memory(kind)`, a `?*const Memory`, to ask for one. The
// places that create Buffers take a kind:
//
//     Buffer.fromBytesOpts(io, platform, shape, bytes,
//         .{ .memory = .host_pinned })
//     Buffer.uninitialized(io, platform, shape,
//         .{ .memory = .host_pinned })
//
// A compiled function may then need to be told where its inputs
// live and to move them: `x.onMemory(.host_pinned)` says "this
// input is in pinned memory", `x.toMemory(.device)` copies it to
// the accelerator inside the graph, and
// `view.createHostPinnedTensor(...)` makes a placeholder whose
// weights the loader will put there. Open `zml/zml/tensor.zig` and
// read the first lines of `onMemory` and `toMemory`:
//
//     switch (ctx.platform.target) {
//         .cpu, .neuron, .metal => return self,
//         .cuda, .rocm, .tpu, .oneapi, .furiosa => {},
//     }
//
// On a CPU the host IS the device. Its three memories are the same
// RAM under three names (`Memory.isOfKind` answers `true` to
// everything there, so whatever kind you ask for, you are handed
// the first), and both functions do nothing. So code written for
// a GPU runs unchanged on your laptop, and nothing in this
// exercise can show you a copy that was saved. What CAN be shown
// is the tool built on pinned memory, which works everywhere:
//
//     zml.Buffer.HostAccessible
//     (zml/zml/buffer/HostAccessible.zig)
//
// A Buffer that the host may read and write IN PLACE.
//
//     var board: zml.Buffer.HostAccessible =
//         try .init(allocator, io, platform, shape, .undonatable);
//     defer board.deinit(allocator);
//
//     board.buffer           the `zml.Buffer` to pass to an Exe.
//                            `deinit` above frees it: never call
//                            `board.buffer.deinit()` yourself.
//     board.view(T)          a `View(T)` over its elements, in
//                            row-major order. T must have the size
//                            of the dtype.
//
//     view.len
//     view.get(i)  view.set(i, value)
//     view.fill(value)
//     view.fillIota(first, step)      first + i * step
//     view.copyFrom(values)           from a `[]const T`, same len
//     view.copyTo(values)             into a `[]T`, same len
//     view.slice(start, len)          a narrower view; `null` for
//                                     "to the end"
//
// A write through a view lands in every copy the devices hold.
// `{f}` prints a HostAccessible with its contents.
//
// The last argument of `init` is a `zml.Buffer.Donation`:
//
//     .undonatable   Exes only READ this Buffer. The host rewrites
//                    it between calls. The devices may share one
//                    allocation.
//     .donatable     an Exe may write its output INTO it
//                    (`reuseBuffer`, exercise 156). Then the call
//                    replaces the handles, and the host must catch
//                    up before it looks again:
//
//                        try board.await(io);
//
//                    It waits for the execution and re-reads where
//                    the memory is. Take a view before that and ZML
//                    stops you, with a message that says what to do.
//
// One rule for both: no view while an execution is using the
// Buffer. Write, call, wait, read.
//
// To do:
//
//  1. `nextPlease`: every window moves on to its next ticket, and
//     the result is cast in the memory of the input. Exercise 156.
//
//  2. Round 1: the windows open at tickets 10, 15, 20 and 25. One
//     call on the view.
//
//  3. Round 3: windows 0 and 1 close for lunch and show 0. Basile
//     closed the wrong two.
//
//  4. The second board is donated to `nextPlease`, three times.
//     Basile looks at it too early.
//
const std = @import("std");
const zml = @import("zml");

/// board: {window}, the ticket served at each window.
/// last: a scalar, the last ticket handed out.
/// -> {window}: how many tickets are still to come at each window.
fn waiting(board: zml.Tensor, last: zml.Tensor) zml.Tensor {
    return last.broad(board.shape()).sub(board);
}

/// Every window calls its next ticket, in place.
fn nextPlease(board: zml.Tensor) zml.Tensor {
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

    const device = platform.devices[0];
    try stdout.print("the memories of one CPU device:", .{});
    for (device.addressable_memories) |memory| try stdout.print(" {s}", .{memory.kind()});
    try stdout.print("\nask it for .host_pinned and you get: {s}\n", .{device.memory(.host_pinned).?.kind()});

    const shape: zml.Shape = .init(.{ .window = 4 }, .i32);

    var waiting_exe = try platform.compileFn(gpa, io, waiting, .{ zml.Tensor.fromShape(shape), zml.Tensor.init(.{}, .i32) }, .{});
    defer waiting_exe.deinit();

    var last: zml.Buffer = try .scalar(io, platform, @as(i32, 50), .i32);
    defer last.deinit();

    // The board the Exe only reads.
    var board: zml.Buffer.HostAccessible = try .init(gpa, io, platform, shape, .undonatable);
    defer board.deinit(gpa);
    const numbers = board.view(i32);

    // Round 1.
    ???;
    try printRound(gpa, io, stdout, 1, &waiting_exe, board, last);

    // Round 2: window 2 jumps to ticket 41.
    numbers.set(2, 41);
    try printRound(gpa, io, stdout, 2, &waiting_exe, board, last);

    // Round 3.
    numbers.slice(2, null).fill(0);
    try printRound(gpa, io, stdout, 3, &waiting_exe, board, last);

    try stdout.print("uploads since the board was hung: 0\n", .{});

    // The board the Exe writes on.
    var next_exe = try platform.compileFn(gpa, io, nextPlease, .{zml.Tensor.fromShape(shape)}, .{});
    defer next_exe.deinit();
    var caller = try next_exe.runner(gpa);
    defer caller.deinit(gpa);

    var counter: zml.Buffer.HostAccessible = try .init(gpa, io, platform, shape, .donatable);
    defer counter.deinit(gpa);
    counter.view(i32).copyFrom(&.{ 7, 30, 12, 99 });

    for (0..3) |_| {
        caller.run(io, .{counter.buffer}, .{&counter.buffer}, .{});
        try stdout.print("next, please: {f}\n", .{&counter});
    }

    var served: [4]i32 = undefined;
    counter.view(i32).copyTo(&served);
    try stdout.print("window 3 is now at ticket {d}\n", .{served[3]});

    try stdout.flush();
}

fn printRound(gpa: std.mem.Allocator, io: std.Io, stdout: *std.Io.Writer, round: usize, exe: *const zml.Exe, board: zml.Buffer.HostAccessible, last: zml.Buffer) !void {
    var result = try exe.eval(gpa, io, .{ board.buffer, last });
    defer result.deinit();
    try stdout.print("round {d}: {f} still to come: {any}\n", .{ round, &board, try result.getValue([4]i32, io) });
}
