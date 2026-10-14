//
// Friday is inventory day. Odile counts everything in the office
// that belongs to somebody, and at closing time the count must be
// zero: every object went back to its owner, or to the auction.
//
// You have freed things since chapter 02 because the examples did.
// Here is the whole table, once, with the reason for each line.
//
//     WHAT                WHO MADE IT           WHO FREES IT
//
//     an input Buffer     fromBytes, scalar,    you: `b.deinit()`. A
//                         fromSlice             call never frees an
//                                               input. (A DONATED
//                                               one is the exception:
//                                               exercise 156.)
//
//     an output Buffer    results.get / fill,   you. EVERY call makes
//                         runner.run, eval      new ones.
//
//     Arguments, Results  exe.args(a),          you: `x.deinit(a)`.
//                         exe.results(a)        Three host
//                                               allocations each.
//                                               They hold handles,
//                                               they own no Buffer.
//
//     a Runner            exe.runner(a)         `runner.deinit(a)`:
//                                               it is one of each.
//
//     a Slice             Slice.alloc(a, ...),  `slice.free(a)`
//                         toSliceAlloc(a, io)
//                         Slice.init, subSlice  nobody: borrowed
//
//     a Bufferized(T)     results.get,          `deinitAll`
//                         bufferize + a loader  (exercise 085), or
//                                               field by field
//
//     an Exe              compile, compileFn    `exe.deinit()`,
//                                               AFTER its runners
//
//     the Platform        Platform.init         last of all
//
// Two rules hide in that table.
//
// THE ALLOCATOR THAT GAVE IS THE ALLOCATOR THAT TAKES BACK. Every
// `deinit(a)` and `free(a)` wants the allocator of the matching
// `init`. ZML keeps no pointer to it: you pass it both times.
//
// A BUFFER IS A VALUE, AND IT IS A TICKET. Its three fields are a
// platform, a Shape and a few handles. `var spare = umbrella;`
// copies the handles: two tickets, one umbrella. `deinit` destroys
// the handles and then EMPTIES THE VARIABLE IT WAS CALLED ON:
//
//     pub fn deinit(self: *Buffer) void {
//         for (self._shards.constSlice()) |buffer| {
//             buffer.deinit(self._platform.pjrt_api);
//         }
//         self._shards = .empty;
//     }
//
// So a second `deinit` on the SAME variable does nothing, and
// `numShards()` on it says 0. But the copy was not emptied. It
// still says 4, all four dangling, and `spare.deinit()` destroys
// them again: the segmentation fault of exercise 156.
//
// Now, who checks all this?
//
// THE HOST SIDE is checked by `init.gpa`, which reports leaks when
// the program ends. On stderr, where zmling does not look. So
// today the office has a turnstile at the door: `Turnstile`, below,
// is an allocator that passes every request on to `gpa` and counts
// who went in and who came out. An `std.mem.Allocator` is a pointer
// and a table of four functions, and anybody can write one.
//
// THE DEVICE SIDE is checked by nobody. PJRT has a question for it,
// `PJRT_Device_MemoryStats`, and the CPU plugin answers
// `Unimplemented` (you have seen that word on stderr, in a line of
// warning, in every exercise since chapter 02: the compiler asks
// the same question). A GPU answers, but with a number of bytes,
// not with the name of your variable. The only ledger of device
// handles is the one you keep. `Ledger`, below, is one: `issue`
// writes a line when a Buffer is born, `claim` strikes it out and
// frees the Buffer. A ticket whose line is already struck out is
// refused with a panic of our own, before it reaches PJRT. It is
// thirty lines of Zig, and it is the only thing in this file that
// touches `_shards`, a field ZML would rather you left alone.
//
// The day's work: three objects are examined (the raw way of
// exercise 082), and a note is written on the host for each. At
// closing time both counts must be zero.
//
// To do:
//
//  1. `examine`: one raw call that waits, and the output goes
//     through `ledger.issue` on its way out.
//
//  2. Run it. The ledger stops Basile in the loop: read what he
//     does with his spare ticket. One umbrella, one claim.
//
//  3. Now it runs, and closing time shows what is still out. The
//     numbers printed on the way tell you where to look: when all
//     is well, an object in the loop is 8 handles and 13
//     allocations. Three things are never given back, and one of
//     them is lost three times.
//
const std = @import("std");
const zml = @import("zml");

/// An allocator that counts. Everything is done by `child`.
const Turnstile = struct {
    child: std.mem.Allocator,
    /// Allocations made and not yet freed.
    out: usize = 0,

    fn allocator(self: *Turnstile) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Turnstile = @ptrCast(@alignCast(ctx));
        const memory = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.out += 1;
        return memory;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Turnstile = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Turnstile = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *Turnstile = @ptrCast(@alignCast(ctx));
        self.out -= 1;
        self.child.rawFree(memory, alignment, ret_addr);
    }
};

/// The device handles this program holds. Nobody else counts them.
const Ledger = struct {
    /// One line per live Buffer: the address of its first handle.
    // shortcut: a dead ticket whose address PJRT has since given to a
    // new Buffer is not caught; compare all handles and a serial number
    // if you ever keep copies around that long.
    lines: [32]?*const anyopaque = @splat(null),
    /// Handles issued and not yet claimed.
    out: usize = 0,

    fn issue(self: *Ledger, b: zml.Buffer) zml.Buffer {
        for (&self.lines) |*line| {
            if (line.* != null) continue;
            line.* = b._shards.get(0);
            self.out += b.numShards();
            return b;
        }
        @panic("the ledger is full");
    }

    /// Frees `b`. A variable that was already emptied is left alone.
    fn claim(self: *Ledger, b: *zml.Buffer) void {
        if (b.numShards() == 0) return;
        const handle: *const anyopaque = b._shards.get(0);
        for (&self.lines) |*line| {
            if (line.* != handle) continue;
            line.* = null;
            self.out -= b.numShards();
            b.deinit();
            return;
        }
        @panic("this ticket was claimed already: two tickets, one umbrella");
    }
};

const objects = [3][3]i32{
    .{ 90, 4, 1 },
    .{ 25, 20, 2 },
    .{ 130, 45, 9 },
};
const names = [3][]const u8{ "an umbrella", "a hat box", "a cello" };

/// object: {side}, in cm -> the length of string to tie it up.
fn stringFor(object: zml.Tensor) zml.Tensor {
    return object.sum(.side).squeeze(.side).scale(2);
}

/// One raw call, waited for. The caller owns what comes back.
fn examine(io: std.Io, ledger: *Ledger, exe: *const zml.Exe, args: *zml.Exe.Arguments, results: *zml.Exe.Results, object: zml.Buffer) zml.Buffer {
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

    var turnstile: Turnstile = .{ .child = gpa };
    const counted = turnstile.allocator();
    var ledger: Ledger = .{};

    const object: zml.Tensor = .init(.{ .side = 3 }, .i32);
    var exe = try platform.compileFn(gpa, io, stringFor, .{object}, .{});
    defer exe.deinit();

    var args = try exe.args(counted);
    try stdout.print("an arguments table: {d} host allocations\n", .{turnstile.out});
    var results = try exe.results(counted);
    try stdout.print("and a results table: {d}\n", .{turnstile.out});
    var runner = try exe.runner(counted);
    try stdout.print("and a runner: {d}\n", .{turnstile.out});

    for (objects, names) |sides, name| {
        var object_buffer = ledger.issue(try .fromBytes(io, platform, object.shape(), std.mem.sliceAsBytes(&sides)));
        defer ledger.claim(&object_buffer);

        var length = examine(io, &ledger, &exe, &args, &results, object_buffer);
        var spare = length; // Basile: "in case I lose the first one"
        defer ledger.claim(&spare);
        defer ledger.claim(&length);

        // The note, on the host.
        const note = try length.toSliceAlloc(counted, io);
        try stdout.print("{s}: {d} cm of string ({d} handles out, {d} allocations)\n", .{
            name, note.constItems(i32)[0], ledger.out, turnstile.out,
        });
    }

    // A runner call, and `eval`, which builds a runner of its own and
    // throws it away.
    var again: zml.Buffer = try .fromBytes(io, platform, object.shape(), std.mem.sliceAsBytes(&objects[0]));
    again = ledger.issue(again);
    var twice: zml.Buffer = undefined;
    runner.run(io, .{again}, .{&twice}, .{ .wait = true });
    twice = ledger.issue(twice);
    var thrice = ledger.issue(try exe.eval(counted, io, .{again}));
    try stdout.print("after a run and an eval: {d} handles out, {d} allocations\n", .{ ledger.out, turnstile.out });

    // Two tickets, one umbrella.
    const spare = twice;
    ledger.claim(&twice);
    ledger.claim(&twice);
    try stdout.print("claimed twice through the same variable: it holds {d} handles, the copy still says {d}\n", .{ twice.numShards(), spare.numShards() });

    ledger.claim(&thrice);
    runner.deinit(counted);
    args.deinit(counted);

    try stdout.print("closing time. host allocations still out: {d}\n", .{turnstile.out});
    try stdout.print("closing time. device handles still out: {d}\n", .{ledger.out});

    try stdout.flush();
}
