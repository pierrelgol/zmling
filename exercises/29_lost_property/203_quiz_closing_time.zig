//
// Saturday, 17:30. Half an hour to closing, a queue to the door of
// the buffet, and Odile has gone to see the stationmaster about
// the kettle. The office is yours.
//
// What you are about to write is, in small, what every inference
// server is: ONE compiled model, a queue of requests that do not
// have the size the model was compiled for, results that are
// started long before anybody reads them, and a rule that at
// closing time nothing is left on the shelves.
//
// THE MODEL. `Appraiser` looks at a batch of 4 objects, each
// described by 3 numbers {b=4,d=3} (size, days in the office,
// condition), and gives 2 numbers for each {b=4,out=2}: its value
// in pence, and the days left before it goes to auction. One
// `dot` over `.d` with `weight` {out=2,d=3}, plus `bias` {out=2}.
// You wrote it as `forward` in exercise 087. Everything is i32.
//
// THE QUEUE. Five people, with 3, 1, 4, 2 and 1 objects. The model
// takes 4. Exercise 087 again: pad with zeros, ignore the rows you
// made up.
//
// THE SERVICE. For each person: pad, upload, FIRE, next person.
// Nobody waits (exercise 197): the runner is called with `.{}` and
// the ticket goes in `tickets`. Only when the queue is empty do
// you wait, once, for the last ticket, and read them all in order.
// (Why not ONE board in the glass cabinet of exercise 201,
// rewritten for each person? Because of its rule: write, call,
// WAIT, read. A board cannot be rewritten while a call is using
// it. Five calls in flight need five inputs.)
//
// THE BOOKS. Every device handle goes through `ledger.issue` when
// it is born, and that includes the weights and the outputs
// (exercise 199). Host memory that outlives a line goes through
// `counted`: here, that is the runner. `main` prints the books
// twice, and the middle print tells you what a correct service
// holds at 17:59:
//
//     2 weights + 5 inputs + 5 tickets = 12 Buffers, 48 handles
//     one runner = 6 host allocations
//
// What is given: the data, the turnstile's frame, the ledger, the
// lines that print, and Basile's closing round, which he wrote
// before leaving for the 17:42.
//
// To do:
//
//  1. `Turnstile.alloc` and `Turnstile.free`: the two that count.
//     Exercise 199 has them.
//
//  2. `Appraiser.forward`.
//
//  3. `main`, down to the middle print. In this order: the
//     platform; the model (two placeholders) and its Exe
//     (`platform.compile`, exercise 087); `weights`, a
//     `zml.Bufferized(Appraiser)`, each Buffer issued by the
//     ledger; `runner`, made with `counted`; then the service,
//     which fills `inputs` and `tickets`.
//
//  4. The collection: one wait, then for each request its rows as
//     a `[4][2]i32`, handed to `printRequest`.
//
//  5. Basile's closing round. Run it and read the last two lines.
//     He gave some things back at the wrong desk, and some not at
//     all. Nothing crashes: that is the problem with leaks.
//
const std = @import("std");
const zml = @import("zml");

const batch = 4;

const Appraiser = struct {
    /// {out=2,d=3}
    weight: zml.Tensor,
    /// {out=2}
    bias: zml.Tensor,

    /// things: {b, d} -> {b, out}
    pub fn forward(self: Appraiser, things: zml.Tensor) zml.Tensor {
        return ???;
    }
};

// value = 2 * size + 10 * condition; days left = 90 - days here.
const weight_values = [2][3]i32{ .{ 2, 0, 10 }, .{ 0, -1, 0 } };
const bias_values = [2]i32{ 0, 90 };

// size, days in the office, condition (0 to 5)
const objects = [11][3]i32{
    .{ 9, 30, 3 },  .{ 1, 80, 1 }, .{ 4, 12, 4 },
    .{ 130, 2, 5 }, .{ 6, 45, 2 }, .{ 3, 89, 3 },
    .{ 8, 60, 4 },  .{ 20, 7, 2 }, .{ 15, 20, 1 },
    .{ 5, 90, 5 },  .{ 1, 1, 5 },
};
const object_names = [11][]const u8{
    "umbrella", "glove", "scarf",
    "cello",    "hat",   "book",
    "cane",     "bag",   "kite",
    "teapot",   "key",
};

/// How many objects each person in the queue brings.
const requests = [5]usize{ 3, 1, 4, 2, 1 };

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
        return ???;
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
        return ???;
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

/// `first`: the index of the person's first object in `objects`.
/// `rows`: what the model said for the padded batch.
fn printRequest(stdout: *std.Io.Writer, number: usize, first: usize, count: usize, rows: [batch][2]i32) !void {
    try stdout.print("request {d}, {d} of {d} rows are real:\n", .{ number, count, batch });
    for (object_names[first..][0..count], rows[0..count]) |name, row| {
        try stdout.print("  {s}: {d}p, {d} days left\n", .{ name, row[0], row[1] });
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    var turnstile: Turnstile = .{ .child = gpa };
    const counted = turnstile.allocator();
    var ledger: Ledger = .{};

    // The platform, the model, its Exe, `weights` and `runner`.
    ???;

    // What goes in, and the tickets for what comes out.
    var inputs: [requests.len]zml.Buffer = undefined;
    var tickets: [requests.len]zml.Buffer = undefined;

    // The service. Nobody waits.
    ???;

    try stdout.print("17:59. {d} requests fired, none read. handles out: {d}, host allocations out: {d}\n", .{ requests.len, ledger.out, turnstile.out });

    // The collection, in the order of the queue.
    ???;

    // Basile's closing round.
    for (&tickets) |*ticket| ledger.claim(ticket);
    ledger.claim(&weights.weight);
    ledger.claim(&weights.bias);
    runner.deinit(gpa);

    try stdout.print("18:00. device handles still out: {d}\n", .{ledger.out});
    try stdout.print("18:00. host allocations still out: {d}\n", .{turnstile.out});

    try stdout.flush();
}
