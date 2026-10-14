//
// Tensor Cross station, the lost property office, a door between
// platform 4 and the buffet. Odile has run it for thirty years.
// Basile, the new porter, started on Monday.
//
// Until now, when a compiled Exe handed you a Buffer, you assumed
// the numbers were in it. This chapter is about what really
// happens AFTER the compile, between ZML and the runtime below it
// (XLA, spoken to through an interface called PJRT), and the first
// thing to know is that the numbers may not be there yet.
//
// Exercise 082 said it in one paragraph. Here is the option again,
// from `zml/zml/exe.zig`:
//
//     pub const CallOpts = struct {
//         /// Waits for execution to complete before returning.
//         wait: bool = false,
//     };
//
// False by default. With `.{}`, `exe.call` hands the work to the
// device and comes back. And yet, right after it:
//
//     var ticket = results.get(zml.Buffer);
//
// works. What you hold is not the umbrella, it is a CLAIM TICKET:
// a handle that PJRT made at once, for a result that will exist.
// Look at what a `zml.Buffer` is made of, in `zml/zml/buffer.zig`:
// a platform, a Shape, and `_shards`, a small array of handles.
// No numbers. The numbers stay on the device.
//
// What can you do with a ticket whose umbrella is still on the
// trolley?
//
//  - WAIT AT THE COUNTER:
//
//        pub fn await(self: Buffer, io: std.Io) !void
//
//    It asks each handle for its "ready event" and sleeps until
//    all of them have fired. It reads nothing.
//
//  - READ IT. `getValue`, `toSlice` and `toSliceAlloc` copy from
//    the device to the host, and that copy waits for the contents
//    by itself. A read is always right. `.wait` never changes the
//    numbers you get, it changes WHEN you pay for them.
//
//  - HAND IT TO ANOTHER EXE. This is the one that matters:
//
//        args_b.set(.{ticket});
//        exe_b.call(io, args_b, &results_b, .{});
//
//    The runtime knows that this input is the output of a job
//    still running, and it puts the second job behind the first.
//    You can queue a whole evening of work this way, each job
//    fed with the tickets of the one before, and never stop.
//    The host only waits once, at the end, when it reads. On a
//    GPU this is the difference between a device that works all
//    the time and one that idles while the host shuffles paper.
//
// Who waits, by default?
//
//     exe.call(io, args, &results, .{})         does not
//     runner.run(io, inputs, outputs, .{})      does not
//     exe.eval(gpa, io, inputs)                 ALWAYS waits
//     Buffer.fromBytes / fromSlice / scalar     wait (the upload)
//     Buffer.fromBytesOpts(io, platform, shape, bytes,
//         .{ .wait = false })                   does not: and then
//                                               the host bytes
//                                               must not change
//                                               until it is done
//
// The price of not waiting is paid in error reports. With
// `.wait = true`, a job that fails stops the program at the call
// that started it (`PJRT execution failed with: ...`). Without, the
// call has long returned: the failure comes back later, as a Zig
// error, from whichever `await` or read meets it first. When you
// debug, wait. When you measure time, wait, or you are timing the
// queue and not the work.
//
// And one trap, which exercise 082 called "get once per call". The
// results table holds the tickets of the LAST call only. If you
// call three times and then `get` three times, you hold three
// tickets for the same umbrella and none for the other two. A
// runner cannot get this wrong: `run` fills your destinations
// before it returns. That is what the three runners below are for.
//
// The evening's work. Three bags came in on the 18:42. For each:
//
//     weigh      the bag {thing=4} -> its weight, a scalar
//     price      the weight -> the storage fee, 3 pence a pound
//     enter      the ledger so far, the fee -> the new ledger
//
// Three Exes, three runners, nine jobs, and `enter` for bag 2
// needs the ledger of bag 1: one chain through the whole evening.
// Nobody waits and nobody reads until the trolley is empty.
//
// To do:
//
//  1. `weigh` has lost its body. Chapter 06.
//
//  2. In the loop, `price` and `enter` are queued. The line that
//     queues `weigh` is yours. Do not wait.
//
//  3. Basile's ledger does not add up: every evening it shows the
//     fee of the last bag and nothing else. One ticket in his
//     chain is not the one that should be passed on.
//
//  4. Before the reading starts, Odile waits at the counter for
//     the LAST ticket of the chain, once. The call is missing.
//
const std = @import("std");
const zml = @import("zml");

const bags = [3][4]i32{
    .{ 3, 1, 4, 1 },
    .{ 5, 9, 2, 6 },
    .{ 5, 3, 5, 8 },
};

/// bag: {thing} -> its weight, a scalar.
fn weigh(bag: zml.Tensor) zml.Tensor {
    return ???;
}

/// Three pence a pound.
fn price(weight: zml.Tensor) zml.Tensor {
    return weight.scale(3);
}

fn enter(ledger: zml.Tensor, fee: zml.Tensor) zml.Tensor {
    return ledger.add(fee);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const bag: zml.Tensor = .init(.{ .thing = 4 }, .i32);
    const amount: zml.Tensor = .init(.{}, .i32);
    const other_amount: zml.Tensor = .init(.{}, .i32);

    var weigh_exe = try platform.compileFn(gpa, io, weigh, .{bag}, .{});
    defer weigh_exe.deinit();
    var price_exe = try platform.compileFn(gpa, io, price, .{amount}, .{});
    defer price_exe.deinit();
    var enter_exe = try platform.compileFn(gpa, io, enter, .{ amount, other_amount }, .{});
    defer enter_exe.deinit();

    var weigher = try weigh_exe.runner(gpa);
    defer weigher.deinit(gpa);
    var pricer = try price_exe.runner(gpa);
    defer pricer.deinit(gpa);
    var bookkeeper = try enter_exe.runner(gpa);
    defer bookkeeper.deinit(gpa);

    // The ledger opens at zero.
    var opening: zml.Buffer = try .scalar(io, platform, @as(i32, 0), .i32);
    defer opening.deinit();

    var bag_buffers: [3]zml.Buffer = undefined;
    var weights: [3]zml.Buffer = undefined;
    var fees: [3]zml.Buffer = undefined;
    var ledgers: [3]zml.Buffer = undefined;

    // The trolley: nine jobs queued, nothing read.
    for (bags, 0..) |things, i| {
        bag_buffers[i] = try .fromBytes(io, platform, bag.shape(), std.mem.sliceAsBytes(&things));

        // Queue the weighing of this bag: its weight goes in `weights[i]`.
        ???;
        pricer.run(io, .{weights[i]}, .{&fees[i]}, .{});

        bookkeeper.run(io, .{ opening, fees[i] }, .{&ledgers[i]}, .{});
    }
    // Every ticket is freed at closing time, after the last read.
    defer for (&bag_buffers, &weights, &fees, &ledgers) |*b, *w, *f, *l| {
        b.deinit();
        w.deinit();
        f.deinit();
        l.deinit();
    };

    try stdout.print("tickets in hand, umbrellas seen: {d}, 0\n", .{weights.len + fees.len + ledgers.len});

    // One wait, for the end of the chain. Whatever it depends on is
    // done by then.
    try ledgers[2].???(io);

    for (0..3) |i| {
        try stdout.print("bag {d}: {d} lb, fee {d}p, ledger {d}p\n", .{
            i + 1,
            try weights[i].getValue(i32, io),
            try fees[i].getValue(i32, io),
            try ledgers[i].getValue(i32, io),
        });
    }

    // `eval` waits, always: it has to, it returns as soon as you read.
    var check = try price_exe.eval(gpa, io, .{ledgers[2]});
    defer check.deinit();
    try stdout.print("and if the fee were charged on the ledger itself: {d}p\n", .{try check.getValue(i32, io)});

    try stdout.flush();
}
