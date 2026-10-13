//
// Three nights in a parked car outside a grain silo, with a
// microphone, a thermos, and Molder's sunflower seeds.
//
// Each night produces a tape, each tape goes through the same
// Exe. That is the normal life of an Exe: compiled once, called
// again and again with fresh Buffers. A chatbot calls its Exe
// once per word.
//
// With the raw interface of the last exercise, every call is
// three steps: `args.set`, `exe.call`, `results.get`. Somebody
// at the Bureau got tired of typing that and bundled it up:
//
//     var runner = try exe.runner(gpa);
//     defer runner.deinit(gpa);
//
// A `Runner` owns an arguments table and a results table for
// one Exe. (It only BORROWS the Exe: the Exe must outlive it.)
// Then one line does a whole call:
//
//     runner.run(io, inputs, outputs, opts);
//
//     inputs    a tuple of Buffers, as for `eval` or `args.set`
//     outputs   a tuple of POINTERS: where to put the results
//     opts      the same options as `exe.call`: `.{ .wait = true }`
//
// The outputs are the new part. Instead of asking for a T and
// getting it returned, you say where each result should land:
//
//     var peak: zml.Buffer = undefined;
//     var alarms: zml.Buffer = undefined;
//     runner.run(io, .{tape}, .{ &peak, &alarms }, .{ .wait = true });
//
// `run` walks through the destinations and overwrites each
// Buffer it finds with the next output. A destination can also
// be a whole struct of Buffers:
//
//     var report: zml.Bufferized(Report) = undefined;
//     runner.run(io, .{tape}, .{&report}, .{ .wait = true });
//
// (The raw interface has this too: `results.fill(.{ &a, &b })`
// is `results.get` with destinations instead of a return value,
// and it is what `run` calls once the Exe has finished.)
//
// Starting from `undefined` is fine: `run` only writes there.
// After the call, the Buffers are yours, and yours to deinit.
// EVERY call makes new output Buffers, so in a loop, every
// iteration has to free its own. A `defer` inside the loop body
// does exactly that.
//
// And `exe.eval`? Here is its entire body, more or less:
//
//     var result: Buffer = undefined;
//     var r = try self.runner(allocator);
//     defer r.deinit(allocator);
//     r.run(io, input_values, .{&result}, .{ .wait = true });
//     return result;
//
// It builds a runner for a single call and throws it away. For
// one output and a handful of calls, that's fine. In a loop that
// runs thousands of times, keep ONE runner outside the loop.
//
// The task. `listen` reports on a tape: the loudest second, and
// how many seconds were louder than a threshold. The threshold
// is an input, but the same one for all three nights: its Buffer
// is made once, outside the loop. Make the runner, and make the
// call.
//
const std = @import("std");
const zml = @import("zml");

const Report = struct {
    /// The loudest second of the night.
    peak: zml.Tensor,
    /// How many seconds were louder than the threshold.
    alarms: zml.Tensor,
};

fn listen(tape: zml.Tensor, threshold: zml.Tensor) Report {
    return .{
        .peak = tape.max(.seconds).squeeze(.seconds),
        .alarms = tape.cmp(.GT, threshold).convert(.i32).sum(.seconds).squeeze(.seconds),
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const tape: zml.Tensor = .init(.{ .seconds = 6 }, .f32);
    const threshold: zml.Tensor = .init(.{}, .f32);

    var exe = try platform.compileFn(gpa, io, listen, .{ tape, threshold }, .{});
    defer exe.deinit();

    var threshold_buffer: zml.Buffer = try .scalar(io, platform, 5, .f32);
    defer threshold_buffer.deinit();

    // One runner for the whole stakeout.
    var runner = ???;
    defer runner.deinit(gpa);

    const nights = [3][6]f32{
        .{ 1, 2, 1, 1, 2, 1 }, // crickets
        .{ 1, 7, 9, 8, 1, 1 }, // a low hum, lights over the silo
        .{ 1, 1, 6, 6, 1, 1 }, // Molder dropped the thermos. Twice.
    };

    for (nights, 1..) |night, n| {
        var tape_buffer: zml.Buffer = try .fromBytes(io, platform, tape.shape(), std.mem.sliceAsBytes(&night));
        defer tape_buffer.deinit();

        var report: zml.Bufferized(Report) = undefined;
        runner.run(io, ???, ???, ???);
        defer report.peak.deinit();
        defer report.alarms.deinit();

        try stdout.print("Night {d}: peak {d}, {d} seconds over the threshold.\n", .{
            n,
            try report.peak.getValue(f32, io),
            try report.alarms.getValue(i32, io),
        });
    }
    try stdout.flush();
}
