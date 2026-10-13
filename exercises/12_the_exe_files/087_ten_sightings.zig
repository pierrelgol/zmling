//
// A busy night over Nevada. Ten sightings come in by fax, and
// the detector from the autopsy has been upgraded to examine
// FOUR sightings per call: its input is now {b=4,d=3}, a batch.
//
// Batching is how every model is run in practice. One call
// working on four rows is much cheaper than four calls working
// on one row each, and on a GPU it's cheaper by a lot.
//
// Time to say out loud what the Exe has been telling you since
// `input_shapes`:
//
//     An Exe is compiled for exact shapes. Calling it again
//     with new VALUES is free. Calling it with a new SHAPE is
//     impossible.
//
// The 4 in {b=4,d=3} is as frozen as the 3. The compiler used
// it to plan every byte of memory in advance; that's a good
// part of why the compiled code is fast. So what is free to
// change between calls?
//
//     the contents of any input Buffer      yes, that's the point
//     which Buffers you pass                yes
//     the weights                           yes! They are inputs.
//                                           Other Buffers of the
//                                           same shapes will do.
//     any shape, any dtype                  no
//     anything that was a Zig value         no
//     during tracing
//
// The first three cost nothing. The last two cost a new compile.
//
// Ten sightings, batches of four: 4 + 4 + 2. The third batch is
// the classic. Three ways out, all of them used in the wild:
//
//   1. PAD. Fill the batch up with dummy rows (zeros), run it,
//      and ignore the results for the rows you made up. One
//      Exe, a little wasted work. The usual answer.
//   2. Compile a second Exe for the odd shape. Compiling is slow
//      (seconds, or minutes for a big model); do this when there
//      are few distinct shapes and you know them in advance.
//   3. Compile for b=1 and call ten times. Simple and slow.
//
// Padding works here because each row is scored on its own: a
// row of zeros can't disturb its neighbours. (When rows DO
// influence each other, padding takes more care. That is a
// story for the chapters on real models.)
//
// The weights, meanwhile, are uploaded once, before the loop,
// and passed along in every call. Passing a Buffer is passing a
// handle; nothing is copied.
//
// About the new `forward`: `x.dot(weight, .d)` with x
// {b=4,d=3} and weight {d_out=2,d=3} gives {b=4,d_out=2}, one
// row of two scores per sighting. The bias {d_out=2} has to be
// stretched to that shape by hand, with `broad` (chapter 05).
//
// The task. The night shift left you this:
//
// 1. `forward` is missing. It is described just above, and you
//    wrote it once already, as `layer` in exercise 067.
//
// 2. The weights are two arrays and no Buffers. Upload them
//    once, as a `zml.Bufferized(Detector)`, with one line to
//    free them all (exercise 085).
//
// 3. One runner for the whole night (exercise 083).
//
// 4. The agent on night duty made each Buffer exactly as big as
//    its batch. Two calls go well; the third one panics, and by
//    now you can read that message in your sleep. Pad the batch
//    instead: a [batch_size][3]f32 full of zeros (`@splat`),
//    with the real rows copied to the front (`@memcpy`).
//
const std = @import("std");
const zml = @import("zml");

const batch_size = 4;

const Detector = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    /// {b,d} -> {b,d_out}
    pub fn forward(self: Detector, x: zml.Tensor) zml.Tensor {
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

    const model: Detector = .{
        .weight = .init(.{ .d_out = 2, .d = 3 }, .f32),
        .bias = .init(.{ .d_out = 2 }, .f32),
    };
    const x: zml.Tensor = .init(.{ .b = batch_size, .d = 3 }, .f32);

    var exe = try platform.compile(gpa, io, model, .forward, .{x}, .{});
    defer exe.deinit();

    //                            glow hum speed
    const weight_data = [2][3]f32{
        .{ 1, 2, 0 }, // alien
        .{ 0, -1, 3 }, // weather balloon
    };
    const bias_data = [2]f32{ 0.5, -1 };

    // Uploaded once. And freed, all of them, in one line.
    var weights: zml.Bufferized(Detector) = ???;
    defer ???;

    var runner = ???;
    defer runner.deinit(gpa);

    // glow, hum, speed
    const sightings = [10][3]f32{
        .{ 2, 1, 4 }, .{ 5, 3, 0 }, .{ 0, 0, 1 }, .{ 9, 9, 9 },
        .{ 1, 0, 0 }, .{ 0, 4, 2 }, .{ 3, 3, 3 }, .{ 0, 1, 0 },
        .{ 7, 0, 1 }, .{ 0, 0, 6 },
    };

    var first: usize = 0;
    while (first < sightings.len) : (first += batch_size) {
        const batch = sightings[first..@min(first + batch_size, sightings.len)];

        // A Buffer made to measure.
        const batch_shape: zml.Shape = .init(.{ .b = batch.len, .d = 3 }, .f32);
        var x_buffer: zml.Buffer = try .fromBytes(io, platform, batch_shape, std.mem.sliceAsBytes(batch));
        defer x_buffer.deinit();

        var result: zml.Buffer = undefined;
        runner.run(io, .{ weights, x_buffer }, .{&result}, .{ .wait = true });
        defer result.deinit();

        // Only the first `batch.len` rows mean anything.
        const scores = try result.getValue([batch_size][2]f32, io);
        for (scores[0..batch.len], first..) |score, n| {
            const verdict = if (score[0] > score[1]) "ALIEN" else "weather balloon";
            try stdout.print("Sighting {d}: {d} against {d}. {s}.\n", .{ n, score[0], score[1], verdict });
        }
    }
    try stdout.flush();
}
