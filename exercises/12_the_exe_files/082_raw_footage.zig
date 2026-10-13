//
// "Scally, I ran the photo analysis. Twice. It crashed. Twice.
// Somebody doesn't want us to see this."
//
// "Molder. Read the message."
//
//     eval expects one output, got 2; use a runner for
//     multiple outputs
//
// `exe.eval` is a convenience for the commonest case: Buffers
// in, ONE Buffer out. This Exe has two outputs, and `eval`
// says so and leaves. (It checks before running anything, so
// you can't even get the first output out of it.)
//
// The message recommends a runner. We'll get to it in the next
// exercise, and it will make more sense once you have seen the
// three raw pieces it is made of. `eval` is made of the same
// three. This is the real interface of an Exe:
//
// 1. THE ARGUMENTS: a table with one slot per input.
//
//        var args = try exe.args(gpa);
//        defer args.deinit(gpa);
//
//        args.set(.{ photo_buffer, other_buffer });
//
//    `set` walks through the tuple you give it, exactly like the
//    compiler walked through your function's arguments, and puts
//    each Buffer it finds into the next slot. It checks every
//    shape against `exe.input_shapes` on the way. It does NOT
//    copy any data and does not take ownership: the Buffers
//    stay yours.
//
// 2. THE RESULTS: a table with one slot per output.
//
//        var results = try exe.results(gpa);
//        defer results.deinit(gpa);
//
// 3. THE CALL:
//
//        exe.call(io, args, &results, .{ .wait = true });
//
//    This executes the compiled code. `call` returns nothing and
//    cannot fail politely: if the execution goes wrong, it
//    panics.
//
//    About `.wait`. It is false by default! Without it, `call`
//    may return as soon as the work has been handed over to the
//    device, while the numbers are still being computed. That's
//    wonderful for a GPU: you can queue the next call right
//    away. With `.wait = true`, `call` only returns when the
//    outputs are ready. Reading a Buffer (`getValue`,
//    `toSliceAlloc`) waits for its contents anyway, so the
//    numbers you read are right in both cases; `.wait` is about
//    WHEN you pay. If you time your model, or want problems to
//    show up at the call that caused them, you wait. `eval`
//    always waits.
//
// Then you collect what's in the results table:
//
//        var out = results.get(T);
//
// `T` describes the outputs you expect, and `get` fills a T
// with Buffers, in output order. Useful values of T:
//
//     zml.Buffer                           one output
//     struct { zml.Buffer, zml.Buffer }    two, as a tuple
//     [2]zml.Buffer                        two, as an array
//     zml.Bufferized(Report)               "Report, with every
//                                          Tensor turned into a
//                                          Buffer"
//
// The last one is the nice one: your function returned a
// `Report` of Tensors, you get a `Bufferized(Report)` with the
// same field names. Every Buffer that comes out of `get` is
// yours to `deinit`.
//
// One warning from Scally: `get` hands out the Buffers of the
// LAST call. Call it once per call. Twice, and you hold two
// handles to the same memory, and freeing both is the kind of
// thing that gets a basement office closed.
//
// The task.
//
// 1. `analyze` has lost its body: for each frame, the brightest
//    pixel, and the sum of all its pixels. Chapter 06, and no
//    size-1 axes left behind.
//
// 2. Molder tried to fill the report with one `eval` per field.
//    Replace his attempt with the real thing: the arguments, the
//    results, a call that waits, and one `get`.
//
// 3. The next night there is a second photo, and Molder did it
//    again. The two tables are not used up by a call: `set` the
//    new Buffer into the SAME `args`, call, and `get` from the
//    SAME `results`. No second pair of tables.
//
// The first report is printed after the second call, on purpose.
// What `get` handed out is yours and stays valid: the second
// call makes new Buffers, it doesn't write into the old ones.
//
const std = @import("std");
const zml = @import("zml");

const Report = struct {
    /// The brightest pixel of each frame.
    brightest: zml.Tensor,
    /// The sum of all pixels of each frame.
    glow: zml.Tensor,
};

fn analyze(photo: zml.Tensor) Report {
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

    const photo: zml.Tensor = .init(.{ .frames = 3, .pixels = 4 }, .f32);

    var exe = try platform.compileFn(gpa, io, analyze, .{photo}, .{});
    defer exe.deinit();

    // Three frames of the night sky over Bellefleur, Oregon.
    var photo_buffer: zml.Buffer = try .fromBytes(io, platform, photo.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 0, 1, 0, 1 },
        .{ 2, 9, 2, 1 },
        .{ 0, 0, 0.5, 0 },
    }));
    defer photo_buffer.deinit();

    // Molder's attempt: one eval per field.
    var report: zml.Bufferized(Report) = .{
        .brightest = try exe.eval(gpa, io, .{photo_buffer}),
        .glow = try exe.eval(gpa, io, .{photo_buffer}),
    };
    defer report.brightest.deinit();
    defer report.glow.deinit();

    // The next night. Same sky, same Exe, same two tables.
    var second_buffer: zml.Buffer = try .fromBytes(io, platform, photo.shape(), std.mem.sliceAsBytes(&[3][4]f32{
        .{ 3, 3, 3, 3 },
        .{ 0, 0, 0, 0 },
        .{ 1, 2, 4, 8 },
    }));
    defer second_buffer.deinit();

    // Molder again. This time: no new tables, the ones from above.
    var second: zml.Bufferized(Report) = .{
        .brightest = try exe.eval(gpa, io, .{second_buffer}),
        .glow = try exe.eval(gpa, io, .{second_buffer}),
    };
    defer second.brightest.deinit();
    defer second.glow.deinit();

    try stdout.print("brightest pixel per frame: {any}\n", .{try report.brightest.getValue([3]f32, io)});
    try stdout.print("total glow per frame:      {any}\n", .{try report.glow.getValue([3]f32, io)});
    try stdout.print("the next night, brightest: {any}\n", .{try second.brightest.getValue([3]f32, io)});
    try stdout.print("the next night, glow:      {any}\n", .{try second.glow.getValue([3]f32, io)});
    try stdout.flush();
}
