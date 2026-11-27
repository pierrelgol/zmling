//
// Every week somebody leaves the office in a coat that is not
// theirs. Odile's rule: before anything crosses the counter, it is
// checked against the ticket. ZML has a counter too, the line
// between your Zig program and a compiled Exe, and this exercise
// is about what is checked there and, above all, what is NOT.
//
// The border is one function, `Arguments.set`, in
// `zml/zml/core/exe.zig`. It walks through what you give it and, for
// each Buffer it meets, in order:
//
//     stdx.debug.assert(
//         expected_shapes[i].eql(buffer.shape()),
//         "Expected argument {} to have shape {f}, got {f}", ...);
//
//     stdx.debug.assert(
//         shard_count == expected,
//         "Argument {} has {d} shards but executable expects {d}",
//         ...);
//
// and then copies the handles in slot i. That is all of it.
//
// WHAT IS CHECKED, and how it fails (a panic, never a Zig error):
//
//  - the dimensions and the dtype. `Shape.eql` is
//
//        std.mem.eql(i64, self.dims(), other.dims())
//            and self.dtype() == other.dtype()
//
//  - the number of handles: a Buffer made on a platform with one
//    device (exercise 198) does not enter an Exe compiled on a
//    platform with four.
//
// WHAT IS NOT CHECKED:
//
//  - THE TAGS. Look at `eql` again. A Buffer {shelf=3,f32} enters
//    a slot compiled for {measure=3,f32} without a word. Tags are
//    a compile-time help: at run time only sizes are left.
//    (`Shape.eqlWithTags` exists, for those who want it.)
//
//  - THE ORDER. Two Buffers of the same shape, swapped: both pass.
//
//  - THE COUNT, and this one is nasty. Give one Buffer too many
//    and you get a bare `index out of bounds` from `exe.zig`,
//    without a message. Give one TOO FEW and nothing happens: the
//    last slot keeps the handle of the PREVIOUS call (or garbage,
//    if there was none). The Exe runs, and answers a question
//    nobody asked. `set` fills slots from 0; it does not know
//    where you meant to stop.
//    (`args.setPartial(buffers, offset)` fills from slot `offset`
//    on purpose: that is how exercise 157's baked weights stay
//    put while the rest changes.)
//
//  - ON THE WAY OUT, nothing at all: `results.get(T)` and
//    `results.fill` trust your T.
//
// So what is a runner? Exercise 083 gave you one ready-made. Here
// is everything `zml.Exe.Runner` has, once more from the source:
//
//     exe: *const Exe,        borrowed: the Exe must outlive it
//     args: Arguments,        one table of each, made once,
//     results: Results,       reused for every call
//
//     pub fn run(self, io, input_values, output_refs, opts) void {
//         self.args.set(input_values);
//         self.exe.call(io, self.args, &self.results, opts);
//         self.exe.deinitDonatedInputs(self.args);
//         self.results.fill(output_refs);
//     }
//
// Four lines. (The third is exercise 156: it destroys the handles
// of donated inputs. `deinitDonatedInputs` is private; nothing is
// donated today.) There is no magic in it, which means you can
// build your own, and yours can be stricter.
//
// `Clerk` is Odile's runner. It checks what `set` does not: the
// count, and the tags. And because it is a Zig function that can
// fail, it says no with an ERROR, which the caller may catch, and
// not with a panic.
//
// The Exe of the day answers "by how many centimetres is this coat
// too SMALL for this person?": for each of the three measures,
// what the person has more than the coat, never less than zero,
// and the sum of that.
//
// To do:
//
//  1. `tooSmall`. `relu` (exercise 090's oven) is the "never less
//     than zero".
//
//  2. `Clerk.init`: the two tables, for this Exe. If the second
//     one cannot be made, the first must not leak (`errdefer`).
//
//  3. `Clerk.serve`: refuse a wrong count with
//     `error.WrongCount`. Then, for each input whose shape `eql`s
//     the expected one but whose tags do not match, refuse with
//     `error.WrongLabel`. (`inputs` is a tuple: it has a `.len`,
//     and `inline for (inputs, 0..) |b, i|` walks it.) Then the
//     three lines: `set`, a `call` that waits, `fill`.
//
//  4. Basile measured Mr Abbot with the office tape, in whole
//     centimetres. The program stops at the border, and by now you
//     read that message at a glance.
//
//  5. The tweed coat is tried on three people, and the answers are
//     wrong: it "fits" the largest man in the station. Nothing
//     panics. Which check would have caught it? None.
//
const std = @import("std");
const zml = @import("zml");

/// coat, person: {measure}, in cm -> a scalar, in cm.
fn tooSmall(coat: zml.Tensor, person: zml.Tensor) zml.Tensor {
    return ???;
}

const Clerk = struct {
    exe: *const zml.Exe,
    args: zml.Exe.Arguments,
    results: zml.Exe.Results,

    pub fn init(exe: *const zml.Exe, allocator: std.mem.Allocator) !Clerk {
        return ???;
    }

    pub fn deinit(self: *Clerk, allocator: std.mem.Allocator) void {
        self.results.deinit(allocator);
        self.args.deinit(allocator);
    }

    /// inputs: a tuple of Buffers. outputs: a tuple of pointers.
    /// Waits for the result.
    pub fn serve(self: *Clerk, io: std.Io, inputs: anytype, outputs: anytype) !void {
        return ???;
    }
};

// chest, sleeve, length
const grey_coat = [3]f32{ 100, 62, 110 };
const tweed_coat = [3]f32{ 96, 60, 104 };

const people = [3][3]f32{
    .{ 100, 62, 110 },
    .{ 92, 58, 100 },
    .{ 96, 61.5, 104 },
};
const people_names = [3][]const u8{ "Mr Abbot", "Miss Finch", "Dr Quill" };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const measures: zml.Shape = .init(.{ .measure = 3 }, .f32);
    var exe = try platform.compileFn(gpa, io, tooSmall, .{ zml.Tensor.fromShape(measures), zml.Tensor.fromShape(measures) }, .{});
    defer exe.deinit();

    var grey: zml.Buffer = try .fromBytes(io, platform, measures, std.mem.sliceAsBytes(&grey_coat));
    defer grey.deinit();
    var tweed: zml.Buffer = try .fromBytes(io, platform, measures, std.mem.sliceAsBytes(&tweed_coat));
    defer tweed.deinit();

    // Mr Abbot, measured at the counter.
    const abbot = [3]i32{ 100, 62, 110 };
    var abbot_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .measure = 3 }, .i32), std.mem.sliceAsBytes(&abbot));
    defer abbot_buffer.deinit();

    // The raw tables, the way of exercise 082.
    {
        var args = try exe.args(gpa);
        defer args.deinit(gpa);
        var results = try exe.results(gpa);
        defer results.deinit(gpa);

        args.set(.{ grey, abbot_buffer });
        exe.call(io, args, &results, .{ .wait = true });
        var first = results.get(zml.Buffer);
        defer first.deinit();
        try stdout.print("raw: the grey coat on Mr Abbot is too small by {d:.1} cm\n", .{try first.getValue(f32, io)});

        // The tweed coat, and nobody in it.
        args.set(.{tweed});
        exe.call(io, args, &results, .{ .wait = true });
        var second = results.get(zml.Buffer);
        defer second.deinit();
        try stdout.print("raw: the tweed coat on nobody is too small by {d:.1} cm (Mr Abbot never left the table)\n", .{try second.getValue(f32, io)});
    }

    var clerk: Clerk = try .init(&exe, gpa);
    defer clerk.deinit(gpa);

    var verdict: zml.Buffer = undefined;
    if (clerk.serve(io, .{tweed}, .{&verdict})) |_| {
        verdict.deinit();
        try stdout.print("Odile: the tweed coat on nobody: served\n", .{});
    } else |err| {
        try stdout.print("Odile: the tweed coat on nobody: {t}\n", .{err});
    }

    // Three numbers that are not measures at all.
    var shelf: zml.Buffer = try .fromBytes(io, platform, .init(.{ .shelf = 3 }, .f32), std.mem.sliceAsBytes(&[3]f32{ 7, 7, 7 }));
    defer shelf.deinit();
    if (clerk.serve(io, .{ tweed, shelf }, .{&verdict})) |_| {
        verdict.deinit();
        try stdout.print("Odile: the tweed coat on shelf 7: served\n", .{});
    } else |err| {
        try stdout.print("Odile: the tweed coat on shelf 7: {t}\n", .{err});
    }

    // Everybody tries the tweed coat.
    for (people, people_names) |person, name| {
        var person_buffer: zml.Buffer = try .fromBytes(io, platform, measures, std.mem.sliceAsBytes(&person));
        defer person_buffer.deinit();

        try clerk.serve(io, .{ person_buffer, tweed }, .{&verdict});
        defer verdict.deinit();
        const cm = try verdict.getValue(f32, io);
        try stdout.print("Odile: the tweed coat on {s}: too small by {d:.1} cm{s}\n", .{ name, cm, if (cm == 0) ", it fits" else "" });
    }

    try stdout.flush();
}
