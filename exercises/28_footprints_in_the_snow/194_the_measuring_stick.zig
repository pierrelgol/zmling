//
// Ingrid measures every print before she casts it. "After the
// plaster sets," she says, "it is too late to find out that it
// was a hare."
//
// Every panic ZML ever gave you about a shape, a dtype or a tag
// came from a line like this one (from `binaryOpHelper`, the
// engine of `add`, `mul` and friends):
//
//     stdx.debug.assert(
//         same_shape,
//         "{s} expects tensor shapes to match, got {f} and {f}",
//         .{ op_name, self._shape, other._shape },
//     );
//
// and `assert` is four lines in `zml/stdx/debug.zig`:
//
//     pub inline fn assert(
//         check: bool, comptime msg: []const u8, args: anytype,
//     ) void {
//         if (!check) {
//             @branchHint(.cold);
//             panic(msg, args);
//         }
//     }
//
// A Zig `if`, on a Zig `bool`, computed from SHAPES. And a
// shape is a Zig value, complete and known while the snow
// falls (exercise 191: `_shape` is right there in the struct).
// So every check ZML makes runs at TRACE time, when your
// function is called by `emitMlir`. It follows that:
//
//   - A shape mistake can never wait for the run. If the
//     compile call returned, every op of the program has the
//     shape it will have for ever. The Exe checks one thing
//     only, the shapes of the Buffers you hand it.
//   - The checks cost nothing at run time. They are not in the
//     program: they are Zig, and Zig is gone. Be generous.
//
// But ZML checks what an OP needs, not what YOUR LAYER means.
// Three things it lets through, all met earlier:
//
//   - element-wise ops on two tensors of the same sizes do not
//     compare the tags: {day=3,print=3} plus {print=3,day=3}
//     runs, position by position, and is wrong;
//   - `Shape.eql` compares the sizes and the dtype, NOT the
//     tags (read it in `shape.zig`: two lines);
//   - the convolutions of chapter 24 never look at tags.
//
// When a square table comes in sideways, nothing panics, and
// the model answers. So a layer with a convention should say
// so, with the same tool, which is public:
//
//     zml.stdx.debug.assert(check, "format", .{ args });
//
// The format string is comptime; `{f}` prints a Tensor as
// `Tensor({day=3,print=3,f32})` and a Shape without the
// wrapper. A good message has three parts: WHO is complaining,
// what it EXPECTED, what it GOT. Helpful things to test:
//
//     x.rank()                  how many axes
//     x.dtype()                 .f32, .i32, ...
//     x.shape().hasTag(.day)    ?u3: the position of the axis
//                               tagged .day, or null
//     x.dim(.day)               its size (panics if there is
//                               no such tag)
//
// And here is the limit. A trace-time check sees shapes, never
// VALUES. "No depth is negative" is a statement about numbers
// that do not exist yet:
//
//     zml.stdx.debug.assert(x.cmp(.LT, zero), ...)
//     error: expected type 'bool', found 'tensor.Tensor'
//
// Same wall as in exercise 076. A check on values has to be IN
// the program, as ops, and its verdict comes out as a result
// that the host reads after the run: count the offenders
// (exercise 151 did it for NaNs: `cmp`, `convert(.i32)`,
// `sum`), or repair them with `select`, `clamp`, `relu`.
//
// The task. `Settle` works out how much of the fresh snow in
// each print is really new: `fresh - old * keep`, where `keep`
// is a weight per print. Its convention: a table is
// {.day, .print}, in that order. Ingrid's old notebook is
// written sideways, {.print, .day}, three prints over three
// days, and `settle` hands it over as it is. Nothing panics.
//
// To do:
//
//  1. `expectTrail`: one tally, then the stick. Rank 2, `.day`
//     first, `.print` second. The message is given: its three
//     arguments are yours.
//  2. `Settle.forward` measures both of its tables.
//  3. Run it: this time the panic is YOURS. Read it, and turn
//     the notebook the right way up in `settle` (`transpose`,
//     chapter 07).
//  4. `belowZero`: how many readings of a table are negative,
//     as an i32 scalar.
//  5. In `main`: `belowZero` is compiled for the shape of the
//     result, and run on it. A result Buffer is a Buffer like
//     any other.
//
const std = @import("std");
const zml = @import("zml");

/// How many times the stick came out.
var measures: usize = 0;

/// Panics, at trace time, unless `x` is a {.day, .print} table.
fn expectTrail(comptime who: []const u8, x: zml.Tensor) void {
    // The message: "{s} expects a {s} table, got {f}", and the
    // table it expects is spelled "{day,print}".
    return ???;
}

const Settle = struct {
    /// {.print}: how much of yesterday's snow is still there.
    keep: zml.Tensor,

    /// fresh, old: {.day, .print} -> {.day, .print}
    pub fn forward(self: Settle, fresh: zml.Tensor, old: zml.Tensor) zml.Tensor {
        return fresh.sub(old.mul(self.keep.broad(old.shape())));
    }
};

/// fresh: {.day, .print}, notebook: {.print, .day} -> {.day, .print}
fn settle(model: Settle, fresh: zml.Tensor, notebook: zml.Tensor) zml.Tensor {
    return model.forward(fresh, notebook);
}

/// How many values of `x` are negative: {} i32.
fn belowZero(x: zml.Tensor) zml.Tensor {
    return ???;
}

const keep_data = [3]f32{ 1, 2, 0.5 };

/// {.day = 3, .print = 3}
const fresh_data = [3][3]f32{
    .{ 10, 12, 14 },
    .{ 20, 22, 24 },
    .{ 30, 32, 34 },
};

/// {.print = 3, .day = 3}: one ROW per print.
const notebook_data = [3][3]f32{
    .{ 1, 4, 70 },
    .{ 2, 5, 8 },
    .{ 3, 6, 9 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const model: Settle = .{ .keep = .init(.{ .print = 3 }, .f32) };
    const fresh: zml.Tensor = .init(.{ .day = 3, .print = 3 }, .f32);
    const notebook: zml.Tensor = .init(.{ .print = 3, .day = 3 }, .f32);

    var exe = try platform.compileFn(gpa, io, settle, .{ model, fresh, notebook }, .{});
    defer exe.deinit();

    try stdout.print("while the snow fell, the stick came out {d} times\n", .{measures});

    var keep_buffer: zml.Buffer = try .fromBytes(io, platform, model.keep.shape(), std.mem.sliceAsBytes(&keep_data));
    defer keep_buffer.deinit();
    const weights: zml.Bufferized(Settle) = .{ .keep = keep_buffer };

    var fresh_buffer: zml.Buffer = try .fromBytes(io, platform, fresh.shape(), std.mem.sliceAsBytes(&fresh_data));
    defer fresh_buffer.deinit();
    var notebook_buffer: zml.Buffer = try .fromBytes(io, platform, notebook.shape(), std.mem.sliceAsBytes(&notebook_data));
    defer notebook_buffer.deinit();

    var settled: zml.Buffer = try exe.eval(gpa, io, .{ weights, fresh_buffer, notebook_buffer });
    defer settled.deinit();

    try stdout.print("new snow {f}:\n", .{settled.shape()});
    for (try settled.getValue([3][3]f32, io), 0..) |row, day| {
        try stdout.print("  day {d}: {d:5.1} {d:5.1} {d:5.1}\n", .{ day, row[0], row[1], row[2] });
    }

    // The check on VALUES: a second little program, run on the result.
    var below_exe = ???;
    defer below_exe.deinit();
    var below: zml.Buffer = ???;
    defer below.deinit();

    try stdout.print("readings below zero: {d}\n", .{try below.getValue(i32, io)});
    try stdout.print("two runs later, the stick came out {d} times\n", .{measures});
    try stdout.flush();
}
