//
// Saturday afternoon, birthday party in the back room. Eleven
// children, one adult, and the oldest problem in mathematics:
// cutting a pizza so that nobody cries.
//
// Cutting a tensor into several pieces along one axis is
// nothing more than several `slice` calls. Since you'd have to
// compute every start and end by hand, ZML has two helpers
// that do the bookkeeping.
//
//     pub fn split(self: Tensor, axis_: anytype, split_sizes: []const i64) []Tensor
//
// `split` cuts where you tell it. You give the SIZE of each
// piece, in order, and get one tensor per size:
//
//     const parts = pizza.split(.slices, &.{ 1, 3, 4 });
//
//     parts[0]    {slices=1}     slice 0
//     parts[1]    {slices=3}     slices 1, 2, 3
//     parts[2]    {slices=4}     slices 4, 5, 6, 7
//
// The sizes must add up to the whole axis, no leftovers and no
// imaginary pizza:
//
//     split expects sum of 'split_sizes' values and axis dimension to be equal, got 7 and 8
//
// The returned `[]Tensor` is allocated by the compiler that is
// tracing your function, and released with it. Don't free it,
// and don't store it anywhere that outlives the function.
//
//     pub fn chunkExact(self: Tensor, axis_: anytype, n_chunks: comptime_int) [n_chunks]Tensor
//
// `chunkExact` cuts into `n_chunks` pieces of EQUAL size. The
// count is comptime, so the result is an array, and an array
// can be destructured:
//
//     const left, const right = pizza.chunkExact(.slices, 2);
//
// The size of the axis must be divisible by the count (it's a
// `@divExact` inside, and it means it).
//
// In both cases every piece keeps the cut axis, shorter: these
// are "start/end" slices, not `.single` ones. Which is
// convenient, because pieces that still have the axis can be
// handed straight back to `concatenate`.
//
// And we'll need to. A function given to `compileFn` MAY return
// several tensors (a `[]Tensor`, a struct of Tensors...), and
// the Exe then has several outputs. But `exe.eval`, the only
// way to run an Exe you know so far, insists on exactly one.
// Running multi-output models is a later chapter. Until then,
// we cut, we rearrange, and we glue the result back into ONE
// tensor before returning it.
//
// The party orders:
//
//   - `bigFirst`: the pizza is shared between a toddler
//     (1 slice), her brother (3 slices) and their cousin
//     (everything else), in that order around the pizza. Serve
//     them biggest share first: return the cousin's slices,
//     then the brother's, then the toddler's, on one axis.
//     Luigi cut it, and Luigi served it. Two complaints.
//
//   - `twins`: the twins get exactly half each, and each will
//     check the other's plate. Return, position by position,
//     how many more olives the first twin got than the second
//     (`a.sub(b)`). Result: {slices=4}.
//
//   - `tally`: the parents want numbers. Same three shares as
//     `bigFirst`; return how many olives each child got in
//     total, toddler first: {slices=3}. Yours to write. You
//     know from the Reduction Kitchen (exercise 040) that
//     `sum` leaves an axis of size 1 behind. For once, that is
//     exactly what you want: don't skim it off, glue.
//
// And in `main`, the Exes are compiled but nobody runs them.
//
const std = @import("std");
const zml = @import("zml");

fn bigFirst(pizza: zml.Tensor) zml.Tensor {
    // Toddler, brother, cousin.
    const shares = pizza.split(.slices, &.{ 1, 3, 3 });

    return zml.Tensor.concatenate(&.{ shares[0], shares[1], shares[2] }, .slices);
}

fn twins(pizza: zml.Tensor) zml.Tensor {
    const first, const second = ???;

    return first.sub(second);
}

/// Olives per child: toddler, brother, cousin. Result: {slices=3}.
fn tally(pizza: zml.Tensor) zml.Tensor {
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

    const pizza: zml.Tensor = .init(.{ .slices = 8 }, .i32);

    // Olives per slice.
    const olives = [_]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var pizza_buffer: zml.Buffer = try .fromBytes(io, platform, pizza.shape(), std.mem.sliceAsBytes(&olives));
    defer pizza_buffer.deinit();

    inline for (.{ bigFirst, twins, tally }, .{ "bigFirst:", "twins:   ", "tally:   " }) |order, name| {
        var exe = try platform.compileFn(gpa, io, order, .{pizza}, .{});
        defer exe.deinit();

        // Run it on the pizza.
        var result: zml.Buffer = ???;
        defer result.deinit();

        const plate: zml.Slice = try result.toSliceAlloc(gpa, io);
        defer plate.free(gpa);

        try stdout.print("{s} {f} {any}\n", .{ name, plate.shape, plate.constItems(i32) });
    }
    try stdout.flush();
}
