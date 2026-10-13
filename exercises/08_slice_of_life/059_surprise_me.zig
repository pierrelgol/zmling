//
// A new kind of customer walks into Tensorino's: "Three slices,
// please. Starting where? Oh, I'll tell you when they're on the
// table. Surprise me. No wait, I'LL surprise YOU."
//
// Until now, every `start` was a Zig number, written in the
// function, and therefore BAKED into the Exe when it was
// compiled (chapter 2: house rules). To cut somewhere else you
// had to compile again. That's fine for "drop the crust". It's
// hopeless for "give me row number `token_id`" or "the next 16
// positions starting at wherever we are now", where the
// position is only known when the data arrives.
//
// So `zml.Tensor.Slice` has a second way to say where to
// start: with a TENSOR.
//
//     pub fn dyn(start: Tensor, len: i64) Slice
//     pub fn dynSingle(start: Tensor) Slice
//
//     x.slice(.slices, .dyn(first, 3))      3 elements starting at
//                                           `first`. Axis kept.
//
//     x.slice(.pizzas, .dynSingle(which))   the element at `which`.
//                                           Axis dropped, like
//                                           `.single`.
//
// `first` and `which` are scalar tensors of an integer dtype:
// placeholders in main, parameters of your function, and at
// run time a Buffer like any other input. One Exe, compiled
// once, cuts wherever each call tells it to.
//
// Look closely at what is a tensor and what is not:
//
//     .dyn(first, 3)
//          ^^^^^  ^
//          |      '-- LENGTH: a Zig number. Compile time.
//          '-- START: a tensor. Run time.
//
// This is not laziness on ZML's part. An Exe is compiled for
// fixed shapes; the shape of the result must be known before
// any data exists. WHERE to cut may wait for the data; HOW MUCH
// comes out may not. (This is the moment of glory promised to
// `len` in exercise 054: for a dynamic slice it's the field
// that matters, since there can't be an `end`.)
//
// Two consequences of the start being unknown when tracing:
//
//   - No bounds check at compile time, obviously. And none at
//     run time either: a start that would make the window
//     stick out is CLAMPED so that the window fits. Asking for
//     3 slices starting at 7 on a pizza of 8 gives you slices
//     5, 6, 7 and no complaint.
//
//   - Negative starts do not count from the end. They are
//     clamped as well, to 0.
//
// A wrong index is thus never a crash, only a wrong (but
// well-shaped) answer. Check your indices before they go in.
//
// And the other direction, since writing is the natural
// companion of reading: `dynamicUpdateSlice` returns a copy of
// a tensor in which a window, located by run-time indices, has
// been replaced.
//
//     tray.dynamicUpdateSlice(.{ .pizzas = which }, fresh)
//
//     tray {pizzas=3,slices=4}, fresh {slices=4}, `which` a scalar:
//     "the same tray, with pizza number `which` replaced by
//     `fresh`". Tags are used to line the update up with the
//     tray. Nothing is modified in place; tensors are values.
//
// Four orders:
//
//   - `window`: 3 consecutive slices of the pizza, starting at
//     `first`. Yours to write.
//
//   - `pick`: pizza number `which` of the tray, OUT of its box.
//     Luigi took this one. Look at the shape it prints.
//
//   - `swap`: (done, by Nonna) the tray with pizza number
//     `which` replaced by a fresh one.
//
//   - `fullest`: the pizza of the tray that has the most
//     olives, out of its box. Nobody says which one it is:
//     the Exe must find out by itself. Yours to write, with
//     three things from the Reduction Kitchen: the total of
//     each pizza (039), the POSITION of the biggest total
//     (`argMax(...).indices`, 042), and `squeeze`, twice,
//     until that position is a scalar (040). A scalar integer
//     tensor is all a dynamic slice asks for. An index that
//     the model computes from the data and uses a moment
//     later: this is what dynamic slices are really for.
//
// `main` has two holes as well. The surprise itself is
// missing: a placeholder for a scalar i32 (exercise 016 all
// over again). And `fullest` is compiled but never run: once
// on the tray, once on the tray that `swap` just returned. A
// result Buffer is a Buffer like any other, and nothing stops
// you from feeding it to another Exe.
//
const std = @import("std");
const zml = @import("zml");

fn window(pizza: zml.Tensor, first: zml.Tensor) zml.Tensor {
    return ???;
}

fn pick(tray: zml.Tensor, which: zml.Tensor) zml.Tensor {
    return tray.slice(.pizzas, .dyn(which, 1));
}

fn swap(tray: zml.Tensor, which: zml.Tensor, fresh: zml.Tensor) zml.Tensor {
    return tray.dynamicUpdateSlice(.{ .pizzas = which }, fresh);
}

/// The pizza with the most olives: {slices=4}.
fn fullest(tray: zml.Tensor) zml.Tensor {
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
    const tray: zml.Tensor = .init(.{ .pizzas = 3, .slices = 4 }, .i32);
    const fresh: zml.Tensor = .init(.{ .slices = 4 }, .i32);
    // The surprise goes here: a placeholder called `index`.

    var pizza_buffer: zml.Buffer = try .fromBytes(io, platform, pizza.shape(), std.mem.sliceAsBytes(&[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 }));
    defer pizza_buffer.deinit();
    var tray_buffer: zml.Buffer = try .fromBytes(io, platform, tray.shape(), std.mem.sliceAsBytes(&[3][4]i32{
        .{ 10, 11, 12, 13 },
        .{ 20, 21, 22, 23 },
        .{ 30, 31, 32, 33 },
    }));
    defer tray_buffer.deinit();
    var fresh_buffer: zml.Buffer = try .fromBytes(io, platform, fresh.shape(), std.mem.sliceAsBytes(&[_]i32{ 99, 99, 99, 99 }));
    defer fresh_buffer.deinit();

    // Each Exe is compiled ONCE and then run with several indices.
    {
        var exe = try platform.compileFn(gpa, io, window, .{ pizza, index }, .{});
        defer exe.deinit();

        // 7 is too far, -2 is not "from the end": watch them get clamped.
        for ([_]i32{ 0, 2, 4, 7, -2 }) |first| {
            var first_buffer: zml.Buffer = try .scalar(io, platform, first, .i32);
            defer first_buffer.deinit();

            var result: zml.Buffer = try exe.eval(gpa, io, .{ pizza_buffer, first_buffer });
            defer result.deinit();

            try stdout.print("window from {d}: {any}\n", .{ first, try result.getValue([3]i32, io) });
        }
    }
    {
        var exe = try platform.compileFn(gpa, io, pick, .{ tray, index }, .{});
        defer exe.deinit();

        for ([_]i32{ 2, 0 }) |which| {
            var which_buffer: zml.Buffer = try .scalar(io, platform, which, .i32);
            defer which_buffer.deinit();

            var result: zml.Buffer = try exe.eval(gpa, io, .{ tray_buffer, which_buffer });
            defer result.deinit();

            try stdout.print("pick {d}: {f} {any}\n", .{ which, exe.output_shapes[0], try result.getValue([4]i32, io) });
        }
    }
    {
        var exe = try platform.compileFn(gpa, io, swap, .{ tray, index, fresh }, .{});
        defer exe.deinit();
        var fullest_exe = try platform.compileFn(gpa, io, fullest, .{tray}, .{});
        defer fullest_exe.deinit();

        var which_buffer: zml.Buffer = try .scalar(io, platform, 1, .i32);
        defer which_buffer.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{ tray_buffer, which_buffer, fresh_buffer });
        defer result.deinit();

        try stdout.print("swap 1: {any}\n", .{try result.getValue([12]i32, io)});

        // Run `fullest_exe` twice: `before` is its result on the
        // tray, `after` its result on the swapped tray.

        try stdout.print("fullest before the swap: {any}\n", .{try before.getValue([4]i32, io)});
        try stdout.print("fullest after the swap: {any}\n", .{try after.getValue([4]i32, io)});
    }
    try stdout.flush();
}
