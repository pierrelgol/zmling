//
// "Integers are easy", says Tock. "No rounding. A gear has
// teeth or it has not." Mr. Tick says nothing and slides a
// tray across the bench: five watches, each with a gear that
// fits and turns and gives the wrong time.
//
// Models are floats, but what goes IN and comes OUT is
// integers: pixels are .u8, token ids are .i32 or .i64,
// labels and positions are integers. Every port has a
// border where integers become floats and back, and that
// border has its own traps. All of these run without a
// word. That is what makes them traps.
//
// TRAP 1: INTEGER DIVISION. `div` on integers truncates.
// A pixel is 0..255, a model wants 0..1:
//
//     pixels.div(.scalar(255, .u8))      0 0 0 0 1
//
// Every pixel is black except pure white. Convert FIRST,
// divide after. Chapter 17 preprocessed MNIST: go and look
// at the order it did things in.
//
// TRAP 2: WRAPAROUND. Integer arithmetic wraps: in .u8,
// 200 + 100 = 44 (300 - 256). Brightening an image in .u8
// turns the brightest pixels dark. Compute in a wider type,
// `clamp` (exercise 69), come back.
//
// TRAP 3: INT TO INT `convert` WRAPS TOO. 200 as .u8 becomes
// -56 as .i8: the same bits, reinterpreted. The famous one:
// token ids saved as .i64 by PyTorch, loaded as .i32. Fine
// until an id above two billion. (There is none. But a -1
// "ignore" label converted to .u32 is 4294967295, and then
// `gather` goes looking for row four billion.)
//
// TRAP 4: FLOAT TO INT `convert` TRUNCATES. Toward zero:
// 2.9 becomes 2, -2.9 becomes -2. It is not `floor` (-2.9
// would give -3) and it is not rounding (2.9 would give 3).
// Positions, bucket indices, image coordinates: say which
// one you mean with `floor`, `ceil` or `roundNearestAfz`
// (exercise 19) BEFORE converting.
//
// TRAP 5: FLOAT TO INT SATURATES. Out of range, a float
// stops at the edge of the integer type: 300.0 as .u8 is
// 255, -3.0 as .u8 is 0. Unlike trap 3, this one is usually
// what you want. But do not expect it of int to int.
//
// And one rule for the other direction: .f32 holds every
// integer up to 2^24 = 16777216 exactly (exercise 148), and
// .bf16 only up to 2^8 = 256. Token ids and positions never
// travel as .bf16.
//
// To do:
//
//  1. `toUnit` is trap 1. Repair it.
//  2. Write `brighten`: add `amount` to every pixel, stop at
//     255, return .u8.
//  3. Write `bucket`: position `p` (a float) falls in the
//     bucket floor(p / width), as an .i32. Negative
//     positions exist: -0.5 is in bucket -1.
//  4. `main` runs the five traps too, so you see each of
//     them once with your own eyes.
//
const std = @import("std");
const zml = @import("zml");

/// pixels: {.px} u8 in 0..255 -> {.px} f32 in 0..1
fn toUnit(pixels: zml.Tensor) zml.Tensor {
    return pixels.div(.scalar(255, .u8)).convert(.f32);
}

/// pixels: {.px} u8 -> {.px} u8, `amount` brighter, at most 255.
fn brighten(pixels: zml.Tensor, amount: u8) zml.Tensor {
    return ???;
}

/// positions: {.px} f32 -> {.px} i32
fn bucket(positions: zml.Tensor, width: f32) zml.Tensor {
    return ???;
}

const Tray = struct {
    trap1: zml.Tensor,
    unit: zml.Tensor,
    trap2: zml.Tensor,
    brighter: zml.Tensor,
    trap3: zml.Tensor,
    trap4: zml.Tensor,
    buckets: zml.Tensor,
    trap5: zml.Tensor,
};

fn tray(pixels: zml.Tensor, positions: zml.Tensor) Tray {
    return .{
        .trap1 = pixels.div(.scalar(255, .u8)).convert(.i32),
        .unit = toUnit(pixels),
        .trap2 = pixels.addConstant(100).convert(.i32),
        .brighter = brighten(pixels, 100).convert(.i32),
        .trap3 = pixels.convert(.i8).convert(.i32),
        .trap4 = positions.divByConst(4).convert(.i32),
        .buckets = bucket(positions, 4),
        .trap5 = positions.scale(100).convert(.u8).convert(.i32),
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

    const pixel_values = [_]u8{ 0, 51, 128, 200, 255 };
    const position_values = [_]f32{ -6, -0.5, 0.5, 3.9, 11.6 };
    const pixels: zml.Tensor = .init(.{ .px = 5 }, .u8);
    const positions: zml.Tensor = .init(.{ .px = 5 }, .f32);

    var pixels_buffer: zml.Buffer = try .fromBytes(io, platform, pixels.shape(), &pixel_values);
    defer pixels_buffer.deinit();
    var positions_buffer: zml.Buffer = try .fromBytes(io, platform, positions.shape(), std.mem.sliceAsBytes(&position_values));
    defer positions_buffer.deinit();

    var exe = try platform.compileFn(gpa, io, tray, .{ pixels, positions }, .{});
    defer exe.deinit();
    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);
    var out: zml.Bufferized(Tray) = undefined;
    runner.run(io, .{ pixels_buffer, positions_buffer }, &out, .{ .wait = true });
    defer zml.Buffer.deinitAll(Tray, &out);

    try stdout.print("pixels:                {any}\n", .{pixel_values});
    inline for (.{
        .{ "trap 1, u8 / 255:      ", out.trap1 },
        .{ "trap 2, u8 + 100:      ", out.trap2 },
        .{ "brighten by 100:       ", out.brighter },
        .{ "trap 3, u8 as i8:      ", out.trap3 },
        .{ "trap 4, p / 4 as i32:  ", out.trap4 },
        .{ "bucket of width 4:     ", out.buckets },
        .{ "trap 5, p * 100 as u8: ", out.trap5 },
    }) |line| {
        const slice = try line[1].toSliceAlloc(gpa, io);
        defer slice.free(gpa);
        try stdout.print("{s}{any}\n", .{ line[0], slice.items(i32) });
    }
    const unit = try out.unit.toSliceAlloc(gpa, io);
    defer unit.free(gpa);
    try stdout.print("to the unit interval: ", .{});
    for (unit.items(f32)) |v| try stdout.print(" {d:.3}", .{v});
    try stdout.print("\n", .{});
    try stdout.flush();
}
