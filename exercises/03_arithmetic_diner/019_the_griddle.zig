//
// Gus has one tool (the griddle) and he applies it to one thing
// at a time. Operations like that, taking a single tensor and
// transforming each element on its own, are called UNARY. They
// take no argument at all, and the result has the shape of the
// input:
//
//     x.abs()       |x|             x.negate()    -x
//     x.sign()      -1, 0 or 1
//
//     x.exp()       e^x             x.log()       natural logarithm
//     x.sqrt()      square root     x.rsqrt()     1 / sqrt(x)
//     x.tanh()      hyperbolic tangent, squashes into (-1, 1)
//     x.sin()       x.cos()         in radians
//
//     x.floor()     round down      x.ceil()      round up
//     x.round()                     round to nearest, ties to EVEN
//     x.roundNearestAfz()           round to nearest, ties Away
//                                   From Zero
//
// One plate at a time, as always:
//
//     x            { -2.5,  0,  1.5 }
//
//     x.abs()      {  2.5,  0,  1.5 }
//     x.negate()   {  2.5,  0, -1.5 }
//     x.sign()     { -1,    0,  1   }
//     x.floor()    { -3,    0,  1   }
//     x.ceil()     { -2,    0,  2   }
//
// Three things the menu doesn't tell you.
//
// 1. Most of these want floats. `abs`, `negate` and `sign` are
//    happy with integers, but ask for the `sqrt` of an .i32 tensor
//    and the compiler behind ZML refuses the graph with a long
//    error about "stablehlo.sqrt ... must be ranked tensor of
//    float". (StableHLO is the name of the instruction set ZML
//    writes its recipes in. You'll see it in error messages.)
//
// 2. ZML checks shapes and dtypes. It can NOT check values: there
//    aren't any while tracing. `log` of a negative number is not a
//    panic, it is a quiet NaN in your result, at run time, exactly
//    like `@log(-1.0)` in Zig.
//
// 3. `round` does "banker's rounding": a value exactly halfway
//    goes to the nearest EVEN integer. 0.5 becomes 0, 1.5 becomes
//    2, 2.5 becomes 2 as well. Statisticians love it; it doesn't
//    drift upward over a million additions. The rounding you
//    learned in school, where 2.5 becomes 3, is
//    `roundNearestAfz`.
//
// To try several recipes without drowning in boilerplate, this
// exercise has a helper, `cook`: it compiles a one-tensor recipe,
// runs it on four numbers, and returns the four results. Nothing
// new in it, which is why two of its lines are missing: the one
// that puts the ingredients on the device and the one that reads
// the four results back (both from exercise 011). The helper will
// come back, so get it right once.
//
// Today's tickets:
//
//   - The griddle wants to be at 350 degrees F. How far off is
//     each of the four burners, in either direction?
//
//   - Coffee cools. Poured at 80 degrees C in a 20 degree room,
//     its temperature after t minutes is 20 + 60 * e^(-t / 10).
//     That's Newton, not Gus. (At t = 0, e^0 is 1 and the cup is
//     at 20 + 60 = 80. After a very long time e^(-t / 10) is
//     nearly 0, and the coffee is as warm as the room.) No
//     single method does all that: it is a chain.
//
//   - The tip jar is shared out in whole dollars. Flo wants
//     halves rounded the way she learned in school.
//
// Gus wrote the first ticket and forgot one word. Nobody wrote
// the second one. The third one runs, and Flo is not happy.
//
const std = @import("std");
const zml = @import("zml");

/// Distance of each burner from 350 degrees, never negative.
fn offBy(burners: zml.Tensor) zml.Tensor {
    return burners.subConstant(350).???();
}

/// 20 + 60 * e^(-minutes / 10)
fn coffee(minutes: zml.Tensor) zml.Tensor {
    return ???;
}

/// Whole dollars. A half dollar rounds up, like in school.
fn wholeDollars(tips: zml.Tensor) zml.Tensor {
    return tips.round();
}

/// Compiles `recipe` for four f32, runs it on `ingredients`.
fn cook(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    comptime recipe: fn (zml.Tensor) zml.Tensor,
    ingredients: [4]f32,
) ![4]f32 {
    const x: zml.Tensor = .init(.{4}, .f32);

    var exe = try platform.compileFn(gpa, io, recipe, .{x}, .{});
    defer exe.deinit();

    // The ingredients, on the device, in the shape of `x`. (A
    // parameter is a constant like any other: Zig lets you take
    // its address.)
    var input: zml.Buffer = ???;
    defer input.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{input});
    defer result.deinit();

    // The four results, back on the host.
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

    const burners = try cook(gpa, io, platform, offBy, .{ 340, 350, 375, 310 });
    try stdout.print("Burners are off by: {any}\n", .{burners});

    const cups = try cook(gpa, io, platform, coffee, .{ 0, 5, 10, 30 });
    try stdout.print("Coffee after 0, 5, 10, 30 minutes:", .{});
    for (cups) |celsius| try stdout.print(" {d:.1}", .{celsius});
    try stdout.print("\n", .{});

    const tips = try cook(gpa, io, platform, wholeDollars, .{ 0.5, 1.5, 2.5, 3.5 });
    try stdout.print("Tips of 0.5, 1.5, 2.5, 3.5 become: {any}\n", .{tips});
    try stdout.flush();
}
