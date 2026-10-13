//
// "How would you like that done?"
//
// Everything so far was arithmetic you could do on a napkin. A
// neural network needs one more ingredient, and it is the reason
// the things work at all: ACTIVATION FUNCTIONS.
//
// Here's the problem they solve. Stack as many "multiply, then
// add" steps as you like and the result is still one big
// "multiply, then add". A straight line, however long you stare
// at it. To learn anything interesting (what a cat looks like,
// when Earl is going to order pie) a network must be able to
// BEND. So between the linear steps it puts each number through
// a small non-linear function: an activation.
//
// In many frameworks activations are objects you instantiate. In
// ZML they are plain Tensor methods, like `abs` or `exp`:
//
//     x.relu()            max(x, 0). Negative? Gone. Brutal,
//                         cheap, and the default for a decade.
//
//     x.leakyReLU(0.1)    like relu, but negative values are
//                         multiplied by the slope instead of
//                         being thrown away. The slope is an f32.
//
//     x.sigmoid()         1 / (1 + e^-x). Squashes anything into
//                         (0, 1): big negative -> 0, big positive
//                         -> 1, zero -> 0.5. A "probability dial".
//
//     x.tanh()            the same idea, into (-1, 1).
//
//     x.silu()            x * sigmoid(x). A smooth relu with a
//                         little dip below zero. Also sold under
//                         the name "swish".
//
//     x.gelu()            another smooth relu, the one inside
//                         most transformers. ZML uses the usual
//                         tanh approximation of it.
//
// And they really are nothing special. This is the whole of
// `silu` in ZML's tensor.zig:
//
//     pub fn silu(x: Tensor) Tensor {
//         return x.mul(.sigmoid(x));
//     }
//
// You could have written it. In fact, you're about to.
//
// (For models whose activation is chosen by a config file, there
// is also `zml.nn.Activation`, a tagged union with a `forward`
// method that calls the functions above. Same thing, one more
// layer of wrapping paper.)
//
// At the diner, Gus grades every steak with a number: negative is
// underdone, positive is overdone, and the customer's reaction
// depends on which activation the customer is. The program prints
// each reaction to the same five steaks.
//
// Two of the recipes are "by hand" versions and must print the
// same numbers as the built-in ones, using only operations you
// already know:
//
//   - leaky relu with slope 0.1. (Hint for the napkin: for a
//     slope below 1, which is larger, x or 0.1 * x? When?)
//
//   - silu.
//
const std = @import("std");
const zml = @import("zml");

const steaks = [5]f32{ -3, -1, 0, 1, 3 };

fn relu(x: zml.Tensor) zml.Tensor {
    return x.relu();
}

fn leaky(x: zml.Tensor) zml.Tensor {
    return x.leakyReLU(0.1);
}

fn sigmoid(x: zml.Tensor) zml.Tensor {
    return x.sigmoid();
}

fn tanh(x: zml.Tensor) zml.Tensor {
    return x.tanh();
}

fn silu(x: zml.Tensor) zml.Tensor {
    return x.silu();
}

fn gelu(x: zml.Tensor) zml.Tensor {
    return x.gelu();
}

/// Leaky relu with a slope of 0.1, without calling `leakyReLU`.
fn leakyByHand(x: zml.Tensor) zml.Tensor {
    // TODO
    return x;
}

/// silu, without calling `silu`.
fn siluByHand(x: zml.Tensor) zml.Tensor {
    // TODO
    return x;
}

/// Compiles `recipe` for five f32, runs it on the steaks.
fn cook(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    comptime recipe: fn (zml.Tensor) zml.Tensor,
) ![5]f32 {
    const x: zml.Tensor = .init(.{ .steaks = 5 }, .f32);

    var exe = try platform.compileFn(gpa, io, recipe, .{x}, .{});
    defer exe.deinit();

    var input: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&steaks));
    defer input.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{input});
    defer result.deinit();

    return result.getValue([5]f32, io);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    try stdout.print("steaks ", .{});
    for (steaks) |s| try stdout.print(" {d:7.3}", .{s});
    try stdout.print("\n", .{});

    // `inline for` over a tuple: each recipe is known at compile
    // time, which is what `compileFn` wants.
    inline for (.{
        .{ "relu   ", relu },
        .{ "leaky  ", leaky },
        .{ "by hand", leakyByHand },
        .{ "sigmoid", sigmoid },
        .{ "tanh   ", tanh },
        .{ "silu   ", silu },
        .{ "by hand", siluByHand },
        .{ "gelu   ", gelu },
    }) |row| {
        const name, const recipe = row;
        const reactions = try cook(gpa, io, platform, recipe);

        try stdout.print("{s}", .{name});
        for (reactions) |r| try stdout.print(" {d:7.3}", .{r});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
