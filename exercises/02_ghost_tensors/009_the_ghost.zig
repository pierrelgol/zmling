//
// Chapter 2: Ghost Tensors
//
// You've left Shifty & Sons (the highway incident was not your
// fault) and found work as the apprentice of Madame Zelda,
// medium, at Graymoor Manor. It is said to be haunted. It is.
//
// The ghosts of Graymoor are called `zml.Tensor`.
//
//     const ghost = zml.Tensor.init(.{ 13, 7 }, .f32);
//
// Look familiar? `Tensor.init` takes exactly what `Shape.init`
// takes. And if you already hold a Shape:
//
//     const ghost = zml.Tensor.fromShape(some_shape);
//
// Now here is the single most important idea of this whole
// course, so we'll say it slowly:
//
//     A zml.Tensor contains NO numbers.
//
// None. Not zeros, not garbage: nothing. A Tensor is a ghost: it
// has a shape (you can see its outline), it has a dtype (you can
// tell what it used to be made of), but there is no body. Your
// hand goes right through.
//
// If you've used numpy or PyTorch, this is the big difference.
// There, `x + y` computes a sum right away. In ZML, a Tensor
// stands for "a value that WILL exist later, when the model
// runs", and operations on Tensors don't compute anything: they
// write down a recipe (a computation graph). ZML then hands the
// whole recipe to a compiler that turns it into fast code for
// your CPU or GPU. Only then do actual numbers flow through.
//
// This is why ZML models are fast, and why they can't crash in
// production because of a shape mismatch: by the time numbers
// show up, every shape was checked.
//
// A ghost made with `init` or `fromShape` is the blankest kind:
// a PLACEHOLDER. "Something of this shape will be supplied
// here." What can you do with a placeholder, outside a séance?
// Only ask about its outline:
//
//     ghost.shape()   -> zml.Shape
//     ghost.dtype()   -> zml.DataType
//     ghost.rank()    ghost.count()    ghost.byteSize()
//     ghost.dim(ax)   ghost.dims()     ghost.axis(ax)
//
// Madame Zelda hands you the floor plan. "Thirteen rooms, seven
// candles each. Summon me something that fits. And something
// small for the cellar, while you're at it."
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const floor_plan = zml.Shape.init(.{ .rooms = 13, .candles = 7 }, .f32);

    // Summon a placeholder Tensor with the shape of the floor plan.
    const ghost = zml.Tensor.fromShape(???);

    try stdout.print("{f} haunts {d} rooms and {d} candle flames in total.\n", .{
        ghost,
        // The size of the .rooms axis. Tensors answer the same
        // questions as their Shape.
        ghost.dim(.candles),
        ghost.count(),
    });
    try stdout.print("It would weigh {d} bytes, if it weighed anything.\n", .{ghost.byteSize()});

    // The cellar: 3 barrels, tagged .barrels, holding 32-bit
    // signed integers. No Shape this time: summon the placeholder
    // directly, the way the first line of the lesson does.
    const cellar_ghost = ???;

    try stdout.print("{f} lurks in the cellar: rank {d}, made of {s}.\n", .{
        cellar_ghost,
        cellar_ghost.rank(),
        @tagName(cellar_ghost.dtype()),
    });
    try stdout.flush();
}
