//
// Welcome to zmling!
//
// ZML is a Zig library for running machine learning models. You
// describe a computation on tensors (n-dimensional arrays), ZML
// compiles it for your hardware (CPU, CUDA, ...), and then you
// feed it real numbers. We will get to all of that. Promise.
//
// But first: every tensor, whether it holds a cat picture or the
// weights of a large language model, starts with two questions:
//
//     1. How many numbers are there, and how are they arranged?
//     2. What KIND of numbers are they?
//
// In ZML both answers live in one small value: a `zml.Shape`.
//
//     const zml = @import("zml");
//
//     //                           dimensions  data type
//     //                               |          |
//     const s = zml.Shape.init(.{ 4, 8, 3 }, .u8);
//
// That's a shape for 4 x 8 x 3 = 96 unsigned bytes. Note that a
// Shape holds NO DATA at all. It is a label on a moving box, not
// the box.
//
// Speaking of boxes: this chapter takes place at "Shifty & Sons",
// the clumsiest moving company in town. You are the new hire. On
// your first morning, Old Shifty hands you a clipboard:
//
//     "One sofa. Two rows of three cushions. Each cushion is
//      stuffed with 32-bit floats. Write it up, kid."
//
// A few things a Shape can tell you about itself:
//
//     s.rank()      how many axes ("dimensions")     -> 3
//     s.count()     how many elements in total       -> 96
//     s.byteSize()  how many bytes they would occupy -> 96
//     s.dtype()     the data type                    -> .u8
//
// The data type is a `zml.DataType`, an enum with members such
// as .f32, .f16, .i64, .u8 and .bool.
//
// Old Shifty's nephew started the paperwork below but could not
// remember what cushions are made of. Please finish the job.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    // zmling checks what your program writes to stdout, so every
    // exercise starts with these three lines. (Zig 0.17 does all
    // I/O through the `std.Io` interface: `init.io`.)
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // Two rows of three cushions, each one a 32-bit float.
    const sofa = zml.Shape.init(.{ 2, 3 }, ???);

    try stdout.print("The sofa has {d} axes, {d} cushions, and weighs {d} bytes.\n", .{
        sofa.rank(),
        sofa.count(),
        sofa.byteSize(),
    });
    try stdout.flush();
}
