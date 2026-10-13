//
// "What's in the booox?!" -- every mover, eventually.
//
// The second half of a Shape is its data type, a `zml.DataType`.
// It is a plain Zig enum, and it has quite a few members:
//
//     .bool
//     .i8  .i16  .i32  .i64        signed integers
//     .u8  .u16  .u32  .u64        unsigned integers
//     .f16 .f32  .f64              IEEE floats
//     .bf16                        "brain float": 16 bits, but
//                                  with the exponent range of f32
//     .c64 .c128                   complex numbers
//     ...and a handful of exotic tiny floats (.f8e4m3, ...) used
//     for quantized models.
//
// Why so many? Because the choice matters a lot. A model with 8
// billion weights weighs 32 GB in .f32, 16 GB in .f16 or .bf16,
// and your graphics card has opinions about that.
//
// A DataType can answer questions:
//
//     const dt: zml.DataType = .i32;
//
//     dt.sizeOf()      -> 4          bytes per element
//     dt.bitSizeOf()   -> 32
//     dt.isFloat()     -> false
//     dt.isInteger()   -> true
//     dt.class()       -> .integer   (.bool, .float, .integer or .complex)
//
// and you can go back and forth between Zig types and DataTypes:
//
//     zml.DataType.fromZigType(u8)   -> .u8
//     zml.DataType.toZigType(.u8)    -> u8      (comptime only)
//
// Since DataType is an enum, `@tagName(dt)` gives you its name.
//
// Today's job: a crate of 1000 marbles. The customer swears the
// marbles are IEEE floating point numbers, and the crate must
// weigh exactly 2000 bytes or the truck tips over.
//
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    // Which data type makes this crate weigh 2000 bytes?
    const marble: zml.DataType = .f32;
    const crate = zml.Shape.init(.{1000}, marble);

    // The receipt has to state how heavy ONE marble is. There's a
    // method for that; multiplying by hand is how the last truck
    // tipped over.
    const one_marble: u16 = marble.???();

    try stdout.print("{d} marbles of {s} ({s}), {d} bytes each: {d} bytes.\n", .{
        crate.count(),
        @tagName(crate.dtype()),
        @tagName(marble.class()),
        one_marble,
        crate.byteSize(),
    });

    // The customer's own inventory software is written in Zig and
    // stores marble serial numbers as `u64`. Which DataType would
    // a crate of serial numbers have? Don't type it, compute it.
    const serial = zml.DataType.fromZigType(???);
    try stdout.print("Serial numbers are {s}: {d} bits.\n", .{ @tagName(serial), serial.bitSizeOf() });
    try stdout.flush();
}
