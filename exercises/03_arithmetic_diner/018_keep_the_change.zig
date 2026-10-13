//
// Half the numbers in a model aren't tensors at all. "Divide by
// 2." "Square it." "Add 8% tax." They are plain constants, known
// when you write the code, and building a scalar placeholder plus
// a Buffer for each of them (like the `cap` of the previous
// exercise) would be silly.
//
// So every arithmetic operation has a form that takes an ordinary
// Zig number. You met two of them at the Manor; here is the set:
//
//     x.addConstant(c)    x + c
//     x.subConstant(c)    x - c
//     x.scale(c)          x * c
//     x.divByConst(c)     x / c
//     x.powByConst(c)     x ^ c
//     x.remainderConst(c) x mod c
//
// (No, the names don't follow a pattern. `scale`, `...Constant`,
// `...ByConst`: Gus named the specials the same way. You get used
// to it.)
//
// There is no magic in them. This is `divByConst`, straight from
// ZML's tensor.zig:
//
//     pub fn divByConst(self: Tensor, b: anytype) Tensor {
//         return self.div(.scalar(b, self.dtype()));
//     }
//
// `.scalar(b, dtype)` creates a rank 0 constant tensor inside the
// graph (more about it in a few exercises), and a rank 0 tensor
// goes with anything. Two consequences worth remembering:
//
//   1. The constant takes the dtype OF THE TENSOR. On an .f32
//      tensor, `scale(2)` multiplies by 2.0. No dtype trouble.
//
//   2. The constant is BAKED IN. It is part of the compiled Exe.
//      To change it you recompile. That is the price; the reward
//      is that the compiler knows the value and can optimize
//      around it.
//
// What you won't find is a "constant on the left" form. There is
// `x.subConstant(10)` for x - 10, and nothing for 10 - x. You
// rearrange: 10 - x is (-1 * x) + 10, so
// `x.scale(-1).addConstant(10)`.
//
// Now, pancakes. Flo prices them by AREA, because Gus pours them
// by diameter and "a 10 inch is not twice a 5 inch, hon". Her
// formula, for a diameter d in inches:
//
//     price = d^2 * 0.25        a quarter per square inch-ish
//     price = price - 1         Tuesday coupon
//     price = price / 2         it's also half-price hour
//     price = price + 0.5       coffee. Coffee is never half price.
//
// A chain of methods reads left to right, in the order things
// happen. The dishwasher typed this one up, and got creative.
// There are two problems: the compiler will find one for you.
// Only Flo will find the other.
//
const std = @import("std");
const zml = @import("zml");

/// Flo's pancake formula: ((d^2 * 0.25) - 1) / 2 + 0.5
fn pancakePrice(diameters: zml.Tensor) zml.Tensor {
    return diameters.scale(0.25).powByConst(2).subConstant(1).???(2).addConstant(0.5);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const diameters: zml.Tensor = .init(.{ .pancakes = 4 }, .f32);

    var exe = try platform.compileFn(gpa, io, pancakePrice, .{diameters}, .{});
    defer exe.deinit();

    // Silver dollar, regular, large, and "the Earl".
    var diameters_buffer: zml.Buffer = try .fromBytes(io, platform, diameters.shape(), std.mem.sliceAsBytes(&[_]f32{ 4, 6, 8, 10 }));
    defer diameters_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{diameters_buffer});
    defer result.deinit();

    try stdout.print("Pancake prices: {any}\n", .{try result.getValue([4]f32, io)});
    try stdout.flush();
}
