//
// Chapter 3: The Arithmetic Diner
//
// Madame Zelda has let you go ("the spirits find you... loud"),
// and the only place hiring is Flo's Diner, off Route 32. Open 24
// hours, bottomless coffee, and a sign in the window:
//
//     ELEMENT-WISE SERVICE ONLY. SAME SHAPE, SAME DTYPE,
//     NO EXCEPTIONS.              -- the management (Flo)
//
// The staff: Flo, who owns the place and does the books. Gus, the
// short-order cook, who has not left the griddle since 1987. And
// Earl, who isn't staff, but has been sitting on stool number 3
// for so long that he's on the insurance.
//
// You already met `add`, `sub`, `mul` and `maximum` at the Manor.
// Here is the whole family of BINARY ELEMENT-WISE operations:
//
//     a.add(b)        a + b
//     a.sub(b)        a - b
//     a.mul(b)        a * b
//     a.div(b)        a / b
//     a.pow(b)        a to the power of b
//     a.maximum(b)    the larger of the two
//     a.minimum(b)    the smaller of the two
//     a.remainder(b)  what is left of a after dividing by b
//
// "Element-wise" means the diner serves one plate at a time:
// element 0 of `a` meets element 0 of `b`, element 1 meets
// element 1, and nobody ever looks at their neighbour's plate.
// The result has the shape of the operands.
//
// If you come from linear algebra, careful: `a.mul(b)` is NOT a
// matrix product. It's plate-by-plate multiplication. (Matrix
// products get their own chapter.)
//
// They all obey the sign in the window:
//
//   - same dtype on both sides, or ZML panics while tracing;
//   - same shape on both sides, or ZML panics while tracing;
//   - except that a rank 0 tensor (a scalar) goes with anything.
//     (A later chapter, "Broadcast News", has the fine print.)
//
// One detail about `div`: on integer tensors it is an integer
// division, and it truncates toward zero, like Zig's `@divTrunc`.
// 7 / 2 is 3, and -7 / 2 is -3.
//
// And since each of these returns a Tensor, you can call the next
// one directly on the result. `a.mul(b).sub(c)` is (a * b) - c.
//
// Your first job: the tab board. Four regulars owe Flo money.
//
//   - A tab grows: every week, it is multiplied by that regular's
//     `interest`. After `weeks` weeks, that's interest ^ weeks.
//   - The regulars came in groups, and the tab is split evenly
//     between the `guests` who ran it up.
//   - Flo has a heart: nobody ever pays more than the `cap`.
//
const std = @import("std");
const zml = @import("zml");

/// What each guest owes: tab * interest^weeks / guests, but never
/// more than the cap.
fn settle(tab: zml.Tensor, interest: zml.Tensor, weeks: zml.Tensor, guests: zml.Tensor, cap: zml.Tensor) zml.Tensor {
    return tab.mul(interest.???(weeks)).???(guests).???(cap);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const regulars = zml.Shape.init(.{ .regulars = 4 }, .f32);

    const tab: zml.Tensor = .fromShape(regulars);
    const interest: zml.Tensor = .fromShape(regulars);
    const weeks: zml.Tensor = .fromShape(regulars);
    const guests: zml.Tensor = .fromShape(regulars);
    // One number for everybody: a scalar.
    const cap: zml.Tensor = .init(.{}, .f32);

    var exe = try platform.compileFn(gpa, io, settle, .{ tab, interest, weeks, guests, cap }, .{});
    defer exe.deinit();

    // Earl is the last one. Earl is always the last one.
    var tab_buffer: zml.Buffer = try .fromBytes(io, platform, regulars, std.mem.sliceAsBytes(&[_]f32{ 10, 20, 8, 100 }));
    defer tab_buffer.deinit();
    var interest_buffer: zml.Buffer = try .fromBytes(io, platform, regulars, std.mem.sliceAsBytes(&[_]f32{ 2, 1.5, 1, 1 }));
    defer interest_buffer.deinit();
    var weeks_buffer: zml.Buffer = try .fromBytes(io, platform, regulars, std.mem.sliceAsBytes(&[_]f32{ 3, 2, 9, 1 }));
    defer weeks_buffer.deinit();
    var guests_buffer: zml.Buffer = try .fromBytes(io, platform, regulars, std.mem.sliceAsBytes(&[_]f32{ 2, 1, 4, 2 }));
    defer guests_buffer.deinit();
    var cap_buffer: zml.Buffer = try .scalar(io, platform, 42, .f32);
    defer cap_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ tab_buffer, interest_buffer, weeks_buffer, guests_buffer, cap_buffer });
    defer result.deinit();

    try stdout.print("Each guest owes: {any}\n", .{try result.getValue([4]f32, io)});
    try stdout.flush();
}
