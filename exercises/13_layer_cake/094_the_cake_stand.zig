//
// Time to put layers ON TOP of each other, which is, after all,
// the business model of this bakery.
//
// A model is a struct whose fields are layers, and layers are
// structs whose fields are Tensors. Structs in structs: nothing
// new since chapter 12. Here is a small cake:
//
//     const Cake = struct {
//         base: zml.nn.Linear,
//         oven: zml.nn.Activation,
//         level: zml.nn.LayerNorm,
//
//         pub fn forward(self: Cake, x: zml.Tensor) zml.Tensor { ... }
//     };
//
// To run it you need a `zml.Bufferized(Cake)`. For three
// exercises now, things have been quietly disappearing from
// those. Time to look at the conjuring trick properly. The doc
// comment of `Bufferized` in zml/mem.zig says:
//
//     Return a clone of a type with Tensors replaced by Buffer.
//     Non-Tensor metadata is stripped out of the resulting
//     struct. Recursively descends into the type.
//
// "Stripped out" means GONE: the field does not exist anymore.
// Side by side:
//
//     Cake                            zml.Bufferized(Cake)
//     ----                            --------------------
//     base: nn.Linear                 base: struct
//       weight: Tensor                  weight: Buffer
//       bias: ?Tensor = null            bias: ?Buffer = null
//       tag: Shape.Tag                  -
//       quantization: ?Quantization     quantization: ?(...)
//     oven: nn.Activation             -
//     level: nn.LayerNorm             level: struct
//       weight: Tensor                  weight: Buffer
//       bias: ?Tensor = null            bias: ?Buffer = null
//       eps: f32 = 1e-5                 -
//
// The rule: a field survives if there is a Tensor somewhere
// inside it. `tag` and `eps` have none. `oven` has none, so the
// whole union vanishes. Two details worth a raised eyebrow:
//
//  - an optional Tensor keeps its `= null` default, so a layer
//    without bias needs no `.bias` line;
//  - `quantization` survives (a Quantization holds Tensors) but
//    loses its default on the way. This is why you have had to
//    write `.quantization = null` all along. Sorry about that.
//
// Now the WHY, which matters more than the list. Tracing and
// running are two different moments:
//
//     while TRACING    `forward` runs once, on ghosts. The tag
//                      picks the axis, the union picks the
//                      activation, eps is written into the
//                      graph as a constant. All of it is frozen
//                      into the Exe.
//
//     while RUNNING    nobody calls `forward`. The Exe wants its
//                      data: one Buffer per Tensor. Nothing
//                      else could possibly be of use to it.
//
// So: a different eps means a new Exe, and different WEIGHTS
// mean... nothing at all. Same Exe, other Buffers. And the other
// way around: a different OVEN means a new Exe, fed with the
// very same Buffers. You will do both:
//
//  1. Write `Cake.forward`: the base, the oven, then the level.
//     (A Linear wants to know the dtype of its output: the one
//     of its input will do.)
//
//  2. One compiled cake, baked with the weekday recipe and then
//     with the Sunday recipe. The weekday Buffers were filled in
//     by somebody who copied the model field by field. Believe
//     the compiler, delete what it tells you doesn't exist, and
//     add what it tells you is missing. Then write the Sunday
//     ones, and the run inside the loop.
//
//  3. Monsieur Perte wants the Sunday cake again, from a gentler
//     oven: a leaky relu with a slope of 0.5. Everything else
//     stays as it is. Decide what must be made again and what
//     can be reused. (A struct is a value: `var copy = cake;`
//     and then change one field, as you did with Scally's doubt
//     in exercise 088.)
//
// The program starts by printing the field names of the
// Bufferized structs, with `@typeInfo`, so you never have to
// take a comment's word for it.
//
const std = @import("std");
const zml = @import("zml");

const Cake = struct {
    base: zml.nn.Linear,
    oven: zml.nn.Activation,
    level: zml.nn.LayerNorm,

    pub fn forward(self: Cake, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

fn printFields(stdout: *std.Io.Writer, comptime name: []const u8, comptime T: type) !void {
    try stdout.print("{s}:", .{name});
    inline for (@typeInfo(T).@"struct".field_names) |field_name| try stdout.print(" {s}", .{field_name});
    try stdout.print("\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    try printFields(stdout, "Cake", Cake);
    try printFields(stdout, "Bufferized(Cake)", zml.Bufferized(Cake));
    try printFields(stdout, "Bufferized(Linear)", zml.Bufferized(zml.nn.Linear));
    try printFields(stdout, "Bufferized(LayerNorm)", zml.Bufferized(zml.nn.LayerNorm));

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const cake: Cake = .{
        .base = .init(.init(.{ .dout = 3, .d = 2 }, .f32), null, .d),
        .oven = .relu,
        .level = .{ .weight = .init(.{ .d = 3 }, .f32), .eps = 1e-5 },
    };
    const x: zml.Tensor = .init(.{ .d = 2 }, .f32);

    // Compiled once.
    var exe = try platform.compile(gpa, io, cake, .forward, .{x}, .{});
    defer exe.deinit();

    var weekday_base: zml.Buffer = try .fromBytes(io, platform, cake.base.weight.shape(), std.mem.sliceAsBytes(&[3][2]f32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 } }));
    defer weekday_base.deinit();
    var sunday_base: zml.Buffer = try .fromBytes(io, platform, cake.base.weight.shape(), std.mem.sliceAsBytes(&[3][2]f32{ .{ 2, 2 }, .{ -1, 0 }, .{ 0, 3 } }));
    defer sunday_base.deinit();
    var level_weight: zml.Buffer = try .fromBytes(io, platform, cake.level.weight.shape(), std.mem.sliceAsBytes(&[3]f32{ 1, 1, 1 }));
    defer level_weight.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&[2]f32{ 1, 3 }));
    defer x_buffer.deinit();

    const weekday: zml.Bufferized(Cake) = .{
        .base = .{
            .weight = weekday_base,
            .tag = .d,
        },
        .oven = .relu,
        .level = .{
            .weight = level_weight,
            .eps = 1e-5,
        },
    };
    // The same, with the Sunday base. The level does not change.
    const sunday: zml.Bufferized(Cake) = ???;

    // Baked twice, with two sets of weights.
    for ([_][]const u8{ "weekday", "sunday " }, [_]zml.Bufferized(Cake){ weekday, sunday }) |day, cake_buffers| {
        var result: zml.Buffer = ???;
        defer result.deinit();

        try stdout.print("{s}", .{day});
        for (try result.getValue([3]f32, io)) |v| try stdout.print(" {d:6.3}", .{v});
        try stdout.print("\n", .{});
    }

    // TODO: the gentler oven. Whatever you make here, free it. The
    // result goes in `gentle`, a zml.Buffer.

    try stdout.print("sunday, gentle oven", .{});
    for (try gentle.getValue([3]f32, io)) |v| try stdout.print(" {d:6.3}", .{v});
    try stdout.print("\n", .{});
    try stdout.flush();
}
