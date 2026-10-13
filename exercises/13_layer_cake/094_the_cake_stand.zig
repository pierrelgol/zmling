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
// mean... nothing at all. Same Exe, other Buffers. Below, one
// compiled cake is baked with the weekday recipe and then with
// the Sunday recipe, without recompiling anything.
//
// The program also prints the field names of the Bufferized
// structs, with `@typeInfo`, so you never have to take a
// comment's word for it. Believe the compiler, then delete
// what it tells you doesn't exist.
//
const std = @import("std");
const zml = @import("zml");

const Cake = struct {
    base: zml.nn.Linear,
    oven: zml.nn.Activation,
    level: zml.nn.LayerNorm,

    pub fn forward(self: Cake, x: zml.Tensor) zml.Tensor {
        const baked = self.oven.forward(self.base.forward(x, x.dtype()));
        return self.level.forward(baked);
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

    // Baked twice, with two sets of weights.
    for ([_][]const u8{ "weekday", "sunday " }, [_]zml.Buffer{ weekday_base, sunday_base }) |day, base_weight| {
        const cake_buffers: zml.Bufferized(Cake) = .{
            .base = .{
                .weight = base_weight,
                .tag = .d,
                .quantization = null,
            },
            .oven = .relu,
            .level = .{
                .weight = level_weight,
                .eps = 1e-5,
            },
        };

        var result: zml.Buffer = try exe.eval(gpa, io, .{ cake_buffers, x_buffer });
        defer result.deinit();

        try stdout.print("{s}", .{day});
        for (try result.getValue([3]f32, io)) |v| try stdout.print(" {d:6.3}", .{v});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
