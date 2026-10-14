//
// Above the tree line there is no trail, only cairns: little
// piles of stones that say "you are on the way to the north
// col". Ingrid builds one at every fork. "For the day it goes
// wrong," she says. It goes wrong today.
//
// PART ONE: CAIRNS. Every line of the program can carry a
// LOCATION: where it comes from. The Compiler keeps the current
// one in a field, `location`, most ops stamp it on the line
// they record (`currentLoc()`), and you can change it:
//
//     compiler.pushLocation(@src(), "name")
//     compiler.pushLocationFmt(@src(), "layer[{d}]", .{i})
//     compiler.popLocation()
//
// A push "must be followed by a `defer compiler.popLocation`",
// says the source. ZML's own MNIST example:
//
//     const compiler = zml.Compiler.current();
//     for (0.., layers) |i, layer| {
//         compiler.pushLocationFmt(@src(), "layer[{d}]", .{i});
//         defer compiler.popLocation();
//         x = layer.forward(x);
//     }
//
// `@src()` is Zig's "this file, this line". A frame is that
// plus your name, and frames nest: "block, called from layer".
// `ops.while` and `ops.if` push frames of their own, with the
// `@typeName` of your struct (and `.cond`, `.body`).
//
// What for? A real program has thousands of lines called
// `%1742`. The locations travel with it into XLA, whose
// profiles and errors can say "layer[7]" instead. And when a
// compile fails, they are in the text ZML prints: part two.
//
// Two traps and one confession:
//
//   - A push without its pop does NOT panic. Every later line
//     is filed under a layer it has nothing to do with.
//   - A pop without a push panics: "unexpected location
//     metadata".
//   - At the commit you are using, the element-wise ops with
//     two operands (`add`, `mul`, `sub`...) are recorded with
//     an unknown location, whatever you pushed. A named layer
//     is a group of named lines with gaps.
//
// An exercise cannot read a name back, but it can ask whether
// one is standing: `compiler.location` is a pointer, and
// `compiler.unknown_location` is the "nowhere" it starts from.
//
// PART TWO: AVALANCHES. After tracing, `compileInternal` can
// fail in two places, and both call `handleCompilationError`:
//
//   "Failed to canonicalize invalid MLIR"
//       what you traced is not a valid program. A panic
//       follows: "ZML generated invalid mlir. Please open a
//       bug report".
//   "Pjrt failed to compile the following MLIR"
//       it is valid, and XLA could not compile it for this
//       device. That one is a Zig error, for your `try`.
//
// In both cases the whole program is printed on stderr, WITH
// its locations. It is the one moment ZML shows you what you
// wrote, so learn to read it. Today's, shortened:
//
//     "builtin.module"() <{sym_name = "zml"}> ({
//       "func.func"() <{...}> ({
//       ^bb0(%arg0: tensor<4xf32> ..., %arg1: tensor<4xf32> ...,
//            %arg2: tensor<i32> ...):
//         %0 = "stablehlo.constant"() <{value = dense<1> ...
//         ...
//         %10:2 = "stablehlo.while"(%1, %9) ({
//         ^bb0(%arg5: tensor<i32> ..., %arg6: tensor<4xf32> ...):
//           ...
//         }, {
//         ^bb0(%arg3: tensor<i32> ..., %arg4: tensor<4xf32> ...):
//           ...
//         }) : ...
//
// `%arg0`...: the arguments, weights first. `%0`, `%1`...: one
// per line, the `_value` of your Tensors. `^bb0(...)` opens a
// block with its own arguments (exercise 193: the State). The
// `while` holds two blocks between `({` and `})`: `cond`, then
// `body`. At the end of a line, its location; the `#loc` lines
// above and under the program give the name and the file.
//
// The classic avalanche is the other half of exercise 193. A
// Tensor made INSIDE a block is a line of that block. Smuggle
// it out (a global, a field) and use it after the loop: the
// function now reads a value that only exists inside the
// `while`, on a trip that may never happen. Tracing does not
// notice, `value()` finds a pointer. MLIR notices:
//
//     error: operand #1 does not dominate this use
//
// "Dominate" is compiler talk for "is certainly computed
// before". And the panic then tells you to open a bug report.
// Don't. This one is yours. What a block computes leaves
// through its door: the State of a `while`, the return value
// of a branch.
//
// (A healthy compile logs its program too, at debug level:
// `pub const std_options: std.Options = .{ .log_level = .debug };`
// in this file, and look at stderr.)
//
// The dump stops at 8192 bytes. For more, there is
// `zml.Compiler.Options`, the last argument of every compile
// call, that you have filled with `.{}` since chapter 02:
//
//     program_name = "zml"   the `sym_name` of the module.
//     xla_dump_to = null     a directory. XLA writes its own
//                            files there (three more
//                            `xla_dump_*` fields say which),
//                            and a dump that was cut goes
//                            there as `<program_name>.mlir`.
//
// The hike. A pack holds four kinds of food, {.item}. Three
// legs first, each with its cairn: on every leg the weight of
// the pack changes by a fixed factor (things fall out, snow
// gets in). Then camp: each day the hikers eat `ration` of each
// item, or what is left of it, for `days` days. Ingrid wants
// the pack on the MORNING of the last day: what is left at the
// end, plus what was eaten that day. `body` knows what was
// eaten. It put it in a pocket.
//
// To do:
//
//  1. Run it and read the avalanche, from the top. Which `%`
//     is used outside of its block? The names at the end of
//     the lines will look odd too: keep that for step 3.
//  2. Empty the pocket: what was eaten leaves through the
//     State. A State needs an initial value of the right shape
//     (`zml.Tensor.zeroes(shape)`).
//  3. The cairns: "back in the open" must say false.
//  4. The program has a name: "the_hike".
//
const std = @import("std");
const zml = @import("zml");

/// Noted at trace time: is a cairn standing?
var cairns: struct { on_the_legs: usize = 0, in_the_open: bool = true, in_camp: bool = false } = .{};

fn cairnStanding() bool {
    const compiler = zml.Compiler.current();
    return compiler.location != compiler.unknown_location;
}

/// What stays in the pack on each of the three legs.
const leg_factors = [3]f32{ 0.5, 0.5, 2 };

/// What was eaten on the last traced day. Filled by `Rations.body`.
var pocket: zml.Tensor = undefined;

const Rations = struct {
    ration: zml.Tensor,
    days: zml.Tensor,

    pub const State = struct {
        day: zml.Tensor,
        pack: zml.Tensor,
    };

    pub fn cond(self: Rations, state: State) zml.Tensor {
        return state.day.cmp(.LT, self.days);
    }

    pub fn body(self: Rations, state: State) State {
        cairns.in_camp = cairnStanding();

        const eaten = state.pack.minimum(self.ration);
        pocket = eaten;
        return .{
            .day = state.day.addConstant(1),
            .pack = state.pack.sub(eaten),
        };
    }
};

const Hike = struct {
    /// {.item}: what is eaten of each item in a day.
    ration: zml.Tensor,

    /// pack: {.item} f32, days: {} i32 -> {.item} f32
    pub fn forward(self: Hike, pack: zml.Tensor, days: zml.Tensor) zml.Tensor {
        const compiler = zml.Compiler.current();

        var x = pack;
        for (leg_factors, 0..) |factor, i| {
            compiler.pushLocationFmt(@src(), "leg[{d}]", .{i});

            if (cairnStanding()) cairns.on_the_legs += 1;
            x = x.scale(factor);
        }
        cairns.in_the_open = cairnStanding();

        const end = zml.ops.@"while"(
            Rations,
            .{ .ration = self.ration, .days = days },
            .{ .day = zml.Tensor.scalar(0, .i32), .pack = x },
        );
        return end.pack.add(pocket);
    }
};

const ration_data = [4]f32{ 1, 2, 0.5, 4 };
const pack_data = [4]f32{ 8, 8, 8, 8 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const hike: Hike = .{ .ration = .init(.{ .item = 4 }, .f32) };
    const pack: zml.Tensor = .init(.{ .item = 4 }, .f32);
    const days: zml.Tensor = .init(.{}, .i32);

    const options: zml.Compiler.Options = .{};
    var exe = try platform.compile(gpa, io, hike, .forward, .{ pack, days }, options);
    defer exe.deinit();

    try stdout.print("\"{s}\" compiled\n", .{options.program_name});
    try stdout.print("cairns: on {d} legs of 3, back in the open: {}, in camp: {}\n", .{ cairns.on_the_legs, cairns.in_the_open, cairns.in_camp });

    var ration_buffer: zml.Buffer = try .fromBytes(io, platform, hike.ration.shape(), std.mem.sliceAsBytes(&ration_data));
    defer ration_buffer.deinit();
    const weights: zml.Bufferized(Hike) = .{ .ration = ration_buffer };

    var pack_buffer: zml.Buffer = try .fromBytes(io, platform, pack.shape(), std.mem.sliceAsBytes(&pack_data));
    defer pack_buffer.deinit();

    for ([_]i32{ 2, 3, 5 }) |n| {
        var days_buffer: zml.Buffer = try .scalar(io, platform, n, .i32);
        defer days_buffer.deinit();

        var morning: zml.Buffer = try exe.eval(gpa, io, .{ weights, pack_buffer, days_buffer });
        defer morning.deinit();

        try stdout.print("the last morning of {d} days: {any}\n", .{ n, try morning.getValue([4]f32, io) });
    }
    try stdout.flush();
}
