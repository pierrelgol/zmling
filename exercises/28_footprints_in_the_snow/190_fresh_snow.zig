//
// The Col du Fantome, 2400 m. Ingrid has tracked the Beast of
// this mountain for forty winters and has never seen it. "You
// don't see it," she says, handing you snowshoes. "It walks
// while the snow falls, and the snow only falls while you are
// not looking. What you get is the trail."
//
// You have followed that trail since chapter 02 without asking
// who made it. Exercise 012: an op outside a compile call
// panics in `Compiler.current`. Exercise 075: your function
// runs ONCE, at trace time, and never again. This chapter opens
// `zml/zml/Compiler.zig` and reads how. Nothing here is magic,
// and it is about sixty lines.
//
// `platform.compileFn(gpa, io, func, args, opts)` is one line:
// it calls `zml.Compiler.compile`, which calls
// `compileInternal`. That one does, in this order:
//
//     var compiler: Compiler = .init(allocator, io, platform, opts);
//     defer compiler.deinit();
//
//     var result = compiler.emitMlir(func, args) ...  // TRACING
//     ...
//     compiler.mlir_pass_manager.runOnOp(...)         // tidy up
//     compiler.compileModuleToPjrtExecutable(opts)    // XLA
//     const exe = try Exe.init(...);
//     return exe;
//
// A Compiler is born, your function is traced into a program,
// the program is cleaned ("canonicalize", "cse": remove the
// duplicates), XLA turns it into machine code, and the Compiler
// DIES. `defer compiler.deinit()`. The Exe that comes back
// holds the machine code and the shapes. No Compiler, no
// program text, and nothing of your Zig function.
//
// And tracing itself, in `emitMlir`:
//
//     var input_info = try createBlockArguments(
//         compiler, fn_scope, &args,
//     );
//
//     var result = blk: {
//         compiler.activate();
//         defer compiler.deactivate();
//         break :blk @call(.auto, func, args);
//     };
//
// First the Tensors found in `args` are declared as the
// arguments of the program (chapter 27 showed how they are
// found). Then, three lines that explain half of ZML:
// `activate` stores the Compiler in
//
//     threadlocal var _current: ?*Compiler = null;
//
// your function is CALLED, a plain Zig call, and `deactivate`
// puts `null` back. That variable is the whole mechanism.
// `x.add(y)` takes no compiler parameter because it asks
// `Compiler.current()`, which is `currentOrNull().?`. Inside a
// compile there is one. Outside, `.?` on null: "attempt to use
// null value", the panic of exercise 012. Both are public:
//
//     zml.Compiler.currentOrNull()   ?*Compiler, never panics
//     zml.Compiler.current()         *Compiler, or the panic
//
// (`threadlocal`: one per thread. Two threads can each compile
// their own model at the same moment. One thread cannot compile
// from inside a `forward`: `activate` asserts that the slot is
// empty.)
//
// So there are not two times, but THREE, and every mistake you
// ever made in this course belongs to exactly one of them:
//
//   COMPTIME    Zig compiles your program. The args tuple of a
//               compile call is checked against the parameters
//               of the function: its type is
//               `std.meta.ArgsTuple(@TypeOf(func))`. One
//               argument too many is a Zig error, before
//               anything runs.
//   TRACE TIME  the snow falls. Your function runs once, a
//               Compiler is current, Tensors have shapes and no
//               values. Shape mistakes panic HERE, with a
//               stack trace that goes through `emitMlir`.
//   RUN TIME    `exe.eval`. The trail is walked, as often as
//               you like. Your function is not called, no
//               Compiler is current. The only thing left to
//               check is that the Buffers have the promised
//               shapes.
//
// The trail of the day: at each step the Beast climbs
// `stride * slope` meters, and `Trail.forward` returns the
// total. `slope` is a weight, `stride` an input. Ingrid wants
// proof of the three times, so `forward` keeps two tallies in
// plain Zig variables: how often it ran, and how often it saw
// a Compiler when it did.
//
// To do:
//
//  1. `Trail.forward`: both tallies, then the climb. `mul`,
//     and a `sum` that leaves a scalar (chapter 06).
//  2. `main` has one mistake per time. Zig finds the first.
//     The second is a panic: read its stack trace and find
//     `emitMlir` in it. The third is a panic too: there is no
//     `emitMlir` in that one, and no `forward` either.
//  3. The second compile, for a trail of 6 steps. Same model
//     type, same function: only the placeholders change.
//
const std = @import("std");
const zml = @import("zml");

/// How many times `Trail.forward` ran.
var passes: usize = 0;
/// How many of those times a Compiler was current.
var sightings: usize = 0;

const Trail = struct {
    /// {.step}: meters climbed per meter walked.
    slope: zml.Tensor,

    /// stride: {.step}, meters walked at each step -> {} meters climbed.
    pub fn forward(self: Trail, stride: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const slope_data = [4]f32{ 0.5, 1, 2, 0.25 };
const strides = [3][4]f32{
    .{ 2, 2, 2, 2 },
    .{ 4, 1, 1, 4 },
    .{ 0, 8, 0, 0 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    try stdout.print("before the snow, a Compiler on the mountain: {}\n", .{zml.Compiler.currentOrNull() != null});

    const trail: Trail = .{ .slope = .init(.{ .step = 4 }, .f32) };
    const stride: zml.Tensor = .init(.{ .step = 5 }, .f32);

    var exe = try platform.compile(gpa, io, trail, .forward, .{ stride, stride }, .{});
    defer exe.deinit();

    try stdout.print("after the snow,  a Compiler on the mountain: {}\n", .{zml.Compiler.currentOrNull() != null});
    try stdout.print("1 compile: forward ran {d} time, and saw the Compiler {d} time\n", .{ passes, sightings });

    var slope_buffer: zml.Buffer = try .fromBytes(io, platform, trail.slope.shape(), std.mem.sliceAsBytes(&slope_data));
    defer slope_buffer.deinit();
    const weights: zml.Bufferized(Trail) = .{ .slope = slope_buffer };

    for (strides, 1..) |walked, hiker| {
        var stride_buffer: zml.Buffer = try .fromBytes(io, platform, .init(.{ .step = 2, .half = 2 }, .f32), std.mem.sliceAsBytes(&walked));
        defer stride_buffer.deinit();

        var climbed: zml.Buffer = try exe.eval(gpa, io, .{ weights, stride_buffer });
        defer climbed.deinit();

        try stdout.print("  hiker {d} climbs {d} m. forward ran {d} time\n", .{ hiker, try climbed.getValue(f32, io), passes });
    }

    // The same function, traced again for a longer trail.
    var long_exe = ???;
    defer long_exe.deinit();

    try stdout.print("2 compiles: forward ran {d} times, and saw the Compiler {d} times\n", .{ passes, sightings });
    try stdout.print("the trails: {f} and {f}\n", .{ exe.input_shapes[1], long_exe.input_shapes[1] });
    try stdout.flush();
}
