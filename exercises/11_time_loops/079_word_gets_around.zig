//
// Loopsville has three streets and no secrets. Whatever Phil
// says at breakfast, every person who knows it tells a few
// others each day. On Main Street everyone tells 2 people, on
// Elm Street 3, on Mill Lane 1 (Mill Lane keeps to itself).
// After n days, how many people on each street have heard?
//
//     crowd = factor * factor * ... * factor      (n times)
//
// A power, by repeated multiplication, and a chance to put the
// two kinds of loop side by side.
//
// THE UNROLLED LOOP. `days` is a Zig number:
//
//     fn spread(factor: zml.Tensor, days: usize) zml.Tensor {
//         var crowd = ...ones...;
//         for (0..days) |_| crowd = crowd.mul(factor);
//         return crowd;
//     }
//
// THE GRAPH LOOP. `days` is a Tensor, and so we need
// `zml.ops.@"while"`. But look: the body must multiply by
// `factor`, and `factor` isn't part of the State. It never
// changes! This is what the `context` argument is for. The
// While struct may have FIELDS, and `cond` and `body` reach
// them through `self`:
//
//     const Rumor = struct {
//         factor: zml.Tensor,     // a tensor from outside the loop
//         limit: i32,             // a Zig value, baked in
//
//         pub const State = struct { ... };
//         pub fn cond(self: Rumor, state: State) zml.Tensor { ... }
//         pub fn body(self: Rumor, state: State) State { ... }
//     };
//
//     zml.ops.@"while"(Rumor, .{ .factor = factor, .limit = 9 }, ...)
//
// As with `zml.ops.@"if"`, the struct is the closure. A tensor
// from outside the loop gets in through a field. (Tensors you
// create inside `cond` or `body`, like `Tensor.scalar(0, .i32)`,
// are fine where they are.)
//
// So: State is what CHANGES, the fields are what STAYS.
//
// Which loop should you use? Both are right; they are for
// different things.
//
//   Zig `for`, unrolled      - the count is known when you
//                              compile: the 12 layers of a
//                              network, the 4 steps of a
//                              schedule.
//                            - each trip may be DIFFERENT: other
//                              weights, other shapes, an `if` on
//                              the loop index.
//                            - the compiler sees one flat graph
//                              and can optimize across the trips.
//                            - but the graph grows with the
//                              count, and the count is frozen.
//
//   `zml.ops.@"while"`       - the count depends on data: "until
//                              the model says stop", "until the
//                              error is small enough".
//                            - the graph stays small, whatever
//                              the count.
//                            - but every trip is the SAME code,
//                              and the State must keep its shape.
//
// In real models, the unrolled loop is the everyday one. Reach
// for the graph loop when the number of trips is genuinely not
// yours to know.
//
// About that last "but". ZML checks it while tracing: if `body`
// hands back a State whose shapes differ from the one it
// received, you get
//
//     zml.ops.while(...) expects body to preserve loop state
//     shape, got ... for operand 1 but expected ...
//
// Operands are the Tensors of the State, counted from 0 in field
// order. (The message has its two shapes the wrong way around:
// the first one is what went IN.)
//
// The task. `spreadLooped` should do the same as
// `spreadUnrolled`, with `days` as a run-time input.
//
//   1. `spreadUnrolled` starts its crowd well, and then the
//      rumor goes nowhere.
//   2. `Rumor` has lost its field, its `cond` and its `body`.
//      `days_left` counts down to zero, like Phil's piano days.
//   3. `main` never compiles `spreadLooped`, and never runs it.
//   4. With all that written, `spreadLooped` panics during
//      tracing. Read the message, then look at how the unrolled
//      version starts its crowd. (`x.broad(shape)` is from
//      chapter 05: it stretches x to the given shape.)
//
const std = @import("std");
const zml = @import("zml");

fn spreadUnrolled(factor: zml.Tensor, days: usize) zml.Tensor {
    // On day 0, one person per street knows: Phil.
    var crowd = zml.Tensor.scalar(1, .i32).broad(factor.shape());
    for (0..days) |_| crowd = ???;
    return crowd;
}

const Rumor = struct {
    // TODO: the field. What does every trip need, that no trip changes?

    pub const State = struct {
        days_left: zml.Tensor,
        crowd: zml.Tensor,
    };

    pub fn cond(self: Rumor, state: State) zml.Tensor {
        _ = self;
        return ???;
    }

    pub fn body(self: Rumor, state: State) State {
        // One day less to go, and the crowd grows on every street.
        return ???;
    }
};

fn spreadLooped(factor: zml.Tensor, days: zml.Tensor) zml.Tensor {
    const end = zml.ops.@"while"(Rumor, .{ .factor = factor }, .{
        .days_left = days,
        .crowd = zml.Tensor.scalar(1, .i32),
    });
    return end.crowd;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const factor: zml.Tensor = .init(.{ .streets = 3 }, .i32);
    const days: zml.Tensor = .init(.{}, .i32);

    var factor_buffer: zml.Buffer = try .fromBytes(io, platform, factor.shape(), std.mem.sliceAsBytes(&[_]i32{ 2, 3, 1 }));
    defer factor_buffer.deinit();

    // Unrolled: 5 days, and not a day more. One input.
    var unrolled_exe = try platform.compileFn(gpa, io, spreadUnrolled, .{ factor, 5 }, .{});
    defer unrolled_exe.deinit();

    var unrolled: zml.Buffer = try unrolled_exe.eval(gpa, io, .{factor_buffer});
    defer unrolled.deinit();
    try stdout.print("unrolled, {d} input,  5 days: {any}\n", .{ unrolled_exe.input_shapes.len, try unrolled.getValue([3]i32, io) });

    // Looped: the days are an input like any other. Two inputs.
    var looped_exe = ???;
    defer looped_exe.deinit();

    for ([_]i32{ 0, 5, 12 }) |n| {
        var days_buffer: zml.Buffer = ???;
        defer days_buffer.deinit();

        var looped: zml.Buffer = ???;
        defer looped.deinit();
        try stdout.print("looped,   {d} inputs, {d} days: {any}\n", .{ looped_exe.input_shapes.len, n, try looped.getValue([3]i32, io) });
    }
    try stdout.flush();
}
