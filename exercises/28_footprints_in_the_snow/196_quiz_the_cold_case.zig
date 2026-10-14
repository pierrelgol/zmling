//
// Quiz. The cold case.
//
// Twenty winters ago a ranger wrote a program to tell how old a
// trail is, and left the valley. Ingrid found it in a drawer.
// "It worked the day he showed it to me," she says. "Once."
//
// What it is supposed to do. Each print has a depth of snow,
// {.print}. Every day the sun takes `melt` off each print (a
// weight, {.print}), down to zero and no further; a trail in
// the SHADE loses only half of that per day. `Case.forward`
// answers: what is left after `days` days? And a second,
// separate program, `visible`, counts the prints of a result
// that are still deeper than a threshold.
//
// The statements of the witnesses:
//
//   A. "It panics before anything is printed. Something went
//      really wrong, it says."
//   B. "I got it past that once, by tracing it for a trail in
//      the shade. I do not know why that helped. Then it fell
//      over further on, in the counting, with the very same
//      message."
//   C. "The day it ran, it gave the same answer for day 1 and
//      for day 6. I had changed `today`, I swear."
//   D. "Shade, sun: same numbers. I had changed `in_shade`
//      too."
//
// You have everything you need: exercise 191 (who can be found
// by `value()`, and where), 192 (photographs), 193 (what goes
// into a block, and the footnote about what gets in by
// accident), and from further back, a decision that depends on
// data is an op (`select`, exercise 076; a scalar bool is
// stretched with `broad` first, as `Tensor.mask` does).
//
// The repaired program has NO global Tensor and no global
// setting: the only global left is the tally `traced`. How many
// days, shade or sun, and the threshold all arrive at run time
// (the shade as an i32, 1 or 0: compare it). One Exe for the
// case, traced once, answers the four questions of `main`.
//
// To do: repair `Melt`, `Case`, `visible` and `main`. Do not
// touch the `print` lines: they are the proof of each repair.
// The first says how many inputs each program takes; the last,
// how often the case was traced.
//
const std = @import("std");
const zml = @import("zml");

/// How many times `Case.forward` ran.
var traced: usize = 0;

/// The ranger's settings. `main` changes them before each question.
var today: i32 = 1;
var in_shade: bool = false;

/// The ranger's shelf.
var shelf: struct { rate: zml.Tensor, threshold: zml.Tensor } = undefined;

const Melt = struct {
    pub const State = struct {
        day: zml.Tensor,
        depth: zml.Tensor,
    };

    pub fn cond(self: Melt, state: State) zml.Tensor {
        _ = self;
        return state.day.cmp(.LT, zml.Tensor.scalar(today, .i32));
    }

    pub fn body(self: Melt, state: State) State {
        _ = self;
        return .{
            .day = state.day.addConstant(1),
            .depth = state.depth.sub(shelf.rate).relu(),
        };
    }
};

const Case = struct {
    /// {.print}: what the sun takes from each print in a day.
    melt: zml.Tensor,

    /// depth: {.print} f32 -> {.print} f32
    pub fn forward(self: Case, depth: zml.Tensor) zml.Tensor {
        traced += 1;

        // In the shade, the sun takes half.
        var rate = self.melt;
        if (in_shade) rate = rate.scale(0.5);
        shelf.rate = rate;

        const end = zml.ops.@"while"(
            Melt,
            .{},
            .{ .day = zml.Tensor.scalar(0, .i32), .depth = depth },
        );
        return end.depth;
    }
};

/// left: {.print} f32 -> {} i32, the prints deeper than the threshold.
fn visible(left: zml.Tensor) zml.Tensor {
    return left.cmp(.GT, shelf.threshold.broad(left.shape())).convert(.i32).sum(.print).squeeze(.print);
}

const melt_data = [4]f32{ 2, 5, 4, 1 };
const depth_data = [4]f32{ 12, 30, 8, 20 };

const Question = struct { days: i32, shade: i32 };
const questions = [4]Question{
    .{ .days = 1, .shade = 0 },
    .{ .days = 6, .shade = 0 },
    .{ .days = 6, .shade = 1 },
    .{ .days = 3, .shade = 1 },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const case: Case = .{ .melt = .init(.{ .print = 4 }, .f32) };
    const depth: zml.Tensor = .init(.{ .print = 4 }, .f32);
    const threshold: zml.Tensor = .init(.{}, .f32);
    shelf.threshold = threshold;

    var case_exe = try platform.compile(gpa, io, case, .forward, .{depth}, .{ .program_name = "the_cold_case" });
    defer case_exe.deinit();
    var visible_exe = try platform.compileFn(gpa, io, visible, .{depth}, .{});
    defer visible_exe.deinit();

    try stdout.print("the case takes {d} inputs, the counting {d}\n", .{ case_exe.input_shapes.len, visible_exe.input_shapes.len });

    var melt_buffer: zml.Buffer = try .fromBytes(io, platform, case.melt.shape(), std.mem.sliceAsBytes(&melt_data));
    defer melt_buffer.deinit();
    const weights: zml.Bufferized(Case) = .{ .melt = melt_buffer };

    var depth_buffer: zml.Buffer = try .fromBytes(io, platform, depth.shape(), std.mem.sliceAsBytes(&depth_data));
    defer depth_buffer.deinit();
    var threshold_buffer: zml.Buffer = try .scalar(io, platform, 9.5, .f32);
    defer threshold_buffer.deinit();

    for (questions) |question| {
        today = question.days;
        in_shade = question.shade == 1;

        var left: zml.Buffer = try case_exe.eval(gpa, io, .{ weights, depth_buffer });
        defer left.deinit();
        var count: zml.Buffer = try visible_exe.eval(gpa, io, .{left});
        defer count.deinit();

        try stdout.print("day {d}, {s}: {any}, {d} deeper than 9.5\n", .{
            question.days,
            if (question.shade == 1) "shade" else "sun  ",
            try left.getValue([4]f32, io),
            try count.getValue(i32, io),
        });
    }

    try stdout.print("{d} questions, traced {d} time\n", .{ questions.len, traced });
    try stdout.flush();
}
