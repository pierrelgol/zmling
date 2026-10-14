//
// The trail goes into a snow cave. Ingrid stops at the mouth.
// "In there you only have what you carried in. And don't count
// on carrying anything out in your pockets."
//
// So far a program was one list of lines. `zml.ops.@"while"`
// (exercise 078) and `zml.ops.@"if"` (exercise 077) are ops
// whose operands are not only values but whole pieces of
// program: `cond` and `body`, `onTrue` and `onFalse`. MLIR
// calls such a piece a BLOCK, and the Compiler keeps a stack of
// the blocks it is writing:
//
//     scopes: stdx.BoundedArray(Scope, 16) = .empty,
//
//     pub const Scope = struct {
//         compiler: *Compiler,
//         block: *mlir.Block,
//         id_to_argument: std.AutoArrayHashMapUnmanaged(
//             Tensor.Id, *const mlir.Value,
//         ),
//         ...
//     };
//
// `emitMlir` pushes the first Scope, the block of your function,
// before calling it. Every op you call says
// `.appendTo(currentBlock())`: the block on TOP of the stack.
// Here is `ops.while` tracing your `body` (`cond` is the same):
//
//     const scope = comp.pushBlock(block);
//     defer scope.pop();
//
//     for (0.., flat_operands) |i, input| {
//         scope.registerTensorAsBlockArgument(input.id, i);
//     }
//
//     const result = captured_context.body(initial_state);
//
// A new block goes on the stack, your `body` is CALLED, once,
// and the block comes off. Whatever `body` records lands in the
// new block. And look at the loop in the middle: the Tensors of
// `initial_state` are registered as the arguments of the new
// block, under THEIR OWN IDS. Inside `body`, `state.depth` is
// "the depth at the start of this trip". Outside, the very same
// Tensor is "the depth before the loop". One id, two scopes,
// two meanings: `value()` asks the scope on top (exercise 191).
//
// Now the trap. Each Scope has its own `id_to_argument`. A
// weight of your model is an argument of the FUNCTION: it is
// in the table of scope 1 and `_value` is null. Use it inside
// `body` and `value()` looks in the table of scope 2, finds
// nothing, finds no pointer, and panics. That is what the
// `context` of `ops.while` and the captures of `ops.if` are
// for. Before pushing anything, both do this to every Tensor of
// the struct you pass:
//
//     fn capture(_: void, tensor: Tensor) Tensor {
//         return Tensor._result(tensor.shape(), tensor.value());
//     }
//
// `value()` is called while the OUTER scope is still on top,
// where the weight can be found, and the copy that your `body`
// receives as `self.something` carries a pointer. A captured
// Tensor works anywhere below.
//
// An honest footnote. The result of an op already carries a
// pointer, and MLIR lets a block read the values of the block
// around it. So an intermediate result used in `body` without
// being captured happens to work, where a weight or an input
// panics. Do not write code whose fate depends on which of the
// two you are holding. What `body` needs from outside goes in
// the context; what changes from trip to trip goes in the
// State; nothing else comes in.
//
// The depth of the stack is a plain number you can read:
//
//     zml.Compiler.current().scopes.len
//
// 1 in `forward`, 2 in a `body`, 3 in an `if` inside that
// body. The array holds 16 and `pushBlock` does not check
// (`appendAssumeCapacity`): sixteen blocks deep is a crash, and
// a sign that something else went wrong long before.
//
// The cave. Each print has a depth of snow, {.print}. Every day
// the sun takes `melt` off each print (a weight, {.print}: some
// prints are in the shade), and a depth never goes below zero.
// Except on the day of the storm: that day nothing melts and 10
// fall everywhere. How many days, and which day the storm is,
// arrive at run time: a graph loop with a graph `if` inside.
//
// Ingrid, again, left what the loop needs on a shelf.
//
// To do:
//
//  1. Run it and read the panic: who called `value`, from
//     which block? Then empty the shelf: `Thaw` and `Weather`
//     get the fields they need, `forward` and `body` fill
//     them.
//  2. `Weather.onFalse` lets a depth go negative.
//  3. `Weather.onTrue` and `onFalse` each note the depth of the
//     stack in `seen.branch` and count themselves in
//     `traced.branches`.
//  4. In `main`, the second run: 6 days, storm on day 2.
//
const std = @import("std");
const zml = @import("zml");

/// The depth of the scope stack, noted at trace time.
var seen: struct { forward: usize = 0, cond: usize = 0, body: usize = 0, branch: usize = 0 } = .{};
/// How many times each piece was traced.
var traced: struct { cond: usize = 0, body: usize = 0, branches: usize = 0 } = .{};

fn stackDepth() usize {
    return zml.Compiler.current().scopes.len;
}

/// Ingrid's shelf: filled by `Slope.forward`, before the loop.
var shelf: struct { melt: zml.Tensor, days: zml.Tensor, storm_day: zml.Tensor } = undefined;

/// One day: a storm, or the sun.
const Weather = struct {
    depth: zml.Tensor,

    /// The storm: 10 everywhere.
    pub fn onTrue(self: Weather) zml.Tensor {
        return self.depth.addConstant(10);
    }

    /// The sun: each print loses its `melt`, down to zero.
    pub fn onFalse(self: Weather) zml.Tensor {
        return self.depth.sub(shelf.melt);
    }
};

const Thaw = struct {
    pub const State = struct {
        day: zml.Tensor,
        depth: zml.Tensor,
    };

    pub fn cond(self: Thaw, state: State) zml.Tensor {
        seen.cond = stackDepth();
        traced.cond += 1;
        _ = self;
        return state.day.cmp(.LT, shelf.days);
    }

    pub fn body(self: Thaw, state: State) State {
        seen.body = stackDepth();
        traced.body += 1;

        _ = self;
        const is_storm = state.day.cmp(.EQ, shelf.storm_day);
        const depth = zml.ops.@"if"(Weather, .{ .depth = state.depth }, is_storm);
        return .{ .day = state.day.addConstant(1), .depth = depth };
    }
};

const Slope = struct {
    /// {.print}: what the sun takes from each print in a day.
    melt: zml.Tensor,

    /// depth: {.print} f32, days: {} i32, storm_day: {} i32 -> {.print} f32
    pub fn forward(self: Slope, depth: zml.Tensor, days: zml.Tensor, storm_day: zml.Tensor) zml.Tensor {
        seen.forward = stackDepth();

        shelf = .{ .melt = self.melt, .days = days, .storm_day = storm_day };

        const end = zml.ops.@"while"(
            Thaw,
            .{},
            .{ .day = zml.Tensor.scalar(0, .i32), .depth = depth },
        );
        return end.depth;
    }
};

const melt_data = [4]f32{ 2, 5, 4, 1 };
const depth_data = [4]f32{ 12, 30, 8, 20 };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const slope: Slope = .{ .melt = .init(.{ .print = 4 }, .f32) };
    const depth: zml.Tensor = .init(.{ .print = 4 }, .f32);
    const days: zml.Tensor = .init(.{}, .i32);
    const storm_day: zml.Tensor = .init(.{}, .i32);

    var exe = try platform.compile(gpa, io, slope, .forward, .{ depth, days, storm_day }, .{});
    defer exe.deinit();

    try stdout.print("the stack: forward {d}, cond {d}, body {d}, a branch {d}\n", .{ seen.forward, seen.cond, seen.body, seen.branch });
    try stdout.print("traced: cond {d} time, body {d} time, branches {d} times\n", .{ traced.cond, traced.body, traced.branches });

    var melt_buffer: zml.Buffer = try .fromBytes(io, platform, slope.melt.shape(), std.mem.sliceAsBytes(&melt_data));
    defer melt_buffer.deinit();
    const weights: zml.Bufferized(Slope) = .{ .melt = melt_buffer };

    var depth_buffer: zml.Buffer = try .fromBytes(io, platform, depth.shape(), std.mem.sliceAsBytes(&depth_data));
    defer depth_buffer.deinit();

    // 3 days of sun: the storm is on a day that never comes.
    {
        var days_buffer: zml.Buffer = try .scalar(io, platform, 3, .i32);
        defer days_buffer.deinit();
        var storm_buffer: zml.Buffer = try .scalar(io, platform, -1, .i32);
        defer storm_buffer.deinit();

        var left: zml.Buffer = try exe.eval(gpa, io, .{ weights, depth_buffer, days_buffer, storm_buffer });
        defer left.deinit();
        try stdout.print("3 days of sun:           {any}\n", .{try left.getValue([4]f32, io)});
    }

    // 6 days, and the storm on day 2 (the third day: days start at 0).
    // It ends with:
    //     try stdout.print("6 days, storm on day 2:  {any}\n", .{try left.getValue([4]f32, io)});

    try stdout.print("after 9 days in the cave, body was traced {d} time\n", .{traced.body});
    try stdout.flush();
}
