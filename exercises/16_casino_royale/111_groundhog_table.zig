//
// Inside ONE run of the graph, the deck is under control: each
// draw passes the new state to the next. But a casino is open
// all night. The Exe is called again, and again, and again...
// and what state does each call start from?
//
// The one you give it. An Exe remembers nothing between two
// calls. If every call receives the same state Buffer, every
// call deals the same cards. The players at table 7 have been
// living the same round since Tuesday. They call it the
// groundhog table. Some of them have stopped aging.
//
// The cure has two halves.
//
// HALF ONE, in the graph: the function must hand the new state
// back. It becomes one more output:
//
//     fn dealRound(rng: zml.Tensor.Rng) struct { zml.Tensor.Rng, zml.Tensor } {
//         const next_rng, const cards = rng.uniform(...);
//         return .{ next_rng, cards };
//     }
//
// A function may return a tuple or a struct: every Tensor in
// there (the one inside the Rng included) becomes an output of
// the Exe, in order.
//
// HALF TWO, in `main`: the new state must go back in at the
// next call. And here `exe.eval` leaves us:
//
//     panic: eval expects one output, got 2; use a runner for
//            multiple outputs
//
// Fair enough. The Runner, from chapter 12:
//
//     var runner = try exe.runner(gpa);
//     defer runner.deinit(gpa);
//
//     var hand: zml.Buffer = undefined;
//     runner.run(io, .{rng_buffer}, .{ &rng_buffer, &hand }, .{ .wait = true });
//     defer hand.deinit();
//
// `run` takes the inputs (a tuple of Buffers, like `eval`),
// then a tuple of POINTERS telling where to put each output,
// in the order the function returns them. `.wait = true` means
// "come back when the numbers are there".
//
// Now look at that line again. `rng_buffer` is on both sides:
// it goes in as the old state, and `&rng_buffer` is where the
// new state lands. After the call, the variable holds the next
// state, ready for the next round. That is the whole pattern:
//
//     state in, state out, into the same variable.
//
// It is not only tidy, it is REQUIRED, for a reason worth
// knowing. Producing a new state from an old one, a thousand
// times a second, would mean a thousand allocations a second.
// So `Rng` tells the compiler that the new state may REUSE THE
// MEMORY of the old one (`Tensor.reuseBuffer`, if you want to
// read about it). The input is "donated" to the Exe:
//
//     after the call, the old state Buffer is gone.
//
// The Runner destroys what is left of it. By pointing the
// output at the very same variable, you replace the dead Buffer
// with the live one in a single move, and there is never a
// moment where you could touch the corpse. You `deinit` it
// once, at the very end: by then it holds the last state.
//
// Every model that keeps something between two calls works
// this way. In ZML's LLM example, the loop that generates text
// threads exactly three things through each call: the last
// token, the cache of the attention layers, and the Rng.
//
// Table 7, then. To do:
//
//  1. `dealRound` promises a single Tensor. That is half of
//     the problem: fix what it returns, then write it. Five
//     cards, 1 to 13, like Monte's in the previous exercise.
//
//  2. In `main`: one state for the whole night, from the seed
//     1234 (and freed at the end of the night), a runner, and
//     three rounds, each one dealt from what the previous one
//     left. The cards of a round go into `round.*`.
//
// And let those poor people go home.
//
const std = @import("std");
const zml = @import("zml");

/// Deals five cards, each between 1 and 13, and returns the
/// state to deal the next round from.
fn dealRound(rng: zml.Tensor.Rng) zml.Tensor {
    const hand: zml.Shape = .init(.{ .card = 5 }, .f32);
    return ???;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const rng: zml.Tensor.Rng = .init();

    var exe = try platform.compileFn(gpa, io, dealRound, .{rng}, .{});
    defer exe.deinit();

    // One seed for the whole night, and a runner.
    ???;

    var rounds: [3][5]i32 = undefined;
    for (&rounds) |*round| {
        // Deal one round into `round.*`. State in, state out.
        ???;
    }

    try stdout.print("Round 2 is round 1 all over again: {}\n", .{std.mem.eql(i32, &rounds[0], &rounds[1])});
    try stdout.print("Round 3 is round 2 all over again: {}\n", .{std.mem.eql(i32, &rounds[1], &rounds[2])});
    try stdout.print("Round 3 is round 1 all over again: {}\n", .{std.mem.eql(i32, &rounds[0], &rounds[2])});
    try stdout.flush();
}
