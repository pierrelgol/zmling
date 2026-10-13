//
// Greedy decoding has a problem, and the House's nephew found
// it: it is BORING. The same input always gives the same
// output. Ask a greedy chatbot for a joke and you get the same
// joke until the end of time; worse, greedy text tends to fall
// into loops, the same sentence again and again and again and
// again and again.
//
// So, the dice. Instead of always taking the best token, DRAW
// one, with chances that follow the scores: a token with a high
// logit comes out often, a token with a low one rarely, but not
// never. You know the function that turns scores into chances,
// it's the `softmax` of chapter 6:
//
//     p(token i)  =  exp(logit i) / (sum of exp(logit j), all j)
//
// Two knobs control how adventurous the draw is. They are the
// two fields of `zml.nn.SamplingStrategy`, and the two numbers
// behind every "creativity" slider you have ever seen on a
// chatbot.
//
// TEMPERATURE. Before the softmax, all logits are divided by T.
//
//     T = 1      the chances are what the model said
//     T < 1      gaps between scores GROW: the favorite gets
//                most of the chances. As T approaches 0, this
//                becomes greedy.
//     T > 1      gaps SHRINK: all candidates get closer to
//                equal chances. At T = 100 it's a lottery.
//
// TOP-K. Only the k best tokens are kept as candidates; the
// rest get no chance at all, whatever the temperature. This
// stops the one-in-a-million disaster token from ever being
// drawn, and with millions of draws a day, one-in-a-million
// happens before lunch. `topk = 1` is, again, greedy.
//
// `sampleTokens` does exactly that, in this order: `topK` over
// `.voc`, divide by the temperature, draw. (For the draw, it
// uses a famous trick: add "Gumbel noise" from `rng.gumbel` to
// each score and take the argMax. It is mathematically the same
// as drawing from the softmax, and it never has to compute an
// exponential. The comments in zml/nn.zig have the link.)
//
// Now dice ARE rolled, so the Rng that comes back is a NEW
// state. Thread it, like at the groundhog table: `&rng_buffer`
// goes in the outputs, at the place of the Rng. And remember
// the order: tokens first.
//
// A strategy is an ordinary Zig value, not a Tensor: it is
// BAKED INTO the Exe when you compile, like the `.f32` of a
// Linear. Another strategy means another Exe. So does the code
// below: one Exe per table.
//
// The roulette. Monte's wheel has six pockets, and (don't tell
// the gaming commission) a small model behind it that scores
// each pocket before every spin:
//
//     pocket   0     1     2     3     4     5
//     logit    0.5   3.0  -1.0   2.0   2.5   0.0
//
// Pocket 1 is the favorite, then 4, then 3. The House wants
// four tables, each with its own character, and a thousand
// test spins of each. The strategies are for you to set. (The
// thousand spins are what makes the report trustworthy: any
// single spin proves nothing. That is how you test a random
// thing.)
//
const std = @import("std");
const zml = @import("zml");

const Table = struct {
    name: []const u8,
    strategy: zml.nn.SamplingStrategy,
};

const spins = 1000;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const logits: zml.Tensor = .init(.{ .voc = 6 }, .f32);
    const rng: zml.Tensor.Rng = .init();

    var logits_buffer: zml.Buffer = try .fromBytes(io, platform, logits.shape(), std.mem.sliceAsBytes(&[_]f32{ 0.5, 3.0, -1.0, 2.0, 2.5, 0.0 }));
    defer logits_buffer.deinit();

    const tables = [_]Table{
        // No dice: the favorite, every time.
        .{ .name = "the sure thing", .strategy = .{ .topk = 6 } },
        // The three best pockets, with the chances the model gave them.
        .{ .name = "the honest table", .strategy = .{ .topk = 6, .temperature = 1.0 } },
        // The three best pockets are candidates, but so cold
        // that the favorite wins all the same.
        .{ .name = "the cold table", .strategy = .{ .topk = 3, .temperature = 100 } },
        // The three best pockets, so hot that the model's
        // opinion hardly counts: about a third each.
        .{ .name = "the hot table", .strategy = .{ .topk = 3, .temperature = 0.01 } },
    };

    // One seed for the whole test, one state threaded through
    // every spin of every table.
    var rng_buffer = try rng.initBuffer(io, platform, 2024);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    for (tables) |table| {
        // `sampleTokens` is a function of Tensors (and one plain
        // struct): it can be compiled as it is.
        var exe = try platform.compileFn(gpa, io, zml.nn.sampleTokens, .{ logits, table.strategy, rng }, .{});
        defer exe.deinit();

        var runner = try exe.runner(gpa);
        defer runner.deinit(gpa);

        const state_before = try rng_buffer._state.getValue(u128, io);

        // How many times each pocket came up.
        var counts: [6]u32 = @splat(0);
        for (0..spins) |_| {
            var pocket: zml.Buffer = undefined;
            runner.run(io, ???, ???, .{ .wait = true });
            defer pocket.deinit();

            counts[@intCast(try pocket.getValue(i32, io))] += 1;
        }

        const state_after = try rng_buffer._state.getValue(u128, io);

        // Facts that hold whatever the seed (give or take a
        // miracle). This is what testing a random thing looks like.
        try stdout.print("{s}\n", .{table.name});
        try stdout.print("  dice were rolled: {}\n", .{state_after != state_before});
        try stdout.print("  an outsider (pocket 0, 2 or 5) came up: {}\n", .{counts[0] + counts[2] + counts[5] > 0});
        try stdout.print("  the favorite came up every single time: {}\n", .{counts[1] == spins});
        try stdout.print("  pocket 3 got at least a quarter of the spins: {}\n", .{counts[3] >= spins / 4});
    }
    try stdout.flush();
}
