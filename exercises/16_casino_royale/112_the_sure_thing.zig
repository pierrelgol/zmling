//
// Why does a course about machine learning end in a casino?
// Because of the last step of every language model.
//
// A language model reads some text and, at the very end of all
// its layers, produces one number per word it knows: a SCORE
// for "this word comes next". Those scores are called LOGITS,
// and by convention the axis they live on is tagged `.voc`, for
// vocabulary:
//
//     logits   {.voc = 50000}     one score per known word
//
// That's it. The model does not answer "the next word is cat".
// It answers with fifty thousand scores, and somebody has to
// turn them into ONE word. This last step is called decoding,
// or sampling, and it is where the dice come in.
//
// Or don't. The simplest strategy uses no dice at all: take the
// word with the highest score. Always. It's called GREEDY
// decoding, and you could write it in chapter 6:
//
//     logits.argMax(.voc).indices.squeeze(.voc)
//
// (`argMax` returns `.values` and `.indices`, and keeps the
// axis with a size of 1; `squeeze` drops it. Other axes, a
// batch for instance, are left alone: each row gets its own
// winner.)
//
// ZML keeps all its decoding strategies behind one function,
// in zml/nn.zig:
//
//     pub const SamplingStrategy = struct {
//         topk: u32 = 1,
//         temperature: f32 = 1.0,
//     };
//
//     pub fn sampleTokens(
//         activations: Tensor,       // the logits
//         opts: SamplingStrategy,
//         rng: Tensor.Rng,
//     ) struct { Tensor, Tensor.Rng }
//
// "Token" is the proper name for what we've been calling a
// word: an entry of the vocabulary, known by its number. With
// the default strategy, `topk = 1`, the first lines of that
// function are:
//
//     if (opts.topk <= 1) {
//         const next_tokens = activations.argMax(.voc).indices.squeeze(.voc);
//         return .{ next_tokens, rng };
//     }
//
// Greedy, word for word. Three things to read off those lines:
//
//  1. The axis is not an argument. `sampleTokens` goes looking
//     for a tag named `.voc`, exactly like the `argMax(.voc)`
//     you would write by hand. Logits tagged any other way get
//     a panic about a missing tag, from whichever of the two is
//     traced first.
//
//  2. It returns the tokens FIRST and the Rng SECOND. Yes, that
//     is the opposite of `uniform`. No, nobody knows why. The
//     pointers you give to `runner.run` must follow the order
//     of the function you compiled, so keep an eye on it.
//
//  3. When greedy, the Rng comes back untouched. No dice were
//     rolled. An Rng must be passed all the same, so that you
//     can switch strategy without rewiring your program.
//
// Monte takes you to the back room, where the House keeps its
// private slot machine. Three reels, six symbols per reel, and
// a small model that scores every symbol of every reel. "The
// boss calls it the Sure Thing," says Monte. "Pull."
//
// You are to compute the outcome twice: once by hand, once with
// `sampleTokens`. The machine was last serviced by a nephew of
// the House.
//
const std = @import("std");
const zml = @import("zml");

/// Greedy decoding, by hand: {.reel, .voc} -> {.reel}
fn greedy(logits: zml.Tensor) zml.Tensor {
    return ???;
}

/// The same, the official way.
fn pull(logits: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
    return zml.nn.sampleTokens(logits, .{ .topk = 1 }, rng);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const symbols = [_][]const u8{ "cherry", "lemon", "bell", "bar", "seven", "skull" };

    // Three reels, one score per symbol.
    const logits: zml.Tensor = .init(.{ .reel = 3, .symbol = 6 }, .f32);
    const rng: zml.Tensor.Rng = .init();

    var logits_buffer: zml.Buffer = try .fromBytes(io, platform, logits.shape(), std.mem.sliceAsBytes(&[3][6]f32{
        // cherry, lemon, bell, bar, seven, skull
        .{ 0.3, -1.0, 0.5, 1.9, 2.0, -3.0 },
        .{ 1.0, 0.0, -0.5, 0.2, 4.5, 0.1 },
        .{ -2.0, 2.4, 0.0, 1.1, 2.5, 2.3 },
    }));
    defer logits_buffer.deinit();

    // By hand: one input, one output, good old `eval`.
    {
        var exe = try platform.compileFn(gpa, io, greedy, .{logits}, .{});
        defer exe.deinit();

        var result: zml.Buffer = try exe.eval(gpa, io, .{logits_buffer});
        defer result.deinit();

        try stdout.print("by hand:     ", .{});
        for (try result.getValue([3]i32, io)) |token| try stdout.print(" {s}", .{symbols[@intCast(token)]});
        try stdout.print("\n", .{});
    }

    // With sampleTokens: two outputs, hence a runner.
    {
        var exe = try platform.compileFn(gpa, io, pull, .{ logits, rng }, .{});
        defer exe.deinit();

        var rng_buffer = try rng.initBuffer(io, platform, 777);
        defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

        var runner = try exe.runner(gpa);
        defer runner.deinit(gpa);

        var result: zml.Buffer = undefined;
        runner.run(io, .{ logits_buffer, rng_buffer }, .{ &result, &rng_buffer }, .{ .wait = true });
        defer result.deinit();

        try stdout.print("sampleTokens:", .{});
        for (try result.getValue([3]i32, io)) |token| try stdout.print(" {s}", .{symbols[@intCast(token)]});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
