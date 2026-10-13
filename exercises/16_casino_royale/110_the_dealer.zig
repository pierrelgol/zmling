//
// Chapter 16: Casino Royale
//
// One more stop. You have shapes, ghosts, layers, attention, and a
// gym membership you will never use again. One thing is still
// missing before you can build a machine that talks: LUCK.
//
// Welcome to the Casino Royale, where the carpets are loud so
// the customers don't hear the math. You are the new assistant
// of Monte, the oldest dealer on the floor. (His sister Carla
// works at the other casino. Monte and Carla. Their parents
// thought it was funny.) Above you both is the House, which you
// will never meet, and which always wins.
//
// Here is the dirty secret of every casino and every computer:
// there is no luck in the building. A computer can only follow
// instructions. What it calls "random numbers" comes out of a
// very ordinary function:
//
//     (state)  ->  (some numbers, a NEW state)
//
// The numbers LOOK like chaos. But feed the same state in, and
// the same numbers come out, every time. The trick is to never
// feed the same state twice: use the new one for the next draw,
// then the newer one, and so on. Like a deck of cards: once a
// card is dealt, you deal the next one from what is LEFT of the
// deck, not from a fresh copy of it.
//
// In ordinary Zig, `std.Random` hides the deck inside a struct
// and quietly mutates it at every call. A ZML graph cannot do
// that. A graph is a recipe, a pure function of its inputs: it
// has no drawer in which to keep a deck between two runs. So in
// ZML the deck is out on the table, where everyone can see it:
//
//     zml.Tensor.Rng
//
// An `Rng` is a small struct holding one Tensor, the state (two
// u64, if you must know). It behaves like everything else in a
// graph:
//
//     const rng: zml.Tensor.Rng = .init();    // a PLACEHOLDER for a state
//
//     // inside the compiled function:
//     const next_rng, const numbers = rng.uniform(shape, .{ .min = 0, .max = 1 });
//
// `uniform` returns TWO things, as a Zig tuple: the new state,
// and a tensor of `shape` filled with floats between `min`
// (included) and `max` (excluded), all equally likely. The
// defaults are 0 and 1. The `const a, const b = ...` syntax is
// plain Zig destructuring; write `_` for a part you don't want.
//
// The doc comment of the function underneath it all, in
// zml/tensor.zig, says it plainly:
//
//     /// The given Rng state should not be used anymore
//     /// (or you'll get the same numbers again).
//
// At run time, the placeholder needs a body like any other.
// The body of a state is made from a SEED, any number you like:
//
//     var rng_buffer = try rng.initBuffer(io, platform, 1234);   // seed: u128
//     defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);
//
// `rng_buffer` is a `zml.Bufferized(zml.Tensor.Rng)`: an Rng is
// a struct of Tensors, so it gets Buffers the same way a model
// does, and it is passed to the Exe the same way too.
//
// Same seed, same state, same numbers. That is not a flaw, it's
// the feature: a result you can reproduce is a result you can
// debug. (Which numbers come out of a given seed depends on the
// device and on the version of the backend, though. So this
// chapter never prints a random number. It prints facts ABOUT
// them, and those hold everywhere.)
//
// Monte deals two hands of five cards, valued 1 (ace) to 13
// (king), from an endless shoe. Tonight the players at table 4
// are complaining: both hands are always the same. Monte blames
// the shoe. The shoe is fine.
//
// To do:
//
//  1. In `deal`, the two draws are written, and one of them is
//     the reason for the complaints. The end of the function is
//     missing: both hands in ONE tensor, stacked along a new
//     `.hand` axis the way you boxed pizzas in exercise 57, and
//     turned into honest cards. A float in [1, 14), rounded
//     down, is 1, 2, ... 13: `floor` (exercise 19) rounds,
//     and `convert` (exercise 21) makes the i32.
//
//  2. `main` prints its report, and nothing else: the
//     placeholder, the Exe, and what happens on each of the
//     three evenings are for you. An evening is: a state made
//     from the seed (and freed), one run, and the cards copied
//     into `evening.*` as a `[2][5]i32`.
//
const std = @import("std");
const zml = @import("zml");

/// Deals two hands of five cards: {.hand = 2, .card = 5}, i32,
/// every card between 1 and 13.
fn deal(rng: zml.Tensor.Rng) zml.Tensor {
    const hand: zml.Shape = .init(.{ .card = 5 }, .f32);

    // A float in [1, 14), rounded down: 1, 2, ... 13.
    _, const first = rng.uniform(hand, .{ .min = 1, .max = 14 });
    _, const second = rng.uniform(hand, .{ .min = 1, .max = 14 });

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

    // `rng`, a placeholder for the state of the generator, and
    // the Exe of `deal`.
    ???;

    // Three evenings at table 4. The first two start from the
    // same seed, the third from another one.
    var evenings: [3][2][5]i32 = undefined;
    for ([_]u128{ 1234, 1234, 777 }, &evenings) |seed, *evening| {
        // A state from `seed`, one run, and the cards in `evening.*`.
        ???;
    }

    const first_hand, const second_hand = evenings[0];
    var all_legal = true;
    for (first_hand ++ second_hand) |card| {
        if (card < 1 or card > 13) all_legal = false;
    }

    try stdout.print("Every card is between 1 and 13: {}\n", .{all_legal});
    try stdout.print("The two hands are the same: {}\n", .{std.mem.eql(i32, &first_hand, &second_hand)});
    try stdout.print("Same seed, same evening: {}\n", .{std.meta.eql(evenings[0], evenings[1])});
    try stdout.print("Another seed, same evening: {}\n", .{std.meta.eql(evenings[0], evenings[2])});
    try stdout.flush();
}
