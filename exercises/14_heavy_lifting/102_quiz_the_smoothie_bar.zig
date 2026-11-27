//
// Quiz! After the workout, the smoothie bar. (The smoothie bar
// is where the Iron Tensor makes its money. The lifting is a
// pretext.)
//
// Behind the counter stands the BLENDTRON, a machine with three
// blenders in a row. You tell it how many cups of banana and of
// spinach you want; it answers with two numbers: the price, in
// dollars, and how good the thing will taste, from 0 up.
//
// Nobody at the gym knows HOW it decides. Denise only knows
// that its weights are in the same locker as everything else,
// under "bar":
//
//     bar.0.weight  bar.0.bias     2 numbers in, 2 numbers out
//     bar.1.weight  bar.1.bias     2 in, 2 out
//     bar.2.weight  bar.2.bias     2 in, 2 out
//
// and the manual, which is a napkin, says:
//
//     blender 0, relu, blender 1, relu, blender 2. NO relu at
//     the end. Do not put your hand in blender 1.
//
// That's a model nobody understands, with weights in a file,
// and a note about the architecture. Welcome to machine
// learning!
//
// Denise has an idea, and Denise's ideas are not suggestions.
// The gym gets an app. A member walks in, the app looks at
// their lifts and at their smoothie order, and answers with
// both the Coach's verdict and the receipt. ONE model for the
// whole gym, loaded in ONE go, with two inputs and two outputs:
//
//     Gym
//       coach   the Coach of exercise 101, under "coach":
//                   coach.0   3 lifts -> 2 hunches (no bias), relu
//                   coach.1   2 hunches -> 3 scores (with a bias)
//                   and the index of the highest score
//       bar     the Blendtron, under "bar"
//
// The struct tree mirrors the names of the file: that is the
// whole trick, and this time you draw the tree. Three model
// structs, each with an `init` that takes a view (and hands
// narrower views to its children) and a `forward`:
//
//     Coach.forward       lifts {.member, .d} -> {.member}, i32
//     Blendtron.forward   cups {.member, .d} -> {.member, .dout}
//     Gym.forward         lifts and cups -> a struct with both:
//                         the printing calls them `verdict` and
//                         `receipt`
//
// plus `linear`, the helper that makes a `zml.nn.Linear` out of
// the "weight" and (if there is one!) the "bias" under a view,
// and something that loads a whole Gym from the store.
//
// Notes from Denise, who has watched people get this wrong:
//
//  - the three blenders are three FIELDS of the Blendtron, one
//    per layer number. Tempting as it is, do not make them an
//    array `[3]zml.nn.Linear`: `zml.mem.bufferize` does not walk
//    through arrays (in zml/core/mem.zig they end up in a branch
//    marked `unreachable`, and it means it);
//  - a `forward` may take several Tensors. They are listed in
//    the same order three times: in its signature, in the tuple
//    given to `platform.compile`, and (after the weights) in
//    the tuple given to the runner;
//  - only ONE struct needs to know how to load itself: the one
//    at the top. `bufferize` and the Loader walk down the whole
//    tree, Coach and Blendtron included;
//  - build it in the order of the tree, leaves first: `linear`,
//    `Coach` (you have it: exercise 101), `Blendtron`, `Gym`,
//    and only then `main`.
//
// Everything you need, you have used in this chapter or the one
// before:
//
//     TensorRegistry.fromPath     the header of the file
//     TensorStore.fromRegistry    the key cabinet, and its `view()`
//     withPrefix / withLayer      where a view stands
//     hasKey                      look before you leap
//     createReplicatedTensor      a placeholder with an address
//     zml.mem.bufferize           the empty bar
//     zml.io.Loader               init, load, await, deinit
//     platform.compile            shapes are enough
//     exe.runner, runner.run      weights first, then the inputs;
//                                 two outputs do not `eval`
//     zml.Buffer.deinitAll        a struct of Buffers, freed at
//                                 once (nested structs and
//                                 optional Buffers included)
//
// The path of the file, the members, and the printing are given.
// The rest of the page is blank, and it is yours.
//
// Three members are waiting. One of them ordered nothing at
// all, and will be charged for it.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/14_heavy_lifting/gym.safetensors";

// TODO: `linear`. A view goes in, a `zml.nn.Linear` comes out.

// TODO: `Coach`, `Blendtron` and `Gym`.

const members = [_][]const u8{ "Olga", "Chad", "You" };

//                            squat, bench, deadlift
const member_lifts = [3][3]f32{
    .{ 2.0, 1.5, 2.5 }, // Olga
    .{ 0.5, 1.5, 1.0 }, // Chad
    .{ 0.5, 0.4, 0.6 }, // You (it's your second week)
};

//                           banana, spinach
const member_cups = [3][2]f32{
    .{ 2, 1 }, // Olga: the Classic
    .{ 1, 3 }, // Chad: the Regret
    .{ 0, 0 }, // You: "just a glass, please"
};

const verdicts = [_][]const u8{
    "DO YOU EVEN LIFT?",
    "LIGHT WEIGHT, BABY!",
    "NEVER SKIP LEG DAY.",
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // TODO: the real address of the file; the logbook; the key
    // cabinet.
    // TODO: the gym, from a view. Mind where that view stands.
    // TODO: the placeholders of the two inputs: three members.
    // TODO: the Exe. Shapes are enough.
    // TODO: the heavy lifting. Free it all at the end.
    // TODO: the lifts and the cups of the three members, as Buffers.
    // TODO: the run. The two results go in `visit`.

    try stdout.print("verdicts {f}, receipts {f}\n", .{ visit.verdict.shape(), visit.receipt.shape() });
    const receipts = try visit.receipt.getValue([3][2]f32, io);
    for (members, try visit.verdict.getValue([3]i32, io), member_cups, receipts) |member, verdict, order, receipt| {
        try stdout.print("  {s:<4} {s:<19}  {d} banana, {d} spinach: ${d:.2}, taste {d}\n", .{
            member,
            verdicts[@intCast(verdict)],
            order[0],
            order[1],
            receipt[0],
            receipt[1],
        });
    }
    try stdout.flush();
}
