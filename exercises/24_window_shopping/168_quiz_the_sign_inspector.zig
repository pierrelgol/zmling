//
// Quiz! The town hall has voted a by-law: every shop of the rue
// des Vitrines shall display, above its window, the letter of
// its trade category. L for laundry, 7 for "open 7 days", U for
// umbrellas, C for cheese, O for opticians. (Nobody knows who
// drew up the categories. Nobody dares to ask.)
//
// Monsieur Tatillon, the inspector, is supposed to walk down
// the street and check every sign. Monsieur Tatillon would
// rather not walk. He had a machine built instead: a small
// convolutional network, in the style of LeNet, the ancestor
// that read the digits on American cheques in the 1990s:
//
//     sign {.b, .h = 14, .w = 14}, 1 for paint, 0 for wall
//       -> one channel               {.b, .c = 1, .h = 14, .w = 14}
//       -> conv1, relu, max pool 2   {.b, .c = 2, .h =  6, .w =  6}
//       -> conv2, relu, max pool 2   {.b, .c = 2, .h =  2, .w =  2}
//       -> flatten                   {.b, .d = 8}
//       -> fc                        {.b, .dout = 5}
//       -> the best letter           {.b}
//
// Check the sizes with the formula of exercise 163 before you
// write a line: 14 - 3 + 1 = 12, halved, 6; 6 - 3 + 1 = 4,
// halved, 2.
//
// His weights are in `inspector.safetensors`, under the names
// PyTorch would give them, in the layouts PyTorch would use:
//
//     conv1.weight  {2, 1, 3, 3}    conv1.bias  {2}
//     conv2.weight  {2, 2, 3, 3}    conv2.bias  {2}
//     fc.weight     {5, 8}          fc.bias     {5}
//
// Nobody trained this network: Odette chose every weight by
// hand (`make_fixtures.py` has the story), so for once you can
// read a CNN like a recipe:
//
//  - conv1 has two STROKE DETECTORS: (-1, 2, -1) on each of
//    its three rows, and the same turned a quarter. Channel 0
//    lights up on vertical strokes one pixel wide, channel 1
//    on horizontal ones;
//  - conv2 counts, for each kind of stroke separately, how
//    much of it there is in a neighbourhood;
//  - after the second pooling, 2 x 2 pixels are left: the four
//    QUARTERS of the sign. So the 8 features are "vertical
//    strokes top-left, top-right, bottom-left, bottom-right",
//    then the same for horizontal strokes;
//  - fc holds one template per letter: +1 for a stroke the
//    letter expects, -1 for one it must not have. An L is
//    "vertical on the left, horizontal at the bottom".
//
// Three things PyTorch does without telling you, and that you
// must now do yourself:
//
//  1. A convolution has a BIAS: one number per OUTPUT channel,
//     added to every pixel of that channel. `conv2d` has no
//     bias option. Remember what exercise 164 found: the
//     output of `conv2d` keeps the tags of its input, so the
//     output channels are on the axis tagged `.c`. Tag the bias
//     accordingly, and `broad` (exercise 33) does the rest.
//
//  2. `x.flatten(1)` in PyTorch walks the axes in the order of
//     the tensor: channel, row, column. The columns of
//     `fc.weight` are in that order. `merge` (exercise 49)
//     glues neighbours in the order they have in YOUR tensor,
//     so it gives PyTorch's order only if your axes are still
//     (c, h, w) at that point. Had you kept the signs channels
//     last, (h, w, c), the merge of those three axes would
//     hand the Linear (row, column, channel). Nothing panics:
//     8 features are 8 features. The inspector just starts
//     taking umbrella shops for cheese shops.
//
//  3. `F.max_pool2d(x, 2)`: window 2, stride 2, and you want
//     the values (exercise 165).
//
// The rest is chapter 14, from memory: the registry, the store
// and its view; an `init(view)` per struct, where each struct
// hands a narrower view to its children; `bufferize`, a
// `Loader`, `load`, `await`; `platform.compile` on `.forward`;
// a runner, because the inspector returns three things:
//
//     Verdict.letter     {.b}, i32      the index of the best score
//     Verdict.scores     {.b, .dout}    the five scores
//     Verdict.strokes    {.b, .d}       the 8 features, flattened
//
// Write a `Conv` struct (weight, bias, `init`, `forward`), the
// `Inspector` (two Convs, one `zml.nn.Linear`, `init`,
// `forward`, and a `load` like the Gym's in exercise 102), and
// `main` up to the printing, which is given.
//
// Six signs are waiting. The last one was painted by Basile,
// the apprentice, on a ladder, on a windy day. Max pooling is
// the reason Monsieur Tatillon's machine does not care.
//
const std = @import("std");
const zml = @import("zml");

const weights_path = "zmling/24_window_shopping/inspector.safetensors";

const letters = "L7UCO";

const shops = [_][]const u8{
    "Aux Parapluies de Cherbourg",
    "Blanchisserie Moderne",
    "L'Epicerie du Coin",
    "Optique Lumiere",
    "La Cave a Fromages",
    "Pressing Basile et Fils",
};

const signs = [shops.len][14]*const [14:0]u8{
    .{
        "..............",
        "..............",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...########...",
        "..............",
        "..............",
    },
    .{
        "..............",
        "..............",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...########...",
        "..............",
        "..............",
    },
    .{
        "..............",
        "..............",
        "...########...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..........#...",
        "..............",
        "..............",
    },
    .{
        "..............",
        "..............",
        "...########...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...#......#...",
        "...########...",
        "..............",
        "..............",
    },
    .{
        "..............",
        "..............",
        "...########...",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...########...",
        "..............",
        "..............",
    },
    .{
        "..............",
        "..#...........",
        "..#...........",
        "..#...........",
        "..#...........",
        "..#...........",
        "..##..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#..........",
        "...#####......",
        ".......#####..",
        "..............",
    },
};

// TODO: `Conv`. Its weight, its bias, an `init` from a view, a
// `forward` from {.b, .c, .h, .w} to {.b, .c, .h, .w}.

// TODO: `Inspector`, with its `Verdict`, `init`, `forward` and
// `load`.

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // The signs, as numbers: 1 where there is paint.
    var pixels: [shops.len][14][14]f32 = undefined;
    for (signs, &pixels) |sign, *image| {
        for (sign, image) |row, *pixel_row| {
            for (row, pixel_row) |char, *pixel| pixel.* = if (char == '#') 1 else 0;
        }
    }

    // TODO: the file, its registry, its store.
    // TODO: the inspector, from a view.
    // TODO: the placeholder of the signs, {.b, .h, .w}: all the
    // shops at once.
    // TODO: the Exe.
    // TODO: the weights. Free them at the end.
    // TODO: `pixels`, as a Buffer.
    // TODO: the run. The three results go in `verdict`.

    try stdout.print("strokes {f}, scores {f}, letters {f}\n", .{
        verdict.strokes.shape(),
        verdict.scores.shape(),
        verdict.letter.shape(),
    });
    try stdout.print("{s:<28} | | | |  - - - -   reads    L   7   U   C   O\n", .{""});
    const all_strokes = try verdict.strokes.getValue([shops.len][8]f32, io);
    const all_scores = try verdict.scores.getValue([shops.len][5]f32, io);
    const all_letters = try verdict.letter.getValue([shops.len]i32, io);
    for (shops, all_strokes, all_scores, all_letters) |shop, strokes, scores, letter| {
        try stdout.print("{s:<28}", .{shop});
        for (strokes[0..4]) |s| try stdout.print(" {d}", .{s});
        try stdout.print(" ", .{});
        for (strokes[4..8]) |s| try stdout.print(" {d}", .{s});
        try stdout.print("     {c}   ", .{letters[@intCast(letter)]});
        for (scores) |s| try stdout.print(" {d:>3}", .{s});
        try stdout.print("\n", .{});
    }
    try stdout.flush();
}
