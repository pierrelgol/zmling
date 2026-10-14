//
// Chapter 26: The Mail Room
//
// You have a diploma, and so Consolidated Tensor Inc. has given
// you a job that needs none: the MAIL ROOM, second basement,
// between the boiler and a vending machine that only takes
// francs. Your boss is Gus. Thirty-one years down here, never
// lost a parcel, lost several interns. Your colleague is Priya,
// who writes the labels, fast.
//
// The parcels are checkpoints. They come down the chute from
// the 14th floor (Research), from the annex, from overseas, and
// no two of them are packed the same way. Chapters 14 to 21
// taught you to open ONE file that you knew by heart. A program
// that people actually run, like `zml/examples/llm`, opens
// whatever it is handed: it reads its command line, finds the
// model, reads a config written by somebody else's library,
// collects weights spread over many files, and only then
// computes anything. This chapter is everything that happens
// before the first tensor.
//
// First thing on a parcel: the LABEL. For a program, that is
// its command line:
//
//     llm --model=hf://meta-llama/Llama-3.2-1B --topk=4 --profile
//
// ZML ships a small parser for those, borrowed from the
// TigerBeetle database: `zml.stdx.flags` (file `stdx/flags.zig`).
// You describe the label as a struct, and the struct IS the
// parser:
//
//     const Args = struct {
//         model: []const u8,           // --model=<text>, required
//         seqlen: u32 = 2048,          // --seqlen=<number>
//         prompt: ?[]const u8 = null,  // --prompt=<text>, or nothing
//         profile: bool = false,       // --profile, no value
//         positional: struct {         // what has no `--`
//             input: []const u8,       // <input>, required
//             count: u32 = 1,          // <count>, optional
//         },
//
//         pub const help = "usage: ...\n";
//     };
//
//     const args = zml.stdx.flags.parse(init.minimal.args, Args);
//
// (`init.minimal.args` is a `std.process.Args`: the command
// line as the operating system delivered it. `parse` skips the
// program's name itself.)
//
// The rules. They are strict on purpose, "make operator errors
// harder to make":
//
//  - One field, one flag. An underscore in the field is a dash
//    on the command line: `max_steps` is `--max-steps`.
//  - A value is glued with `=`. `--seqlen 64` is refused: is
//    64 the value, or a positional argument?
//  - A field without a default is REQUIRED. A flag given twice
//    is refused. A flag after a positional argument is refused.
//  - Types: integers, floats, `[]const u8`, enums (the value
//    is the name of a tag), `ByteSize` (below), and `?T` of
//    those. A `bool` is a flag WITHOUT value, and its default
//    must be `false`. A `?T` must default to `null`.
//    Those two are checked when your program compiles.
//  - The field named `positional` is special: a struct, whose
//    fields are filled in order. Its optional ones come last.
//  - `pub const help` is printed for `-h` and `--help`. Nothing
//    writes it for you: Zig cannot read your comments.
//  - A label that breaks a rule is not a Zig error to `try`.
//    `parse` prints one line on stderr and EXITS the program
//    with code 1. At this point there is nothing to clean up
//    and nobody to report to but the person at the keyboard.
//
// `zml.stdx.flags.ByteSize` is for sizes: `--chunk=64KiB`
// (also MiB, GiB, TiB, or a bare number of bytes). It keeps
// what was typed, `{ .value = 64, .unit = .kib }`, and
// `size.bytes()` does the multiplication.
//
// One practical problem. The zmling runner starts your program
// with NO arguments, always. So in this chapter the labels are
// written in the source: on Linux and macOS a `std.process.Args`
// is nothing but a list of C strings,
//
//     flags.parse(.{ .vector = &.{ "mailroom", "--fragile", "x" } }, Label)
//
// and `parse` cannot tell the difference. In your own programs,
// pass `init.minimal.args`. (And one more habit of
// `examples/llm`: `bazel run` starts a binary inside Bazel's
// runfiles tree, not where you typed the command. Bazel leaves
// the real directory in the environment variable
// BUILD_WORKING_DIRECTORY, and the example moves back there
// before it parses a single path.)
//
// Gus drops three parcels on your desk. "Priya did the labels.
// The machine reads them, so teach the machine." The help text
// below is the specification: Gus wrote it, it is right.
//
//  1. `Label` is missing three fields. Their names, types and
//     defaults are in the help text.
//  2. Then the compiler stops inside `stdx/flags.zig`. Read
//     what it says: it names a flag.
//  3. Then the program runs, and dies on the second label, and
//     after that on the third. Priya writes fast. The message
//     of `parse` tells you which rule she broke; fix the label,
//     not the parser.
//
const std = @import("std");
const zml = @import("zml");

const flags = zml.stdx.flags;

const stations = [_][]const u8{ "dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement" };

const Label = struct {
    // --max-steps, --chunk and --service go here.
    fragile: bool = true,
    positional: struct {
        repo: []const u8,
        start: u32 = 0,
    },

    pub const help =
        \\ mailroom [options] <repo> [<start>]
        \\
        \\ Routes a parcel with the model found in <repo>.
        \\
        \\   <repo>               the model repository (required)
        \\   <start>              number of the station to start from (default: 0)
        \\   --max-steps=<n>      give up after n stations (default: ask the config)
        \\   --chunk=<size>       size of a loading buffer (default: 4KiB)
        \\   --service=<name>     standard or express (default: standard)
        \\   --fragile            handle with care
        \\
    ;
};

fn describe(stdout: *std.Io.Writer, label: Label) !void {
    try stdout.print("  repo: {s}\n", .{label.positional.repo});
    try stdout.print("  start: {s}\n", .{stations[label.positional.start]});
    if (label.max_steps) |n| {
        try stdout.print("  max steps: {d}\n", .{n});
    } else {
        try stdout.print("  max steps: ask the config\n", .{});
    }
    try stdout.print("  chunk: {d} bytes\n", .{label.chunk.bytes()});
    try stdout.print("  service: {s}\n", .{@tagName(label.service)});
    try stdout.print("  fragile: {}\n", .{label.fragile});
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    // The real command line of this very program. The runner gives
    // it no arguments: all there is in it is the program's name.
    try stdout.print("arguments from the runner: {d}\n", .{init.minimal.args.vector.len - 1});

    try stdout.print("\nmailroom parcels/floor14\n", .{});
    try describe(stdout, flags.parse(.{ .vector = &.{ "mailroom", "parcels/floor14" } }, Label));
    try stdout.flush();

    try stdout.print("\nmailroom --max-steps=3 --chunk=64KiB parcels/overseas 2\n", .{});
    try describe(stdout, flags.parse(.{ .vector = &.{ "mailroom", "--max-steps=3", "--chunk", "64KiB", "parcels/overseas", "2" } }, Label));
    try stdout.flush();

    try stdout.print("\nmailroom --service=express --fragile parcels/annex\n", .{});
    try describe(stdout, flags.parse(.{ .vector = &.{ "mailroom", "parcels/annex", "--service=express", "--fragile" } }, Label));
    try stdout.flush();
}
