//
// The zmling runner.
//
// Adapted from Ziglings' `rivendell/elrond.zig` (MIT, https://codeberg.org/ziglings/exercises).
//
// ZML can only be built with Bazel, so instead of `zig run` this program:
//   1. points the symlink `zml/zmling` at the exercise tree, which makes every
//      chapter directory a Bazel package of the ZML workspace,
//   2. runs `bazel build //zmling/<chapter>:<exercise>` (compiler errors go
//      straight to your terminal),
//   3. runs the binary with `bazel run`, and compares its stdout with the
//      expected output from the table at the bottom of this file.
//
// Exit codes:
//   0  all requested exercises passed
//   1  an exercise failed (compile error, output mismatch, runtime error)

const std = @import("std");

const Io = std.Io;
const Process = std.process;
const print = std.debug.print;
const cutPrefix = std.mem.cutPrefix;

const progress_filename = ".progress.txt";

pub const logo =
    \\                     _ _
    \\     ____ _ __ ___ | (_)_ __   __ _
    \\    |_  /| '_ ` _ \| | | '_ \ / _` |
    \\     / / | | | | | | | | | | | (_| |
    \\    /___||_| |_| |_|_|_|_| |_|\__, |
    \\                              |___/
    \\
    \\    "Look out! Broken tensors below!"
    \\
    \\
;

const Mode = enum { normal, named, start };

pub const Exercise = struct {
    // Path below exercises/, format: chapter_dir/key_name.zig
    main_file: []const u8,

    // Desired stdout. A program passes if its stdout, excluding trailing whitespace, equals this string.
    output: []const u8,

    // Optional hint shown if the program does not succeed.
    hint: ?[]const u8 = null,

    pub fn chapter(self: Exercise) []const u8 {
        return std.fs.path.dirname(self.main_file).?;
    }

    // Name of the main file with directory and .zig stripped. This is also the Bazel target name.
    pub fn name(self: Exercise) []const u8 {
        return std.fs.path.stem(self.main_file);
    }

    // "01_shape_shifters/001_hello.zig" -> 1
    pub fn number(self: Exercise) usize {
        const base = self.name();
        const end_index = std.mem.indexOfScalar(u8, base, '_') orelse unreachable;
        return std.fmt.parseInt(usize, base[0..end_index], 10) catch unreachable;
    }
};

const Context = struct {
    io: std.Io,
    arena: std.mem.Allocator,
    bazel_exe: []const u8,
    zml_path: []const u8,
};

const Error = error{Failed};

const C = struct {
    var red_bold: []const u8 = "\x1b[31;1m";
    var green: []const u8 = "\x1b[32m";
    var yellow: []const u8 = "\x1b[33m";
    var bold: []const u8 = "\x1b[1m";
    var reset: []const u8 = "\x1b[0m";
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    var args_it = try init.minimal.args.iterateAllocator(arena);
    if (!args_it.skip()) @panic("expected self arg");

    const NO_COLOR = try init.minimal.environ.containsUnempty(arena, "NO_COLOR");
    const CLICOLOR_FORCE = try init.minimal.environ.containsUnempty(arena, "CLICOLOR_FORCE");
    if (try Io.Terminal.Mode.detect(io, Io.File.stderr(), NO_COLOR, CLICOLOR_FORCE) != .escape_codes) {
        C.red_bold = "";
        C.green = "";
        C.yellow = "";
        C.bold = "";
        C.reset = "";
    }

    if (!validateExercises()) std.process.exit(1);

    var bazel_exe: []const u8 = "bazel";
    var work_path: []const u8 = "exercises";
    var root_path: []const u8 = ".";
    var mode: Mode = .normal;
    var n_arg: usize = 0;

    while (args_it.next()) |arg| {
        if (cutPrefix(u8, arg, "--bazel=")) |v| {
            bazel_exe = v;
        } else if (cutPrefix(u8, arg, "--work-path=")) |v| {
            work_path = v;
        } else if (cutPrefix(u8, arg, "--root-path=")) |v| {
            root_path = v;
        } else if (cutPrefix(u8, arg, "--only=")) |v| {
            n_arg = parseNumber(v);
            mode = .named;
        } else if (cutPrefix(u8, arg, "--start=")) |v| {
            n_arg = parseNumber(v);
            mode = .start;
        } else {
            print("unknown argument: {s}\n", .{arg});
            std.process.exit(2);
        }
    }

    print("{s}{s}{s}", .{ C.yellow, logo, C.reset });

    const ctx: Context = .{
        .io = io,
        .arena = arena,
        .bazel_exe = bazel_exe,
        .zml_path = try std.fs.path.join(arena, &.{ root_path, "zml" }),
    };

    linkExercises(ctx, work_path) catch |err| {
        print(
            \\{s}error:{s} unable to create the symlink {s}/zmling: {s}
            \\Did you forget the ZML submodule?  Try: git submodule update --init
            \\
        , .{ C.red_bold, C.reset, ctx.zml_path, @errorName(err) });
        std.process.exit(1);
    };

    switch (mode) {
        .named => runOne(ctx, exercises[n_arg - 1], .named) catch std.process.exit(1),
        .start => iterateFrom(ctx, n_arg - 1) catch std.process.exit(1),
        .normal => {
            // Start after the last solved exercise recorded in .progress.txt.
            const solved = readProgress(io, arena);
            const start_index = for (exercises, 0..) |ex, idx| {
                if (solved < ex.number()) break idx;
            } else {
                print("{s}All exercises completed!{s}\n", .{ C.green, C.reset });
                return;
            };
            iterateFrom(ctx, start_index) catch std.process.exit(1);
        },
    }
}

fn parseNumber(v: []const u8) usize {
    const n = std.fmt.parseInt(usize, v, 10) catch 0;
    if (n == 0 or n > exercises.len) {
        print("unknown exercise number: {s}\n", .{v});
        std.process.exit(1);
    }
    return n;
}

// Makes `<zml>/zmling` a symlink to `../<work_path>`, so that Bazel sees the
// exercises as packages of the ZML workspace.
fn linkExercises(ctx: Context, work_path: []const u8) !void {
    const dir = try Io.Dir.cwd().openDir(ctx.io, ctx.zml_path, .{});
    defer dir.close(ctx.io);
    const target = try std.fs.path.join(ctx.arena, &.{ "..", work_path });

    var buf: [Io.Dir.max_path_bytes]u8 = undefined;
    if (dir.readLink(ctx.io, "zmling", &buf)) |len| {
        if (std.mem.eql(u8, buf[0..len], target)) return;
        try dir.deleteFile(ctx.io, "zmling");
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }
    try dir.symLink(ctx.io, target, "zmling", .{ .is_directory = true });
}

// Iterates exercises from `start_index` to the end, stopping at the first failure.
fn iterateFrom(ctx: Context, start_index: usize) Error!void {
    for (exercises[start_index..]) |ex| {
        try runOne(ctx, ex, .normal);
    }
}

fn runOne(ctx: Context, ex: Exercise, mode: Mode) Error!void {
    printProgress(ex.number(), exercises.len);

    runExe(ctx, ex) catch {
        if (ex.hint) |hint|
            print("\n{s}{s}zmling hint: {s}{s}", .{ C.bold, C.green, hint, C.reset });
        print("\n{s}Edit exercises/{s} and run '{s}' again.{s}\n", .{
            C.red_bold, ex.main_file, if (mode == .named) "zig build -Dn=..." else "zig build", C.reset,
        });
        return Error.Failed;
    };

    writeProgress(ctx.io, ctx.arena, ex.number()) catch {};
}

fn printProgress(num: usize, max: usize) void {
    const bar_width = 60;
    const filled_len = (num * bar_width) / max;

    var bar_buf: [bar_width]u8 = undefined;
    for (0..bar_width) |n| {
        bar_buf[n] = switch (std.math.order(n, filled_len)) {
            .lt => '#',
            .eq => '>',
            .gt => '-',
        };
    }
    print("\rProgress: [{s}]  {d}/{d}\n\n", .{ &bar_buf, num, max });
}

fn runExe(ctx: Context, ex: Exercise) !void {
    const io = ctx.io;
    const arena = ctx.arena;
    const target = std.fmt.allocPrint(arena, "//zmling/{s}:{s}", .{ ex.chapter(), ex.name() }) catch @panic("OOM");

    print("Compiling {s}...\n", .{ex.main_file});

    // Build with inherited stdio: the Zig compiler errors are the lesson, and
    // the very first build of ZML takes a while, so progress must be visible.
    var build = Process.spawn(io, .{
        .argv = &.{ ctx.bazel_exe, "build", "--ui_event_filters=-info", "--show_result=0", target },
        .cwd = .{ .path = ctx.zml_path },
    }) catch |err| {
        print("{s}error:{s} unable to run '{s}': {s}\n", .{ C.red_bold, C.reset, ctx.bazel_exe, @errorName(err) });
        return err;
    };
    const build_term = try build.wait(io);
    if (!build_term.success()) return Error.Failed;

    print("Checking {s}...\n", .{ex.main_file});

    const result = try Process.run(arena, io, .{
        .argv = &.{ ctx.bazel_exe, "run", "--ui_event_filters=-info,-stdout,-stderr", "--noshow_progress", target },
        .cwd = .{ .path = ctx.zml_path },
    });

    if (!result.term.success()) {
        // Runtime failure: usually a ZML assert while tracing the graph. Its message is on stderr.
        const diag = std.mem.trimEnd(u8, result.stderr, " \r\n");
        if (diag.len > 0) print("{s}\n", .{diag});
        return Error.Failed;
    }

    const output = trimLines(arena, result.stdout) catch @panic("OOM");
    if (!std.mem.eql(u8, output, ex.output)) {
        print(
            \\
            \\{s}========= expected this output: =========={s}
            \\{s}
            \\{s}========= but found: ====================={s}
            \\{s}
            \\{s}=========================================={s}
        ++ "\n", .{ C.red_bold, C.reset, ex.output, C.red_bold, C.reset, output, C.red_bold, C.reset });
        return Error.Failed;
    }

    print("{s}PASSED:\n{s}{s}\n\n", .{ C.green, output, C.reset });
}

// Removes trailing whitespace per line and any trailing LF at the end.
fn trimLines(arena: std.mem.Allocator, buf: []const u8) ![]const u8 {
    var list = try std.ArrayList(u8).initCapacity(arena, buf.len + 1);
    var iter = std.mem.splitScalar(u8, buf, '\n');
    while (iter.next()) |line| {
        list.appendSliceAssumeCapacity(std.mem.trimEnd(u8, line, " \r"));
        list.appendAssumeCapacity('\n');
    }
    return std.mem.trimEnd(u8, list.items, "\n");
}

// Reads the last solved exercise number from .progress.txt; 0 if absent.
fn readProgress(io: std.Io, arena: std.mem.Allocator) usize {
    const contents = Io.Dir.cwd().readFileAlloc(io, progress_filename, arena, .limited(64)) catch return 0;
    return std.fmt.parseInt(usize, std.mem.trim(u8, contents, " \r\n"), 10) catch 0;
}

fn writeProgress(io: std.Io, arena: std.mem.Allocator, number: usize) !void {
    const progress = try std.fmt.allocPrint(arena, "{d}", .{number});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = progress_filename, .data = progress });
}

// Checks that exercise numbers form the sequence 1..N and that output lines
// have no trailing whitespace (it would be trimmed from the program's output).
fn validateExercises() bool {
    for (exercises, 1..) |ex, i| {
        if (ex.number() != i) {
            print("exercise {s} has an incorrect number: expected {}\n", .{ ex.main_file, i });
            return false;
        }
        var iter = std.mem.splitScalar(u8, ex.output, '\n');
        while (iter.next()) |line| {
            if (std.mem.trimEnd(u8, line, " \r").len != line.len) {
                print("exercise {s} output field lines have trailing whitespace\n", .{ex.main_file});
                return false;
            }
        }
    }
    return true;
}

test trimLines {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("a\n\nb", try trimLines(arena.allocator(), "a  \n\nb \r\n\n"));
}

test validateExercises {
    try std.testing.expect(validateExercises());
}

const exercises = [_]Exercise{
    // Chapter 01: Shape Shifters
    .{
        .main_file = "01_shape_shifters/001_hello_shape.zig",
        .output = "The sofa has 2 axes, 6 cushions, and weighs 24 bytes.",
        .hint =
        \\DON'T PANIC!
        \\Read the compiler messages above, then open the source file
        \\noted below and read the comments. All of them. They are the
        \\course.
        \\
        ,
    },
    .{
        .main_file = "01_shape_shifters/002_piano_dims.zig",
        .output = "88 keys, 2 hammers each. Full manifest: { 3, 88, 2 }",
        .hint = "Axes are numbered from 0. A rank 3 shape has no axis 3.\n",
    },
    .{
        .main_file = "01_shape_shifters/003_whats_in_the_box.zig",
        .output =
        \\1000 marbles of f16 (float), 2 bytes each: 2000 bytes.
        \\Serial numbers are u64: 64 bits.
        ,
    },
    .{
        .main_file = "01_shape_shifters/004_the_lonely_sock.zig",
        .output =
        \\sock:          rank 0, count 1, dims {  }
        \\paranoid sock: rank 3, count 1, dims { 1, 1, 1 }
        \\Same thing? false
        \\Is the sock a scalar? true
        ,
    },
    .{
        .main_file = "01_shape_shifters/005_label_your_boxes.zig",
        .output = "{shelves=5,eggs=12,u8}: 12 eggs per shelf, shelves are axis 0, axis 1 is called 'eggs'.",
        .hint = "The first line of the panic names the shape and the tag it could not find.\n",
    },
    .{
        .main_file = "01_shape_shifters/006_repacking.zig",
        .output = "cupboard {15,8,u16}, dishwasher {20,6,u16}, insurance {120,u16}",
    },
    .{
        .main_file = "01_shape_shifters/007_truck_tetris.zig",
        .output =
        \\Loaded: {layers=2,rows=4,boxes=5,u8}, 40 boxes in total.
        \\Aftermath: {layers=2,rows=4,i64}.
        ,
        .hint = "Shapes are values: `truck = truck.someMethod(...)`.\n",
    },
    .{
        .main_file = "01_shape_shifters/008_quiz_moving_day.zig",
        .output =
        \\archive:    {n=32,h=480,w=640,c=3,u8}, 28 MiB
        \\studio:     {n=32,c=3,h=480,w=640,u8}
        \\floats:     {n=32,c=3,h=480,w=640,f32}, 112 MiB
        \\thumbnails: {n=32,h=48,w=64,f32}
        \\cabinet:    {n=32,pixels=3072,f32}, 3072 pixels per photo
        ,
    },
};
