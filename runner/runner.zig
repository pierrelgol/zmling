//
// The zmling runner.
//
// Adapted from Ziglings' `rivendell/elrond.zig` (MIT, https://codeberg.org/ziglings/exercises).
//
// ZML can only be built with Bazel, so instead of `zig run` this program:
//   1. points the symlink `zml/zmling` at the exercise tree, which makes every
//      chapter directory a Bazel package of the ZML workspace,
//   2. runs `bazel build //zmling/<chapter>:<number>` (compiler errors go
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

    // Name of the main file with directory and .zig stripped.
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
    // Binaries are named by number only, see exercises/harness/defs.bzl.
    const target = std.fmt.allocPrint(arena, "//zmling/{s}:{d:0>3}", .{ ex.chapter(), ex.number() }) catch @panic("OOM");

    print("Compiling {s}...\n", .{ex.main_file});

    // Build with inherited stdio: the Zig compiler errors are the lesson, and
    // the very first build of ZML takes a while, so progress must be visible.
    var build = Process.spawn(io, .{
        // `--disk_cache=` keeps the ~1 GB binary out of a disk cache, if you have one configured.
        .argv = &.{ ctx.bazel_exe, "build", "--disk_cache=", "--experimental_check_output_files", "--ui_event_filters=-info", "--show_result=0", target },
        .cwd = .{ .path = ctx.zml_path },
    }) catch |err| {
        print("{s}error:{s} unable to run '{s}': {s}\n", .{ C.red_bold, C.reset, ctx.bazel_exe, @errorName(err) });
        return err;
    };
    const build_term = try build.wait(io);
    if (!build_term.success()) return Error.Failed;

    print("Checking {s}...\n", .{ex.main_file});

    const result = try Process.run(arena, io, .{
        .argv = &.{ ctx.bazel_exe, "run", "--disk_cache=", "--experimental_check_output_files", "--ui_event_filters=-info,-stdout,-stderr", "--noshow_progress", target },
        .cwd = .{ .path = ctx.zml_path },
    });

    // The binary links all of ZML statically and weighs about 1 GB. Bazel
    // rebuilds it on demand (that's what --experimental_check_output_files is
    // for), so don't keep a hundred of them around.
    const binary = std.fmt.allocPrint(arena, "{s}/bazel-bin/zmling/{s}/{d:0>3}", .{ ctx.zml_path, ex.chapter(), ex.number() }) catch @panic("OOM");
    Io.Dir.cwd().deleteFile(io, binary) catch {};

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

    // Chapter 02: Ghost Tensors
    .{
        .main_file = "02_ghost_tensors/009_the_ghost.zig",
        .output =
        \\Tensor({rooms=13,candles=7,f32}) haunts 13 rooms and 91 candle flames in total.
        \\It would weigh 364 bytes, if it weighed anything.
        ,
    },
    .{
        .main_file = "02_ghost_tensors/010_the_seance.zig",
        .output = "The seance takes 1 input {candles=7,f32} and returns 1 output {candles=7,f32}.",
    },
    .{
        .main_file = "02_ghost_tensors/011_ectoplasm.zig",
        .output = "The flames rise: { 2, 4, 6, 8, 10, 12, 14 }",
        .hint = "The panic message has both byte counts. How big is an f32?\n",
    },
    .{
        .main_file = "02_ghost_tensors/012_do_not_touch.zig",
        .output = "A chill runs down your spine: { 7, 18, 29 }",
        .hint =
        \\Skip the first lines of the stack trace and look for the last
        \\frame that is in YOUR file: that's the ghost being touched.
        \\
        ,
    },
    .{
        .main_file = "02_ghost_tensors/013_a_slice_of_afterlife.zig",
        .output = "Result is a {ghosts=5,f32} slice. Happy birthday: 101 201 301 401 501",
        .hint = "Nothing crashed, and that's the scary part. `items(T)` believes you.\n",
    },
    .{
        .main_file = "02_ghost_tensors/014_one_knock_for_yes.zig",
        .output =
        \\volume 0.5: { 0.5, 1, 1.5 }
        \\volume 1: { 1, 2, 3 }
        \\volume 10: { 10, 20, 30 }
        ,
        .hint = "A sock in a box is not a sock.\n",
    },
    .{
        .main_file = "02_ghost_tensors/015_house_rules.zig",
        .output = "Spookiest, hour by hour: { 5, 9, 4, 8 }",
    },
    .{
        .main_file = "02_ghost_tensors/016_quiz_cold_spots.zig",
        .output =
        \\intensity 1: -40.0 32.0 50.0 98.6
        \\intensity 2.5: -148.0 32.0 77.0 198.5
        ,
    },
    // Chapter 03: The Arithmetic Diner
    .{
        .main_file = "03_arithmetic_diner/017_the_tab.zig",
        .output = "Each guest owes: { 40, 42, 2, 42 }",
        .hint = "The comment above the function is the formula. The lesson has the method names.\n",
    },
    .{
        .main_file = "03_arithmetic_diner/018_keep_the_change.zig",
        .output = "Pancake prices: { 2, 4.5, 8, 12.5 }",
        .hint = "Flo's formula has four steps, in an order. Does the chain have the same order?\n",
    },
    .{
        .main_file = "03_arithmetic_diner/019_the_griddle.zig",
        .output =
        \\Burners are off by: { 10, 0, 25, 40 }
        \\Coffee after 0, 5, 10, 30 minutes: 80.0 56.4 42.1 23.0
        \\Tips of 0.5, 1.5, 2.5, 3.5 become: { 1, 2, 3, 4 }
        ,
        .hint = "Three recipes, three problems. For the tips: which rounding did Flo learn in school?\n",
    },
    .{
        .main_file = "03_arithmetic_diner/020_how_would_you_like_it.zig",
        .output =
        \\steaks   -3.000  -1.000   0.000   1.000   3.000
        \\relu      0.000   0.000   0.000   1.000   3.000
        \\leaky    -0.300  -0.100   0.000   1.000   3.000
        \\by hand  -0.300  -0.100   0.000   1.000   3.000
        \\sigmoid   0.047   0.269   0.500   0.731   0.953
        \\tanh     -0.995  -0.762   0.000   0.762   0.995
        \\silu     -0.142  -0.269   0.000   0.731   2.858
        \\by hand  -0.142  -0.269   0.000   0.731   2.858
        \\gelu     -0.004  -0.159   0.000   0.841   2.996
        ,
        .hint = "Compare each 'by hand' row with the row above it.\n",
    },
    .{
        .main_file = "03_arithmetic_diner/021_no_substitutions.zig",
        .output = "Pay envelopes are {cooks=3,i32}: { 155, 165, 157 }",
        .hint = "The first line of each panic names the two dtypes that met. Which one did you want?\n",
    },
    .{
        .main_file = "03_arithmetic_diner/022_from_scratch.zig",
        .output =
        \\Tickets: { 101, 102, 103, 104, 105 }
        \\Each seat pays: { { 3, 5, 3 }, { 10, 3, 7 } }
        ,
        .hint = "What dtype does iota return? And along which axis do the SEATS count?\n",
    },
    .{
        .main_file = "03_arithmetic_diner/023_order_up.zig",
        .output = "Doneness: 0.201 0.474 0.974 1.701",
        .hint = "NaN means the square root got something negative. What is under the square root in the formula?\n",
    },
    .{
        .main_file = "03_arithmetic_diner/024_quiz_the_dinner_rush.zig",
        .output = "Ring up {tables=4,i32}: { 8, 2, 14, 19 }",
    },
    // Chapter 04: Luggage Tags
    .{
        .main_file = "04_luggage_tags/025_unclaimed_baggage.zig",
        .output =
        \\Delivered: Tensor({3,4,f32})
        \\Labeled:   Tensor({carousels=3,bags=4,f32})
        \\3 carousels, 4 bags on each. Bags are axis 1.
        \\The delivery slip still says {3,4,f32}.
        ,
        .hint = "Count the axes of the shape in the panic. Count the tags. Is a kilogram an axis?\n",
    },
    .{
        .main_file = "04_luggage_tags/026_half_a_label.zig",
        .output =
        \\Tensor({7,2,bags=5,cm=3,f32}): fully tagged? false
        \\bags: axis 2, days: axis null
        \\Tensor({days=7,2,bags=5,cm=3,f32}): fully tagged? false
        \\days: axis 0, and the mystery axis has 2 of something called '_'.
        ,
        .hint = "Each panic prints the shape: count its axes, and look for the one without a name.\n",
    },
    .{
        .main_file = "04_luggage_tags/027_the_label_maker.zig",
        .output =
        \\Manifest: Tensor({flights=3,bags=4,2,f32})
        \\Customs:  Tensor({departures=3,items=4,2,f32})
        \\Complete: Tensor({departures=3,items=4,measures=2,f32}), fully tagged? true
        \\Archive:  Tensor({flights=3,items=4,measures=2,f32}), 4 items per flight.
        ,
        .hint = "In rename, the field name is the OLD tag. And a shape should not print the same tag twice.\n",
    },
    .{
        .main_file = "04_luggage_tags/028_row_seven.zig",
        .output =
        \\Tensor({days=7,2,bags=5,cm=3,f32})
        \\Bags per scan: 5
        \\cm is axis 3; second from the end is axis 2
        \\The mystery axis has size 2
        \\Tensor({days=7,sides=2,bags=5,cm=3,f32}): fully tagged? true
        \\210 measurements, and count() agrees: true
        ,
        .hint = "An underscore alone and an underscore with a digit mean very different things.\n",
    },
    .{
        .main_file = "04_luggage_tags/029_same_size_different_bag.zig",
        .output =
        \\weights vs fees:    dims true, dims+dtype true, dims+dtype+tags false
        \\weights vs serials: dims true, dims+dtype false, dims+dtype+tags false
        \\weights + fees is {bags=4,f32}
        \\fees + weights is {flights=4,f32}
        \\Kilograms plus dollars: { 73, 58, 90, 151 }. Unit: unknown.
        ,
        .hint = "Read the three definitions again. And which operand does the result take its tags from?\n",
    },
    .{
        .main_file = "04_luggage_tags/030_the_wrong_carousel.zig",
        .output = "Owed, as {flights=2,bags=3,f32}: { { 11, 23, 35 }, { 42, 54, 66 } }",
        .hint = "Same tags, different order: the stickers are right, the suitcases need to move.\n",
    },
    .{
        .main_file = "04_luggage_tags/031_layover.zig",
        .output =
        \\Old machine {n=1,h=2,w=5,c=3,f32}: a pixel of 10 becomes 1.000
        \\New machine {n=1,c=3,h=2,w=5,f32}: a pixel of 10 becomes 1.000
        ,
        .hint = "The function says dim(1). Which axis is that on the NEW machine?\n",
    },
    .{
        .main_file = "04_luggage_tags/032_quiz_flight_zg404.zig",
        .output =
        \\Report {flights=2,bags=3,f32}, bags on axis 1, fully tagged: true
        \\Excess: { { 0, 1, 0 }, { 3, 0, 2 } }
        ,
    },
    // Chapter 05: Broadcast News
    .{
        .main_file = "05_broadcast_news/033_dead_air.zig",
        .output =
        \\The forecast is a {city=3,day=4,f32}.
        \\Springfield  16.0 17.0 15.0 18.0
        \\Shelbyville  16.5 15.5 17.5 16.5
        \\Ogdenville   30.0 31.0 29.0 33.0
        ,
        .hint = "The panic names the operation and both shapes. One of them is too small.\n",
    },
    .{
        .main_file = "05_broadcast_news/034_stage_left.zig",
        .output =
        \\Morning Zoo   120  124  122  180
        \\Noon News      90   84   93  240
        \\Late Movie     55   56   55   68
        ,
        .hint = "Shows are the first axis of the table, weeks are the last one.\n",
    },
    .{
        .main_file = "05_broadcast_news/035_make_room.zig",
        .output =
        \\{anchor=3,segment=4,f32}
        \\       news weather sports cat
        \\Brock   120      30     50  20
        \\Linda   240      60    100  40
        \\Gail    180      45     75  30
        ,
        .hint = "Where should the size 1 axis go: before or after the axis that is already there?\n",
    },
    .{
        .main_file = "05_broadcast_news/036_the_fine_print.zig",
        .output =
        \\Brock 0.50 0.75 0.00 1.00
        \\Linda 0.50 0.25 0.90 0.00
        \\Gail  0.90 0.80 0.50 0.00
        ,
        .hint = "Only the right side of an operation gets stretched for free.\n",
    },
    .{
        .main_file = "05_broadcast_news/037_technical_difficulties.zig",
        .output =
        \\{city=3,day=3,f32}
        \\             Fri  Sat  Sun
        \\Springfield  16.0 17.0 15.0
        \\Shelbyville  16.5 15.5 17.5
        \\Ogdenville   30.0 31.0 29.0
        ,
        .hint = "No panic this time. Which shape does `column` really have?\n",
    },
    .{
        .main_file = "05_broadcast_news/038_quiz_live_at_six.zig",
        .output =
        \\On air: {n=2,c=3,h=2,w=2,f32}
        \\frame 0 red  : 100.00 60.00 70.00 160.00
        \\frame 0 green: 100.00 60.00 70.00 160.00
        \\frame 0 blue : 120.00 70.00 80.00 180.00
        \\frame 1 red  : 50.00 30.00 35.00 80.00
        \\frame 1 green: 52.50 31.25 36.25 82.50
        \\frame 1 blue : 55.00 32.50 37.50 85.00
        ,
        .hint = "Doug's note: rule 3 wants the same tags, in the same order, on both sides.\n",
    },
    // Chapter 06: The Reduction Kitchen
    .{
        .main_file = "06_reduction_kitchen/039_reduce_the_sauce.zig",
        .output =
        \\{table=3,dish=4,i32} reduced to {table=1,dish=4,i32}
        \\Fire 3 soup!
        \\Fire 4 duck!
        \\Fire 5 risotto!
        \\Fire 7 tart!
        ,
        .hint = "The axis you name is the one that goes away.\n",
    },
    .{
        .main_file = "06_reduction_kitchen/040_skim_it_off.zig",
        .output =
        \\Jean-Luc carries {table=3,i32}: { 5, 5, 9 }
        \\Chef Odile gets {i32}: 19 plates
        ,
        .hint = "squeeze only removes an axis of size 1. Which axis has size 1 after that sum?\n",
    },
    .{
        .main_file = "06_reduction_kitchen/041_the_critics.zig",
        .output =
        \\{dish=3,f32}
        \\soup: 7.67
        \\duck: 9.00
        \\tart: 4.00
        ,
    },
    .{
        .main_file = "06_reduction_kitchen/042_dish_of_the_day.zig",
        .output =
        \\The blackboard is a {day=5,i32}.
        \\Tuesday: soup
        \\Wednesday: risotto
        \\Thursday: duck
        \\Friday: duck
        \\Saturday: tart
        ,
        .hint = "eval says how many outputs it got. After that, look at the shape on the first line.\n",
    },
    .{
        .main_file = "06_reduction_kitchen/043_running_low.zig",
        .output =
        \\Butter: 7.0 2.5 10.5 5.5 -0.5 -2.5 7.5 3.5
        \\Lowest point: -2.5 kg. We ran out. Nobody tell Chef Odile.
        ,
        .hint = "First a panic about bytes: how many numbers should come out of `stock`?\n",
    },
    .{
        .main_file = "06_reduction_kitchen/044_plating.zig",
        .output =
        \\softmax:
        \\  guest 0: 0.644 0.237 0.032 0.087  sum 1.00
        \\  guest 1: 0.017 0.028 0.017 0.937  sum 1.00
        \\by hand:
        \\  guest 0: 0.644 0.237 0.032 0.087  sum 1.00
        \\  guest 1: 0.017 0.028 0.017 0.937  sum 1.00
        ,
        .hint = "Which axis should add up to 1?\n",
    },
    .{
        .main_file = "06_reduction_kitchen/045_quiz_saturday_night.zig",
        .output =
        \\bills {table=3,f32}: { 60, 97, 124 }
        \\average bill {f32}: 93.67
        \\best seller {i32}: tart
        \\shares {dish=4,f32}: soup 0.085 duck 0.370 risotto 0.320 tart 0.224
        ,
    },
    // Chapter 07: Origami Club
    .{
        .main_file = "07_origami_club/046_the_first_fold.zig",
        .output =
        \\grid:    {rows=3,cols=4,i32} { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        \\booklet: {pages=2,rows=3,cols=2,i32} { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        \\unfold:  {12,i32} { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        ,
        .hint = "Two things to fix. One is a panic: its message has both shapes in it.\n",
    },
    .{
        .main_file = "07_origami_club/047_turn_the_paper.zig",
        .output =
        \\turn:      {cols=3,rows=2,i32}
        \\  { 0, 3 }
        \\  { 1, 4 }
        \\  { 2, 5 }
        \\restack:   {rows=3,cols=4,sheets=2,i32}
        \\flip:      {rows=3,sheets=2,cols=4,i32}
        \\insideOut: {cols=4,rows=3,sheets=2,i32}
        ,
        .hint = "A permutation says, for each NEW position, which OLD axis goes there.\n",
    },
    .{
        .main_file = "07_origami_club/048_same_shape_different_bird.zig",
        .output =
        \\Scoreboard {rounds=3,members=2,i32}
        \\round 1: Ingrid 9, Timmy 2
        \\round 2: Ingrid 8, Timmy 3
        \\round 3: Ingrid 10, Timmy 1
        ,
        .hint = "The shape is already right. The shape was never the problem.\n",
    },
    .{
        .main_file = "07_origami_club/049_cut_along_the_crease.zig",
        .output = "{entries=6,i32} { 37, 37, 28, 23, 4, 12 }",
        .hint = "How many cranes of 4 squares are there in a strip of 8?\n",
    },
    .{
        .main_file = "07_origami_club/050_neighbours_only.zig",
        .output =
        \\{rows=3,strip=8,i32}
        \\  { 0, 1, 2, 3, 12, 13, 14, 15 }
        \\  { 4, 5, 6, 7, 16, 17, 18, 19 }
        \\  { 8, 9, 10, 11, 20, 21, 22, 23 }
        ,
        .hint = "The axes to merge must be side by side first.\n",
    },
    .{
        .main_file = "07_origami_club/051_the_copy_machine.zig",
        .output = "Garland {beads=12,i32}: 3 3 2 2 1 1 3 3 2 2 1 1",
        .hint = "Right length, wrong order. One copier copies the whole motif, the other copies each bead.\n",
    },
    .{
        .main_file = "07_origami_club/052_mind_the_margin.zig",
        .output =
        \\Mounted on {rows=4,cols=5,i32}
        \\  { 0, 0, 0, 0, 0 }
        \\  { 0, 1, 2, 3, 0 }
        \\  { 0, 4, 5, 6, 0 }
        \\  { 0, 0, 0, 0, 0 }
        ,
        .hint = "Once it compiles, count the columns of the card.\n",
    },
    .{
        .main_file = "07_origami_club/053_quiz_a_thousand_cranes.zig",
        .output =
        \\{patches=4,squares=4,i32}
        \\  { 10, 11, 14, 15 }
        \\  { 8, 9, 12, 13 }
        \\  { 2, 3, 6, 7 }
        \\  { 0, 1, 4, 5 }
        ,
        .hint = "After the two cuts you have four axes. Which two must become neighbours?\n",
    },
    // Chapter 08: Slice of Life
    .{
        .main_file = "08_slice_of_life/054_just_a_slice.zig",
        .output =
        \\firstThree: {slices=3,i32} { 3, 1, 4 }
        \\lastTwo:    {slices=2,i32} { 2, 6 }
        \\middle:     {slices=4,i32} { 4, 1, 5, 9 }
        \\everyOther: {slices=4,i32} { 1, 1, 9, 6 }
        ,
        .hint = "The first slice is slice 0, and `end` is excluded. Luigi disagrees three times.\n",
    },
    .{
        .main_file = "08_slice_of_life/055_with_or_without_the_box.zig",
        .output =
        \\extraBasil: {slices=4,i32} { 105, 106, 207, 208 }
        \\lastBoxed:  {pizzas=1,slices=4,i32} { 9, 10, 11, 12 }
        ,
        .hint = "The panic prints the two shapes that should have been equal.\n",
    },
    .{
        .main_file = "08_slice_of_life/056_no_crust_please.zig",
        .output =
        \\noCrust: {rows=2,cols=4,i32}
        \\  { 11, 12, 13, 14 }
        \\  { 21, 22, 23, 24 }
        \\leftEdge: {rows=2,i32}
        \\  { 0, 20 }
        ,
        .hint = "The n-th Slice cuts the n-th axis of the tuple.\n",
    },
    .{
        .main_file = "08_slice_of_life/057_boxing_night.zig",
        .output =
        \\party:    {slices=8,i32} { 1, 2, 3, 4, 5, 6, 7, 8 }
        \\delivery: {boxes=2,slices=4,i32} { 1, 2, 3, 4, 5, 6, 7, 8 }
        \\tasting:  {slices=4,kinds=2,i32} { 1, 5, 2, 6, 3, 7, 4, 8 }
        ,
        .hint = "Does the order need a NEW axis? And where does it go?\n",
    },
    .{
        .main_file = "08_slice_of_life/058_fair_shares.zig",
        .output =
        \\bigFirst: {slices=8,i32} { 5, 9, 2, 6, 1, 4, 1, 3 }
        \\twins:    {slices=4,i32} { -2, -8, 2, -5 }
        ,
        .hint = "Somebody at the party is not getting all the pizza they were promised.\n",
    },
    .{
        .main_file = "08_slice_of_life/059_surprise_me.zig",
        .output =
        \\window from 0: { 0, 1, 2 }
        \\window from 2: { 2, 3, 4 }
        \\window from 4: { 4, 5, 6 }
        \\window from 7: { 5, 6, 7 }
        \\window from -2: { 0, 1, 2 }
        \\pick 2: {slices=4,i32} { 30, 31, 32, 33 }
        \\pick 0: {slices=4,i32} { 10, 11, 12, 13 }
        \\swap 1: { 10, 11, 12, 13, 99, 99, 99, 99, 30, 31, 32, 33 }
        ,
        .hint = "An input that the function ignores cannot change the output.\n",
    },
    .{
        .main_file = "08_slice_of_life/060_quiz_the_usual.zig",
        .output =
        \\The usual is a {boxes=2,slices=5,i32}.
        \\pizza 0: his box { 0, 2, 4, 5, 6 }, her sample { 23, 24, 25, 26, 27 }
        \\pizza 1: his box { 10, 12, 14, 15, 16 }, her sample { 23, 24, 25, 26, 27 }
        \\pizza 2: his box { 20, 22, 24, 25, 26 }, her sample { 23, 24, 25, 26, 27 }
        ,
        .hint = "Only `which` is a tensor. Every size in the order is a plain number.\n",
    },
    // Chapter 09: Dot Com Bubble
    .{
        .main_file = "09_dot_com_bubble/061_no_matmul.zig",
        .output =
        \\The result is a {f32}: no axes left.
        \\Revenue: $1400. Cost of the Super Bowl ad: $2000000.
        ,
        .hint = "One method, and the name of the axis the two lists share.\n",
    },
    .{
        .main_file = "09_dot_com_bubble/062_synergy.zig",
        .output =
        \\revenue {days=3,f32}: { 90, 250, 1500 }
        \\parts {days=3,parts=2,f32}:
        \\  day 1: 12 rocks, 4 googly eyes
        \\  day 2: 29 rocks, 18 googly eyes
        \\  day 3: 158 rocks, 36 googly eyes
        ,
        .hint =
        \\The panic names a tensor and a tag: which axis do BOTH operands have?
        \\Then compare the shape that is printed with the one you were asked for.
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/063_branch_offices.zig",
        .output =
        \\revenue {office=2,days=3,f32}: { { 90, 250, 1500 }, { 4, 0, 27 } }
        \\synergy {office=2,partner=2,f32}:
        \\  { 1002925, 702320 }
        \\  { 702320, 491841 }
        ,
        .hint = "Same name on both sides means \"pair them up\". The slide wants them NOT paired up.\n",
    },
    .{
        .main_file = "09_dot_com_bubble/064_the_price_grid.zig",
        .output =
        \\price grid {products=3,tiers=3,f32}:
        \\    5.00    2.50    0.50
        \\   20.00   10.00    2.00
        \\ 1000.00  500.00  100.00
        ,
        .hint = "The function is fine. What does `broad` match on?\n",
    },
    .{
        .main_file = "09_dot_com_bubble/065_legacy_code.zig",
        .output =
        \\parts {2,3,2,f32}:
        \\  quarry 1: { { 12, 4 }, { 29, 18 }, { 158, 36 } }
        \\  quarry 2: { { 2, 1 }, { 0, 0 }, { 8, 4 } }
        ,
        .hint =
        \\Write the two shapes down and number their axes from 0.
        \\Which pair is contracted? Which pair is "for each quarry"?
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/066_the_valuation_engine.zig",
        .output =
        \\output {startup=2,d_out=2,f32}
        \\PetRocks.com: valued at $156.20M, burning $2.87M a month
        \\Boring Inc.: valued at $40.00M, burning $3.50M a month
        ,
        .hint = "The panic shows both shapes. One of them has to grow before the `add`.\n",
    },
    .{
        .main_file = "09_dot_com_bubble/067_quiz_ipo_day.zig",
        .output =
        \\verdicts {startup=3,verdict=2,f32}
        \\PetRocks.com: buy 12, sell 2. BUY!
        \\Gravel.net: buy 16, sell 0. BUY!
        \\Foosball.org: buy 21, sell -3. BUY!
        ,
    },
    // Chapter 10: Treasure Maps
    .{
        .main_file = "10_treasure_maps/068_the_parrot.zig",
        .output =
        \\Polly answers with a {chest=6,bool}.
        \\Worth the shovel: { true, false, false, true, true, false }
        ,
        .hint = "Doubloons are counted in whole numbers on this ship.\n",
    },
    .{
        .main_file = "10_treasure_maps/069_the_ledger.zig",
        .output =
        \\The ledger: { 30, 5, -13, 0, 50, 0 }
        \\The crew's purse: 72 doubloons.
        ,
        .hint = "Where the mask is TRUE, the value is KEPT.\n",
    },
    .{
        .main_file = "10_treasure_maps/070_x_marks_the_index.zig",
        .output =
        \\clues {stop=4,clue=3,i32}:
        \\  stop 1: 41 paces north, 42 paces east, dig 43 feet
        \\  stop 2: 1 paces north, 2 paces east, dig 3 feet
        \\  stop 3: 21 paces north, 22 paces east, dig 23 feet
        \\  stop 4: 11 paces north, 12 paces east, dig 13 feet
        ,
        .hint =
        \\One stop is right, and only by accident.
        \\What is the index of the FIRST island?
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/071_latitude_longitude.zig",
        .output =
        \\under the X's {x=3,i32}: { 9, 8, 7 }
        \\best of each latitude {lat=4,i32}: { 7, 9, 4, 8 }
        ,
        .hint = "First the order of the fields. Then look at the shape that is printed before the crash.\n",
    },
    .{
        .main_file = "10_treasure_maps/072_roll_call.zig",
        .output =
        \\the pay line {pirate=5,i32}: { 80, 80, 30, 12, 5 }
        \\their ages   {rank=5,i32}: { 19, 27, 41, 35, 63 }
        ,
        .hint = "The panic counts three places where one tag shows up. Two is the limit.\n",
    },
    .{
        .main_file = "10_treasure_maps/073_the_three_best.zig",
        .output =
        \\The Exe has 2 outputs: {best=3,i32} and {best=3,i32}.
        \\Load chest 4: 95 doubloons.
        \\Load chest 2: 80 doubloons.
        \\Load chest 0: 30 doubloons.
        ,
        .hint = "The compile error quotes an example call. It is a very good example.\n",
    },
    .{
        .main_file = "10_treasure_maps/074_quiz_buried_treasure.zig",
        .output =
        \\landing parties: {best=2,role=2,i32}
        \\Island 4: 85 doubloons. Send 6 diggers and 3 lookouts.
        \\Island 3: 70 doubloons. Send 5 diggers and 2 lookouts.
        ,
    },
    // Chapter 11: Time Loops
    .{
        .main_file = "11_time_loops/075_six_oclock.zig",
        .output =
        \\The Exe has 1 input. While tracing, the alarm rang 3 times.
        \\Phil wakes up at 80 and goes to bed at 10.
        \\Phil wakes up at 64 and goes to bed at 8.
        \\Phil wakes up at 8 and goes to bed at 1.
        \\Phil wakes up at 0 and goes to bed at 0.
        \\Four runs later, the alarm has rung 3 times in total.
        ,
        .hint = "The loop body is plain Zig. How many days did the story ask for?\n",
    },
    .{
        .main_file = "11_time_loops/076_dont_drive_angry.zig",
        .output =
        \\The thermometers say   { -4, 0, 3, 7 }
        \\An optimist feels      { -8, 0, 3, 7 }
        \\A pessimist feels      { -18, -10, -4, 2 }
        ,
        .hint = "Zig can't branch on a ghost. Chapter 10 had an operation that chooses element by element.\n",
    },
    .{
        .main_file = "11_time_loops/077_the_shadow.zig",
        .output =
        \\0 minutes of sun. Gus says { 6, 8, 15 }, and Gus's lawyer says { 6, 8, 15 }.
        \\45 minutes of sun. Gus says { -9, -7, 0 }, and Gus's lawyer says { -9, -7, 0 }.
        ,
        .hint = "The panic says what dtype it wanted. Where do bool tensors come from?\n",
    },
    .{
        .main_file = "11_time_loops/078_piano_lessons.zig",
        .output =
        \\After 0 days, Phil has practiced 0 minutes.
        \\After 4 days, Phil has practiced 10 minutes.
        \\After 100 days, Phil has practiced 5050 minutes.
        \\After 3650 days, Phil has practiced 6663075 minutes.
        ,
        .hint = "cond returns a Tensor, not a Zig bool. Today's practice time is already in the State.\n",
    },
    .{
        .main_file = "11_time_loops/079_word_gets_around.zig",
        .output =
        \\unrolled, 1 input,  5 days: { 32, 243, 1 }
        \\looped,   2 inputs, 0 days: { 1, 1, 1 }
        \\looped,   2 inputs, 5 days: { 32, 243, 1 }
        \\looped,   2 inputs, 12 days: { 4096, 531441, 1 }
        ,
        .hint = "The panic prints two shapes: the crowd that went in and the crowd that came out. The unrolled version starts its crowd differently.\n",
    },
    .{
        .main_file = "11_time_loops/080_quiz_getting_out.zig",
        .output =
        \\Starting from 1: free after 0 steps. Before coffee, Phil stops at 0.
        \\Starting from 6: free after 8 steps. Before coffee, Phil stops at 8.
        \\Starting from 7: free after 16 steps. Before coffee, Phil stops at 16.
        \\Starting from 27: free after 111 steps. Before coffee, Phil stops at 20.
        \\Starting from 97: free after 118 steps. Before coffee, Phil stops at 20.
        ,
        .hint = "Which of the three numbers are known at trace time, and which only at run time? Exercises 078 and 079 have the shape of the answer.\n",
    },
    // Chapter 12: The Exe Files
    .{
        .main_file = "12_the_exe_files/081_the_envelope.zig",
        .output =
        \\This Exe takes 3 Buffers:
        \\  input 0: {frames=4,pixels=6,f32}
        \\  input 1: {seconds=10,f32}
        \\  input 2: {f32}
        \\and gives back 3:
        \\  output 0: {frames=4,f32}
        \\  output 1: {f32}
        \\  output 2: {frames=4,pixels=6,f32}
        ,
        .hint = "The shapes are spelled out in the lesson. The Exe has two lists; you printed the first one already.\n",
    },
    .{
        .main_file = "12_the_exe_files/082_raw_footage.zig",
        .output =
        \\brightest pixel per frame: { 1, 9, 0.5 }
        \\total glow per frame:      { 2, 14, 0.5 }
        ,
        .hint = "eval is out. The lesson lists the replacement in order: args, results, set, call, get.\n",
    },
    .{
        .main_file = "12_the_exe_files/083_the_stakeout.zig",
        .output =
        \\Night 1: peak 2, 0 seconds over the threshold.
        \\Night 2: peak 9, 3 seconds over the threshold.
        \\Night 3: peak 6, 2 seconds over the threshold.
        ,
        .hint = "run takes the inputs as a tuple, then a tuple of pointers to the destinations, then the call options.\n",
    },
    .{
        .main_file = "12_the_exe_files/084_the_autopsy.zig",
        .output =
        \\The Exe has 3 inputs: {d_out=2,d=3,f32}, {d_out=2,f32} and {d=3,f32}.
        \\alien: 4.5, weather balloon: 10
        ,
        .hint = "compile wants the method's name as an enum literal. Then: which Buffers fill slots 0 and 1?\n",
    },
    .{
        .main_file = "12_the_exe_files/085_redacted.zig",
        .output =
        \\A Polygraph has 4 fields, its Bufferized twin has 2. Each Exe has 3 inputs.
        \\Witness 1: Scally's polygraph says lying=false, Molder's says lying=false.
        \\Witness 2: Scally's polygraph says lying=false, Molder's says lying=true.
        \\Witness 3: Scally's polygraph says lying=true, Molder's says lying=true.
        ,
        .hint = "Zig names the field that doesn't exist. And which of the two agents is the paranoid one?\n",
    },
    .{
        .main_file = "12_the_exe_files/086_it_goes_all_the_way_up.zig",
        .output =
        \\3 layers, 7 inputs.
        \\What reaches the top: { 2.5, 4.5 }
        ,
        .hint = "A Bufferized(Layer) is a struct of two Buffers. deinitAll is happy with a Layer, just not with a slice of them.\n",
    },
    .{
        .main_file = "12_the_exe_files/087_ten_sightings.zig",
        .output =
        \\Sighting 0: 4.5 against 10. weather balloon.
        \\Sighting 1: 11.5 against -4. ALIEN.
        \\Sighting 2: 0.5 against 2. weather balloon.
        \\Sighting 3: 27.5 against 17. ALIEN.
        \\Sighting 4: 1.5 against -1. ALIEN.
        \\Sighting 5: 8.5 against 1. ALIEN.
        \\Sighting 6: 9.5 against 5. ALIEN.
        \\Sighting 7: 2.5 against -2. ALIEN.
        \\Sighting 8: 7.5 against 2. ALIEN.
        \\Sighting 9: 0.5 against 17. weather balloon.
        ,
        .hint = "Read the two shapes in the panic. Which of the three ways out does the task ask for?\n",
    },
    .{
        .main_file = "12_the_exe_files/088_quiz_the_truth_is_in_there.zig",
        .output =
        \\Each agent's Exe has 5 inputs and 2 outputs.
        \\Case 1: score 3. Molder: Aliens! Scally: Swamp gas.
        \\Case 2: score 11. Molder: Aliens! Scally: ...Aliens.
        \\Case 3: score 4. Molder: Aliens! Scally: Swamp gas.
        \\Case 4: score 9. Molder: Aliens! Scally: Swamp gas.
        \\Case 5: score -1. Molder: Inconclusive. Scally: Swamp gas.
        ,
        .hint = "Follow the TODOs in order. What do the two agents NOT have in common?\n",
    },
    // Chapter 13: Layer Cake
    .{
        .main_file = "13_layer_cake/089_the_first_tier.zig",
        .output =
        \\scores {cake=3,dout=2,f32}
        \\  sponge    sweetness  3.25  richness  1.50
        \\  brioche   sweetness  3.00  richness  6.50
        \\  meringue  sweetness  5.00  richness -1.00
        ,
        .hint = "No label, no cake. Which axis does `forward` contract, and does the input have it?\n",
    },
    .{
        .main_file = "13_layer_cake/090_oven_settings.zig",
        .output =
        \\relu         0.000   0.000   0.500   2.000
        \\leakyReLU   -0.200  -0.050   0.500   2.000
        \\tanh        -0.964  -0.462   0.462   0.964
        \\sigmoid      0.119   0.378   0.622   0.881
        ,
        .hint = "Four identical rows: nobody turned the oven on. And how does Zig write a union value that carries a payload?\n",
    },
    .{
        .main_file = "13_layer_cake/091_level_the_tiers.zig",
        .output =
        \\  2  4  4  6  ->  7.172 10.000 10.000 12.828
        \\ 10 20 30 40  ->  7.317  9.106 10.894 12.683
        \\  5  5  5  5  -> 10.000 10.000 10.000 10.000
        ,
        .hint = "nan? Work out what gets divided by what for the flat cake. And which field sets the mean, which the spread?\n",
    },
    .{
        .main_file = "13_layer_cake/092_no_such_struct.zig",
        .output =
        \\(  3,  4)   rms norm  1.697  1.131   unit length  0.600  0.800
        \\( 30, 40)   rms norm  1.697  1.131   unit length  0.600  0.800
        \\(  0, -5)   rms norm  0.000 -1.414   unit length  0.000 -1.000
        ,
        .hint = "The numbers of one cake live along one axis. And `rmsNorm` only normalizes: the weight is your job.\n",
    },
    .{
        .main_file = "13_layer_cake/093_the_menu.zig",
        .output =
        \\order {s=4,d=3,f32}
        \\  croissant  sweet 0.2  buttery 0.9  crunchy 0.7
        \\  eclair     sweet 0.8  buttery 0.6  crunchy 0.1
        \\  eclair     sweet 0.8  buttery 0.6  crunchy 0.1
        \\  macaron    sweet 0.9  buttery 0.2  crunchy 0.5
        ,
        .hint = "The panic names the dtype it received. The ids are written on the menu, in the comments.\n",
    },
    .{
        .main_file = "13_layer_cake/094_the_cake_stand.zig",
        .output =
        \\Cake: base oven level
        \\Bufferized(Cake): base level
        \\Bufferized(Linear): weight bias quantization
        \\Bufferized(LayerNorm): weight bias
        \\weekday -1.336  0.267  1.069
        \\sunday   0.579 -1.407  0.828
        ,
        .hint = "The compiler names a field that does not exist. If Bufferized dropped it, the Exe does not need it.\n",
    },
    .{
        .main_file = "13_layer_cake/095_sweet_or_salty.zig",
        .output =
        \\verdicts {cake=4,i32}
        \\  plain          -> rejected
        \\  strawberry     -> served
        \\  olive          -> served
        \\  salted caramel -> rejected
        ,
        .hint = "The panic shows the two shapes that met in `dot`: which tag is missing? Then: is there an oven between the tiers?\n",
    },
    .{
        .main_file = "13_layer_cake/096_quiz_the_wedding_cake.zig",
        .output =
        \\occasions {order=3,i32}
        \\  2 croissants, 2 baguettes -> breakfast
        \\  2 eclairs, 2 macarons     -> tea time
        \\  3 brownies, 1 eclair      -> a wedding
        ,
        .hint = "Seven steps, in order. A panic names the shapes it got: compare them with the tags that step expects.\n",
    },
    // Chapter 14: Heavy Lifting
    .{
        .main_file = "14_heavy_lifting/097_the_locker.zig",
        .output =
        \\The header is 800 bytes of JSON:
        \\{
        \\"__metadata__":{"gym":"Iron Tensor"},
        \\"plates":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},
        \\"locker_combination":{"dtype":"I32","shape":[3],"data_offsets":[16,28]},
        \\"coach.0.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[28,52]},
        \\"coach.1.weight":{"dtype":"F32","shape":[3,2],"data_offsets":[52,76]},
        \\"coach.1.bias":{"dtype":"F32","shape":[3],"data_offsets":[76,88]},
        \\"bar.0.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[88,104]},
        \\"bar.0.bias":{"dtype":"F32","shape":[2],"data_offsets":[104,112]},
        \\"bar.1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[112,128]},
        \\"bar.1.bias":{"dtype":"F32","shape":[2],"data_offsets":[128,136]},
        \\"bar.2.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[136,152]},
        \\"bar.2.bias":{"dtype":"F32","shape":[2],"data_offsets":[152,160]}
        \\}
        \\
        \\The plates, in kg: { 20, 10, 5, 2.5 }
        ,
        .hint = "Two things about the file layout: which end of the integer comes first, and where byte 0 of the data is.\n",
    },
    .{
        .main_file = "14_heavy_lifting/098_the_front_desk.zig",
        .output =
        \\11 tensors, 160 bytes of weights.
        \\
        \\name                 shape       bytes   at
        \\plates               {4,f32}        16   808
        \\locker_combination   {3,i32}        12   824
        \\coach.0.weight       {2,3,f32}      24   836
        \\coach.1.weight       {3,2,f32}      24   860
        \\coach.1.bias         {3,f32}        12   884
        \\bar.0.weight         {2,2,f32}      16   896
        \\bar.0.bias           {2,f32}         8   912
        \\bar.1.weight         {2,2,f32}      16   920
        \\bar.1.bias           {2,f32}         8   936
        \\bar.2.weight         {2,2,f32}      16   944
        \\bar.2.bias           {2,f32}         8   960
        \\
        \\The plates start at byte 808 of the file.
        ,
        .hint = "FileNotFound, for a file that is right there? The lesson has a paragraph about a quirk.\n",
    },
    .{
        .main_file = "14_heavy_lifting/099_the_key_ring.zig",
        .output =
        \\11 entries in the locker, 3 of them under "coach".
        \\plates: Tensor({plate=4,f32})
        \\
        \\layer "coach.0."
        \\  weight: Tensor({dout=2,d=3,f32})
        \\  bias:   null
        \\
        \\layer "coach.1."
        \\  weight: Tensor({dout=3,d=2,f32})
        \\  bias:   Tensor({dout=3,f32})
        ,
        .hint = "Each panic names an entry of the file. Compare what was asked with what the logbook of 098 says.\n",
    },
    .{
        .main_file = "14_heavy_lifting/100_the_spotter.zig",
        .output =
        \\plates      {plate=4,f32}: { 20, 10, 5, 2.5 }
        \\combination {turn=3,i32}: { 36, 24, 36 }
        ,
        .hint = "Two moves: an empty bar, then the spotter. Four calls and a `defer`.\n",
    },
    .{
        .main_file = "14_heavy_lifting/101_meet_the_coach.zig",
        .output =
        \\Olga  LIGHT WEIGHT, BABY!
        \\Chad  NEVER SKIP LEG DAY.
        \\You   DO YOU EVEN LIFT?
        ,
        .hint = "If Coach says LIGHT WEIGHT about a 0.4 bench press, a tensor of the file is not being used.\n",
    },
    .{
        .main_file = "14_heavy_lifting/102_quiz_the_smoothie_bar.zig",
        .output =
        \\receipt {order=3,dout=2,f32}
        \\  2 banana, 1 spinach: $8.50, taste 3
        \\  1 banana, 3 spinach: $7.50, taste 0
        \\  0 banana, 0 spinach: $0.50, taste 0
        ,
    },
    // Chapter 15: Pay Attention
    .{
        .main_file = "15_pay_attention/103_who_listens_to_whom.zig",
        .output =
        \\weights {q=4,k=4,f32}, each row is a listener:
        \\        Tina  Kuzco Paca  Dolly
        \\  Tina  0.123 0.505 0.249 0.123
        \\  Kuzco 0.505 0.123 0.249 0.123
        \\  Paca  0.308 0.308 0.308 0.075
        \\  Dolly 0.250 0.250 0.250 0.250
        \\heard {q=4,hd=2,f32}:
        \\  Tina   2.597  6.544
        \\  Kuzco  6.421  2.720
        \\  Paca   4.700  4.775
        \\  Dolly  4.000  4.250
        ,
        .hint = "Whose attention must add up to 1: each row (a listener) or each column (a speaker)?\n",
    },
    .{
        .main_file = "15_pay_attention/104_no_spoilers.zig",
        .output =
        \\no spoilers:
        \\        Tina  Kuzco Paca  Dolly
        \\  Tina  1.000 0.000 0.000 0.000
        \\  Kuzco 0.804 0.196 0.000 0.000
        \\  Paca  0.333 0.333 0.333 0.000
        \\  Dolly 0.250 0.250 0.250 0.250
        \\a memory of 2 llamas:
        \\        Tina  Kuzco Paca  Dolly
        \\  Tina  1.000 0.000 0.000 0.000
        \\  Kuzco 0.804 0.196 0.000 0.000
        \\  Paca  0.000 0.500 0.500 0.000
        \\  Dolly 0.000 0.000 0.500 0.500
        ,
        .hint = "The panic shows the two shapes that met in `add`. Which one is not a table of listeners and speakers? Then: what must happen BEFORE the 100% is shared out?\n",
    },
    .{
        .main_file = "15_pay_attention/105_the_official_channel.zig",
        .output =
        \\by hand  {h=1,s=4,hd=2,f32}:
        \\  Tina  10.000  0.000
        \\  Kuzco  8.044  1.956
        \\  Paca   5.000  5.000
        \\  Dolly  4.000  4.250
        \\official {h=1,s=4,hd=2,f32}:
        \\  Tina  10.000  0.000
        \\  Kuzco  8.044  1.956
        \\  Paca   5.000  5.000
        \\  Dolly  4.000  4.250
        ,
        .hint = "The panic lists the tags `sdpa` wants. `byHand` does every rename you need, and builds the mask.\n",
    },
    .{
        .main_file = "15_pay_attention/106_two_ears.zig",
        .output =
        \\1 head(s), heard {s=4,d=4,f32}:
        \\         Tina  Kuzco | Tina  Kuzco
        \\  Tina   0.803 0.009 | 0.803 0.009
        \\  Kuzco  0.009 0.803 | 0.009 0.803
        \\  Paca   0.210 0.210 | 0.210 0.210
        \\  Dolly  0.175 0.175 | 0.175 0.175
        \\2 head(s), heard {s=4,d=4,f32}:
        \\         Tina  Kuzco | Tina  Kuzco
        \\  Tina   0.250 0.250 | 0.890 0.002
        \\  Kuzco  0.002 0.890 | 0.250 0.250
        \\  Paca   0.011 0.786 | 0.786 0.011
        \\  Dolly  0.199 0.199 | 0.199 0.199
        ,
        .hint = "How many heads, and who does the division? Then: `merge` undoes exactly what `splitAxis` did.\n",
    },
    .{
        .main_file = "15_pay_attention/107_spin_the_llama.zig",
        .output =
        \\after the spin {s=4,hd=2,f32}:
        \\  position 0: ( 1.000,  0.000)  length 1.000
        \\  position 1: ( 0.540,  0.841)  length 1.000
        \\  position 2: (-0.416,  0.909)  length 1.000
        \\  position 3: (-0.990,  0.141)  length 1.000
        \\scores at positions { 0, 1, 2, 3 }:
        \\   1.000  0.540 -0.416 -0.990
        \\   0.540  1.000  0.540 -0.416
        \\  -0.416  0.540  1.000  0.540
        \\  -0.990 -0.416  0.540  1.000
        \\scores at positions { 100, 101, 102, 103 }:
        \\   1.000  0.540 -0.416 -0.990
        \\   0.540  1.000  0.540 -0.416
        \\  -0.416  0.540  1.000  0.540
        \\  -0.990 -0.416  0.540  1.000
        ,
        .hint = "A score depends on the distance only if BOTH llamas have turned. Compare the two tables.\n",
    },
    .{
        .main_file = "15_pay_attention/108_the_gossip_archive.zig",
        .output =
        \\Tina  arrives at row 0 and hears 10.000  0.000
        \\Kuzco arrives at row 1 and hears  8.044  1.956
        \\Paca  arrives at row 2 and hears  5.000  5.000
        \\Dolly arrives at row 3 and hears  4.000  4.250
        \\the archive of keys {k=4,hd=2,f32}: (0,2) (2,0) (1,1) (0,0)
        ,
        .hint = "Keys and values are filed the same way. For the mask: which rows are filled once the llama at `pos` is in?\n",
    },
    .{
        .main_file = "15_pay_attention/109_quiz_the_whole_herd.zig",
        .output =
        \\after one round of gossip {s=4,d=4,f32}:
        \\  Tina   0.000  0.000  2.500  0.000
        \\  Kuzco  1.203  0.000  0.750  0.000
        \\  Paca   1.401  0.000  2.011  0.000
        \\  Dolly  0.173  1.452  0.567  1.988
        ,
        .hint = "Steps 1 to 5, one at a time. When ZML panics, the shapes in the message tell you which tag is off.\n",
    },
    // Chapter 16: Casino Royale
    .{
        .main_file = "16_casino_royale/110_the_dealer.zig",
        .output =
        \\Every card is between 1 and 13: true
        \\The two hands are the same: false
        \\Same seed, same evening: true
        \\Another seed, same evening: false
        ,
        .hint = "Which state does the second hand come from?\n",
    },
    .{
        .main_file = "16_casino_royale/111_groundhog_table.zig",
        .output =
        \\Round 2 is round 1 all over again: false
        \\Round 3 is round 2 all over again: false
        \\Round 3 is round 1 all over again: false
        ,
        .hint = "Two halves: the function returns the new state, and `main` puts it back where the old one was.\n",
    },
    .{
        .main_file = "16_casino_royale/112_the_sure_thing.zig",
        .output =
        \\by hand:      seven seven seven
        \\sampleTokens: seven seven seven
        ,
        .hint = "Read the panic: which tag is it looking for, and what are your logits tagged?\n",
    },
    .{
        .main_file = "16_casino_royale/113_let_it_ride.zig",
        .output =
        \\the sure thing
        \\  dice were rolled: false
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: true
        \\  pocket 3 got at least a quarter of the spins: false
        \\the honest table
        \\  dice were rolled: true
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: false
        \\  pocket 3 got at least a quarter of the spins: false
        \\the cold table
        \\  dice were rolled: true
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: true
        \\  pocket 3 got at least a quarter of the spins: false
        \\the hot table
        \\  dice were rolled: true
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: false
        \\  pocket 3 got at least a quarter of the spins: true
        ,
        .hint = "The comment above each table is its strategy. Inputs go in a tuple, outputs are pointers: tokens first.\n",
    },
    .{
        .main_file = "16_casino_royale/114_quiz_the_house_always_wins.zig",
        .output = "the house always wins please play again.",
    },
};
