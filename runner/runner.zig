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
        .output =
        \\88 keys, 2 hammers each. Full manifest: { 3, 88, 2 }
        \\528 parts in the crate, 1056 bytes.
        ,
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
        \\boxed sock:    rank 1, count 1, dims { 1 }
        \\paranoid sock: rank 3, count 1, dims { 1, 1, 1 }
        \\Same thing? false
        \\Is the sock a scalar? true
        ,
    },
    .{
        .main_file = "01_shape_shifters/005_label_your_boxes.zig",
        .output =
        \\{shelves=5,eggs=12,u8}: 12 eggs per shelf, shelves are axis 0, axis 1 is called 'eggs'.
        \\{drawers=3,pizzas=4,u8}: 12 pizzas, and the last axis is called 'pizzas'.
        ,
        .hint = "The first line of the panic names the shape and the tag it could not find.\n",
    },
    .{
        .main_file = "01_shape_shifters/006_repacking.zig",
        .output =
        \\cupboard {15,8,u16}, dishwasher {20,6,u16}, insurance {120,u16}
        \\moving {crates=5,plates=24,u16}: 24 plates per crate
        ,
    },
    .{
        .main_file = "01_shape_shifters/007_truck_tetris.zig",
        .output =
        \\Loaded: {layers=2,rows=4,boxes=5,u8}, 40 boxes in total.
        \\Aftermath: {layers=2,rows=4,i64}.
        \\Depot: {pallets=6,layers=2,rows=4,i64}, rank 3.
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
        \\invoice:    {n=32,u16}, 64 bytes
        \\total:      {f64}, rank 0, 8 bytes
        ,
    },

    // Chapter 02: Ghost Tensors
    .{
        .main_file = "02_ghost_tensors/009_the_ghost.zig",
        .output =
        \\Tensor({rooms=13,candles=7,f32}) haunts 13 rooms and 91 candle flames in total.
        \\It would weigh 364 bytes, if it weighed anything.
        \\Tensor({barrels=3,i32}) lurks in the cellar: rank 1, made of i32.
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
        \\A whisper of 3 echoes back as 9.
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
        \\The wild night moved {spots=4,f32} by: -108.0 0.0 27.0 99.9
        ,
    },
    // Chapter 03: The Arithmetic Diner
    .{
        .main_file = "03_arithmetic_diner/017_the_tab.zig",
        .output = "Each guest owes: { 40, 42, 2, 42 }",
        .hint = "The comment above the function is the formula. And count: how many inputs, how many Buffers?\n",
    },
    .{
        .main_file = "03_arithmetic_diner/018_keep_the_change.zig",
        .output =
        \\Pancake prices: { 2, 4.5, 8, 12.5 }
        \\Change from a twenty: { 18, 15.5, 12, 7.5 }
        ,
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
        \\by hand   0.047   0.269   0.500   0.731   0.953
        \\tanh     -0.995  -0.762   0.000   0.762   0.995
        \\silu     -0.142  -0.269   0.000   0.731   2.858
        \\by hand  -0.142  -0.269   0.000   0.731   2.858
        \\gelu     -0.004  -0.159   0.000   0.841   2.996
        ,
        .hint = "Compare each 'by hand' row with the row above it. And the leaky row with the menu.\n",
    },
    .{
        .main_file = "03_arithmetic_diner/021_no_substitutions.zig",
        .output =
        \\Pay envelopes are {cooks=3,i32}: { 155, 165, 157 }
        \\Per hour, as {cooks=3,f32}: 19.375 16.500 26.167
        ,
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
        .output = "Doneness: -0.298 -0.221 0.375 1.403",
        .hint = "NaN means the square root got something negative. What is under the square root in the formula?\nAnd a variance is not a standard deviation.\n",
    },
    .{
        .main_file = "03_arithmetic_diner/024_quiz_the_dinner_rush.zig",
        .output =
        \\The register rings up {tables=4,i32}
        \\coupon 10: { 8, 2, 14, 19 }
        \\coupon 0: { 20, 11, 24, 33 }
        ,
        .hint = "Follow one table through the five rules on a napkin. Where does the dtype change, and what must rule 5 add to?\nAnd the tables are numbered from 1.\n",
    },
    // Chapter 04: Luggage Tags
    .{
        .main_file = "04_luggage_tags/025_unclaimed_baggage.zig",
        .output =
        \\Delivered: Tensor({3,4,f32})
        \\Labeled:   Tensor({carousels=3,bags=4,f32})
        \\3 carousels, 4 bags on each. Bags are axis 1.
        \\The delivery slip still says {3,4,f32}.
        \\In pounds, {carousels=3,bags=4,f32}. Carousel 0: 22.0 44.0 33.0 11.0
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
        \\Tensor({bags=5,cm=3,f32}): bags on axis 0
        \\Tensor({2,bags=5,cm=3,f32}): bags on axis 1
        \\Tensor({4,7,2,bags=5,cm=3,f32}): bags on axis 3
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
        \\Returns:  Tensor({to=3,from=5,f32}), back to 3 flights.
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
        \\Tensor({days=7,bags=5,cm=3,sides=2,f32}): still 210
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
        \\checked sum is    {bags=4,f32}
        \\Kilograms plus dollars: { 73, 58, 90, 151 }. Unit: unknown.
        ,
        .hint = "Read the three definitions again. And which operand does the result take its tags from?\nA failed assert panics with 'reached unreachable code': the stack trace has the line.\n",
    },
    .{
        .main_file = "04_luggage_tags/030_the_wrong_carousel.zig",
        .output =
        \\Owed, as {flights=2,bags=3,f32}: { { 11, 23, 35 }, { 42, 54, 66 } }
        \\Tuesday, as {flights=3,bags=3,f32}: { { 11, 24, 37 }, { 42, 55, 68 }, { 73, 86, 99 } }
        ,
        .hint = "Same tags, different order: the stickers are right, the suitcases need to move.\nTuesday's numbers are off? Look at how you described accounting's table.\n",
    },
    .{
        .main_file = "04_luggage_tags/031_layover.zig",
        .output =
        \\Old machine {n=1,h=2,w=5,c=3,f32}: a pixel of 10 becomes 1.000
        \\New machine {n=1,c=3,h=2,w=5,f32}: a pixel of 10 becomes 1.000
        \\Handheld    {h=2,w=5,f32}: a pixel of 10 becomes 1.000
        ,
        .hint = "The function says dim(1) and dim(2). Which axes are those on the NEW machine? And does the handheld have an axis 2 at all?\n",
    },
    .{
        .main_file = "04_luggage_tags/032_quiz_flight_zg404.zig",
        .output =
        \\Report {flights=2,bags=3,f32}, bags on axis 1, fully tagged: true
        \\Excess: { { 0, 1, 0 }, { 4, 0, 3 } }
        ,
        .hint = "The airline's table needs new stickers AND a quarter turn; stickers alone move no suitcase.\nIf the assert fails ('reached unreachable code'), print both shapes: what still differs?\n",
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
        .hint = "Three panics, one after the other. Each one names what it got and what it wanted.\n",
    },
    .{
        .main_file = "05_broadcast_news/034_stage_left.zig",
        .output =
        \\{show=3,week=4,f32}
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
        \\Budget left:
        \\Brock 0.50 0.75 -0.30 1.00
        \\Linda 0.50 0.25 0.90 -0.30
        \\Gail  0.90 0.80 0.50 0.00
        \\Dollars over budget:
        \\Brock   0   0  30   0
        \\Linda   0   0   0  60
        \\Gail    0   0   0   0
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
        \\The witness agrees: true
        ,
        .hint = "No panic this time. Which shape does `column` really have? Ask the witness.\n",
    },
    .{
        .main_file = "05_broadcast_news/038_quiz_live_at_six.zig",
        .output =
        \\On air: {n=2,c=3,h=2,w=2,f32}
        \\frame 0 red  : 100.00 60.00 70.00 160.00
        \\frame 0 green: 100.00 60.00 70.00 160.00
        \\frame 0 blue : 120.00 70.00 80.00 160.00
        \\frame 1 red  : 50.00 35.00 35.00 80.00
        \\frame 1 green: 52.50 35.00 36.25 82.50
        \\frame 1 blue : 55.00 35.00 37.50 85.00
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
        \\z-scores {dish=3,critic=5,f32}
        \\soup: 0.00 0.37 -1.86 1.12 0.37
        \\duck: 0.42 0.42 -1.99 0.72 0.42
        \\tart: -0.12 0.18 -1.35 1.72 -0.43
        ,
        .hint = "Every mean keeps its size 1 axis, so each step lines up with the scores.\n",
    },
    .{
        .main_file = "06_reduction_kitchen/042_dish_of_the_day.zig",
        .output =
        \\The blackboard is a {day=5,i32}, with a {day=5,f32} of portions.
        \\Tuesday: soup, 12 sold
        \\Wednesday: risotto, 14 sold
        \\Thursday: duck, 11 sold
        \\Friday: duck, 16 sold
        \\Saturday: tart, 21 sold
        ,
        .hint = "eval says how many outputs it got. After that, look at the shape on the first line.\n",
    },
    .{
        .main_file = "06_reduction_kitchen/043_running_low.zig",
        .output =
        \\Butter: 7.0 2.5 10.5 5.5 -0.5 -2.5 7.5 3.5
        \\Chef Odile reads a {f32} and a {i32}.
        \\Lowest point: -2.5 kg, after hour 5. We ran out. Nobody tell Chef Odile.
        ,
        .hint = "First a panic about bytes: how many numbers should come out of `stock`?\n",
    },
    .{
        .main_file = "06_reduction_kitchen/044_plating.zig",
        .output =
        \\1. by hand:
        \\  guest 0: 0.644 0.237 0.032 0.087  sum 1.00
        \\  guest 1: 0.017 0.028 0.017 0.937  sum 1.00
        \\  guest 2: nan nan nan nan  sum nan
        \\2. by hand, stable:
        \\  guest 0: 0.644 0.237 0.032 0.087  sum 1.00
        \\  guest 1: 0.017 0.028 0.017 0.937  sum 1.00
        \\  guest 2: 0.087 0.237 0.644 0.032  sum 1.00
        \\3. softmax:
        \\  guest 0: 0.644 0.237 0.032 0.087  sum 1.00
        \\  guest 1: 0.017 0.028 0.017 0.937  sum 1.00
        \\  guest 2: 0.087 0.237 0.644 0.032  sum 1.00
        \\4. logSumExp:
        \\  guest 0: 0.644 0.237 0.032 0.087  sum 1.00
        \\  guest 1: 0.017 0.028 0.017 0.937  sum 1.00
        \\  guest 2: 0.087 0.237 0.644 0.032  sum 1.00
        ,
        .hint = "The nan row of table 1 is expected. In the others: which axis should add up to 1?\n",
    },
    .{
        .main_file = "06_reduction_kitchen/045_quiz_saturday_night.zig",
        .output =
        \\bills {table=3,f32}: { 60, 97, 124 }
        \\average bill {f32}: 93.67
        \\best seller {i32}: tart
        \\shares {dish=4,f32}: soup 0.085 duck 0.370 risotto 0.320 tart 0.224
        \\against the average {table=3,f32}: 0.641 1.036 1.324
        ,
        .hint = "Each printed shape tells you what is left to squeeze. A true scalar prints as {f32} or {i32}.\n",
    },
    // Chapter 07: Origami Club
    .{
        .main_file = "07_origami_club/046_the_first_fold.zig",
        .output =
        \\grid:    {rows=3,cols=4,i32} { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        \\booklet: {pages=2,rows=3,cols=2,i32} { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        \\unfold:  {12,i32} { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }
        ,
        .hint = "Fill in the blanks first. Then one panic is left: its message has both shapes in it.\n",
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
        \\Gaps {rounds=3,i32}: { 7, 5, 9 }
        ,
        .hint = "The shape is already right. The shape was never the problem. Then count the rounds in your output.\n",
    },
    .{
        .main_file = "07_origami_club/049_cut_along_the_crease.zig",
        .output =
        \\{entries=6,i32} { 37, 37, 28, 23, 4, 12 }
        \\shares {members=3,cranes=2,f32}
        \\  Ingrid: 0.500 0.500
        \\  Coach's nephew: 0.549 0.451
        \\  Timmy: 0.250 0.750
        ,
        .hint = "How many cranes of 4 squares are there in a strip of 8? And a share of WHOSE total?\n",
    },
    .{
        .main_file = "07_origami_club/050_neighbours_only.zig",
        .output =
        \\longStrips: {rows=3,strip=8,i32}
        \\  { 0, 1, 2, 3, 12, 13, 14, 15 }
        \\  { 4, 5, 6, 7, 16, 17, 18, 19 }
        \\  { 8, 9, 10, 11, 20, 21, 22, 23 }
        \\columnStrips: {cols=4,strip=6,i32}
        \\  { 0, 12, 4, 16, 8, 20 }
        \\  { 1, 13, 5, 17, 9, 21 }
        \\  { 2, 14, 6, 18, 10, 22 }
        \\  { 3, 15, 7, 19, 11, 23 }
        \\weights: {rows=3,i32}
        \\  { 60, 92, 124 }
        ,
        .hint = "The axes to merge must be side by side first. And a sum always leaves a souvenir behind.\n",
    },
    .{
        .main_file = "07_origami_club/051_the_copy_machine.zig",
        .output =
        \\Garland {beads=12,i32}: 3 3 2 2 1 1 3 3 2 2 1 1
        \\Poster {rows=4,cols=4,i32}
        \\  { 1, 1, 2, 2 }
        \\  { 1, 1, 2, 2 }
        \\  { 3, 3, 4, 4 }
        \\  { 3, 3, 4, 4 }
        ,
        .hint = "Right length, wrong order. One copier copies the whole motif, the other copies each bead.\n",
    },
    .{
        .main_file = "07_origami_club/052_mind_the_margin.zig",
        .output =
        \\mount: {rows=4,cols=5,i32}
        \\  { 0, 0, 0, 0, 0 }
        \\  { 0, 1, 2, 3, 0 }
        \\  { 0, 4, 5, 6, 0 }
        \\  { 0, 0, 0, 0, 0 }
        \\spread: {rows=3,cols=5,i32}
        \\  { 1, 0, 2, 0, 3 }
        \\  { 0, 0, 0, 0, 0 }
        \\  { 4, 0, 5, 0, 6 }
        \\showpiece: {rows=5,cols=7,i32}
        \\  { 0, 0, 0, 0, 0, 0, 0 }
        \\  { 0, 1, 0, 2, 0, 3, 0 }
        \\  { 0, 0, 0, 0, 0, 0, 0 }
        \\  { 0, 4, 0, 5, 0, 6, 0 }
        \\  { 0, 0, 0, 0, 0, 0, 0 }
        ,
        .hint = "The compile error starts inside ZML; the `note:` under it points at your line. Once it compiles, count the columns of the card.\n",
    },
    .{
        .main_file = "07_origami_club/053_quiz_a_thousand_cranes.zig",
        .output =
        \\{patches=4,squares=4,i32}
        \\  { 10, 11, 14, 15 }
        \\  { 8, 9, 12, 13 }
        \\  { 2, 3, 6, 7 }
        \\  { 0, 1, 4, 5 }
        \\pooled {rows=2,cols=2,f32}
        \\  2.5 4.5
        \\  10.5 12.5
        ,
        .hint = "After the two cuts you have four axes. Which two must become neighbours? And what shape does `mean` hand you?\n",
    },
    // Chapter 08: Slice of Life
    .{
        .main_file = "08_slice_of_life/054_just_a_slice.zig",
        .output =
        \\firstThree: {slices=3,i32} { 3, 1, 4 }
        \\lastTwo:    {slices=2,i32} { 2, 6 }
        \\middle:     {slices=4,i32} { 4, 1, 5, 9 }
        \\everyOther: {slices=4,i32} { 1, 1, 9, 6 }
        \\leftovers:  {slices=3,i32} { 6, 2, 9 }
        ,
        .hint = "The first slice is slice 0, and `end` is excluded. Negative numbers count from the end.\n",
    },
    .{
        .main_file = "08_slice_of_life/055_with_or_without_the_box.zig",
        .output =
        \\extraBasil: {slices=4,i32} { 105, 106, 207, 208 }
        \\lastBoxed:  {pizzas=1,slices=4,i32} { 9, 10, 11, 12 }
        \\compared:   {pizzas=3,slices=4,i32} { 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8 }
        ,
        .hint = "The panic prints the two shapes that should have been equal. Stretching needs an axis to stretch.\n",
    },
    .{
        .main_file = "08_slice_of_life/056_no_crust_please.zig",
        .output =
        \\noCrust: {rows=2,cols=4,i32}
        \\  { 11, 12, 13, 14 }
        \\  { 21, 22, 23, 24 }
        \\leftEdge: {rows=2,i32}
        \\  { 0, 20 }
        \\corners: {rows=2,cols=2,i32}
        \\  { 0, 5 }
        \\  { 30, 35 }
        ,
        .hint = "The n-th Slice cuts the n-th axis of the tuple.\n",
    },
    .{
        .main_file = "08_slice_of_life/057_boxing_night.zig",
        .output =
        \\party:    {slices=8,i32} { 1, 2, 3, 4, 5, 6, 7, 8 }
        \\delivery: {boxes=2,slices=4,i32} { 1, 2, 3, 4, 5, 6, 7, 8 }
        \\tasting:  {slices=4,kinds=2,i32} { 1, 5, 2, 6, 3, 7, 4, 8 }
        \\byHand:   {boxes=2,slices=4,i32} { 1, 2, 3, 4, 5, 6, 7, 8 }
        ,
        .hint = "Does the order need a NEW axis? And where does it go?\n",
    },
    .{
        .main_file = "08_slice_of_life/058_fair_shares.zig",
        .output =
        \\bigFirst: {slices=8,i32} { 5, 9, 2, 6, 1, 4, 1, 3 }
        \\twins:    {slices=4,i32} { -2, -8, 2, -5 }
        \\tally:    {slices=3,i32} { 3, 6, 22 }
        ,
        .hint = "Somebody at the party is not getting all the pizza they were promised. Then look at who is served first.\n",
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
        \\fullest before the swap: { 30, 31, 32, 33 }
        \\fullest after the swap: { 99, 99, 99, 99 }
        ,
        .hint = "Out of its box means the .pizzas axis is gone. Compare the shape printed with the shape promised. For `fullest`, write down the shape after every step.\n",
    },
    .{
        .main_file = "08_slice_of_life/060_quiz_the_usual.zig",
        .output =
        \\The usual is a {boxes=2,slices=5,i32}.
        \\pizza 0: his box { 0, 2, 4, 5, 6 }, her sample { 23, 24, 25, 26, 27 }, bill { 17, 125 }
        \\pizza 1: his box { 10, 12, 14, 15, 16 }, her sample { 23, 24, 25, 26, 27 }, bill { 67, 125 }
        \\pizza 2: his box { 20, 22, 24, 25, 26 }, her sample { 23, 24, 25, 26, 27 }, bill { 117, 125 }
        ,
        .hint = "Only `which` is a tensor. Every size in the order is a plain number.\n",
    },
    // Chapter 09: Dot Com Bubble
    .{
        .main_file = "09_dot_com_bubble/061_no_matmul.zig",
        .output =
        \\The result is a {f32}: no axes left.
        \\Revenue: $1400. Cost of the Super Bowl ad: $2000000.
        \\Profit: $840. Chad has already spent it on beanbags.
        ,
        .hint =
        \\One method, and the name of the axis the two lists share.
        \\When `dot` panics, compare the tags of the two placeholders it was given.
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/062_synergy.zig",
        .output =
        \\revenue {days=3,f32}: { 90, 250, 1500 }
        \\parts {days=3,parts=2,f32}:
        \\  day 1: 12 rocks, 4 googly eyes
        \\  day 2: 29 rocks, 18 googly eyes
        \\  day 3: 158 rocks, 36 googly eyes
        \\best day: day 3. Chad was on holiday.
        ,
        .hint =
        \\The panic names a tensor and a tag: which axis do BOTH operands have?
        \\Then compare the shape that is printed with the one you were asked for.
        \\The best day is a position, and a reduction leaves an axis of size 1 behind.
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/063_branch_offices.zig",
        .output =
        \\revenue {office=2,days=3,f32}: { { 90, 250, 1500 }, { 4, 0, 27 } }
        \\totals {office=2,f32}: { 1840, 31 }
        \\synergy {office=2,partner=2,f32}:
        \\  { 1002925, 702320 }
        \\  { 702320, 491841 }
        ,
        .hint =
        \\Same name on both sides means "pair them up". The slide wants them NOT paired up.
        \\The accountants want {office=2}, not {office=2,days=1}.
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/064_the_price_grid.zig",
        .output =
        \\price grid {products=3,tiers=3,f32}:
        \\    5.00    2.50    0.50
        \\   20.00   10.00    2.00
        \\ 1000.00  500.00  100.00
        \\stock sold out, per tier {tiers=3,f32}: $1700.00 $850.00 $170.00
        ,
        .hint =
        \\`priceGrid` is fine. What does `broad` match on?
        \\Which axis do the grid and the stock have in common?
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/065_legacy_code.zig",
        .output =
        \\per quarry {quarry=2,parts=2,f32}: { { 199, 58 }, { 10, 5 } }
        \\parts {2,3,2,f32}:
        \\  quarry 1: { { 12, 4 }, { 29, 18 }, { 158, 36 } }
        \\  quarry 2: { { 2, 1 }, { 0, 0 }, { 8, 4 } }
        ,
        .hint =
        \\Write the two shapes down and number their axes from 0.
        \\Which pair is contracted? Which pair is "for each quarry"?
        \\Name the axes first: a reduction is easier to ask for by tag.
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/066_the_valuation_engine.zig",
        .output =
        \\output {startup=2,d_out=2,f32}, runway {startup=2,f32}
        \\PetRocks.com: valued at $156.20M, burning $2.87M a month, 54.4 months to live
        \\Boring Inc.: valued at $40.00M, burning $3.50M a month, 11.4 months to live
        ,
        .hint =
        \\The panic shows both shapes. One of them has to grow before the `add`.
        \\The runway is one slice of the output divided by another.
        \\
        ,
    },
    .{
        .main_file = "09_dot_com_bubble/067_quiz_ipo_day.zig",
        .output =
        \\scores {startup=3,verdict=2,f32}, picks {startup=3,i32}, sure {startup=3,f32}
        \\PetRocks.com: buy 12, sell 4. BUY! (99.97% sure)
        \\Gravel.net: buy 22, sell 24. SELL! (88.08% sure)
        \\Foosball.org: buy 21, sell -3. BUY! (100.00% sure)
        ,
        .hint =
        \\`layer` is the previous exercise with `tag` where the name of the axis was.
        \\The three shapes on the first line of the output tell you what to squeeze.
        \\
        ,
    },
    // Chapter 10: Treasure Maps
    .{
        .main_file = "10_treasure_maps/068_the_parrot.zig",
        .output =
        \\Polly answers with a {chest=6,bool}.
        \\Worth the shovel: { true, false, false, true, true, false }
        \\Holes to dig: 3. Polly will supervise.
        ,
        .hint =
        \\Doubloons are counted in whole numbers on this ship.
        \\Nobody can add up bools. Integers, on the other hand...
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/069_the_ledger.zig",
        .output =
        \\The ledger: { 30, 5, -13, 0, 50, 0 }
        \\The crew's purse: 72 doubloons.
        ,
        .hint =
        \\Where the mask is TRUE, the value is KEPT.
        \\`select` belongs to the bool tensor, and its branches come in the order "true, false".
        \\The purse is the ledger, reduced, with no souvenir left.
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/070_x_marks_the_index.zig",
        .output =
        \\clues {stop=4,clue=3,i32}:
        \\  stop 1: 41 paces north, 42 paces east, dig 43 feet
        \\  stop 2: 1 paces north, 2 paces east, dig 3 feet
        \\  stop 3: 21 paces north, 22 paces east, dig 23 feet
        \\  stop 4: 11 paces north, 12 paces east, dig 13 feet
        \\The whole trip: 74 paces north, 78 paces east, 82 feet of digging.
        ,
        .hint =
        \\One stop is right, and only by accident.
        \\What is the index of the FIRST island?
        \\If the first line differs: the lesson says what the axis of `route` is called.
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/071_latitude_longitude.zig",
        .output =
        \\under the X's {x=3,i32}: { 8, 7, 9 }
        \\dig X number 2 first.
        \\best of each latitude {lat=4,i32}: { 7, 9, 4, 8 }
        ,
        .hint =
        \\First the order of the fields. Then look at the shape that is printed before the crash:
        \\which tag would make `best_lon` walk in step with the sea?
        \\`firstToDig` wants WHERE the maximum is, not how much, and no axis around the answer.
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/072_roll_call.zig",
        .output =
        \\the pay line {pirate=5,i32}: { 80, 80, 30, 12, 5 }
        \\their ages   {rank=5,i32}: { 19, 27, 41, 35, 63 }
        \\the first three are 29.0 years old on average.
        ,
        .hint =
        \\The panic counts three places where one tag shows up. Two is the limit.
        \\The surgeon's number: slice, convert, reduce, and leave no axis behind.
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/073_the_three_best.zig",
        .output =
        \\The Exe has 2 outputs: {best=3,i32} and {best=3,i32}.
        \\Load chest 4: 95 doubloons.
        \\Load chest 2: 80 doubloons.
        \\Load chest 0: 30 doubloons.
        \\Top-3 accuracy of the crew: 0.50
        ,
        .hint =
        \\The compile error starts inside ZML (the `note:` lines under it lead back to yours),
        \\and it quotes an example call. It is a very good example.
        \\For the bets: build the Shape of the grid once, and `broad` both sides to it.
        \\
        ,
    },
    .{
        .main_file = "10_treasure_maps/074_quiz_buried_treasure.zig",
        .output =
        \\landing parties: {best=2,role=2,i32}
        \\Island 4: 85 doubloons. Send 6 diggers and 3 lookouts.
        \\Island 3: 70 doubloons. Send 5 diggers and 2 lookouts.
        \\That makes 11 diggers and 5 lookouts. Polly stays on the ship.
        ,
        .hint =
        \\Mask first, then pick. And "safe" is the opposite of "more than 3 sharks".
        \\`topK` hands you positions along .island under a new name: just what `gather` likes.
        \\Two outputs: `eval` refuses, the runner of exercise 73 does not.
        \\
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
        .hint = "The loop is plain Zig, and so is the counter. How many days did the story ask for? The Buffer and the call are chapter 02.\n",
    },
    .{
        .main_file = "11_time_loops/076_dont_drive_angry.zig",
        .output =
        \\The thermometers say   { -4, 0, 3, 7 }
        \\An optimist feels      { -8, 0, 3, 7 }
        \\A pessimist feels      { -18, -10, -4, 2 }
        \\Hours of frostbite: 1 for the optimist, 2 for the pessimist.
        ,
        .hint = "Zig can't branch on a ghost. Chapter 10 had an operation that chooses element by element. The count starts with a call to the function you just fixed.\n",
    },
    .{
        .main_file = "11_time_loops/077_the_shadow.zig",
        .output =
        \\0 minutes of sun. Gus says { 6, 8, 15 }, and Gus's lawyer says { 6, 8, 15 }.
        \\45 minutes of sun. Gus says { -9, -7, 0 }, and Gus's lawyer says { -9, -7, 0 }.
        ,
        .hint = "A branch is an ordinary method: it uses what the struct holds. Then the panic says what dtype it wanted. Where do bool tensors come from?\n",
    },
    .{
        .main_file = "11_time_loops/078_piano_lessons.zig",
        .output =
        \\After 0 days, Phil has practiced 0 minutes.
        \\After 4 days, Phil has practiced 10 minutes.
        \\After 100 days, Phil has practiced 5050 minutes.
        \\After 3650 days, Phil has practiced 6663075 minutes.
        \\By hand, 100 days: 5050 minutes, in 100 calls. The loop Exe needs 1.
        ,
        .hint = "cond returns a Tensor, not a Zig bool. Today's practice time is already in the State. In part 2 the loop is plain Zig, and the lesson shows how a result becomes the next input.\n",
    },
    .{
        .main_file = "11_time_loops/079_word_gets_around.zig",
        .output =
        \\unrolled, 1 input,  5 days: { 32, 243, 1 }
        \\looped,   2 inputs, 0 days: { 1, 1, 1 }
        \\looped,   2 inputs, 5 days: { 32, 243, 1 }
        \\looped,   2 inputs, 12 days: { 4096, 531441, 1 }
        ,
        .hint = "The call to the loop already says which field Rumor needs. Later, the panic prints two shapes: the crowd that went in and the crowd that came out. The unrolled version starts its crowd differently.\n",
    },
    .{
        .main_file = "11_time_loops/080_quiz_getting_out.zig",
        .output =
        \\Starting from 1: free after 0 steps. Before coffee, Phil stops at 0.
        \\Starting from 6: free after 8 steps. Before coffee, Phil stops at 8.
        \\Starting from 7: free after 16 steps. Before coffee, Phil stops at 16.
        \\Starting from 27: free after 111 steps. Before coffee, Phil stops at 20.
        \\Starting from 97: free after 118 steps. Before coffee, Phil stops at 20.
        \\By hand from 27: 111 calls to the Exe, and the highest number on the way was 9232.
        ,
        .hint = "Which numbers are known at trace time, and which only at run time? Exercise 079 has the shape of part 1, the second half of 078 the shape of part 2.\n",
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
        \\In all, 35 numbers go in and 29 come out.
        ,
        .hint = "The shapes are spelled out in the lesson, and the output shapes tell you whether your Report is right. The Exe has two lists; the first one is printed already.\n",
    },
    .{
        .main_file = "12_the_exe_files/082_raw_footage.zig",
        .output =
        \\brightest pixel per frame: { 1, 9, 0.5 }
        \\total glow per frame:      { 2, 14, 0.5 }
        \\the next night, brightest: { 3, 0, 8 }
        \\the next night, glow:      { 12, 0, 15 }
        ,
        .hint = "eval is out. The lesson lists the replacement in order: args, results, set, call, get. The second photo needs only the last three.\n",
    },
    .{
        .main_file = "12_the_exe_files/083_the_stakeout.zig",
        .output =
        \\Night 1: peak 2, 0 seconds over the threshold.
        \\Night 2: peak 9, 3 seconds over the threshold.
        \\Night 3: peak 6, 2 seconds over the threshold.
        ,
        .hint = "run takes the inputs as a tuple, then a tuple of pointers to the destinations, then the call options. A count is a sum of bools that became numbers. And the two defers: one per Buffer the call made. Nobody checks those but you.\n",
    },
    .{
        .main_file = "12_the_exe_files/084_the_autopsy.zig",
        .output =
        \\The Exe has 3 inputs: {d_out=2,d=3,f32}, {d_out=2,f32} and {d=3,f32}.
        \\alien: 4.5, weather balloon: 10
        \\The detector's verdict: weather balloon.
        ,
        .hint = "compile wants the method's name as an enum literal, whatever the method is called. Then: which Buffers fill slots 0 and 1?\n",
    },
    .{
        .main_file = "12_the_exe_files/085_redacted.zig",
        .output =
        \\A Polygraph has 4 fields, its Bufferized twin has 2. Each Exe has 3 inputs.
        \\Witness 1: Scally's polygraph says lying=false, Molder's says lying=false.
        \\Witness 2: Scally's polygraph says lying=false, Molder's says lying=true.
        \\Witness 3: Scally's polygraph says lying=true, Molder's says lying=true.
        ,
        .hint = "The settings are Zig values: forward reads them at trace time. Zig names the field that doesn't exist. And which of the two agents is the paranoid one?\n",
    },
    .{
        .main_file = "12_the_exe_files/086_it_goes_all_the_way_up.zig",
        .output =
        \\3 layers, 7 inputs.
        \\What reaches the top: { 2.5, 4.5 }
        ,
        .hint = "The lesson gives the formula of a layer and the loop of the network. A Bufferized(Layer) is a struct of two Buffers. deinitAll is happy with a Layer, just not with a slice of them. If Zig then asks for a const, it is right.\n",
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
        .hint = "The first three items are things you have written before. Then read the two shapes in the panic. Which of the three ways out does the task ask for?\n",
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
        \\With Skimmer's weights, Scally's Exe finds aliens in 5 cases out of 5. Budget approved.
        ,
        .hint = "Follow the TODOs in order. What do the two agents NOT have in common? And what does Skimmer change: the model, or the numbers?\n",
    },
    // Chapter 13: Layer Cake
    .{
        .main_file = "13_layer_cake/089_the_first_tier.zig",
        .output =
        \\scores {cake=3,dout=2,f32}
        \\  sponge    sweetness  3.25  richness  1.50
        \\  brioche   sweetness  3.00  richness  6.50
        \\  meringue  sweetness  5.00  richness -1.00
        \\By hand {cake=3,dout=2,f32}: the same six numbers.
        ,
        .hint = "No label, no cake: which axis gets contracted, and does the input have it? Then read the compiler: it names the field that is missing.\n",
    },
    .{
        .main_file = "13_layer_cake/090_oven_settings.zig",
        .output =
        \\relu         0.000   0.000   0.500   2.000
        \\leakyReLU   -0.200  -0.050   0.500   2.000
        \\tanh        -0.964  -0.462   0.462   0.964
        \\sigmoid      0.119   0.378   0.622   0.881
        ,
        .hint = "Four identical rows: nobody turned the oven on. How does Zig write a union value that carries a payload? And which fields of a Tier hold a Tensor?\n",
    },
    .{
        .main_file = "13_layer_cake/091_level_the_tiers.zig",
        .output =
        \\From the pantry:
        \\  2  4  4  6  ->  7.172 10.000 10.000 12.828
        \\ 10 20 30 40  ->  7.317  9.106 10.894 12.683
        \\  5  5  5  5  -> 10.000 10.000 10.000 10.000
        \\By hand:
        \\  2  4  4  6  ->  7.172 10.000 10.000 12.828
        \\ 10 20 30 40  ->  7.317  9.106 10.894 12.683
        \\  5  5  5  5  -> 10.000 10.000 10.000 10.000
        ,
        .hint = "nan? Work out what gets divided by what for the flat cake. Which field sets the mean, which the spread? By hand: `mean` keeps the axis it reduces, with a size of 1.\n",
    },
    .{
        .main_file = "13_layer_cake/092_no_such_struct.zig",
        .output =
        \\(  3,  4)   rms norm  1.697  1.131   unit length  0.600  0.800
        \\( 30, 40)   rms norm  1.697  1.131   unit length  0.600  0.800
        \\(  0, -5)   rms norm  0.000 -1.414   unit length  0.000 -1.000
        ,
        .hint = "The numbers of one cake live along one axis. `rmsNorm` only normalizes: the weight is your job. And an f32 is not a Tensor.\n",
    },
    .{
        .main_file = "13_layer_cake/093_the_menu.zig",
        .output =
        \\order {s=4,d=3,f32}
        \\  croissant  sweet 0.2  buttery 0.9  crunchy 0.7
        \\  eclair     sweet 0.8  buttery 0.6  crunchy 0.1
        \\  eclair     sweet 0.8  buttery 0.6  crunchy 0.1
        \\  macaron    sweet 0.9  buttery 0.2  crunchy 0.5
        \\By hand {s=4,d=3,f32}: the same twelve numbers.
        \\The whole order {d=3,f32}: sweet 0.675  buttery 0.575  crunchy 0.350
        ,
        .hint = "The panic names the dtype it received. The ids are written on the menu, in the comments. By hand: which axis of the table do the ids replace?\n",
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
        \\sunday, gentle oven  0.587 -1.408  0.821
        ,
        .hint = "The compiler names a field that does not exist: if Bufferized dropped it, the Exe does not need it. The oven is frozen into the Exe, the weights are not.\n",
    },
    .{
        .main_file = "13_layer_cake/095_sweet_or_salty.zig",
        .output =
        \\verdicts {cake=4,i32}, confidence {cake=4,f32}
        \\  plain          -> rejected (62.2% sure)
        \\  strawberry     -> served   (81.8% sure)
        \\  olive          -> served   (81.8% sure)
        \\  salted caramel -> rejected (92.4% sure)
        ,
        .hint = "The panic names a tag, and the shape that does not have it. Then: is there an oven between the tiers? Two outputs need a runner.\n",
    },
    .{
        .main_file = "13_layer_cake/096_quiz_the_wedding_cake.zig",
        .output =
        \\occasions {order=3,i32}, confidence {order=3,f32}
        \\  2 croissants, 2 baguettes -> breakfast (87.3% sure)
        \\  2 eclairs, 2 macarons     -> tea time  (70.0% sure)
        \\  3 brownies, 1 eclair      -> a wedding (97.5% sure)
        ,
        .hint = "Seven steps, in order. A panic names the shapes it got: hold them against the table of shapes in the lesson, and you know which step is off.\n",
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
        \\The combination of locker 3: { 36, 24, 36 }
        ,
        .hint = "Three things about the file layout: which end of the integer comes first, where the header starts, and where byte 0 of the data is. The header you print tells you the rest.\n",
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
        \\The coach: 3 tensors, 60 bytes.
        \\The plates start at byte 808 of the file.
        \\coach.0.bias: not in the logbook.
        ,
        .hint = "FileNotFound, for a file that is right there? The lesson has a paragraph about a quirk. An iterator entry holds two pointers.\n",
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
        .hint = "Each panic names an entry of the file. Compare what was asked with what the logbook of 098 says. Views chain: a prefix, then a layer.\n",
    },
    .{
        .main_file = "14_heavy_lifting/100_the_spotter.zig",
        .output =
        \\plates      {plate=4,f32}: { 20, 10, 5, 2.5 }
        \\combination {turn=3,i32}: { 36, 24, 36 }
        \\on the bar  {f32}: 95 kg
        ,
        .hint = "Two moves: an empty bar, then the spotter. Four calls and a `defer`. A sum keeps its axis, with a size of 1.\n",
    },
    .{
        .main_file = "14_heavy_lifting/101_meet_the_coach.zig",
        .output =
        \\Olga  LIGHT WEIGHT, BABY!
        \\Chad  NEVER SKIP LEG DAY.
        \\You   DO YOU EVEN LIFT?
        ,
        .hint = "A panic about `0.weight`: where does the view stand? And if Coach says LIGHT WEIGHT about a 0.4 bench press, a tensor of the file is not being used.\n",
    },
    .{
        .main_file = "14_heavy_lifting/102_quiz_the_smoothie_bar.zig",
        .output =
        \\verdicts {member=3,i32}, receipts {member=3,dout=2,f32}
        \\  Olga LIGHT WEIGHT, BABY!  2 banana, 1 spinach: $8.50, taste 3
        \\  Chad NEVER SKIP LEG DAY.  1 banana, 3 spinach: $7.50, taste 0
        \\  You  DO YOU EVEN LIFT?    0 banana, 0 spinach: $0.50, taste 0
        ,
        .hint = "Draw the tree of the file first: every dot in a name is one struct deeper. Not every layer in there has a bias. Build leaves first: `linear`, `Coach`, `Blendtron`, `Gym`, then `main`.\n",
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
        .hint = "Step 2: whose attention must add up to 1, each row (a listener) or each column (a speaker)? And when ZML panics about a tag: values belong to speakers.\n",
    },
    .{
        .main_file = "15_pay_attention/104_no_spoilers.zig",
        .output =
        \\no spoilers:
        \\  the mask {q=4,k=4,f32}:
        \\        0  -inf  -inf  -inf
        \\        0     0  -inf  -inf
        \\        0     0     0  -inf
        \\        0     0     0     0
        \\  the official mask agrees: true
        \\  the weights:
        \\        Tina  Kuzco Paca  Dolly
        \\  Tina  1.000 0.000 0.000 0.000
        \\  Kuzco 0.804 0.196 0.000 0.000
        \\  Paca  0.333 0.333 0.333 0.000
        \\  Dolly 0.250 0.250 0.250 0.250
        \\a memory of 2 llamas:
        \\  the mask {q=4,k=4,f32}:
        \\        0  -inf  -inf  -inf
        \\        0     0  -inf  -inf
        \\     -inf     0     0  -inf
        \\     -inf  -inf     0     0
        \\  the official mask agrees: true
        \\  the weights:
        \\        Tina  Kuzco Paca  Dolly
        \\  Tina  1.000 0.000 0.000 0.000
        \\  Kuzco 0.804 0.196 0.000 0.000
        \\  Paca  0.000 0.500 0.500 0.000
        \\  Dolly 0.000 0.000 0.500 0.500
        ,
        .hint = "Two aranges, two tags, one `broad` each, and the table knows its rows and its columns. An Exe without inputs is run with `.{}`. Then: what must happen BEFORE the 100% is shared out?\n",
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
        .hint = "`byHand` is 103 + 104 with renames at both ends; `official` is the same renames around one call. The last panic lists the tags `sdpa` wants: which one has `main` never heard of?\n",
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
        .hint = "How many heads, and who does the division? `merge` undoes exactly what `splitAxis` did. And count a llama's ears before you trust the loop.\n",
    },
    .{
        .main_file = "15_pay_attention/107_spin_the_llama.zig",
        .output =
        \\after the spin {s=4,hd=2,f32}:
        \\  position 0: ( 1.000,  0.000)  length 1.000
        \\  position 1: ( 0.540,  0.841)  length 1.000
        \\  position 2: (-0.416,  0.909)  length 1.000
        \\  position 3: (-0.990,  0.141)  length 1.000
        \\facing (3, 4), spun by hand, and by rope:
        \\  position 0: ( 3.000,  4.000)  ( 3.000,  4.000)
        \\  position 1: (-1.745,  4.686)  (-1.745,  4.686)
        \\  position 2: (-4.886,  1.063)  (-4.886,  1.063)
        \\  position 3: (-3.534, -3.537)  (-3.534, -3.537)
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
        .hint = "For the formula: slice `.hd` in two halves of size 1, and give the angles the shape of a half. For the tables: a score depends on the distance only if BOTH llamas have turned.\n",
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
        .hint = "Keys and values are filed the same way. For the mask: which rows are filled once the llama at `pos` is in? In `main`, the archive that comes out replaces the one that went in.\n",
    },
    .{
        .main_file = "15_pay_attention/109_quiz_the_whole_herd.zig",
        .output =
        \\after 1 round(s) of gossip {s=4,d=4,f32}:
        \\  Tina   0.000  0.000  2.500  0.000
        \\  Kuzco  1.203  0.000  0.750  0.000
        \\  Paca   1.401  0.000  2.011  0.000
        \\  Dolly  0.173  1.452  0.567  1.988
        \\after 2 round(s) of gossip {s=4,d=4,f32}:
        \\  Tina   0.000  0.000  6.250  0.000
        \\  Kuzco  1.428  0.000  2.783  0.000
        \\  Paca   2.059  0.000  4.656  0.000
        \\  Dolly  0.294  2.721  1.779  5.376
        \\after 3 round(s) of gossip {s=4,d=4,f32}:
        \\  Tina   0.000  0.000 15.625  0.000
        \\  Kuzco  1.681  0.000  7.252  0.000
        \\  Paca   3.085  0.000 11.602  0.000
        \\  Dolly  0.441  5.442  4.448 16.127
        ,
        .hint = "Steps 1 to 5, one at a time. When ZML panics, the shapes in the message tell you which tag is off. In `main`: four Linears want four placeholders of their own, and the result of a round is the input of the next one.\n",
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
        .hint = "Which state does the second hand come from? `Tensor.stack` wants the tensors, a position and a tag for the new axis.\n",
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
        .hint = "Tokens first, Rng second, in `pull` and in the pointers. Then read the panic: which tag is it looking for, and what are the logits tagged?\n",
    },
    .{
        .main_file = "16_casino_royale/113_let_it_ride.zig",
        .output =
        \\the sure thing
        \\  the odds: pocket 1 1.000
        \\  dice were rolled: false
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: true
        \\  pocket 3 got at least a quarter of the spins: false
        \\the honest table
        \\  the odds: pocket 1 0.506 pocket 4 0.307 pocket 3 0.186
        \\  dice were rolled: true
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: false
        \\  pocket 3 got at least a quarter of the spins: false
        \\the cold table
        \\  the odds: pocket 1 1.000 pocket 4 0.000 pocket 3 0.000
        \\  dice were rolled: true
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: true
        \\  pocket 3 got at least a quarter of the spins: false
        \\the hot table
        \\  the odds: pocket 1 0.335 pocket 4 0.333 pocket 3 0.332
        \\  dice were rolled: true
        \\  an outsider (pocket 0, 2 or 5) came up: false
        \\  the favorite came up every single time: false
        \\  pocket 3 got at least a quarter of the spins: true
        ,
        .hint = "`odds`: top-k, temperature, softmax, and mind which axis is left. The comment above each table is its strategy. Inputs go in a tuple, outputs are pointers: tokens first.\n",
    },
    .{
        .main_file = "16_casino_royale/114_quiz_the_house_always_wins.zig",
        .output =
        \\the house always wins please play again.
        \\After a few drinks, the sign went off script: true
        \\It still got the next word right more often than not: true
        ,
        .hint = "Build it in the order of the chapters: fields, `init`, `forward`, `load`, then `main`. Run the sober Exe first; the night shift is the same loop with a counter instead of a print.\n",
    },
    // Chapter 17: The Customs Office
    .{
        .main_file = "17_customs_office/115_nothing_to_declare.zig",
        .output =
        \\In the suitcase: 4 tensors
        \\  fc1.bias   {500,f32}
        \\  fc1.weight {500,784,f32}
        \\  fc2.bias   {10,f32}
        \\  fc2.weight {10,500,f32}
        \\Every tensor is f32: true
        \\Parameters: 397510
        \\What the matrices say:
        \\  fc1.weight: 784 numbers in, 500 numbers out
        \\  fc2.weight: 500 numbers in, 10 numbers out
        ,
        .hint = "Three loops over `registry.iterator()`: `entry.key_ptr.*` is the name, `entry.value_ptr.shape` the shape. A Linear weight is {out, in}.\n",
    },
    .{
        .main_file = "17_customs_office/116_fill_in_the_form.zig",
        .output =
        \\fc1.weight {dout=500,d=784,f32}
        \\fc1.bias   {dout=500,f32}
        \\fc2.weight {dout=10,d=500,f32}
        \\fc2.bias   {dout=10,f32}
        \\{d=784,f32} in, {dout=10,f32} out
        \\In a blank form, M. Nist sees a 5
        ,
        .hint = "Fill in the blanks first, then run and read the two panics one after the other: the first quotes a name, the second quotes two shapes. Compare each with what exercise 115 printed.\n",
    },
    .{
        .main_file = "17_customs_office/117_the_handwriting_test.zig",
        .output =
        \\The register: 10000 parcels of 28 x 28 pixels
        \\Parcel 0:
        \\............................
        \\............................
        \\............................
        \\............................
        \\............................
        \\............................
        \\............................
        \\.......###..................
        \\......###############.......
        \\..........############......
        \\...................##.......
        \\...................##.......
        \\..................##........
        \\.................###........
        \\.................##.........
        \\................###.........
        \\................##..........
        \\................##..........
        \\...............##...........
        \\..............###...........
        \\.............###............
        \\.............##.............
        \\............##..............
        \\...........###..............
        \\...........###..............
        \\...........###..............
        \\...........##...............
        \\............................
        \\parcel    0: M. Nist reads a 7
        \\parcel    1: M. Nist reads a 2
        \\parcel    2: M. Nist reads a 1
        \\parcel    3: M. Nist reads a 0
        \\parcel    4: M. Nist reads a 4
        \\parcel  115: M. Nist reads a 4
        \\parcel  674: M. Nist reads a 5
        \\parcel 1522: M. Nist reads a 9
        \\Same as PyTorch: 8 / 8
        ,
        .hint = "If parcel 0 does not look like a 7, the bug is in `Register`. If it does and the digits are off, compare `forward` with the seven Python lines, one by one.\n",
    },
    .{
        .main_file = "17_customs_office/118_trust_but_verify.zig",
        .output =
        \\Tolerances: absolute 0.001, relative 0.01, close fraction 0.999
        \\Digits like PyTorch:       8 / 8
        \\Hidden layer like PyTorch: 8 / 8
        \\Scores like PyTorch:       8 / 8
        ,
        .hint = "With a relative tolerance of 1, how far from 30 may a score be? Fix `opts`, write the hidden check, then look at which of the two counts is 0, and at the sign of the scores PyTorch gets.\n",
    },
    .{
        .main_file = "17_customs_office/119_rush_hour.zig",
        .output =
        \\200 parcels in one call: {b=200,i32}
        \\parcel 149: M. Nist reads a 9, the label says 2
        \\Like PyTorch: 200 / 200
        \\Like the label: 199 / 200
        ,
        .hint = "Read the shapes in the panic: who made a `.d` of 156800? Name the axes instead of counting them: `merge` for the pixels, a tag for `argMax` and `squeeze`.\n",
    },
    .{
        .main_file = "17_customs_office/120_quiz_port_of_entry.zig",
        .output =
        \\In the suitcase: 4 tensors, 397510 parameters
        \\The register: 10000 parcels of 28 x 28 pixels
        \\Scores like PyTorch on the first 200 parcels: true
        \\Digits like PyTorch: 1000 / 1000
        \\Digits like the label: 985 / 1000
        ,
        .hint = "Build it in the order of the list, and run after each step. Exercises 115 to 119 each hold one piece. If the scores do not match, the report on stderr says how wrong: rounding, or a different computation?\n",
    },
    // Chapter 18: Lost in Translation
    .{
        .main_file = "18_lost_in_translation/121_the_resume.zig",
        .output =
        \\Bertie's resume:
        \\  layers: 2
        \\  hidden size: 128
        \\  heads: 2, with 64 numbers each
        \\  intermediate size: 512
        \\  vocabulary: 30522 tokens
        \\  longest sentence: 512 tokens
        \\  activation: gelu
        \\  layer norm eps: 1e-12
        \\The checkpoint:
        \\  word table {30522,128,f32}, as announced: true
        \\  layers in the file: 2, as announced: true
        ,
        .hint = "`error.UnknownField` is about the parser's options, `error.MissingField` about a field with no default. Where does HuggingFace keep the values a config.json does not mention?\n",
    },
    .{
        .main_file = "18_lost_in_translation/122_the_glossary.zig",
        .output =
        \\The capital of France is [MASK].
        \\  [CLS] the capital of france is [MASK] . [SEP]
        \\  ours:    { 101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102 }
        \\  PyTorch: { 101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102 }
        \\  same ids: true
        \\  [MASK] is token 103, at position 6
        \\Unintelligible, said Bertie.
        \\  un ##int ##elli ##gible , said bertie .
        \\  and back: unintelligible, said bertie.
        ,
        .hint = "Count the ids on the two lines. What does Python put around a sentence that the ZML encoder does not?\n",
    },
    .{
        .main_file = "18_lost_in_translation/123_three_dictionaries.zig",
        .output =
        \\in the file, under bert.embeddings.
        \\  LayerNorm.bias {128,f32}
        \\  LayerNorm.weight {128,f32}
        \\  position_embeddings.weight {512,128,f32}
        \\  token_type_embeddings.weight {2,128,f32}
        \\  word_embeddings.weight {30522,128,f32}
        \\embeddings: {s=9,d=128,f32}
        \\same as PyTorch: true
        ,
        .hint = "A panic naming a tensor: compare it, letter by letter, with the names printed just above. If it says `false`: `token_type_ids=None` does not mean \"no token types\".\n",
    },
    .{
        .main_file = "18_lost_in_translation/124_a_single_thought.zig",
        .output =
        \\attention half, same as PyTorch: true
        \\whole layer, same as PyTorch: true
        \\whole layer, strictly the same: true
        ,
        .hint = "Where is the LayerNorm in `BertOutput.forward`: before the dense layers, or around the sum? If only the strict check fails, look at which GELU is called.\n",
    },
    .{
        .main_file = "18_lost_in_translation/125_second_thoughts.zig",
        .output =
        \\layers on the resume: 2
        \\layers in the model: 2, in the weights: 2
        \\compared with bert.encoder.layer.1.out
        \\hidden states: {s=9,d=128,f32}
        \\same as PyTorch: true
        ,
        .hint = "The view given to `Encoder.init` stands on \"bert\". Layer names count from 0. Buffers first, then the slice that held them.\n",
    },
    .{
        .main_file = "18_lost_in_translation/126_the_missing_page.zig",
        .output =
        \\cls.predictions.decoder.weight in the file: false
        \\cls.predictions.bias in the file: true
        \\scores: {voc=30522,f32}, same as PyTorch: true
        \\The capital of France is [MASK].
        \\[MASK] = france
        ,
        .hint = "No tensor named `cls.predictions.decoder.weight`: correct, and the file is not broken. What does the reference say about that name?\n",
    },
    .{
        .main_file = "18_lost_in_translation/127_quiz_mind_the_gap.zig",
        .output =
        \\He was born in New [MASK] City.
        \\  Bertie: york, PyTorch: york
        \\He opened the [MASK] and went inside.
        \\  Bertie: door, PyTorch: door
        \\The prime [MASK] of Japan.
        \\  Bertie: minister, PyTorch: minister
        \\The interpreter speaks English and [MASK].
        \\  Bertie: english, PyTorch: english
        \\They agree on 4 sentences out of 4.
        ,
        .hint = "Build it in the order of the chapter, and reuse your own exercises: 121 for the config, 122 for the ids, 123 to 125 for the encoder, 126 for the head.\n",
    },
    // Chapter 19: Once Upon a Time
    .{
        .main_file = "19_once_upon_a_time/128_the_donation_box.zig",
        .output =
        \\The box: 108 tensors, 3745984 numbers.
        \\The biggest: transformer.wte.weight {50257,64,f32}, 85% of the numbers.
        \\lm_head.weight is in the box: false
        \\Blocks: 8
        \\The Linears of block 0:
        \\  attn.attention.q_proj    {64,64,f32}   bias: no
        \\  attn.attention.k_proj    {64,64,f32}   bias: no
        \\  attn.attention.v_proj    {64,64,f32}   bias: no
        \\  attn.attention.out_proj  {64,64,f32}   bias: yes
        \\  mlp.c_fc                 {256,64,f32}  bias: yes
        \\  mlp.c_proj               {64,256,f32}  bias: yes
        \\The voice of block 0 is wired: 4 Linears, 1 bias.
        ,
        .hint = "Two panics wait behind the `???`s, and each one names the tensor it could not find. Compare that name, letter by letter, with the names listed in the lesson. Then: is every Linear supposed to have that tensor?\n",
    },
    .{
        .main_file = "19_once_upon_a_time/129_page_numbers.zig",
        .output =
        \\"Once upon a time, there was a little dragon"
        \\10 tokens: { 7454, 2402, 257, 640, 11, 612, 373, 257, 1310, 10441 }
        \\pieces: [Once][Ġupon][Ġa][Ġtime][,][Ġthere][Ġwas][Ġa][Ġlittle][Ġdragon]
        \\same ids as PyTorch: true
        \\the position table: {pos=2048,d=64,f32}
        \\token vectors match PyTorch: true
        \\position vectors match PyTorch: true
        \\what block 0 receives matches PyTorch: true
        ,
        .hint = "Four `false`, and stderr complains about shapes before it complains about numbers. Count the tokens, and look at the first piece: who asked for it?\n",
    },
    .{
        .main_file = "19_once_upon_a_time/130_no_square_roots.zig",
        .output =
        \\16 heads of 4 numbers each
        \\blocks: global local global local global local global local
        \\a local block hears the last 256 tokens
        \\block 0 (global),  10 tokens, matches PyTorch: true
        \\block 1 (local),   10 tokens, matches PyTorch: true
        \\block 1 (local),  300 tokens, matches PyTorch: true
        ,
        .hint = "Each fix turns different lines true. All three false: what does `sdpa` do to the scores when you tell it nothing? Only the 300-token line false: which blocks are local, and who tells the mask?\n",
    },
    .{
        .main_file = "19_once_upon_a_time/131_bath_first.zig",
        .output =
        \\activation_function: gelu_new
        \\the activation matches PyTorch, strictly: true
        \\ln_1 matches PyTorch: true
        \\attn matches PyTorch: true
        \\mlp matches PyTorch: true
        \\the whole block matches PyTorch: true
        ,
        .hint = "`mlp` true and the whole block false: the parts are fine, their order is not. Put Pip's two lines next to the two lines of the lesson. The strict line is about one function only, and about one key of config.json.\n",
    },
    .{
        .main_file = "19_once_upon_a_time/132_the_same_book_twice.zig",
        .output =
        \\8 blocks between the two tables and the head
        \\what the head receives matches PyTorch: true
        \\the scores match PyTorch: true
        \\"Once upon a time, there was a little dragon" and then:
        \\  1. "."
        \\  2. " named"
        \\  3. " who"
        ,
        .hint = "`Checkpoint has no tensor named lm_head.weight`: it is not coming. Which tensor does the class tie it to? If both checks then say false, compare `hidden` with the last line of the PyTorch loop.\n",
    },
    .{
        .main_file = "19_once_upon_a_time/133_one_page_one_word.zig",
        .output =
        \\10 tokens on a page of 43, the scores match PyTorch: true
        \\Once upon a time, there was a little dragon. He lived in a big castle with his family. One day, the dragon was very hungry. He wanted to eat some food, but he was too hungry.
        ,
        .hint = "A story that falls apart right after the prompt: the scores are read at the wrong place of the page. The place of the last real token changes at every turn of the loop, so it cannot be baked into the Exe.\n",
    },
    .{
        .main_file = "19_once_upon_a_time/134_quiz_the_owl_who_read.zig",
        .output =
        \\9 tokens in, 33 tokens out
        \\The old owl wanted to read a story to the owl. He asked the owl if he could read the story. The owl said yes and the owl was happy.
        \\the same story as PyTorch: true
        ,
        .hint = "Build it in the order of the chapter: Config, linear, Attention, Mlp, Block, the model, then `main`. If the owl goes off script, the checks of exercises 129 to 132 still work on your structs: `activations.safetensors` is still in the directory.\n",
    },
    // Chapter 20: The Autopsy
    .{
        .main_file = "20_the_autopsy/135_the_reference_body.zig",
        .output =
        \\transformer.h.0.ln_1: ok
        \\transformer.h.0.attn.attention: ok
        \\transformer.h.0.attn.attention.q_proj: ok
        \\transformer.h.0.mlp: ok
        \\transformer.h.0: ok
        \\transformer.ln_f: ok
        \\The control, block 0 on the table with the papers of block 1:
        \\transformer.h.1: DIVERGES
        \\Dr. Mortis was fooled: false
        ,
        .hint = "`runAlone` is exercise 101 in four lines: a placeholder, `platform.compile(..., layer, .forward, .{x}, .{})`, `exe.eval` with the weights and the input. `examine` needs two keys, `name.in` and `name.out`: one `bufPrint` each.\n",
    },
    .{
        .main_file = "20_the_autopsy/136_where_it_starts.zig",
        .output =
        \\The chained walk, each stage fed by the port:
        \\transformer.drop.in: ok
        \\transformer.h.0.out: ok
        \\transformer.h.1.out: ok
        \\transformer.h.2.out: ok
        \\transformer.h.3.out: ok
        \\transformer.h.4.out: ok
        \\transformer.h.5.out: ok
        \\transformer.h.6.out: ok
        \\transformer.h.7.out: ok
        \\transformer.ln_f.out: ok
        \\The trouble starts at: nowhere
        \\The isolated walk, each layer fed by PyTorch:
        \\transformer.h.0: ok
        \\transformer.h.1: ok
        \\transformer.h.2: ok
        \\transformer.h.3: ok
        \\transformer.h.4: ok
        \\transformer.h.5: ok
        \\transformer.h.6: ok
        \\transformer.h.7: ok
        \\transformer.ln_f: ok
        \\Inside block 0:
        \\transformer.h.0.ln_1: ok
        \\transformer.h.0.attn.attention: ok
        \\transformer.h.0.ln_2: ok
        \\transformer.h.0.mlp: ok
        \\last_logits: ok
        \\Once upon a time, there was a little dragon. He lived in a big castle with his family. One day, the dragon was very hungry. He wanted to eat some food, but he was too hungry.
        \\
        \\The dragon said, "
        ,
        .hint = "For the walk: keep one `stream` Buffer, and replace it after each stage (deinit the old one). Then let the verdicts choose what you read: which part of block 0 is sick, and is there any code of yours in it?\n",
    },
    .{
        .main_file = "20_the_autopsy/137_between_the_organs.zig",
        .output =
        \\Block 0, part by part:
        \\transformer.h.0.ln_1: ok
        \\transformer.h.0.attn.attention: ok
        \\transformer.h.0.ln_2: ok
        \\transformer.h.0.mlp: ok
        \\The attention of block 0, leaf by leaf:
        \\transformer.h.0.attn.attention.q_proj: ok
        \\transformer.h.0.attn.attention.k_proj: ok
        \\transformer.h.0.attn.attention.v_proj: ok
        \\transformer.h.0.attn.attention.out_proj: ok
        \\The probe, in the middle of the glue:
        \\transformer.h.0.attn.attention.attn_dropout.in: ok
        \\last_logits: ok
        \\Once upon a time, there was a little dragon. He lived in a big castle with his family. One day, the dragon was very hungry. He wanted to eat some food, but he was too hungry.
        \\
        \\The dragon said, "
        ,
        .hint = "The probe test is `runAlone` with `.probs` instead of `.forward`. For the fix, put `probs` next to the six PyTorch lines of the lesson and count the operations.\n",
    },
    .{
        .main_file = "20_the_autopsy/138_mislabelled.zig",
        .output =
        \\The labels, as ZML reads them (PyTorch says [out, in]):
        \\c_fc.weight {dout=256,d=64,f32}
        \\q_proj.weight {dout=64,d=64,f32}
        \\The isolated walk, each layer fed by PyTorch:
        \\transformer.h.0: ok
        \\transformer.h.1: ok
        \\transformer.h.2: ok
        \\transformer.h.3: ok
        \\transformer.h.4: ok
        \\transformer.h.5: ok
        \\transformer.h.6: ok
        \\transformer.h.7: ok
        \\transformer.ln_f: ok
        \\Inside block 0:
        \\transformer.h.0.ln_1: ok
        \\transformer.h.0.attn.attention: ok
        \\transformer.h.0.attn.attention.q_proj: ok
        \\transformer.h.0.attn.attention.k_proj: ok
        \\transformer.h.0.attn.attention.v_proj: ok
        \\transformer.h.0.attn.attention.out_proj: ok
        \\transformer.h.0.ln_2: ok
        \\transformer.h.0.mlp: ok
        \\transformer.h.0.mlp.c_fc: ok
        \\transformer.h.0.mlp.c_proj: ok
        \\last_logits: ok
        \\Once upon a time, there was a little dragon. He lived in a big castle with his family. One day, the dragon was very hungry. He wanted to eat some food, but he was too hungry.
        \\
        \\The dragon said, "
        ,
        .hint = "Ten calls of `m.examine`, with `block.attn.q_proj` and `weights.attn.q_proj` style pairs. For the fix: compare the two shapes printed at the top with PyTorch's [out, in].\n",
    },
    .{
        .main_file = "20_the_autopsy/139_looks_alive.zig",
        .output =
        \\The isolated walk, default tolerance:
        \\transformer.h.0: ok
        \\transformer.h.1: ok
        \\transformer.h.2: ok
        \\transformer.h.3: ok
        \\transformer.h.4: ok
        \\transformer.h.5: ok
        \\transformer.h.6: ok
        \\transformer.h.7: ok
        \\transformer.ln_f: ok
        \\Inside block 7, default tolerance:
        \\transformer.h.7.ln_1: ok
        \\transformer.h.7.attn.attention: ok
        \\transformer.h.7.attn.attention.q_proj: ok
        \\transformer.h.7.attn.attention.k_proj: ok
        \\transformer.h.7.attn.attention.v_proj: ok
        \\transformer.h.7.attn.attention.out_proj: ok
        \\transformer.h.7.ln_2: ok
        \\transformer.h.7.mlp: ok
        \\transformer.h.7.mlp.c_fc: ok
        \\transformer.h.7.mlp.c_proj: ok
        \\The isolated walk, strict:
        \\transformer.h.0: ok
        \\transformer.h.1: ok
        \\transformer.h.2: ok
        \\transformer.h.3: ok
        \\transformer.h.4: ok
        \\transformer.h.5: ok
        \\transformer.h.6: ok
        \\transformer.h.7: ok
        \\transformer.ln_f: ok
        \\Inside block 0, strict:
        \\transformer.h.0.ln_1: ok
        \\transformer.h.0.attn.attention: ok
        \\transformer.h.0.attn.attention.q_proj: ok
        \\transformer.h.0.attn.attention.k_proj: ok
        \\transformer.h.0.attn.attention.v_proj: ok
        \\transformer.h.0.attn.attention.out_proj: ok
        \\transformer.h.0.ln_2: ok
        \\transformer.h.0.mlp: ok
        \\transformer.h.0.mlp.c_fc: ok
        \\transformer.h.0.mlp.c_proj: ok
        \\last_logits: ok
        \\Once upon a time, there was a little dragon. He lived in a big castle with his family. One day, the dragon was very hungry. He wanted to eat some food, but he was too hungry.
        \\
        \\The dragon said, "
        ,
        .hint = "`strict` is a `CompareOpts` with two fields set. When ln_1 and ln_2 are sick in every block and they are library code, look at what they were given, and where `config.json` spells it.\n",
    },
    .{
        .main_file = "20_the_autopsy/140_quiz_the_cold_case.zig",
        .output =
        \\transformer.drop.in: ok
        \\transformer.h.0: ok
        \\transformer.h.1: ok
        \\transformer.h.2: ok
        \\transformer.h.3: ok
        \\transformer.h.4: ok
        \\transformer.h.5: ok
        \\transformer.h.6: ok
        \\transformer.h.7: ok
        \\transformer.ln_f: ok
        \\transformer.h.0.ln_1: ok
        \\transformer.h.0.attn.attention: ok
        \\transformer.h.0.attn.attention.q_proj: ok
        \\transformer.h.0.attn.attention.k_proj: ok
        \\transformer.h.0.attn.attention.v_proj: ok
        \\transformer.h.0.attn.attention.out_proj: ok
        \\transformer.h.0.ln_2: ok
        \\transformer.h.0.mlp: ok
        \\transformer.h.0.mlp.c_fc: ok
        \\transformer.h.0.mlp.c_proj: ok
        \\last_logits: ok
        \\Once upon a time, there was a little dragon. He lived in a big castle with his family. One day, the dragon was very hungry. He wanted to eat some food, but he was too hungry.
        \\
        \\The dragon said, "
        ,
        .hint = "Fix what the verdicts point at, run again: a parent whose children are all ok is sick in its glue, and that goes for a Block too. The lesson has the author's code for each place.\n",
    },
    // Chapter 21: The Graduation
    .{
        .main_file = "21_the_graduation/141_the_transcript.zig",
        .output =
        \\tensors in the file: 47
        \\  named _orig_mod.something: 47
        \\  biases: 0
        \\the token table: {512,64,f32}
        \\wq: {64,64,f32}  wk: {32,64,f32}
        \\            config  checkpoint
        \\vocabulary     512         512
        \\dim             64          64
        \\layers           5           5
        \\kv heads         4           4
        \\hidden size, by the formula of model.py: 172
        \\hidden size, by the shape of w1: 172
        \\a separate output matrix: false
        ,
        .hint = "`error.NoSuchTensor`: the names of model.py are not the names of the file, and `main` prints how every one of them starts. `hasKey` matches the START of a name. In Zig, `/` on integers already rounds down.\n",
    },
    .{
        .main_file = "21_the_graduation/142_cap_and_gown.zig",
        .output =
        \\w1 {dout=172,d=64,f32}, w3 {dout=172,d=64,f32}, w2 {dout=64,d=172,f32}
        \\the gown {s=5,d=64,f32} matches PyTorch: true
        \\the cap {s=5,d=64,f32} matches PyTorch: true
        ,
        .hint = "The gown: `zml.nn.rmsNorm` is `_norm`, the weight is yours (a `broad` away). The cap prints `false`: in the Python, which of the two wide matrices sits inside `F.silu(...)`? The missing test is the one above it with other names.\n",
    },
    .{
        .main_file = "21_the_graduation/143_turn_the_tassel.zig",
        .output =
        \\the queries, after the turn: {s=5,h=8,hd=8,f32}
        \\  the first token matches PyTorch: true
        \\  all 5 tokens match PyTorch: true
        ,
        .hint = "`splitAxis(.dout, .{ .h = ..., .hd = .auto })`. `true` then `false`: position 0 is turned by an angle of 0, so it proves nothing. `reshape(..., -1, 2)` pairs each number with the one right after it: which `Layout` is that?\n",
    },
    .{
        .main_file = "21_the_graduation/144_sharing_notes.zig",
        .output =
        \\8 heads ask, 4 heads answer: 2 share each set of notes
        \\the attention {s=5,d=64,f32} matches PyTorch: true
        ,
        .hint = "Three Linears, three splits (not all with the same number of heads), `rope` on q and k, renames to `.q` and `.k`, `sdpa` with a causal mask, `merge`, `wo`. The panic shows k and v with `hd=4`: 32 numbers cut in how many heads?\n",
    },
    .{
        .main_file = "21_the_graduation/145_the_speech.zig",
        .output =
        \\the prompt, in tokens: { 1, 403, 407, 261, 378 }
        \\the logits {s=5,voc=512,f32} match PyTorch: true
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do.
        ,
        .hint = "The block is the two lines of Python, with `rmsNorm(x, weight)` for the norms. `init` allocates the slice of blocks; `forward` ends with a `dot` against the token table. The prefix belongs where the view is made. Four prompt tokens instead of five: who is missing?\n",
    },
    .{
        .main_file = "21_the_graduation/146_by_heart.zig",
        .output =
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do.
        \\the same speech as PyTorch, token for token: true
        \\positions run through the model, by heart: 105
        \\positions run through the model, yesterday: 10706
        ,
        .hint = "`positions`: an `arange` along `.s`, plus `pos`. `filed`: two `dynamicUpdateSlice`, offsets `.layer` and `.k`, and the keys are spun BEFORE they are filed (as `.k`, not `.s`). The mask is exercise 108's with one row per query. In `main`: the prefill runs once at position 0; each decode feeds the token that just came out, at ITS position.\n",
    },
    .{
        .main_file = "21_the_graduation/147_quiz_the_graduation.zig",
        .output =
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do.
        \\The speech is PyTorch's, token for token: true
        \\For the photograph, dice in hand, the same speech again: true
        \\At the party, Tiny told another story: true
        \\It started with "Once upon a time": true
        \\Every word of it was one of her 512: true
        \\Class dismissed.
        ,
        .hint = "Build it in the order of the chapter: 142, 143, 144, 145, and get the greedy speech right WITHOUT a cache first. Then 146. Then `sampleTokens` instead of the argMax, and the Rng threaded like the cache. A wrong first line? Compare with `activations.safetensors`, layer by layer.\n",
    },
    .{
        .main_file = "22_the_watchmaker/148_half_a_tick.zig",
        .output =
        \\f32:  0.1000 3.1416 1000.0010 70000.0000 16777216.0000
        \\bf16: 0.1001 3.1406 1000.0000 70144.0000 16777216.0000
        \\f16:  0.1000 3.1406 1000.0000 inf inf
        \\One layer in f32:  67108864 bytes
        \\One layer in bf16: 33554432 bytes
        \\One layer in f16:  33554432 bytes
        ,
        .hint = "Down then back up: two `convert`. A Shape can change dtype with `withDtype`, and knows its `byteSize`.\n",
    },
    .{
        .main_file = "22_the_watchmaker/149_under_the_loupe.zig",
        .output =
        \\bits: 3f800000 3dcccccd c0200000 3e200000
        \\{part=3,reading=4,u32}
        \\1: sign 0, exponent 127 (2^0), mantissa 000000
        \\0.1: sign 0, exponent 123 (2^-4), mantissa 4ccccd
        \\-2.5: sign 1, exponent 128 (2^1), mantissa 200000
        \\0.15625: sign 0, exponent 124 (2^-3), mantissa 200000
        \\{reading=4,bitcast=4,u8}
        \\1.0 on disk: 00 00 80 3f
        ,
        .hint = "The first line should show 3f800000 for 1.0, not 00000001. For the parts: shift right, then AND with a mask of the right width (8 bits, 23 bits).\n",
    },
    .{
        .main_file = "22_the_watchmaker/150_the_slow_watch.zig",
        .output =
        \\bf16 ticks near 16: 0.125
        \\bf16 ticks near 32: 0.25
        \\f32 ticks near 32: 3.8146973e-6
        \\The customer's watch: 32.1
        \\After the repair: 100.0
        ,
        .hint = "`spacing`: 2 to the power (floor(log2(value)) - mantissa_bits). `windWell`: the step is rounded to bf16 once, then everything happens in f32 until the very last `convert`.\n",
    },
    .{
        .main_file = "22_the_watchmaker/151_overwound.zig",
        .output =
        \\The overwound row:
        \\  naive: 3 NaN, 3 not finite
        \\  by hand: 0 not finite
        \\  library: 0 not finite
        \\  by hand: 0.6652 0.2447 0.0900 0.0000
        \\  same as the library: true
        \\  naive logsumexp: 1 not finite
        \\  logsumexp by hand: 1000.4076, library: 1000.4076
        \\The silent row:
        \\  naive: 4 NaN, 4 not finite
        \\  by hand: 4 not finite
        \\  library: 0 not finite
        ,
        .hint = "A bool tensor becomes countable with `.convert(.i32)`; `flatten()` then `sum(0)` then `squeeze(0)` gives a rank-0 count. For the softmax: subtract `max(.score)` before `exp`.\n",
    },
    .{
        .main_file = "22_the_watchmaker/152_the_wrong_gears.zig",
        .output =
        \\pixels:                { 0, 51, 128, 200, 255 }
        \\trap 1, u8 / 255:      { 0, 0, 0, 0, 1 }
        \\trap 2, u8 + 100:      { 100, 151, 228, 44, 99 }
        \\brighten by 100:       { 100, 151, 228, 255, 255 }
        \\trap 3, u8 as i8:      { 0, 51, -128, -56, -1 }
        \\trap 4, p / 4 as i32:  { -1, 0, 0, 0, 2 }
        \\bucket of width 4:     { -2, -1, 0, 0, 2 }
        \\trap 5, p * 100 as u8: { 0, 0, 50, 255, 255 }
        \\to the unit interval:  0.000 0.200 0.502 0.784 1.000
        ,
        .hint = "Each trap is cured by choosing WHERE the `convert` goes. `brighten`: widen to .i32, add, `clamp`, come back. `bucket`: `floor` before `convert`.\n",
    },
    .{
        .main_file = "22_the_watchmaker/153_a_cheaper_movement.zig",
        .output =
        \\The file is an f16, bit for bit: true
        \\The file is a bf16, bit for bit: true
        \\1e-5, seen by the file: 0
        \\1e-5, seen by a real f16: 1.001e-5
        \\The squares, seen by an f16: { 0, 0, 0, 0 }
        \\eps, seen by an f16: 0
        \\The cheap norm: 4 values out of 4 are not finite
        \\The careful norm: 0.098 -0.197 0.294 0.098
        \\Same as zml.nn.rmsNorm, bit for bit: true
        ,
        .hint = "The numbers are in the table of exercise 148: (exponent bits, mantissa bits). `carefulNorm` is `cheapNorm` with one `convert` at the start and one at the end.\n",
    },
    .{
        .main_file = "22_the_watchmaker/154_quiz_the_grand_complication.zig",
        .output =
        \\Wheels: 24
        \\The cheap movement leaves the reference at wheel: 3
        \\The mixed movement stays with the reference: true
        \\At the last wheel, cheap is further off than mixed: true
        \\The verdict: mixed goes to the collector
        ,
        .hint = "Write `movement` first and test `reference` alone. The stream is passed through `asSeenBy` once before the first wheel and once after each `add`; the weights once, before the loop. `worst` reads rows of 8 floats: `a[wheel * 8 ..][0..8]`.\n",
    },
    // Chapter 23: Hot Off the Press
    .{
        .main_file = "23_hot_off_the_press/155_the_forme.zig",
        .output =
        \\the forme holds 16 tokens; presses set up for 3 openings: 2
        \\"Once upon a time": 5 tokens and 11 blanks
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted
        \\"One day, a little dog": 8 tokens and 8 blanks
        \\One day, a little dog named Max went to the park with his mom. They saw a big box with a big box. Max was very happy. He wanted to
        \\"Lily and her mom went to the park": 12 tokens and 4 blanks
        \\Lily and her mom went to the park. They saw a big box with a big box. Lily was very happy. She wanted to play with it, but she did not want to
        \\the first story is PyTorch's, token for token: true
        ,
        .hint = "`forward` is exercise 146's without the `slice`: every position goes through the norm and the head. The story is not PyTorch's? The forme has 16 slugs and the opening 5: after WHICH one does the story go on? The decode loop is exercise 146's, and `len` is the true length.\n",
    },
    .{
        .main_file = "23_hot_off_the_press/156_hot_metal.zig",
        .output =
        \\tensors the press takes: 51
        \\tensors it melts down and recasts: 3
        \\bytes of cache that are no longer copied, per token: 163840
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do.
        \\the same story as PyTorch, token for token: true
        \\by hand: after the BOS comes token 403
        \\by hand: the plate that went in can no longer be read: true
        ,
        .hint = "`recasts: 0`: nothing in `forward` says `reuseBuffer`. Three outputs, three inputs, and the tokens need a `convert` first. In `feed`, the destinations are the very pointers you were given. A segfault in `PJRT_Buffer_Destroy`? You freed what the runner had already destroyed.\n",
    },
    .{
        .main_file = "23_hot_off_the_press/157_standing_type.zig",
        .output =
        \\plates the press takes: 51
        \\standing type, locked in once: 47
        \\plates set for each page: 4
        \\plates melted down and recast: 3
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do.
        \\the same story as PyTorch, token for token: true
        ,
        .hint = "`ordered prefix`: which field is baked, and where does it sit in `Input`? `step` is last night's `forward` with everything read from `input` and both results named. `compile` wants a tuple holding ONE struct. The weights are given to the runner once, by name.\n",
    },
    .{
        .main_file = "23_hot_off_the_press/158_stop_the_presses.zig",
        .output =
        \\the page opens with: { 1, 403, 407, 261, 378 }
        \\the presses stop on: { 1, 2 }
        \\--- 512 tokens of paper
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
        \\Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
        \\Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
        \\Lily and her mom went to the park to play. They played together and had fun. After a while, Lily's mom came into the room and saw the park. She was so happy and said, "Thank you, Lily! You are a good friend."
        \\--- 346 tokens printed, and the presses stopped: the story was over
        \\--- 40 tokens of paper
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big,
        \\--- 40 tokens printed, and the presses stopped: no more paper
        ,
        .hint = "The first line of output should be `{ 1, 403, 407, 261, 378 }`: `tokenId`, then `appendTokens` and `encode` on the same writer. `why` is a `while (true)` with two `break`s that carry a value, in the order of the lesson. The second text holds the first one: how many decoders are there?\n",
    },
    .{
        .main_file = "23_hot_off_the_press/159_on_the_wire.zig",
        .output =
        \\in the type case: [<s>][▁Once][▁upon][▁a][▁time][,][▁there][▁was][▁a][▁little][▁g][ir][l]
        \\off the decoder:  [][Once][ upon][ a][ time][,][ there][ was][ a][ little][ g][ir][l]
        \\--- on the tape
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
        \\Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
        \\Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
        \\Lily and her mom went to the park to play. They played together and had fun. After a while, Lily's mom came into the room and saw the park. She was so happy and said, "Thank you, Lily! You are a good friend."
        \\--- 346 tokens printed, and the presses stopped: the story was over
        ,
        .hint = "`feedOne`, `writeAll`, `flush`, in that order, for the opening and then for each new token; `finalize` at the end. The proof: print each piece before the next `feedOne`. A space before `Once`? That decoder had already read thirteen tokens.\n",
    },
    .{
        .main_file = "23_hot_off_the_press/160_the_dials.zig",
        .output =
        \\presses set up: 2, for 6 editions
        \\plates melted down and recast at each run: 4
        \\morning (top_k 1): the story of PyTorch: true
        \\noon (top_p 0): the morning edition again: true
        \\afternoon (min_p 1): the morning edition again: true
        \\evening (top_k 8, the dice roll): another story: true
        \\  it opens like the others: true
        \\  every word of it is one of Tiny's 512: true
        \\evening, reprinted from the same seed: the evening edition again: true
        \\night (top_k 50, in a press that holds 8): the evening edition again: true
        ,
        .hint = "The draw gives two things back, and `Output` wants three. Four dials come from `setting`, and four Buffers need freeing. Night differs from evening: how many candidates can this press hold? Fix the reprint last: when were its dice seeded?\n",
    },
    .{
        .main_file = "23_hot_off_the_press/161_quiz_the_front_page.zig",
        .output =
        \\Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
        \\Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
        \\Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
        \\Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
        \\Lily and her mom went to the park to play. They played together and had fun. After a while, Lily's mom came into the room and saw the park. She was so happy and said, "Thank you, Lily! You are a good friend."
        \\--- morning edition: 346 tokens, and the presses stopped: the story was over
        \\its first 106 tokens are PyTorch's: true
        \\--- late edition: the dice rolled, and it is another story: true
        \\it opens like the morning one: true
        \\reprinted from the same seed, it is the same: true
        \\--- the shop: standing type 47, set for each run 9, recast at each run 4
        \\Put the paper to bed.
        ,
        .hint = "Build it night by night: a greedy press that tells `reference_story` first (155 to 157), then the stops and the page (158), the tape (159), and only then the dials (160). The names the last lines want are listed in the lesson.\n",
    },
    // Chapter 24: Window Shopping
    .{
        .main_file = "24_window_shopping/162_nose_against_the_glass.zig",
        .output =
        \\the street {t=8,f32}: { 1, 1, 1, 6, 6, 6, 2, 2 }
        \\by hand    {t=6,f32}: { 0, 5, 5, 0, -4, -4 }
        \\conv1d     {t=6,f32}: { 0, 5, 5, 0, -4, -4 }
        \\same: true
        ,
        .hint = "`glance`: a slice of the street is still tagged `.t`, and `dot` wants `.k` on both sides. The last place to stand is where the kernel still fits. The panic counts the axes of the two tensors: which one was left behind?\n",
    },
    .{
        .main_file = "24_window_shopping/163_how_many_steps.zig",
        .output =
        \\the regular             Odette says 6, ZML says 6: { 123, 234, 345, 456, 567, 678 }
        \\the jogger              Odette says 3, ZML says 3: { 123, 345, 567 }
        \\the council             Odette says 8, ZML says 8: { 12, 123, 234, 345, 456, 567, 678, 780 }
        \\the daydreamer          Odette says 4, ZML says 4: { 135, 246, 357, 468 }
        \\the daydreamer, padded  Odette says 8, ZML says 8: { 13, 24, 135, 246, 357, 468, 570, 680 }
        \\all of it               Odette says 3, ZML says 3: { 24, 246, 468 }
        \\the astronomer          Odette says 0, ZML says 0: {  }
        ,
        .hint = "`@divFloor`, and a walk with no place to stand gives 0, not a negative number. `samePad`: put out = in and stride = 1 in the formula. Thirteen glances for the daydreamer: who got spread apart, the kernel or the street?\n",
    },
    .{
        .main_file = "24_window_shopping/164_the_right_side_of_the_glass.zig",
        .output =
        \\the photo    {b=1,h=3,w=7,c=3,f32}
        \\careless     {b=1,h=1,w=5,c=1,f32}: { 3, 5, 7, 9, 13 }
        \\by options   {b=1,h=1,w=5,c=1,f32}: { 0, 3, 3, 0, -3 }
        \\by transpose {b=1,h=1,w=5,c=1,f32}: { 0, 3, 3, 0, -3 }
        ,
        .hint = "Line 1106 compares the kernel's axis 1 with the photo's channels: tags do not move data, and they do not tell `conv2d` anything. `byOptions`: six options, three for the input, three for the output, all from `photo.axis(...)`. Compare the SHAPES of the last two lines.\n",
    },
    .{
        .main_file = "24_window_shopping/165_the_best_of_the_window.zig",
        .output =
        \\best              {b=1,c=1,h=2,w=2,f32}: 5 2 6 8
        \\its column        {b=1,c=1,h=2,w=2,i32}: 0 3 1 2
        \\pairs             {b=1,c=1,h=4,w=2,f32}: 3 2 5 1 0 7 6 8
        \\average, folded   {b=1,c=1,h=2,w=2,f32}: 3.25 1 2 5.75
        \\average, stride 2 {b=1,c=1,h=2,w=2,f32}: 3.25 1 2 5.75
        \\average, stride 1 {b=1,c=1,h=3,w=3,f32}: 3.25 1.5 1 2.25 3 3 2 5.25 5.75
        \\overall           {b=1,c=1,f32}: 3
        ,
        .hint = "Both compile errors are about what the max pools take and return: read the two signatures in the lesson. `reduceWindow` wants one entry per axis, batch and channels included, and returns an array. A sum is not an average yet.\n",
    },
    .{
        .main_file = "24_window_shopping/166_the_blow_up.zig",
        .output =
        \\nearest {b=1,c=1,h=4,w=4,f32}
        \\   0   0   4   4
        \\   0   0   4   4
        \\   8   8  12  12
        \\   8   8  12  12
        \\nearest, the photo {b=1,h=4,w=4,c=1,f32}
        \\   0   0   4   4
        \\   0   0   4   4
        \\   8   8  12  12
        \\   8   8  12  12
        \\bilinear {b=1,c=1,h=4,w=4,f32}
        \\   0   2   4   4
        \\   4   6   8   8
        \\   8  10  12  12
        \\   8  10  12  12
        \\the stripes, shrunk {b=1,c=1,h=2,w=2,f32}
        \\   0   0
        \\   0   0
        \\the stripes, averaged {b=1,c=1,h=2,w=2,f32}
        \\   4   4
        \\   4   4
        ,
        .hint = "`{b=1,c=1,4,4}`: two tags are missing. `resizeBilinear` takes sizes, not factors. `blowUpPhoto`: `upsample` scales axes 2 and 3, wherever your height and width are; you already have a function for (b, c, h, w).\n",
    },
    .{
        .main_file = "24_window_shopping/167_one_dresser_per_colour.zig",
        .output =
        \\the dressers {b=1,c=3,h=2,w=2,f32}: Rose, Olive, Marine
        \\    1   2       9   9       9  15
        \\    3   4       9   9       9  15
        \\the decorator, 1 x 1 convolution {b=1,c=2,h=2,w=2,f32}
        \\   19  26      -8 -13
        \\   21  28      -6 -11
        \\the decorator, Linear {b=1,c=2,h=2,w=2,f32}
        \\   19  26      -8 -13
        \\   21  28      -6 -11
        \\same: true
        \\weights: 27 + 6 = 33, against 54 for one full 3 x 3 convolution
        ,
        .hint = "Three dressers, one colour each: how many input channels does ONE kernel see? Then tell `conv2d` about the groups. `blendLinear`: drop the two axes of size 1, name the weight's axes the way a Linear wants them, and mind where `.dout` lands.\n",
    },
    .{
        .main_file = "24_window_shopping/168_quiz_the_sign_inspector.zig",
        .output =
        \\strokes {b=6,d=8,f32}, scores {b=6,dout=5,f32}, letters {b=6,i32}
        \\                             | | | |  - - - -   reads    L   7   U   C   O
        \\Aux Parapluies de Cherbourg  4 4 4 4  0 0 3 3     U     -2 -14  10  -6   6
        \\Blanchisserie Moderne        4 0 4 0  0 0 3 4     L      7 -23   3   3  -1
        \\L'Epicerie du Coin           0 4 0 4  4 3 0 0     7    -23   7 -11 -13  -1
        \\Optique Lumiere              4 4 4 4  3 3 3 3     O     -8  -8   4   0  12
        \\La Cave a Fromages           4 0 4 0  3 4 3 4     C      0 -16  -4  10   6
        \\Pressing Basile et Fils      3 0 3 0  0 0 2 3     L      3 -19  -1  -1  -5
        ,
        .hint = "Leaves first: `Conv`, `Inspector`, `main`. The bias has one number per output channel, and the output channels are tagged `.c`. One channel must be inserted before the first convolution. `maxPool2d` returns a struct. Merge while the axes are still (c, h, w).\n",
    },
    // Chapter 25: The Petting Zoo
    .{
        .main_file = "25_the_petting_zoo/169_the_head_count.zig",
        .output =
        \\"which one is loud?"
        \\pooling        rabbit    goose tortoise
        \\[CLS] token     0.800    0.000    1.000   -> the tortoise
        \\plain mean      1.000   -0.982    0.737   -> the rabbit
        \\masked mean     0.700    0.883   -0.789   -> the goose
        \\the card says mean pooling, {doc=3,s=4,d=2,f32} -> {doc=3,f32}: go and pet the goose
        ,
        .hint = "Divide by how many real tokens there are, not by the length of the row: the mask knows, once it is f32 and has the shape of `hidden`. A cosine above 1 means one side is not a unit vector.\n",
    },
    .{
        .main_file = "25_the_petting_zoo/170_the_short_queue.zig",
        .output =
        \\the short queue, as the mask sees it (x: may not listen):
        \\  (pad) x x x
        \\  hen   x . x
        \\  duck  x . .
        \\after 2 rounds, in the batch {b=2,s=3,d=2,f32}:
        \\  goat   1.000  0.000
        \\  pig    0.638  0.362
        \\  cow    0.699  0.500
        \\  (pad)  0.000  0.000
        \\  hen    0.000  2.000
        \\  duck   1.727  0.273
        \\the short queue alone {b=1,s=2,d=2,f32}:
        \\  hen    0.000  2.000
        \\  duck   1.727  0.273
        \\the padding is invisible: true
        ,
        .hint = "The picture shows which mask is missing. If you see `nan`: which row has nobody to listen to, and what does `Tensor.softmax` do for such a row?\n",
    },
    .{
        .main_file = "25_the_petting_zoo/171_the_feeding_board.zig",
        .output =
        \\{s=3,d=4,f32} keepers read {m=4,dm=3,f32} notes: {s=3,d=4,f32}
        \\     buckets         food  water
        \\Ada   2.086  0.000  0.971  0.029
        \\Bo    4.683  0.000  0.106  0.894
        \\Cy    3.000  0.000  0.667  0.333
        ,
        .hint = "Keys and values come from `memory`, and their Linears contract `.dm`. The panic names a tag: who declared a Tensor without it?\n",
    },
    .{
        .main_file = "25_the_petting_zoo/172_who_stands_where.zig",
        .output =
        \\the clocks {s=3,d=4,f32}:
        \\  position 0:  0.000  1.000  0.000  1.000
        \\  position 1:  0.841  0.540  0.010  1.000
        \\  position 2:  0.909 -0.416  0.020  1.000
        \\the bias {q=3,k=3,f32}, one row per listener:
        \\   0 -2 -2
        \\   2  0 -2
        \\   2  2  0
        \\"goat bites boy" and "boy bites goat":
        \\  no positions  (1.000 1.000 0.387 0.387) (1.000 1.000 0.387 0.387)  the same file: true
        \\  sinusoidal    (1.396 1.703 0.316 1.507) (1.575 1.445 0.356 1.508)  the same file: false
        \\  relative bias (1.236 0.764 0.600 0.363) (0.916 1.084 0.139 0.223)  the same file: false
        ,
        .hint = "`stack` on `.last` then `merge` interleaves. In the bias, row = listener: a speaker BEHIND her has a smaller position. `forward` must read `self.positions` (and `const` may have to become `var`).\n",
    },
    .{
        .main_file = "25_the_petting_zoo/173_second_opinions.zig",
        .output =
        \\4 vets, 2 opinions per animal: {b=1,s=3,d=2,f32} -> ids {b=1,s=3,topk=2,i32}
        \\  pony   0.88 Dr. Hoof + 0.12 Dr. Both  ->  3.762 bandages, 0.000 syrup
        \\  goose  0.88 Dr. Lung + 0.12 Dr. Both  ->  0.000 bandages, 3.762 syrup
        \\  goat   0.73 Dr. Hoof + 0.27 Dr. Both  ->  2.462 bandages, 0.000 syrup
        \\the backend ZML picks for this platform: UnimplementedMoEBackend
        \\.stablehlo is available: true
        \\the reference {b=1,s=3,d=2,f32} agrees: true
        ,
        .hint = "`topK` names its new axis; the weights must sum to 1 over it. `zml.moe.Options` has no defaults: all three fields. The two Linears of `forwardMoe` take the stacked weights and the tag they contract.\n",
    },
    .{
        .main_file = "25_the_petting_zoo/174_the_tortoise.zig",
        .output =
        \\asked for 2 signs, his mood: curious 0.787, grumpy 0.935
        \\asked for 3 signs, his mood: curious 0.978, grumpy 0.759
        \\asked for 4 signs, his mood: curious 0.879, grumpy 0.997
        \\asked for 9 signs, his mood: curious 0.879, grumpy 0.997
        \\the trail {s=4,d=2,f32}:
        \\  after LETTUCE   0.918  0.000
        \\  after NO DOGS   0.787  0.935
        \\  after LETTUCE   0.978  0.759
        \\  after CLOSED    0.879  0.997
        \\unrolled, 6 inputs against 7: the same mood: true
        ,
        .hint = "The context holds what never changes, the State what does. `body` must return every Tensor with the shape and dtype it came in with. Mind the 9: there are only 4 signs.\n",
    },
    .{
        .main_file = "25_the_petting_zoo/175_quiz_the_new_arrival.zig",
        .output =
        \\{b=4,s=3,i32} sentences, {b=3,f32} cosines
        \\  like the rabbit    0.996
        \\  like the goose     0.061
        \\  like the tortoise -0.252
        \\the new arrival moves in with the rabbit
        ,
        .hint = "Leaves first: `Cell`, `Clinic`, then `Zoo`. A local named like one of the data arrays is a compile error. The mask is only needed in the pooling; the residual is `x.add(...)` with matching tags.\n",
    },
    // Chapter 26: The Mail Room
    .{
        .main_file = "26_the_mail_room/176_the_address_label.zig",
        .output =
        \\arguments from the runner: 0
        \\
        \\mailroom parcels/floor14
        \\  repo: parcels/floor14
        \\  start: dock
        \\  max steps: ask the config
        \\  chunk: 4096 bytes
        \\  service: standard
        \\  fragile: false
        \\
        \\mailroom --max-steps=3 --chunk=64KiB parcels/overseas 2
        \\  repo: parcels/overseas
        \\  start: x-ray
        \\  max steps: 3
        \\  chunk: 65536 bytes
        \\  service: standard
        \\  fragile: false
        \\
        \\mailroom --service=express --fragile parcels/annex
        \\  repo: parcels/annex
        \\  start: dock
        \\  max steps: ask the config
        \\  chunk: 4096 bytes
        \\  service: express
        \\  fragile: true
        ,
        .hint = "The help text gives each name, type and default; an underscore in a field is a dash in the flag. The compile error names the flag. Then `parse` says what is wrong with a label on stderr: `=` glues a value, and options come before positional arguments.\n",
    },
    .{
        .main_file = "26_the_mail_room/177_the_customs_slip.zig",
        .output =
        \\floor14
        \\  model type: chute
        \\  layers: 0, tied head: false, stored as float32
        \\  a route starts at: dock
        \\  and ends at: [the 14th floor]
        \\  or after 20 stations
        \\overseas
        \\  model type: conveyor
        \\  layers: 2, tied head: false, stored as float16
        \\  a route starts at: dock
        \\  and ends at: [return to sender] [the basement]
        \\  or after 20 stations
        \\annex
        \\  model type: conveyor
        \\  layers: 2, tied head: true, stored as bfloat16
        \\  a route starts at: dock
        \\  and ends at: [the 14th floor]
        \\  or after 6 stations
        \\pneumatic_tube: error.UnknownModelType
        \\sampling: max_new_tokens 8, top_k 50, temperature 0.7
        ,
        .hint = "`error.UnexpectedToken`: std.json needs to be told how to pick a tag of the union. `error.UnknownField`: exercise 121. `error.InvalidEnumTag`: read annex/config.json. A default only covers an ABSENT key; null is a value.\n",
    },
    .{
        .main_file = "26_the_mail_room/178_three_boxes_one_parcel.zig",
        .output =
        \\the index of overseas, by hand:
        \\  5 tensors, 528 bytes
        \\  model-00001-of-00003.safetensors: 1
        \\  model-00002-of-00003.safetensors: 3
        \\  model-00003-of-00003.safetensors: 1
        \\
        \\floor14: entry point model.safetensors, 2 tensors, 512 bytes
        \\overseas: entry point model.safetensors.index.json, 5 tensors, 528 bytes
        \\annex: entry point model.safetensors.index.json, 3 tensors, 384 bytes
        \\mop_closet: error.FileNotFound
        \\the shelf itself: error.FileNotFound
        \\
        \\overseas, tensor by tensor:
        \\  model.embed_tokens.weight {8,8,f16} in model-00001-of-00003.safetensors
        \\  model.layers.0.fc.weight {8,8,f16} in model-00002-of-00003.safetensors
        \\  model.layers.0.fc.bias {8,f16} in model-00002-of-00003.safetensors
        \\  model.layers.1.fc.weight {8,8,f16} in model-00002-of-00003.safetensors
        \\  lm_head.weight {8,8,f16} in model-00003-of-00003.safetensors
        \\layers: 2
        \\missing for a conveyor: 0
        ,
        .hint = "Count what Priya's registry holds against what the index promised. `fromRepo` takes the directory. Then read the names on the list: where do the layers live?\n",
    },
    .{
        .main_file = "26_the_mail_room/179_the_loading_dock.zig",
        .output =
        \\zmling/26_the_mail_room/parcels/floor14
        \\  model.safetensors
        \\file://zmling/26_the_mail_room/parcels/floor14
        \\  model.safetensors
        \\file://zmling/26_the_mail_room/parcels/overseas
        \\  model.safetensors.index.json
        \\file://zmling/26_the_mail_room/parcels/mop_closet
        \\  error.FileNotFound
        \\hf://consolidated-tensor/router-v2
        \\  error.InvalidPath
        \\
        \\in the parcel:
        \\  model.embed_tokens.weight
        \\  lm_head.weight
        \\
        \\route: dock > scale > x-ray > belt A > the 14th floor
        ,
        .hint = "The dock is in the lesson. The panic names the tensor ZML looked for; the program printed the two names the file really has. `argMax` keeps the axis it reduced.\n",
    },
    .{
        .main_file = "26_the_mail_room/180_the_forklift.zig",
        .output =
        \\forklift: 4 at a time, 4 buffers of 65536 bytes
        \\the slip says float16, the table is Tensor({voc=8,d=8,f16})
        \\in the boxes: 528 bytes
        \\on the device: 528 bytes
        \\route: dock > scale > x-ray > belt B > the basement
        ,
        .hint = "`Loader.init` takes all three fields; `chunk_size.bytes()`. 512 bytes out of 528: which tensor is 16 bytes? Then: `Linear.forward` computes in the dtype of its INPUT. Convert once, early.\n",
    },
    .{
        .main_file = "26_the_mail_room/181_the_sorting_table.zig",
        .output =
        \\floor14: a chute, 2 tensors
        \\  dock > scale > x-ray > belt A > the 14th floor
        \\overseas: a conveyor, 5 tensors
        \\  dock > scale > x-ray > belt B > the basement
        \\overseas: a conveyor, 5 tensors
        \\  belt A > the basement
        \\annex: a conveyor, 3 tensors
        \\  dock > scale > belt A > belt B > belt A > belt B ... (gave up after 6 stations)
        \\annex: a conveyor, 3 tensors
        \\  dock > scale > belt A > belt B ... (gave up after 4 stations)
        ,
        .hint = "The panic in `main` names two fields of the union: one switch, compiled once per tag. A weight that may be absent is an optional field. And compare the number of layers on the slip with the number Priya allocated.\n",
    },
    .{
        .main_file = "26_the_mail_room/182_quiz_special_delivery.zig",
        .output =
        \\parcel: floor14
        \\  entry point: model.safetensors
        \\  slip: chute, 0 layers, tied head: false, float32
        \\  boxes: 2 tensors, 512 bytes
        \\  device: 512 bytes
        \\  route: dock > scale > x-ray > belt A > the 14th floor
        \\parcel: overseas
        \\  entry point: model.safetensors.index.json
        \\  slip: conveyor, 2 layers, tied head: false, float16
        \\  boxes: 5 tensors, 528 bytes
        \\  device: 528 bytes
        \\  route: dock > scale > x-ray > belt B > the basement
        \\parcel: tube
        \\  entry point: model.safetensors
        \\  slip: pneumatic_tube: error.UnknownModelType
        \\parcel: annex
        \\  entry point: model.safetensors.index.json
        \\  slip: conveyor, 2 layers, tied head: true, bfloat16
        \\  boxes: 3 tensors, 384 bytes
        \\  device: 384 bytes
        \\  route: x-ray > belt A > belt B > belt A > belt B ... (gave up after 5 stations)
        ,
        .hint = "Everything is in exercises 176 to 181. An unknown architecture is an error to catch and report, not to `try`. stdout stays on `init.io`.\n",
    },
    // Chapter 27: The Hall of Mirrors
    .{
        .main_file = "27_the_hall_of_mirrors/183_the_inspection.zig",
        .output =
        \\Madame Ottoline's, pane by pane:
        \\  {ticket=3,i32}
        \\  {out=4,d=4,f32}
        \\  {out=4,f32}
        \\  {out=4,d=4,f32}
        \\  {out=4,d=4,f32}
        \\  {out=4,f32}
        \\  {d=4,f32}
        \\  {d=4,f32}
        \\8 panes, 67 numbers, 268 bytes.
        \\With the cheap glass: 8 panes, 67 numbers, 140 bytes.
        \\The last mirror is now Tensor({out=4,d=4,f16}).
        ,
        .hint = "`visit` wants a pointer to the thing, and reads what to look for off your callback. A callback that writes needs everything on the way to be writable.\n",
    },
    .{
        .main_file = "27_the_hall_of_mirrors/184_single_file.zig",
        .output =
        \\Mr. Pym's list:
        \\  slot 0: {d=3,f32}
        \\  slot 1: {d=3,f32}
        \\  slot 2: {f32}
        \\  slot 3: {d=3,f32}
        \\The Exe has 4 slots. Same list: true
        \\Gus's crate: { 30, 40, 50 }
        \\No crate at all: { 30, 40, 50 }
        ,
        .hint = "Nothing but the place in the walk connects a Buffer to its Tensor. Read the slot number in the panic, then Mr. Pym's list. For the crate: which declaration decides the order?\n",
    },
    .{
        .main_file = "27_the_hall_of_mirrors/185_the_other_side.zig",
        .output =
        \\A Hall has 3 fields, its reflection has 2.
        \\The reflection of the sign is void: true
        \\A Mirror can `forward`: true. Its reflection: false.
        \\Fresh from `bufferize`: 6 crates, the first one says {d=2,f32}, full: false
        \\After the pour: full: true, 0 numbers left on the strip.
        \\A visitor { 3, 1 } comes out as { 4, 70 }
        ,
        .hint = "`bufferize` cannot write into a `[]const`, and has no case for arrays: `reached unreachable code` in `bufferizeInner` is that. Each crate already knows its shape: that says how much of the strip it takes.\n",
    },
    .{
        .main_file = "27_the_hall_of_mirrors/186_the_blueprint.zig",
        .output =
        \\Madame Ottoline's: built from the blueprint, 7 slots.
        \\The name was copied, not borrowed: true
        \\The ledger of Madame Ottoline's:
        \\  door: 3 numbers, 12 bytes
        \\  corridor[0]: glass 9 numbers, frame 3, curved: false
        \\  corridor[1]: glass 9 numbers, frame 0, curved: true
        \\  corridor[2]: glass 9 numbers, frame 3, curved: true
        ,
        .hint = "The callback of `mapAlloc` always takes two parameters. If the Exe has fewer slots than Mr. Pym counted, look at what the blueprint says about each frame, and at what it does NOT say.\n",
    },
    .{
        .main_file = "27_the_hall_of_mirrors/187_the_twins.zig",
        .output =
        \\The Twins have the same shape: true, the same id: false
        \\  left <- bar.2.weight, right <- bar.2.weight
        \\A copy of the left one has its id: true, and <- bar.2.weight
        \\A lookalike has its id: false, and <- nothing
        \\The hall:
        \\  entrance.glass <- bar.0.weight
        \\  entrance.frame <- bar.0.bias
        \\  middle.glass <- bar.1.weight
        \\  middle.frame <- bar.1.bias
        \\Three mirrors on the way, 5 slots.
        \\A visitor { 3, 1 } comes out as { 13, 1 }
        ,
        .hint = "`0 shards`: the Loader found no source for that id. Look at the list printed above the panic. A tie is one field used twice in `forward`, never two fields.\n",
    },
    .{
        .main_file = "27_the_hall_of_mirrors/188_staff_only.zig",
        .output =
        \\A Hall has 4 fields, its reflection has 3.
        \\The walk sees 6 panes. The Exe has 7 slots: the visitor, and a spare nobody looks into.
        \\Pane by pane: {d=4,f32} {out=4,d=4,f32} {out=4,f32} {out=4,d=4,f32} {out=4,f32} {out=4,d=4,f32}
        \\In all: 60 numbers of f32 glass.
        ,
        .hint = "The id in the panic is the placeholder you passed as the visitor: where else does the walk meet it? `collectBuf` fills the seats it is given and says nothing about the guests left standing.\n",
    },
    .{
        .main_file = "27_the_hall_of_mirrors/189_quiz_the_grand_tour.zig",
        .output =
        \\The Grand Hall, pane by pane:
        \\  0  hall.doors.entry           4  {d=4,f32}
        \\  1  hall.doors.exit            4  {d=4,f32}
        \\  2  hall.corridor[0].glass    24  {out=6,d=4,f32}
        \\  3  hall.corridor[0].frame     6  {out=6,f32}
        \\  4  hall.corridor[1].glass    30  {out=5,d=6,f32}
        \\  5  hall.corridor[2].glass    20  {out=4,d=5,f32}
        \\  6  hall.corridor[2].frame     4  {out=4,f32}
        \\  7  hall.twins[0]              4  {d=4,f32}
        \\  8  hall.twins[1]              4  {d=4,f32}
        \\  9  visitor                    4  {d=4,f32}
        \\10 panes, 104 numbers.
        \\`zml.meta` agrees: true
        \\The Exe agrees: true
        ,
        .hint = "Start with the Tensor case and the struct case, run, and read the register: it tells you which kind you are still walking past. An extra pane called `hall.notice.visitor` means a door was left open.\n",
    },
    // Chapter 28: Footprints in the Snow
    .{
        .main_file = "28_footprints_in_the_snow/190_fresh_snow.zig",
        .output =
        \\before the snow, a Compiler on the mountain: false
        \\after the snow,  a Compiler on the mountain: false
        \\1 compile: forward ran 1 time, and saw the Compiler 1 time
        \\  hiker 1 climbs 7.5 m. forward ran 1 time
        \\  hiker 2 climbs 6 m. forward ran 1 time
        \\  hiker 3 climbs 8 m. forward ran 1 time
        \\2 compiles: forward ran 2 times, and saw the Compiler 2 times
        \\the trails: {step=4,f32} and {step=6,f32}
        ,
        .hint = "Three mistakes, three times. Zig stops at the first (count the fields of the args tuple, and the parameters of `forward` after `self`). The second trace goes through `emitMlir`: compare the two shapes in the message. The third does not: which Buffer is `argument 1`?\n",
    },
    .{
        .main_file = "28_footprints_in_the_snow/191_plaster_casts.zig",
        .output =
        \\in main, the placeholder carries a pointer: false
        \\while tracing: depth false, wind false, their sum true
        \\two placeholders, two compiles: 2 + 2 inputs
        \\morning: { 15, 33, 11, 23 }
        \\evening: { 9, 18, 7, 13 }
        ,
        .hint = "`_value` is an optional pointer: `!= null`. Then the panic: `evening` asks `value()` for a Tensor that is not in the args of ITS compile call.\n",
    },
    .{
        .main_file = "28_footprints_in_the_snow/192_carved_in_ice.zig",
        .output =
        \\carved: 1 input. surveyed: 2 inputs.
        \\the camp is at 1200 m
        \\  carved:        { 1201, 1202, 1205, 1205.5, 1207 }
        \\  surveyed:      { 1201, 1202, 1205, 1205.5, 1207 }
        \\the camp moves to 1500 m
        \\  carved:        { 1201, 1202, 1205, 1205.5, 1207 }
        \\  surveyed:      { 1501, 1502, 1505, 1505.5, 1507 }
        \\  carved again:  { 1501, 1502, 1505, 1505.5, 1507 }
        ,
        .hint = "`constantTensor(shape, bytes)` wants a Shape and `std.mem.sliceAsBytes`. If `carved again` still shows 1201: which Exe is answering, and when was it traced?\n",
    },
    .{
        .main_file = "28_footprints_in_the_snow/193_the_snow_cave.zig",
        .output =
        \\the stack: forward 1, cond 2, body 2, a branch 3
        \\traced: cond 1 time, body 1 time, branches 2 times
        \\3 days of sun:           { 6, 15, 0, 17 }
        \\6 days, storm on day 2:  { 12, 15, 0, 25 }
        \\after 9 days in the cave, body was traced 1 time
        ,
        .hint = "Follow the stack trace up from `value`: `cond`, then later `onFalse`. Both read the shelf. A Tensor reaches a block through the struct you pass to `ops.while` / `ops.if`, nothing else. Snow has no negative depth: one op floors at zero.\n",
    },
    .{
        .main_file = "28_footprints_in_the_snow/194_the_measuring_stick.zig",
        .output =
        \\while the snow fell, the stick came out 2 times
        \\new snow {day=3,print=3,f32}:
        \\  day 0:   9.0   8.0  12.5
        \\  day 1:  16.0  12.0  21.0
        \\  day 2: -40.0  16.0  29.5
        \\readings below zero: 1
        \\two runs later, the stick came out 2 times
        ,
        .hint = "`hasTag(.day)` is a `?u3`: it can be compared with a number. Once the panic is your own message, fix the caller, not the stick. `belowZero`: `cmp`, `convert(.i32)`, then a `sum` that leaves a scalar.\n",
    },
    .{
        .main_file = "28_footprints_in_the_snow/195_cairns_and_avalanches.zig",
        .output =
        \\"the_hike" compiled
        \\cairns: on 3 legs of 3, back in the open: false, in camp: true
        \\the last morning of 2 days: { 3, 2, 3.5, 0 }
        \\the last morning of 3 days: { 2, 0, 3, 0 }
        \\the last morning of 5 days: { 0, 0, 2, 0 }
        ,
        .hint = "In the dump, find the `%` that is defined between `({` and `})` and used after it. It leaves the loop as a third field of the State. `back in the open: true`: every push needs its `defer ... popLocation()`.\n",
    },
    .{
        .main_file = "28_footprints_in_the_snow/196_quiz_the_cold_case.zig",
        .output =
        \\the case takes 4 inputs, the counting 2
        \\day 1, sun  : { 10, 25, 4, 19 }, 3 deeper than 9.5
        \\day 6, sun  : { 0, 0, 0, 14 }, 1 deeper than 9.5
        \\day 6, shade: { 6, 15, 0, 17 }, 2 deeper than 9.5
        \\day 3, shade: { 9, 22.5, 2, 18.5 }, 2 deeper than 9.5
        \\4 questions, traced 1 time
        ,
        .hint = "Same message twice, two different Tensors: read which function is in the stack trace. Then the silent ones: anything read from a Zig variable inside `forward`, `cond` or `body` is a photograph. 4 inputs: melt, depth, days, shade.\n",
    },
};
