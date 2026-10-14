//
// Last day of the fair. Mr. Pym is back, and this year the
// county wants more than a head count: a REGISTER. Every pane
// of the Grand Hall, by NAME, in the order the public meets
// them, with its size. And the register must agree with the
// machine, or the hall does not open.
//
// By name. That is the one thing `zml.meta` will not do for
// you: `visit` hands your callback a pointer and nothing else.
// "Argument 5", says the panic. Which one is 5?
//
// So you will write the walk yourself, once, with what
// `meta.zig` is written with. Its vocabulary is short:
//
//     @TypeOf(v)                    the type of a value
//     @typeInfo(T)                  what KIND of type T is: a
//                                   tagged union, to `switch` on
//     .@"struct" => |info|          info.field_names, a list of
//                                   strings known at compile time
//                                   (also info.field_types and
//                                   info.field_attrs, in step)
//     .pointer => |info|            info.size is .slice, .one, ...
//     .array, .optional, ...        the other kinds
//     inline for (info.field_names) |name|
//                                   a loop the COMPILER unrolls:
//                                   `name` is comptime in the body
//     @field(v, name)               `v.name`, for a comptime name
//     @hasDecl(T, "some_name")      does T declare that?
//
// (If you have read older Zig: `std.meta.fields` is gone. The
// three parallel lists are what there is.)
//
// A function that takes `v: anytype` is compiled once per type
// it is called with. So a function that calls ITSELF on every
// field of `v` is not really recursive: it is a family of
// functions, one per type in your model, each calling the
// next. `walk(Hall)` calls `walk(Doors)` calls `walk(Tensor)`.
// That is exactly how `visit` is built, and why it costs
// nothing at run time.
//
// The route is the one of exercise 183, and you now know all
// of its rules:
//
//   - a Tensor: that's a pane. Register it;
//   - a struct: its fields, in order, unless the door says
//     staff only (exercise 188);
//   - a slice or an array: its elements;
//   - an optional: what is inside, if anything;
//   - anything else: walk on.
//
// The NAME of a pane is the path that leads to it, and only
// the walker knows it. It grows on the way down:
//
//     hall  ->  hall.corridor  ->  hall.corridor[1]
//           ->  hall.corridor[1].glass
//
// `std.fmt.bufPrint(&buf, "{s}.{s}", .{ path, name })` writes
// into a buffer of yours and returns the slice it wrote: a
// local `var buf: [64]u8` in each `walk` is enough, since a
// path only has to live as long as the walk below it. (An
// optional adds nothing to the path.)
//
// What to write:
//
//  1. The two `forward`. A mirror is the one of exercise 186.
//     The hall: in by the `entry` door (it scales each
//     measurement), through the corridor, then the twins, a
//     stretch and a shift (exercise 184), and out by the
//     `exit` door, which scales again.
//
//  2. `Report`. It holds a `*std.Io.Writer` to print to, room
//     for 32 Shapes, and two counters, `panes` and `numbers`.
//     Its `walk(self, path, v)` prints one line per pane found
//     in `v`:
//
//         "{d:>3}  {s:<24} {d:>3}  {f}\n"
//
//     its number, its path, how many numbers it holds, its
//     Shape. And notes the Shape, and counts. `walk` can fail
//     in two ways. Give it an explicit error set:
//
//         std.Io.Writer.Error || std.fmt.BufPrintError
//
//  3. `sameShapes`: same length, and `eql` all the way.
//
//  4. In `main`: the register, for the hall then for the
//     visitor, the way the Compiler takes its arguments. Then
//     the two witnesses:
//
//       - `zml.meta`: the Shapes of the same two things, in
//         ONE collected slice (exercises 184 and 188);
//       - the Exe (exercise 081).
//
// The staff notice, the frameless mirror in the middle and the
// name on the sign are there to keep you honest.
//
const std = @import("std");
const zml = @import("zml");

/// What a visitor must look like.
const Notice = struct {
    pub const zml_no_meta_visit: void = {};

    visitor: zml.Tensor,
};

const Mirror = struct {
    glass: zml.Tensor,
    frame: ?zml.Tensor,
    curved: bool,

    pub fn forward(self: Mirror, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const Doors = struct {
    entry: zml.Tensor,
    exit: zml.Tensor,
};

const Hall = struct {
    name: []const u8,
    notice: Notice,
    doors: Doors,
    corridor: []Mirror,
    /// Left and right of the exit: a stretch and a shift.
    twins: [2]zml.Tensor,

    pub fn forward(self: Hall, visitor: zml.Tensor) zml.Tensor {
        return ???;
    }
};

// TODO: `Report`, with its `walk`.

fn sameShapes(a: []const zml.Shape, b: []const zml.Shape) bool {
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

    // The corridor narrows and widens: 4 -> 6 -> 5 -> 4.
    var corridor = [_]Mirror{
        .{ .glass = .init(.{ .out = 6, .d = 4 }, .f32), .frame = .init(.{ .out = 6 }, .f32), .curved = false },
        .{ .glass = .init(.{ .out = 5, .d = 6 }, .f32), .frame = null, .curved = true },
        .{ .glass = .init(.{ .out = 4, .d = 5 }, .f32), .frame = .init(.{ .out = 4 }, .f32), .curved = true },
    };
    const hall: Hall = .{
        .name = "The Grand Hall",
        .notice = .{ .visitor = .init(.{ .d = 4 }, .f32) },
        .doors = .{ .entry = .init(.{ .d = 4 }, .f32), .exit = .init(.{ .d = 4 }, .f32) },
        .corridor = &corridor,
        .twins = .{ .init(.{ .d = 4 }, .f32), .init(.{ .d = 4 }, .f32) },
    };
    const visitor = hall.notice.visitor;

    try stdout.print("{s}, pane by pane:\n", .{hall.name});

    // `report`: the register of "hall", then of "visitor".
    // `register`: the Shapes it noted, no more than it found.
    // `walked`: what `zml.meta` collects from the same two.
    // `exe`.
    ???;

    try stdout.print("{d} panes, {d} numbers.\n", .{ report.panes, report.numbers });
    try stdout.print("`zml.meta` agrees: {}\n", .{sameShapes(register, walked)});
    try stdout.print("The Exe agrees: {}\n", .{sameShapes(register, exe.input_shapes)});
    try stdout.flush();
}
