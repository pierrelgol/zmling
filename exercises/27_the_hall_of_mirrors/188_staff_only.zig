//
// There is one door in the hall that is not a mirror. It says
// STAFF ONLY, it has no handle on the public side, and Mr. Pym
// walks past it every year without writing anything down.
//
// The walk goes everywhere a Tensor could be. Sometimes that is
// one room too many, and `meta.zig` has a way to close a door:
//
//     const Notice = struct {
//         pub const zml_no_meta_visit: void = {};
//         ...
//     };
//
// A struct (or union) with a declaration of that name is
// walked PAST. `visit` does not enter it, and `Contains`, the
// function that answers "can this type hold a T?", says no
// without looking. So, for every machine of this chapter, the
// Tensors in there do not exist: they are no input of the Exe,
// `Bufferized` drops the field, `count` does not count them,
// the Loader does not load them.
//
// ZML uses it once, on `zml.Mesh`: every Shape points to the
// mesh it is sharded on, and without that sign every walk over
// every model would wander through the sharding tables.
//
// When would YOU close a door? When a struct keeps Tensors for
// the staff: the placeholders of its own inputs, say, kept next
// to the weights so that nobody has to remember that a visitor
// is {.d = 4}. Convenient, and a trap twice over. Left open,
// that placeholder is one more "weight" of the model: one more
// slot, one more Buffer to provide. And when you then pass it
// as the argument it was meant to be, the walk meets it twice:
// exercise 015's panic, again.
//
// The sign has a price. A Tensor behind the door is no input,
// so `forward` cannot use it, unless it ALSO came in as an
// argument. If it did not:
//
//     Something went really wrong, tensor is not an argument
//     nor has an mlir.Value
//
// Two other things the walk does that you may not expect:
//
//   - a `comptime` field is skipped, always;
//   - a Tensor field that `forward` never uses is an input
//     all the same. The walk reads the struct, not the code:
//     the Exe has a slot for it and wants a Buffer in it.
//
// THE COLLECTORS. You have written three visitors by hand now,
// each a little struct with a callback. The usual ones are in
// `meta.zig` already, all built on `visit`:
//
//     zml.meta.count(T, &v)        u32: how many T
//     zml.meta.first(T, v)         the first T. BY VALUE, no `&`.
//                                  None at all: "attempt to use
//                                  null value"
//     zml.meta.collectAlloc(func, ctx, allocator, &v)
//                                  a slice of func(t), one per
//                                  T, in order. Yours to free
//     zml.meta.collectBuf(func, ctx, &v, out)
//                                  the same into a slice you
//                                  provide
//     zml.meta.collectPtrs(T, allocator, &v)
//                                  a slice of `*const T`
//
// `func` is `fn (T) R`, or `fn (Ctx, T) R` when it needs
// something. With one parameter, `ctx` is `{}`. And a METHOD is
// a function of its `self`: `zml.Tensor.shape` is an
// `fn (Tensor) Shape`, ready to use.
//
// `collectBuf` wants a seat for every guest, no more, no less.
// Too many seats and it trips an assert (`reached unreachable
// code`, in `collectBuf`). Too FEW and it seats the first ones
// and reports nothing.
//
// (There is also `zml.meta.zip`, which merges N structs of the
// same type field by field, and which ZML keeps for itself.)
//
// Gus has pinned a notice to the hall: what a visitor must
// look like. Nobody can compile the hall since.
//
//  1. The notice is for the staff.
//
//  2. `Hall.forward`: through the door (it scales each
//     measurement, exercise 184), then through every mirror.
//     The spare mirror is in the cellar: not on the way.
//
//  3. The Shape of every pane, in order: `shapes`. One call,
//     and one thing to free.
//
//  4. The total does not add up to what the list says.
//
const std = @import("std");
const zml = @import("zml");

/// What a visitor must look like.
const Notice = struct {
    visitor: zml.Tensor,
};

const Mirror = struct {
    glass: zml.Tensor,
    frame: zml.Tensor,

    pub fn forward(self: Mirror, x: zml.Tensor) zml.Tensor {
        return self.glass.dot(x, .d).add(self.frame).rename(.{ .out = .d });
    }
};

const Hall = struct {
    notice: Notice,
    door: zml.Tensor,
    corridor: []Mirror,
    /// In the cellar, in case one breaks.
    spare: zml.Tensor,

    pub fn forward(self: Hall, visitor: zml.Tensor) zml.Tensor {
        return ???;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    var corridor: [2]Mirror = undefined;
    for (&corridor) |*mirror| mirror.* = .{
        .glass = .init(.{ .out = 4, .d = 4 }, .f32),
        .frame = .init(.{ .out = 4 }, .f32),
    };
    const hall: Hall = .{
        .notice = .{ .visitor = .init(.{ .d = 4 }, .f32) },
        .door = .init(.{ .d = 4 }, .f32),
        .corridor = &corridor,
        .spare = .init(.{ .out = 4, .d = 4 }, .f32),
    };

    var exe = try platform.compile(gpa, io, hall, .forward, .{hall.notice.visitor}, .{});
    defer exe.deinit();

    try stdout.print("A Hall has {d} fields, its reflection has {d}.\n", .{
        @typeInfo(Hall).@"struct".field_names.len,
        @typeInfo(zml.Bufferized(Hall)).@"struct".field_names.len,
    });
    try stdout.print("The walk sees {d} panes. The Exe has {d} slots: the visitor, and a spare nobody looks into.\n", .{
        zml.meta.count(zml.Tensor, &hall),
        exe.input_shapes.len,
    });

    // `shapes`.
    ???;

    try stdout.print("Pane by pane:", .{});
    for (shapes) |shape| try stdout.print(" {f}", .{shape});
    try stdout.print("\n", .{});

    var sizes: [4]usize = undefined;
    zml.meta.collectBuf(zml.Tensor.count, {}, &hall, &sizes);

    var total: usize = 0;
    for (sizes) |size| total += size;
    try stdout.print("In all: {d} numbers of {s} glass.\n", .{ total, @tagName(zml.meta.first(zml.Tensor, hall).dtype()) });
    try stdout.flush();
}
