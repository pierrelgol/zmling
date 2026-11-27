//
// In the caravan, above the kettle, hangs the blueprint of the
// hall: the same corridors, drawn. And in a drawer, the ledger:
// the same corridors, priced. Ottoline's father made both by
// hand, forty years ago, and they have been wrong ever since
// the first mirror was moved.
//
// `visit` gives you every Tensor of a model, in order, and
// that is all it gives you: by the time your callback runs,
// `corridor[1].glass` is "the fourth one". Often you want the
// opposite: the same STRUCT, with the same names, holding
// something else. `Bufferized` is one such reflection. Its
// machine has a more accommodating sister:
//
//     zml.meta.MapType(From, To).map(T)
//
// Same idea: a type computed from a type, every `From` turned
// into a `To`. But `MapType` KEEPS the other fields. Where
// `MapRestrict` throws away the name on the sign and the
// `curved: bool`, `MapType` copies them as they are.
//
//     const Blueprint = zml.meta.MapType(zml.Tensor, zml.Shape).map(Hall);
//
// is a Hall whose `door` is a Shape and whose `corridor` is a
// slice of things with a `glass: Shape`, a `frame: ?Shape` and
// a `curved: bool`. (Still no methods. No reflection has any.)
//
// A type is half of it. To fill a VALUE of that type from a
// value of the other:
//
//     try zml.meta.mapAlloc(callback, allocator, ctx, from, &to);
//
//   - `callback` is `fn (ctx: Ctx, x: From) To`. TWO parameters,
//     always, even when the first is `_: void`. `mapAlloc`
//     reads From and To off its signature, like `visit` did.
//     Give it a one-parameter function and it takes your From
//     for the Ctx: "expected type 'shape.Shape', found 'void'",
//     pointing at the `{}` you passed, is that mistake;
//   - `from` is walked field by field, and `to`, a pointer, is
//     written field by field. They are matched BY NAME: `to`
//     need not be the `MapType` of `from`, only to have fields
//     called the same. Your real model struct, methods and
//     all, is a perfectly good `to`;
//   - every From on the way goes through the callback, and
//     everything else is copied;
//   - slices and pointers that lead to a From are ALLOCATED,
//     with the allocator you gave. Give it an arena and forget
//     about them. The innocent ones, with no From inside, like
//     a `[]const u8` name, are copied AS THEY ARE: the slice,
//     not the bytes. Both structs then share the same name.
//
// And `To` must be a struct
// (a Shape, a Buffer, one of yours): `mapAlloc` looks at each
// field of the destination to decide whether to dig or to
// copy, and a plain integer is always copied. Map Tensors to
// `usize` and you are told, from inside `meta.zig`,
// "expected type 'usize', found 'tensor.Tensor'".
//
// One trap is built into the type: `MapType` (like
// `MapRestrict`) gives every `?Tensor` field it maps a default
// value of `null`, whether the original field had one or not.
// `Mirror` insists on a `frame`. Its reflections do not.
//
// Two reflections today, in the two directions.
//
// BLUEPRINT -> HALL. The blueprint below is typed in by hand:
// Shapes and settings. From it, a `Hall` of fresh placeholders,
// one per Shape (`Tensor.fromShape`, chapter 02), ready to
// compile. A model built from its description: that is every
// `init` you have written since chapter 13, in one call.
//
// HALL -> LEDGER. A Hall where every pane is replaced by its
// `Price`: how many numbers, how many bytes. With the names
// still on.
//
//  1. `Mirror.forward`: the layer of exercise 086, with a
//     frame that may not be there, and a `relu` only when the
//     mirror is curved. Out comes a `.d` again.
//
//  2. `pane` does not have the signature `mapAlloc` wants.
//
//  3. `Ledger`, the type. And the call that fills `ledger`
//     from `hall`: `price` is its callback.
//
//  4. Mr. Pym counts seven slots on the blueprint: the door,
//     three mirrors, three frames... no. Two frames: the
//     second mirror hangs from the ceiling. The Exe should
//     say 7, with the visitor. Does it?
//
const std = @import("std");
const zml = @import("zml");

const Mirror = struct {
    glass: zml.Tensor,
    frame: ?zml.Tensor,
    curved: bool,

    pub fn forward(self: Mirror, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const Hall = struct {
    name: []const u8,
    door: zml.Tensor,
    corridor: []Mirror,

    pub fn forward(self: Hall, visitor: zml.Tensor) zml.Tensor {
        var v = visitor.mul(self.door);
        for (self.corridor) |mirror| v = mirror.forward(v);
        return v;
    }
};

const Blueprint = zml.meta.MapType(zml.Tensor, zml.Shape).map(Hall);
const DrawnMirror = zml.meta.MapType(zml.Tensor, zml.Shape).map(Mirror);

const Price = struct {
    numbers: usize,
    bytes: usize,
};

const Ledger = ???;

/// One Shape of the blueprint -> one placeholder.
fn pane(shape: zml.Shape) zml.Tensor {
    return .fromShape(shape);
}

/// One pane -> what it holds.
fn price(_: void, t: zml.Tensor) Price {
    return .{ .numbers = t.count(), .bytes = t.byteSize() };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Everything `mapAlloc` allocates goes here.
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const glass: zml.Shape = .init(.{ .out = 3, .d = 3 }, .f32);
    const frame: zml.Shape = .init(.{ .out = 3 }, .f32);

    var drawn = [_]DrawnMirror{
        .{ .glass = glass, .frame = frame, .curved = false },
        .{ .glass = glass, .frame = null, .curved = true },
        .{ .glass = glass, .curved = true },
    };
    const blueprint: Blueprint = .{
        .name = "Madame Ottoline's",
        .door = .init(.{ .d = 3 }, .f32),
        .corridor = &drawn,
    };

    var hall: Hall = undefined;
    try zml.meta.mapAlloc(pane, arena.allocator(), {}, blueprint, &hall);

    const visitor: zml.Tensor = .init(.{ .d = 3 }, .f32);
    var exe = try platform.compile(gpa, io, hall, .forward, .{visitor}, .{});
    defer exe.deinit();

    try stdout.print("{s}: built from the blueprint, {d} slots.\n", .{ hall.name, exe.input_shapes.len });
    try stdout.print("The name was borrowed, not copied: {}\n", .{hall.name.ptr == blueprint.name.ptr});

    var ledger: Ledger = undefined;
    // Fill the ledger from the hall.
    ???;

    try stdout.print("The ledger of {s}:\n", .{ledger.name});
    try stdout.print("  door: {d} numbers, {d} bytes\n", .{ ledger.door.numbers, ledger.door.bytes });
    for (ledger.corridor, 0..) |mirror, i| {
        try stdout.print("  corridor[{d}]: glass {d} numbers, frame {d}, curved: {}\n", .{
            i,
            mirror.glass.numbers,
            if (mirror.frame) |f| f.numbers else 0,
            mirror.curved,
        });
    }
    try stdout.flush();
}
