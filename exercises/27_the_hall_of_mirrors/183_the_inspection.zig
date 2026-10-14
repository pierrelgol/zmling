//
// THE HALL OF MIRRORS
//
// The funfair comes to town every October. Between the ghost
// train and the candy floss stands Madame Ottoline's Hall of
// Mirrors: a maze of glass in which everything has a
// reflection. Gus sells the tickets. And this morning the
// county has sent an inspector, Mr. Pym, with a clipboard.
//
// For twenty-five chapters you have handed ZML a struct full of
// Tensors and it has "found the Tensors inside, in field
// order". Exercise 081 said so, and you took its word for it.
// This chapter opens the machine that does the finding. It is
// one file, `zml/zml/meta.zig`, and its main entrance is one
// function:
//
//     zml.meta.visit(callback, ctx, &value)
//
// `visit` walks through `value` and calls `callback` on every
// T it meets. Which T? It READS IT OFF YOUR CALLBACK:
//
//     fn callback(ctx: Ctx, t: *const T) void
//
//   - the second parameter must be a pointer, and what it
//     points to is what `visit` looks for. `*const zml.Tensor`
//     finds Tensors, `*const zml.Buffer` finds Buffers, and
//     `*const Mirror` would find Mirrors;
//   - the first parameter is yours: `visit` passes `ctx` along
//     untouched on every call. Usually a pointer to a struct
//     in which you add things up. `void` (pass `{}`) when you
//     need nothing;
//   - the callback may return an error. Then `visit` returns
//     that error too, and wants a `try`.
//
// It is all decided at compile time: `visit` is an `inline for`
// over the fields of your type, and what it compiles to is the
// list of calls you would have typed by hand.
//
// The route, which never changes:
//
//   - a struct: field by field, in DECLARATION order;
//   - an array or a slice: element by element;
//   - an optional: inside, when it is not null;
//   - a union: the active field;
//   - a `*T` pointer: followed.
//
// And what it walks past: every field whose TYPE cannot contain
// a T (a bool, an f32, a `[]const u8`: decided on the type, the
// value is never looked at) and `comptime` fields. A `[*]T`
// pointer has no length: that one is a compile error.
//
// Two rules about the third argument.
//
// It must be a POINTER to the thing. Give it the thing and you
// get
//
//     zml.meta.visit(...) is expecting a pointer/slice input,
//     but received: ...
//
// And a callback that takes `*T` instead of `*const T` may
// WRITE to what it visits: that is how ZML fills a struct of
// Buffers with the results of an Exe. But then everything on
// the way must be writable: a `var`, not a `const`, and a
// `[]Mirror`, not a `[]const Mirror`. Otherwise Zig stops deep
// inside `meta.zig` with
//
//     expected type '*tensor.Tensor', found '*const tensor.Tensor'
//
// which, now that you know what `visit` does, says exactly
// what is wrong.
//
// This walk IS the Compiler's: `createBlockArguments`, in
// `zml/zml/Compiler.zig`, is a `visit` with a `*const Tensor`
// callback that declares one input of the program per call.
// What Mr. Pym counts this morning is what an Exe would ask
// for. (Chapter 28 shows what happens next.)
//
// The inspection has four parts.
//
//  1. The hall itself. The corridor has three mirrors: each
//     needs its OWN placeholders (exercise 086). `glass_shape`
//     and `frame_shape` are there. The middle one has lost its
//     frame.
//
//  2. `Census.see`: one more pane, and its numbers, and its
//     bytes. A Tensor knows both (chapter 01 asked a Shape).
//     Then somebody must walk the inspector through. Gus
//     tried.
//
//  3. `list` prints one line per pane and may fail, as
//     printing does. Its walk is missing. Its ctx is `stdout`.
//
//  4. Ottoline's savings plan: `cheapen` replaces every f32
//     pane by an f16 one, IN PLACE. It is written. It does not
//     compile, and the reason is not in `cheapen`.
//
const std = @import("std");
const zml = @import("zml");

const Mirror = struct {
    glass: zml.Tensor,
    frame: ?zml.Tensor,
    curved: bool,
};

const Hall = struct {
    name: []const u8,
    /// The price list at the door, in pennies.
    prices: zml.Tensor,
    corridor: []const Mirror,
    /// Left and right of the exit.
    twins: [2]zml.Tensor,
    lights: u8,
};

const Census = struct {
    panes: usize = 0,
    numbers: usize = 0,
    bytes: usize = 0,

    fn see(self: *Census, pane: *const zml.Tensor) void {
        return ???;
    }
};

fn list(out: *std.Io.Writer, pane: *const zml.Tensor) !void {
    try out.print("  {f}\n", .{pane.shape()});
}

/// f32 glass is expensive. f16 glass is half the price.
fn cheapen(_: void, pane: *zml.Tensor) void {
    if (pane.dtype() == .f32) pane.* = .fromShape(pane.shape().withDtype(.f16));
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const glass_shape: zml.Shape = .init(.{ .out = 4, .d = 4 }, .f32);
    const frame_shape: zml.Shape = .init(.{ .out = 4 }, .f32);

    var corridor: [3]Mirror = undefined;
    // Three mirrors. Only the first one is flat, and the middle
    // one has no frame.
    ???;

    const hall: Hall = .{
        .name = "Madame Ottoline's",
        .prices = .init(.{ .ticket = 3 }, .i32),
        .corridor = &corridor,
        .twins = .{ .init(.{ .d = 4 }, .f32), .init(.{ .d = 4 }, .f32) },
        .lights = 40,
    };

    var census: Census = .{};
    zml.meta.visit(Census.see, &census, hall);

    try stdout.print("{s}, pane by pane:\n", .{hall.name});
    // `list` goes through the hall.
    ???;
    try stdout.print("{d} panes, {d} numbers, {d} bytes.\n", .{ census.panes, census.numbers, census.bytes });

    zml.meta.visit(cheapen, {}, &hall);

    var after: Census = .{};
    zml.meta.visit(Census.see, &after, &hall);
    try stdout.print("With the cheap glass: {d} panes, {d} numbers, {d} bytes.\n", .{ after.panes, after.numbers, after.bytes });
    try stdout.print("The last mirror is now {f}.\n", .{hall.corridor[2].glass});
    try stdout.flush();
}
