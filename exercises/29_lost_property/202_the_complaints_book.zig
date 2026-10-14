//
// On the counter, chained to it, is the complaints book. Odile
// reads it every evening. "People who shout," she says, "have
// usually told you nothing. The ones who write have."
//
// The runtime under ZML complains in three different ways, and
// you need to know which is which before something goes wrong on
// a machine that is not yours.
//
// 1. A ZIG ERROR. Everything PJRT can refuse comes back as one of
//    the sixteen codes of `pjrt.ApiError` (`zml/pjrt/pjrt.zig`):
//
//        Cancelled, Unknown, InvalidArgument, DeadlineExceeded,
//        NotFound, AlreadyExists, PermissionDenied,
//        ResourceExhausted, FailedPrecondition, Aborted,
//        OutOfRange, Unimplemented, Internal, Unavailable,
//        DataLoss, Unauthenticated
//
//    (`ResourceExhausted` is the one you will meet for real: it is
//    "out of memory" on a GPU.) You get them from whatever you
//    `try`: making a Buffer, `await`, `getValue`, `toSlice`,
//    compiling. `Platform.init` adds its own:
//    `error.InvalidDeviceCount`, `error.MissingDevices`, and
//    `error.Unavailable` for a target this binary was not built
//    with.
//
// 2. A PANIC. `exe.call` returns `void`. It has no way to say no,
//    so it stops the program:
//
//        PJRT_LoadedExecutable_Execute failed with: error.X
//            the call was refused at the door
//        PJRT execution failed with: error.X
//            it started, you asked to wait, and it failed
//
//    `runner.run` and `eval` are built on `call`: same thing. And
//    the border checks of exercise 200 are panics too.
//
// 3. A LINE OF LOG, and this is the one people throw away. An
//    error code is one word. The sentence that explains it only
//    ever exists in the log. Every PJRT error goes through this
//    function on its way to you:
//
//        inline fn interpretPjrtError(api, pjrt_error, context)
//            ApiError
//        {
//            defer pjrt_error.deinit(api);
//            const err_code = pjrt_error.getCode(api).toApiError();
//            log.warn("[{s}] {t}: {s}", .{ context, err_code,
//                pjrt_error.getMessage(api) });
//            return err_code;
//        }
//
//    which is why stderr shows, for example:
//
//        warning(pjrt): [PJRT_Event_Await] InvalidArgument:
//        ToLiteral() called on deleted or donated buffer: ...
//
//        warning        the level
//        (pjrt)         the SCOPE: who is talking
//        [PJRT_...]     the PJRT function that failed
//        InvalidArgument  the code: the Zig error you receive
//        the rest       XLA's own words. Read them.
//
//    You have seen one such line in every exercise since chapter
//    02: `[PJRT_Device_MemoryStats] Unimplemented: ...`. At each
//    compile, ZML asks the device how much memory it has, the CPU
//    plugin answers that it cannot say, and ZML carries on:
//
//        pub fn memoryStats(self: Device) pjrt.Device.MemoryStats {
//            return self.pjrt_device.memoryStats(...) catch .zeroes;
//        }
//
//    An error that is caught and replaced by zeros leaves no trace
//    but its line of log.
//
// Who is talking? Each ZML file logs under a scope:
// `std.log.scoped(.zml)` in `platform.zig` and `buffer.zig`,
// `.pjrt` in the bindings, `.@"zml/pjrtx"`, `.@"zml/Compiler"`,
// `.@"zml/io"`, `.@"zml/tokenizer"`... (`grep -rn "log.scoped"
// zml/zml` lists them.) And the ROOT FILE of a Zig program decides
// what is kept, with one declaration:
//
//     pub const std_options: std.Options = .{
//         .log_level = .warn,
//         .log_scope_levels = &.{
//             .{ .scope = .@"zml/Compiler", .level = .debug },
//         },
//         .logFn = myLogFunction,
//     };
//
//     log_level          the floor for everybody: .err, .warn,
//                        .info or .debug
//     log_scope_levels   exceptions, scope by scope, up or down
//     logFn              where the lines go. The default writes them
//                        on stderr; yours may do anything, and may
//                        call `std.log.defaultLog` to keep that.
//
// The root file of an exercise is zmling's harness, which says
// `.log_level = .warn` unless the exercise declares its own
// `std_options`: then it uses yours (see
// `exercises/harness/shim.zig`). The filtering is done at compile
// time: a silenced line costs nothing.
//
// Last, since `Platform.init` came up. Its signature is
// `init(allocator, io, target, options)`, and `options` is the
// `CreateOptions` of exercise 198: a field per target (`.cpu` with
// its device count, `.cuda` with the share of GPU memory XLA may
// take, 90% by default, ...). There is also
//
//     zml.Platform.auto(allocator, io, options)
//
// which tries `.tpu, .neuron, .rocm, .cuda, .oneapi, .metal,
// .furiosa, .cpu`, in that order, and keeps the first that starts.
// That is what a program meant for other people's machines calls.
// zmling says `.cpu` so that everybody gets the same numbers.
//
// Today the office keeps a book of its own: `record` writes every
// line of log on stderr, as usual, AND notes who said it and how
// loud. The program prints the book at the end. Basile has a scope
// too, and he logs.
//
// To do:
//
//  1. `std_options`. Odile wants in the book: from PJRT, warnings
//     and worse; from everybody else, errors only. Basile wrote
//     one level for all, and the book is full of the weather.
//
//  2. `full`: ask the PJRT device ITSELF for its memory stats,
//     `device.pjrt_device.memoryStats(platform.pjrt_api)`, an
//     `ApiError!MemoryStats`. The name of the error it answers
//     (`@errorName`), or "no error".
//
//  3. `readable`: can this Buffer still be read? Wait for it at
//     the counter (exercise 197), and see whether that fails.
//
//  4. The bag is melted by a raw call, the way of exercise 082:
//     that block is yours. The frees of exercise 156 follow it.
//
//  5. Then Basile weighs the bag. Read the panic, then scroll up
//     ONE line on stderr: that is where the reason is.
//
const std = @import("std");
const zml = @import("zml");

const basile = std.log.scoped(.basile);

const Complaint = struct { level: std.log.Level, scope: []const u8 };

var book: [32]Complaint = undefined;
var book_len: usize = 0;

/// On stderr like every line of log, and in the book.
fn record(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    std.log.defaultLog(level, scope, format, args);
    if (book_len == book.len) return;
    book[book_len] = .{ .level = level, .scope = @tagName(scope) };
    book_len += 1;
}

pub const std_options: std.Options = .{
    .log_level = .warn,
    .logFn = record,
};

/// The bag is recast in place, one pound heavier: wet.
fn soak(bag: zml.Tensor) zml.Tensor {
    return bag.addConstant(1).reuseBuffer(bag);
}

fn weigh(bag: zml.Tensor) zml.Tensor {
    return bag.sum(.thing).squeeze(.thing);
}

/// True when the contents of `b` are there, or will be.
fn readable(io: std.Io, b: zml.Buffer) bool {
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
    try stdout.print("the office opens. complaints: {d}\n", .{book_len});

    basile.warn("it is raining again", .{});
    basile.warn("platform 4 is cold", .{});

    const bag: zml.Tensor = .init(.{ .thing = 4 }, .i32);
    var soak_exe = try platform.compileFn(gpa, io, soak, .{bag}, .{});
    defer soak_exe.deinit();
    var weigh_exe = try platform.compileFn(gpa, io, weigh, .{bag}, .{});
    defer weigh_exe.deinit();
    try stdout.print("two Exes compiled. complaints: {d}\n", .{book_len});

    // The question behind those complaints, asked twice.
    const device = platform.devices[0];
    const full: []const u8 = ???;
    try stdout.print("PJRT, how full is the device? {s}\n", .{full});
    try stdout.print("ZML, how full is the device? {d} bytes, and no error\n", .{device.memoryStats().bytes_in_use});

    basile.err("the kettle is broken", .{});

    var bag_buffer: zml.Buffer = try .fromBytes(io, platform, bag.shape(), std.mem.sliceAsBytes(&[4]i32{ 3, 1, 4, 1 }));
    try stdout.print("the bag can be read: {}\n", .{readable(io, bag_buffer)});

    // The bag is left in the rain: `soaked`, by a raw call that waits.
    var args = try soak_exe.args(gpa);
    defer args.deinit(gpa);
    var results = try soak_exe.results(gpa);
    defer results.deinit(gpa);
    ???;

    // After a raw call, the dead handle and the output are both yours.
    defer bag_buffer.deinit();
    defer soaked.deinit();

    try stdout.print("the bag that went in can be read: {}\n", .{readable(io, bag_buffer)});
    if (bag_buffer.getValue([4]i32, io)) |_| {
        try stdout.print("reading it anyway: fine\n", .{});
    } else |err| {
        try stdout.print("reading it anyway: {t}\n", .{err});
    }

    var weight = try weigh_exe.eval(gpa, io, .{bag_buffer});
    defer weight.deinit();
    try stdout.print("the wet bag weighs {d} lb\n", .{try weight.getValue(i32, io)});

    // An office with three counters.
    if (zml.Platform.init(gpa, io, .cpu, .{ .cpu = .{ .device_count = 3 } })) |annex| {
        annex.deinit(gpa, io);
        try stdout.print("an annex with 3 devices: open\n", .{});
    } else |err| {
        try stdout.print("an annex with 3 devices: {t}\n", .{err});
    }

    try stdout.print("the complaints book, {d} lines:\n", .{book_len});
    for (book[0..book_len], 1..) |complaint, i| {
        try stdout.print("  {d}. {t}({s})\n", .{ i, complaint.level, complaint.scope });
    }

    try stdout.flush();
}
