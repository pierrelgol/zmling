//
// So far every parcel was on the shelf behind you:
// `zmling/26_the_mail_room/parcels/...`, a directory on this
// machine. The label of a real parcel says where it comes FROM:
//
//     llm --model=/data/models/Llama-3.2-1B
//     llm --model=hf://meta-llama/Llama-3.2-1B-Instruct
//     llm --model=s3://our-bucket/checkpoints/run-0412
//     llm --model=gs://another-bucket/run-0412
//
// and `examples/llm/main.zig` has no `if (startsWith("hf://"))`
// anywhere. It calls `resolveModelRepo(io, args.model)`, then
// `fromRepo`, then the Loader, exactly as you did in exercise
// 178. The whole difference is in ONE variable: `io`.
//
// You have passed an `io` to every function that touches the
// outside world since chapter 02, without looking at it.
// A `std.Io` is two pointers: some state, and a table of
// functions (`dirOpenDir`, `dirOpenFile`, `fileReadPositional`,
// `fileClose`...). `init.io` is the one the operating system
// stands behind. Anybody can build another one. ZML did:
//
//     zml.io.VFS      (file `vfs/vfs.zig`)
//
// a Virtual File System. It is a `std.Io` that looks at the
// SCHEME of a path, the word before `://`, and hands the call
// to the backend registered under that word:
//
//     var vfs_file: zml.io.VFS.File = .init(gpa, init.io, .{});
//     defer vfs_file.deinit();
//
//     var vfs: zml.io.VFS = try .init(gpa, init.io);
//     defer vfs.deinit();
//     try vfs.register("file", vfs_file.io());
//
//     const io = vfs.io();
//
// From that line on, `io` goes everywhere: the Platform, the
// registry, the Loader. `init.io` keeps one customer: stdout.
//
// The backends ZML has (each is a `std.Io` too):
//
//     VFS.File   file://    this machine's disks
//     VFS.HF     hf://owner/model[@revision]   the HuggingFace hub
//     VFS.S3     s3://bucket/path              Amazon S3 and friends
//     VFS.GCS    gs://bucket/path              Google Cloud Storage
//     VFS.HTTP   https://host/path             a plain web server
//
// The four network ones want a `*std.http.Client`, and the
// first three have an `.auto(gpa, io, &http_client, environ_map)`
// that finds your credentials where each service keeps them:
// HF_TOKEN or `~/.cache/huggingface/token`, the AWS_* variables,
// GOOGLE_APPLICATION_CREDENTIALS. They read a remote file by
// asking the server for byte ranges. A safetensors header is at
// the front of the file (exercise 097), so a registry of an
// 80 GB model costs a few kilobytes of network. Nothing here
// needs a network, and the mail room has none: `file` is the
// only backend you can run today. `zml/examples/io/main.zig` is
// a small `ls`/`cat`/`cp` over all five when you have one.
//
// How a path is routed (the function `lookupDir`), which is
// what you need to predict a failure:
//
//  1. The path has a scheme: the backend registered for it
//     gets the call, without the `scheme://`. NO backend under
//     that name: `error.VFSNotRegistered` inside the VFS. But
//     `openDir` has no such error in its list, so the VFS logs
//
//       error(zml/vfs): Failed to lookup backend for opening dir 'hf://...'
//
//     on stderr, and `resolveModelRepo` hands you a flat
//     `error.InvalidPath`. Read stderr.
//  2. No scheme, absolute or relative: the call goes straight
//     to the io the VFS was built on. Old paths keep working.
//  3. A path relative to a Dir that the VFS opened: the VFS
//     remembers which backend opened that Dir. This is why
//     `fromRepo` takes a `std.Io.Dir`: open the repository
//     once, wherever it is, and `config.json`, the index and
//     every piece are found next to it.
//
// One trap. A Dir or a File opened through `vfs.io()` is a
// ticket number in the VFS's own table, not a real file
// descriptor. Use it with `vfs.io()` only: handed to `init.io`
// it means some other open file, or nothing. It cuts both
// ways. `std.Io.File.stdout()` is the real descriptor 1, and
// at this commit ticket 1 of the table is stdin: print through
// `vfs.io()` under the runner, and `flush` answers
// `error.WriteFailed`. Hence stdout on `init.io`, below.
//
// Gus points at the hatch in the wall. "Loading dock. Parcels
// come through there now. All of them."
//
//  1. Build the dock: the VFS, with the `file` backend.
//  2. `knock`: is there a model at this address? It returns
//     the name of the entry point (exercise 178), and closes
//     what it opened.
//  3. The parcel of the 14th floor comes in through
//     `file://`. It holds a CHUTE, the simplest router there
//     is: the station a parcel is at (a number) picks a row of
//     a table, one Linear turns that row into a score per
//     station, and the best score is the next station.
//     `init` panics: read the name ZML asks for, then the
//     names the program prints.
//  4. `forward`, from chapters 14 and 15: `TokenEmbedding`,
//     `Linear` (its second argument is the dtype you want
//     out), `argMax`. The result is a rank-0 i32.
//
const std = @import("std");
const zml = @import("zml");

const View = zml.io.TensorStore.View;

const parcels = "zmling/26_the_mail_room/parcels/";

const stations = [_][]const u8{ "dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement" };

/// floor14's config.json: "eos_token_id": 5.
const last_stop = 5;

const Chute = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    lm_head: zml.nn.Linear,

    pub fn init(view: View) Chute {
        return .{
            .embed_tokens = .{ .weight = view.withPrefix("model").withPrefix("embed_tokens").createReplicatedTensor("weight", .{ .voc, .d }) },
            .lm_head = .init(view.withPrefix("model").withPrefix("lm_head").createReplicatedTensor("weight", .{ .dout, .d }), null, .d),
        };
    }

    /// station: a rank-0 i32. Returns the next station, a rank-0 i32.
    pub fn forward(self: Chute, station: zml.Tensor) zml.Tensor {
        return ???;
    }
};

/// The name of the entry point of the repository at `address`.
fn knock(io: std.Io, address: []const u8) ![]const u8 {
    return ???;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;

    // The loading dock: a VFS that knows `file://`.
    const io = ???;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const addresses = [_][]const u8{
        parcels ++ "floor14",
        "file://" ++ parcels ++ "floor14",
        "file://" ++ parcels ++ "overseas",
        "file://" ++ parcels ++ "mop_closet",
        "hf://consolidated-tensor/router-v2",
    };
    for (addresses) |address| {
        if (knock(io, address)) |entry_point| {
            try stdout.print("{s}\n  {s}\n", .{ address, entry_point });
        } else |err| {
            try stdout.print("{s}\n  {}\n", .{ address, err });
        }
    }
    try stdout.flush();

    // The parcel of the 14th floor, through the dock.
    var repo = try zml.safetensors.resolveModelRepo(io, "file://" ++ parcels ++ "floor14");
    defer repo.close(io);

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(gpa, io, repo);
    defer registry.deinit();
    try stdout.print("\nin the parcel:\n", .{});
    for (registry.tensors.keys()) |name| try stdout.print("  {s}\n", .{name});
    try stdout.flush();

    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const model: Chute = .init(store.view());

    var exe = try platform.compile(gpa, io, model, .forward, .{zml.Tensor.init(.{}, .i32)}, .{});
    defer exe.deinit();

    var weights = try zml.mem.bufferize(gpa, Chute, &model);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Chute, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Chute, &model, &weights, &store, .{});
    try loader.await(io);

    // One call of the model per station, until the parcel is home.
    var station: i32 = 0;
    var count: usize = 1;
    try stdout.print("\nroute: {s}", .{stations[0]});
    while (station != last_stop and count < 8) : (count += 1) {
        var here: zml.Buffer = try .scalar(io, platform, station, .i32);
        defer here.deinit();
        var next = try exe.eval(gpa, io, .{ weights, here });
        defer next.deinit();
        station = try next.getValue(i32, io);
        try stdout.print(" > {s}", .{stations[@intCast(station)]});
    }
    try stdout.print("\n", .{});
    try stdout.flush();
}
