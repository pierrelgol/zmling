//
// Three parcels on the table, and until now one program per
// parcel. `examples/llm` is ONE binary, and it runs a Llama, a
// Qwen, a Qwen with a mixture of experts and an LFM2: four
// architectures, with different structs, different `forward`s,
// different caches. Which one? The label does not say. The
// slip does: `"model_type": "llama"`.
//
// The whole trick is in `examples/llm/models.zig`, and it is
// three pieces of plain Zig.
//
// 1. AN ENUM, one tag per architecture the program knows, and
//    the string of the slip turned into a tag (exercise 177):
//
//     pub const ModelType = enum { lfm2, llama, qwen3_5, qwen3_5_moe };
//     std.meta.stringToEnum(ModelType, parsed.value.model_type)
//         orelse return error.UnknownModelType
//
// 2. A TAGGED UNION over that enum, one model struct per tag:
//
//     pub const LoadedModel = union(ModelType) {
//         lfm2: lfm2.LoadedModel,
//         llama: llama.LoadedModel,
//         ...
//     };
//
//    A value of this type IS one of the models, and remembers
//    which. Reading the wrong field is a panic, not a guess:
//
//     access of union field 'chute' while field 'conveyor' is active
//
// 3. `inline else`, to write each operation once:
//
//     pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
//         switch (self.*) {
//             inline else => |*m| m.deinit(allocator),
//         }
//     }
//
//    A plain `else` prong is one piece of code for all the
//    remaining tags, so its capture can have only one type. An
//    `inline else` is compiled ONCE PER TAG: in each copy, `m`
//    has the type of that tag's field, and `m.deinit` is that
//    struct's own `deinit`. Capture the tag too if you need
//    it: `inline else => |m, tag|`, and `tag` is known at
//    compile time, good for `@tagName(tag)`. To BUILD the
//    union from an enum value:
//
//     return switch (model_type) {
//         inline else => |tag| @unionInit(Router, @tagName(tag), try .init(...)),
//     };
//
//    `@unionInit(Union, "field", value)` is `.{ .field = value }`
//    for a field name computed at compile time.
//
// Nothing is looked up at run time, there is no table of
// function pointers. The price: every struct of the union must
// answer the same calls with the same arguments. A chute needs
// neither an allocator nor the config to be built. It takes
// them anyway. That is the contract of the sorting table, and
// each model directory of the example signs it: `init`,
// `deinit`, `compile`, `loadBuffers`, `unloadBuffers`.
//
// A model's weights are a `zml.Bufferized(Chute)` or a
// `zml.Bufferized(Conveyor)`: two more types. The example
// declares a second union for them (`Buffers`), a third for
// the compiled programs, a fourth for a chat session. A
// smaller program gets away with one generic function,
//
//     fn deliver(comptime Model: type, model: Model, ...) !void
//
// called from a single `inline else`: inside it, `Model` is
// one concrete type and everything is as it was in exercise
// 180.
//
// THE SAME ARCHITECTURE, PACKED DIFFERENTLY. Dispatching on the
// type is half of the sorting. Two conveyors can differ in
// what they carry: the overseas one has a bias in one layer
// and an `lm_head`; the annex one has no bias anywhere, and
// its slip says `"tie_word_embeddings": true`: no `lm_head`
// tensor at all, the scores are a `dot` of the last activation
// with the token table itself (exercise 145).
//
// A struct field for a weight that may not exist is an
// optional: `lm_head: ?zml.nn.Linear`, exactly as in the
// example's Llama. Do not "tie" by putting the table's Tensor
// in two fields: a model is flattened into the list of its
// Tensors to make the arguments of the compiled program, and
// the same Tensor twice is the panic of chapter 02,
//
//     Tensor with id ... has already been used once as an argument
//
// Who decides, the slip or the boxes? Both are available:
// `config.tie_word_embeddings`, and `view.hasKey("lm_head")`.
// Exercise 141: a config is a claim, a shape is a fact. The
// example trusts the config and lets `createTensor` panic if
// it lied. You choose; the parcels here are honest.
//
// Gus: "One program. Any parcel. I am not learning three."
//
//  1. `Router.init`: from the slip to the right struct.
//  2. `main` delivers every parcel as if it were a chute.
//     Read the panic, then write the switch.
//  3. The annex parcel panics in `Conveyor.init`: it has no
//     head to give. Change the field, `init` and `forward`.
//     The table has the tags `.voc` and `.d`, and it is
//     stored in bf16.
//  4. Everything runs, and the overseas parcel goes from the
//     dock straight to the basement. In exercise 180 it took
//     four hops. Priya built that conveyor.
//
const std = @import("std");
const zml = @import("zml");

const flags = zml.stdx.flags;
const json = zml.stdx.json;
const View = zml.io.TensorStore.View;

const parcels = "zmling/26_the_mail_room/parcels/";

const stations = [_][]const u8{ "dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement" };

const Label = struct {
    max_steps: ?u32 = null,
    positional: struct {
        repo: []const u8,
        start: u32 = 0,
    },
};

const EosTokens = union(enum) {
    int: u32,
    ints: []const u32,

    const Helpers = json.UnionHelpers(@This());
    pub const jsonParse = Helpers.jsonParse;
    pub const jsonParseFromValue = Helpers.jsonParseFromValue;
    pub const jsonStringify = Helpers.jsonStringify;

    pub fn contains(self: EosTokens, token: u32) bool {
        return switch (self) {
            .int => |id| id == token,
            .ints => |ids| std.mem.findScalar(u32, ids, token) != null,
        };
    }
};

const Config = struct {
    model_type: []const u8,
    num_hidden_layers: u32 = 0,
    eos_token_id: EosTokens,
    max_length: json.NeverNull(u32, 20) = .{},
    tie_word_embeddings: bool = false,
};

fn parseJson(comptime T: type, gpa: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, name: []const u8) !std.json.Parsed(T) {
    const file = try repo.openFile(io, name, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(gpa, &file_reader.interface);
    defer reader.deinit();
    return try std.json.parseFromTokenSource(T, gpa, &reader, .{ .ignore_unknown_fields = true });
}

fn linear(view: View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        view.maybeCreateTensor("bias", .{.dout}, .replicated, .{}),
        .d,
    );
}

const Chute = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    lm_head: zml.nn.Linear,

    pub fn init(_: std.mem.Allocator, view: View, _: Config) !Chute {
        return .{
            .embed_tokens = .{ .weight = view.withPrefix("model").withPrefix("embed_tokens").createReplicatedTensor("weight", .{ .voc, .d }) },
            .lm_head = linear(view.withPrefix("lm_head")),
        };
    }

    pub fn deinit(_: Chute, _: std.mem.Allocator) void {}

    pub fn unload(weights: *zml.Bufferized(Chute), _: std.mem.Allocator) void {
        zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Chute, weights);
    }

    /// station: a rank-0 i32. Returns the next station, a rank-0 i32.
    pub fn forward(self: Chute, station: zml.Tensor) zml.Tensor {
        const x = self.embed_tokens.forward(station).convert(.f32);
        return self.lm_head.forward(x, .f32).argMax(.dout).indices.squeeze(.dout);
    }
};

const Conveyor = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []zml.nn.Linear,
    lm_head: zml.nn.Linear,

    pub fn init(gpa: std.mem.Allocator, view: View, _: Config) !Conveyor {
        const model = view.withPrefix("model");
        // Priya: "every conveyor I have seen has one layer."
        const layers = try gpa.alloc(zml.nn.Linear, 1);
        for (layers, 0..) |*layer, i| layer.* = linear(model.withPrefix("layers").withLayer(i).withPrefix("fc"));
        return .{
            .embed_tokens = .{ .weight = model.withPrefix("embed_tokens").createReplicatedTensor("weight", .{ .voc, .d }) },
            .layers = layers,
            .lm_head = linear(view.withPrefix("lm_head")),
        };
    }

    pub fn deinit(self: Conveyor, gpa: std.mem.Allocator) void {
        gpa.free(self.layers);
    }

    pub fn unload(weights: *zml.Bufferized(Conveyor), gpa: std.mem.Allocator) void {
        zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Conveyor, weights);
        gpa.free(weights.layers);
    }

    /// station: a rank-0 i32. Returns the next station, a rank-0 i32.
    pub fn forward(self: Conveyor, station: zml.Tensor) zml.Tensor {
        var x = self.embed_tokens.forward(station).convert(.f32);
        for (self.layers) |layer| x = layer.forward(x, .f32).relu().rename(.{ .dout = .d });
        const scores = self.lm_head.forward(x, .f32).rename(.{ .dout = .voc });
        return scores.argMax(.voc).indices.squeeze(.voc);
    }
};

const ModelType = enum { chute, conveyor };

const Router = union(ModelType) {
    chute: Chute,
    conveyor: Conveyor,

    pub fn init(gpa: std.mem.Allocator, view: View, config: Config) !Router {
        return ???;
    }

    pub fn deinit(self: Router, gpa: std.mem.Allocator) void {
        switch (self) {
            inline else => |model| model.deinit(gpa),
        }
    }
};

/// What `deliver` needs besides the model.
const Desk = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    stdout: *std.Io.Writer,
};

/// Compiles `model`, loads its weights and prints the route of a
/// parcel from `start`, until a station of `eos` or `max_steps`
/// stations.
fn deliver(comptime Model: type, model: Model, desk: Desk, start: u32, eos: EosTokens, max_steps: u32) !void {
    const gpa = desk.gpa;
    const io = desk.io;

    var exe = try desk.platform.compile(gpa, io, model, .forward, .{zml.Tensor.init(.{}, .i32)}, .{});
    defer exe.deinit();

    var weights = try zml.mem.bufferize(gpa, Model, &model);
    defer Model.unload(&weights, gpa);
    var loader: zml.io.Loader = try .init(gpa, desk.platform, .default);
    defer loader.deinit();
    try loader.load(io, Model, &model, &weights, desk.store, .{});
    try loader.await(io);

    var station = start;
    var count: u32 = 1;
    try desk.stdout.print("  {s}", .{stations[station]});
    while (!eos.contains(station) and count < max_steps) : (count += 1) {
        var here: zml.Buffer = try .scalar(io, desk.platform, station, .i32);
        defer here.deinit();
        var next = try exe.eval(gpa, io, .{ weights, here });
        defer next.deinit();
        station = @intCast(try next.getValue(i32, io));
        try desk.stdout.print(" > {s}", .{stations[station]});
    }
    if (!eos.contains(station)) try desk.stdout.print(" ... (gave up after {d} stations)", .{count});
    try desk.stdout.print("\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const labels = [_][]const [*:0]const u8{
        &.{ "mailroom", parcels ++ "floor14" },
        &.{ "mailroom", parcels ++ "overseas" },
        &.{ "mailroom", parcels ++ "overseas", "3" },
        &.{ "mailroom", parcels ++ "annex" },
        &.{ "mailroom", "--max-steps=4", parcels ++ "annex" },
    };

    for (labels) |arguments| {
        const label = flags.parse(.{ .vector = arguments }, Label);

        var repo = try zml.safetensors.resolveModelRepo(io, label.positional.repo);
        defer repo.close(io);

        const parsed = try parseJson(Config, gpa, io, repo, "config.json");
        defer parsed.deinit();
        const config = parsed.value;

        var registry: zml.safetensors.TensorRegistry = try .fromRepo(gpa, io, repo);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
        defer store.deinit();

        const router: Router = try .init(gpa, store.view(), config);
        defer router.deinit(gpa);

        try stdout.print("{s}: a {s}, {d} tensors\n", .{
            std.fs.path.basename(label.positional.repo),
            @tagName(router),
            registry.tensors.count(),
        });

        const desk: Desk = .{ .gpa = gpa, .io = io, .platform = platform, .store = &store, .stdout = stdout };
        const max_steps = label.max_steps orelse config.max_length.value;
        try deliver(Chute, router.chute, desk, label.positional.start, config.eos_token_id, max_steps);
        try stdout.flush();
    }
}
