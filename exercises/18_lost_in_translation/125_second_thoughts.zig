//
// One layer works. Bertie has two. His cousin at the big
// agency, `bert-base`, has twelve, and `bert-large` twenty-four.
// In Python that difference is one line:
//
//     class BertEncoder(nn.Module):
//         def __init__(self, config):
//             self.layer = nn.ModuleList(
//                 [BertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
//
//         def forward(self, hidden_states, ...):
//             for i, layer_module in enumerate(self.layer):
//                 hidden_states = layer_module(hidden_states, ...)
//
// A list whose length comes from the resume. An `nn.ModuleList`
// called `layer` names its children by their index, and that
// is where the numbers in the file come from:
//
//     bert.encoder.layer.0.attention.self.query.weight
//     bert.encoder.layer.1.attention.self.query.weight
//
// You have built numbered layers before (exercise 101, with
// `view.withLayer(i)`), but as two fields written by hand. A
// port cannot do that: the count is not known when you write
// the struct. The field becomes a SLICE:
//
//     const Encoder = struct {
//         embeddings: Embeddings,
//         layers: []Layer,
//         ...
//     };
//
// A slice needs memory, so `init` now takes an allocator and
// can fail:
//
//     pub fn init(gpa: std.mem.Allocator, view: ..., config: Config) !Encoder {
//         const layers = try gpa.alloc(Layer, config.num_hidden_layers);
//         for (layers, 0..) |*layer, i| layer.* = ...;
//         ...
//     }
//
// and whoever calls it frees `model.layers`.
//
// `forward` is the Python loop. A Zig `for` over a slice,
// inside a function that gets compiled, is not a loop in the
// compiled program: tracing runs your Zig once, and each turn
// adds one more layer to the graph. Two layers, or twenty-four,
// one after the other in the Exe. (Exercise 75, if that sounds
// new.)
//
// The weights follow. `zml.mem.bufferize(gpa, Encoder, &model)`
// walks through the model and, when it meets a slice, ALLOCATES
// a slice of the same length for the Buffers. So the Bufferized
// now owns two kinds of memory, and each one has its own way
// out:
//
//     var weights = try zml.mem.bufferize(gpa, Encoder, &model);
//     defer gpa.free(weights.layers);
//     defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Encoder, &weights);
//
// Defers run from the bottom up, and that order matters:
//
//  1. first every Buffer inside `weights` is released (the
//     function with the long name: it frees the memory of the
//     device, and leaves your struct alone);
//
//  2. then the slice that held them.
//
// The other way around, step 1 would walk through a slice
// that is gone. And `zml.Buffer.deinitAll`, your friend since
// exercise 100, is for structs WITHOUT slices.
//
// Forget the `gpa.free`, and the program still prints the
// right answer. Then `init.gpa`, which counts, writes on
// stderr where the leaked memory was allocated. Read stderr
// once in a while.
//
// PORTING RULE 7: what the reference builds in a loop over the
// config, you build in a loop over the config. If you ever
// type a layer number, you are porting one checkpoint, not a
// model.
//
// To check it, the recording again. This time from the very
// beginning: the ids go in, and what comes out must be what
// came out of the LAST layer in PyTorch. `main` builds that
// name from the config too.
//
// `Embeddings` and `Layer` are yours from the last two
// exercises: they are given. To do:
//
//  1. `Encoder.init`. The view stands on "bert": the
//     embeddings are under "embeddings", the layers under
//     "encoder.layer.0", "encoder.layer.1"...
//
//  2. `Encoder.forward`.
//
//  3. In `main`: free the model. Then the weights: make room,
//     load, and give everything back, in the right order.
//
//  4. And then `error.NotFound`: the trainee asked the
//     recording for a layer Bertie does not have.
//
const std = @import("std");
const zml = @import("zml");

const dir = "zmling/18_lost_in_translation/";

const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    // Not in this config.json: the default of `BertConfig`.
    layer_norm_eps: f32 = 1e-12,
};

/// A torch `nn.Linear`, with its bias: "weight" is {out, in}.
fn linear(view: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        view.createReplicatedTensor("weight", .{ .dout, .d }),
        view.createReplicatedTensor("bias", .{.dout}),
        .d,
    );
}

fn layerNorm(view: zml.io.TensorStore.View, eps: f32) zml.nn.LayerNorm {
    return .{
        .weight = view.createReplicatedTensor("weight", .{.d}),
        .bias = view.createReplicatedTensor("bias", .{.d}),
        .eps = eps,
    };
}

/// The GELU BERT was trained with: x/2 * (1 + erf(x / sqrt(2))).
fn geluExact(x: zml.Tensor) zml.Tensor {
    return x.scale(0.5).mul(x.scale(1.0 / @sqrt(2.0)).erf().addConstant(1));
}

const Embeddings = struct {
    word: zml.nn.TokenEmbedding,
    position: zml.Tensor,
    token_type: zml.Tensor,
    norm: zml.nn.LayerNorm,

    /// `view` must stand on "bert.embeddings".
    pub fn init(view: zml.io.TensorStore.View, config: Config) Embeddings {
        return .{
            .word = .{ .weight = view.withPrefix("word_embeddings").createReplicatedTensor("weight", .{ .voc, .d }) },
            .position = view.withPrefix("position_embeddings").createReplicatedTensor("weight", .{ .pos, .d }),
            .token_type = view.withPrefix("token_type_embeddings").createReplicatedTensor("weight", .{ .typ, .d }),
            .norm = layerNorm(view.withPrefix("LayerNorm"), config.layer_norm_eps),
        };
    }

    /// tokens: {.s} i32 -> {.s, .d}
    pub fn forward(self: Embeddings, tokens: zml.Tensor) zml.Tensor {
        var x = self.word.forward(tokens);
        x = x.add(self.position.slice(.pos, .{ .end = tokens.dim(.s) }).rename(.{ .pos = .s }));
        x = x.add(self.token_type.slice(.typ, .{ .end = 1 }).squeeze(.typ).broad(x.shape()));
        return self.norm.forward(x);
    }
};

const Layer = struct {
    query: zml.nn.Linear,
    key: zml.nn.Linear,
    value: zml.nn.Linear,
    attn_out: zml.nn.Linear,
    attn_norm: zml.nn.LayerNorm,
    up: zml.nn.Linear,
    down: zml.nn.Linear,
    out_norm: zml.nn.LayerNorm,
    heads: u32,

    /// `view` must stand on one layer: "bert.encoder.layer.0".
    pub fn init(view: zml.io.TensorStore.View, config: Config) Layer {
        const self = view.withPrefix("attention").withPrefix("self");
        const attn_output = view.withPrefix("attention").withPrefix("output");
        return .{
            .query = linear(self.withPrefix("query")),
            .key = linear(self.withPrefix("key")),
            .value = linear(self.withPrefix("value")),
            .attn_out = linear(attn_output.withPrefix("dense")),
            .attn_norm = layerNorm(attn_output.withPrefix("LayerNorm"), config.layer_norm_eps),
            .up = linear(view.withPrefix("intermediate").withPrefix("dense")),
            .down = linear(view.withPrefix("output").withPrefix("dense")),
            .out_norm = layerNorm(view.withPrefix("output").withPrefix("LayerNorm"), config.layer_norm_eps),
            .heads = config.num_attention_heads,
        };
    }

    /// x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        const split = .{ .h = self.heads, .hd = .auto };
        const q = self.query.forward(x, .f32).splitAxis(.dout, split).rename(.{ .s = .q });
        const k = self.key.forward(x, .f32).splitAxis(.dout, split).rename(.{ .s = .k });
        const v = self.value.forward(x, .f32).splitAxis(.dout, split).rename(.{ .s = .k });
        const heard = zml.nn.sdpa(q, k, v, .{}).merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const y = self.attn_norm.forward(self.attn_out.forward(heard, .f32).rename(.{ .dout = .d }).add(x));

        const up = geluExact(self.up.forward(y, .f32)).rename(.{ .dout = .d });
        return self.out_norm.forward(self.down.forward(up, .f32).rename(.{ .dout = .d }).add(y));
    }
};

const Encoder = struct {
    embeddings: Embeddings,
    layers: []Layer,

    /// `view` must stand on "bert". The caller frees `layers`.
    pub fn init(gpa: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Encoder {
        return ???;
    }

    /// tokens: {.s} i32 -> what the last layer thinks, {.s, .d}
    pub fn forward(self: Encoder, tokens: zml.Tensor) zml.Tensor {
        return ???;
    }
};

/// The whole file, as bytes. The caller frees them.
fn readFile(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(gpa, try file.length(io));
}

/// One tensor of a file of weights (or of activations), as a Buffer.
fn bufferFromStore(
    gpa: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const bytes = try gpa.alloc(u8, shape.byteSize());
    defer gpa.free(bytes);

    var io_buffer: [4096]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);

    return .fromBytes(io, platform, shape, bytes);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const config_bytes = try readFile(gpa, io, dir ++ "config.json");
    defer gpa.free(config_bytes);
    const parsed = try std.json.parseFromSlice(Config, gpa, config_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const config = parsed.value;

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "model.safetensors", gpa);
    defer gpa.free(path);
    var registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{platform.replicated_mesh});
    defer store.deinit();

    const acts_path = try std.Io.Dir.cwd().realPathFileAlloc(io, dir ++ "activations.safetensors", gpa);
    defer gpa.free(acts_path);
    var acts_registry: zml.safetensors.TensorRegistry = try .fromPath(gpa, io, acts_path);
    defer acts_registry.deinit();
    var acts: zml.io.TensorStore = .fromRegistry(gpa, &acts_registry, &.{platform.replicated_mesh});
    defer acts.deinit();

    const model: Encoder = try .init(gpa, store.view().withPrefix("bert"), config);
    // The model owns memory now.
    ???;

    // `weights`: room for them, a Loader to fill it, and the way
    // out for both kinds of memory.
    ???;

    try stdout.print("layers on the resume: {d}\n", .{config.num_hidden_layers});
    try stdout.print("layers in the model: {d}, in the weights: {d}\n", .{ model.layers.len, weights.layers.len });

    var ids = try bufferFromStore(gpa, io, platform, &acts, "input_ids");
    defer ids.deinit();

    // The output of the last layer, whichever it is.
    var key_buffer: [64]u8 = undefined;
    const last = try std.fmt.bufPrint(&key_buffer, "bert.encoder.layer.{d}.out", .{config.num_hidden_layers});
    var want = try bufferFromStore(gpa, io, platform, &acts, last);
    defer want.deinit();

    const tokens: zml.Tensor = .init(.{ .s = ids.shape().dim(0) }, .i32);
    var exe = try platform.compile(gpa, io, model, .forward, .{tokens}, .{});
    defer exe.deinit();

    var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, ids });
    defer got.deinit();

    const same = if (zml.testing.expectClose(io, want, got, .{})) |_| true else |_| false;
    try stdout.print("compared with {s}\n", .{last});
    try stdout.print("hidden states: {f}\n", .{got.shape()});
    try stdout.print("same as PyTorch: {}\n", .{same});
    try stdout.flush();
}
