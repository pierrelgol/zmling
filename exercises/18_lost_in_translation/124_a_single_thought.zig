//
// After the dictionaries, Bertie thinks. Twice: he has two
// layers, and they are the same code with different weights.
// So port ONE, and test it ALONE.
//
// Alone is possible because of the recording. It holds what
// the first layer received, and what it gave back:
//
//     bert.encoder.layer.0.in     {9,128}
//     bert.encoder.layer.0.out    {9,128}
//
// Feed your layer the first, compare with the second. Your
// embeddings are not involved, your tokenizer is not involved:
// if the answer is `false`, the bug is in the forty lines in
// front of you. (Chapter 20 turns this into a method. Today
// it's a first taste.)
//
// One layer of BERT is four Python classes, and they are named
// as if to confuse you. Here is who calls whom, with the names
// the weights have in the file, under `bert.encoder.layer.0`:
//
//     BertLayer
//       attention          BertAttention
//         self             BertSelfAttention   query, key, value
//         output           BertSelfOutput      dense, LayerNorm
//       intermediate       BertIntermediate    dense
//       output             BertOutput          dense, LayerNorm
//
// TWO things are called "output", at two depths. Seven
// Linears and LayerNorms, fourteen tensors.
//
// The attention is the one of chapter 15, multi-head
// (exercise 106), with two differences from the llamas:
//
//     self.query = nn.Linear(config.hidden_size, self.all_head_size)
//     ...
//     query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
//
//  - `nn.Linear` has a bias unless it says `bias=False`. These
//    three have one, and so have all the others in BERT. The
//    `linear` helper below asks for it.
//
//  - no mask. Bertie is an ENCODER: every token listens to
//    every token, the ones after it included. That is how he
//    can guess a [MASK] in the middle of a sentence.
//
// The `.view(...).transpose(1, 2)` is your `splitAxis`: tags
// spare you the transpose. And Python scales the scores by
// `attention_head_size ** -0.5`, which is what `zml.nn.sdpa`
// does when you don't tell it otherwise.
//
// Then, twice, the same little class:
//
//     class BertSelfOutput(nn.Module):
//         def forward(self, hidden_states, input_tensor):
//             hidden_states = self.dense(hidden_states)
//             hidden_states = self.dropout(hidden_states)
//             hidden_states = self.LayerNorm(hidden_states + input_tensor)
//             return hidden_states
//
// (`BertOutput.forward` is the same, word for word.)
// `input_tensor` is what entered the block: this is the
// residual connection. Look at WHERE the LayerNorm is: around
// the sum. Add, then normalize. That is called POST-NORM, and
// it is how the first transformers were built. The llamas of
// the next chapter, and nearly everything since, do the
// opposite: normalize, compute, then add (PRE-NORM). Both run.
// Both give numbers of a sensible size. One of them is the
// model.
//
// In between, the MLP (`BertIntermediate`): a dense layer up
// to `intermediate_size`, then `self.intermediate_act_fn`.
//
// `intermediate_act_fn` is `ACT2FN[config.hidden_act]`, and
// the resume says "gelu". ZML has `x.gelu()` (exercise 20).
// Done? Read ZML's doc comment for it:
//
//     /// We use an approximation of the erf function using tanh:
//     ///   gelu(x) ≃ 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
//
// and HuggingFace's "gelu", in activations.py:
//
//     return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
//
// Two functions with one name. (HuggingFace calls the tanh
// one "gelu_new".) They differ in the fourth decimal, and here
// is the nasty part: `expectClose` forgives that. Its default
// tolerance is about 1e-3, made for comparing whole models,
// where rounding errors pile up. So the program compares the
// layer a second time, strictly:
//
//     zml.testing.expectClose(io, want, got, .{
//         .absolute_tolerance = 1e-5,
//         .relative_tolerance = 1e-4,
//     })
//
// A single layer, in f32, on the same input, should pass that.
// When it passes the lenient check and fails the strict one,
// nothing is broken, something is APPROXIMATED, and the
// activation is the first suspect. Small errors don't stay
// small in a model of 24 layers.
//
// PORTING RULE 6: a name is not a definition. "gelu",
// "layer norm", "attention" each cover several formulas. Open
// the reference, find the one that is used, and port that.
//
// `erf` exists: `x.erf()`. So do `x.scale(number)` and
// `x.addConstant(number)`.
//
// The model has a second public method, `attend`, the
// attention half of the layer. `platform.compile` takes any
// method name, not only `.forward`, and the recording has the
// reference for that half too. Two checks instead of one, to
// know which half to look at.
//
// To do:
//
//  1. `init`: the four parts that come after the attention
//     itself. The tree above says where they are.
//
//  2. `attend`: the attention, then the first "output" block.
//
//  3. `forward` runs, and is wrong twice: once loudly, once in
//     the fourth decimal. For the second one, `geluExact` is
//     waiting for a body, and for somebody to call it.
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
    return ???;
}

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
            .attn_norm = ???,
            .up = ???,
            .down = ???,
            .out_norm = ???,
            .heads = config.num_attention_heads,
        };
    }

    /// BertAttention: everybody listens to everybody, then
    /// BertSelfOutput. x: {.s, .d} -> {.s, .d}
    pub fn attend(self: Layer, x: zml.Tensor) zml.Tensor {
        return ???;
    }

    /// The whole BertLayer. x: {.s, .d} -> {.s, .d}
    pub fn forward(self: Layer, x: zml.Tensor) zml.Tensor {
        const y = self.attend(x);
        const up = self.up.forward(self.out_norm.forward(y), .f32).gelu().rename(.{ .dout = .d });
        const down = self.down.forward(up, .f32).rename(.{ .dout = .d });
        return down.add(y);
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

/// `expectClose`, as a yes or a no.
fn close(io: std.Io, want: zml.Buffer, got: zml.Buffer, opts: zml.testing.CompareOpts) bool {
    zml.testing.expectClose(io, want, got, opts) catch return false;
    return true;
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

    // The first layer, and nothing else.
    const model: Layer = .init(store.view().withPrefix("bert.encoder.layer").withLayer(0), config);

    var weights = try zml.mem.bufferize(gpa, Layer, &model);
    defer zml.Buffer.deinitAll(Layer, &weights);
    var loader: zml.io.Loader = try .init(gpa, platform, .default);
    defer loader.deinit();
    try loader.load(io, Layer, &model, &weights, &store, .{});
    try loader.await(io);

    // What the layer received in PyTorch. The recording has no
    // tags: the placeholder provides them.
    var input = try bufferFromStore(gpa, io, platform, &acts, "bert.encoder.layer.0.in");
    defer input.deinit();
    const x: zml.Tensor = .init(.{ .s = input.shape().dim(0), .d = config.hidden_size }, .f32);

    {
        var want = try bufferFromStore(gpa, io, platform, &acts, "bert.encoder.layer.0.attention.out");
        defer want.deinit();
        var exe = try platform.compile(gpa, io, model, .attend, .{x}, .{});
        defer exe.deinit();
        var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, input });
        defer got.deinit();
        try stdout.print("attention half, same as PyTorch: {}\n", .{close(io, want, got, .{})});
    }
    {
        var want = try bufferFromStore(gpa, io, platform, &acts, "bert.encoder.layer.0.out");
        defer want.deinit();
        var exe = try platform.compile(gpa, io, model, .forward, .{x}, .{});
        defer exe.deinit();
        var got: zml.Buffer = try exe.eval(gpa, io, .{ weights, input });
        defer got.deinit();
        try stdout.print("whole layer, same as PyTorch: {}\n", .{close(io, want, got, .{})});

        const strict: zml.testing.CompareOpts = .{ .absolute_tolerance = 1e-5, .relative_tolerance = 1e-4 };
        try stdout.print("whole layer, strictly the same: {}\n", .{close(io, want, got, strict)});
    }
    try stdout.flush();
}
