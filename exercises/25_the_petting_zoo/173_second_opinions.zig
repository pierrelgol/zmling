//
// Fifth enclosure: the clinic. Marge's zoo is small, but it has
// FOUR vets, because nobody is good at everything: Dr. Hoof
// does legs, Dr. Lung does coughs, Dr. Both treats whichever
// is worse, and Dr. Nope only sees the healthy ones.
//
// Sending every animal to every vet would be thorough, and
// four times the bill. So there is a receptionist. She looks at
// the animal, picks the TWO vets who fit best, and the
// prescription is a blend of their two opinions, weighted by
// how sure she was of each.
//
// That is a MIXTURE OF EXPERTS (MoE), and it replaces the MLP
// of a transformer block (the "each token thinks alone" half
// of exercise 131) in Mixtral, DeepSeek, Qwen's MoE models and
// many others. The deal: a model with E experts has the
// WEIGHTS of E MLPs, and each token pays the COMPUTE of only k
// of them. Knowledge grows, the bill does not.
//
// Per token x, {.d}:
//
//  1. ROUTER. One Linear, d -> E: a score per expert. Softmax
//     over `.expert`: the receptionist's confidence.
//  2. TOP-K. Keep the k best experts: their ids, and their
//     confidences. You know the op since exercise 073:
//
//         probs.topK(.{ .topk = .expert }, k, .{})
//
//     reads "the top k along `.expert`, and call the new axis
//     `.topk`". It returns `.values` and `.indices` (i32),
//     best first.
//  3. RENORMALIZE. The k confidences no longer sum to 1 (the
//     losers took their share with them): divide by their sum
//     over `.topk`.
//  4. THE EXPERTS. Each one is a small MLP: up, activation,
//     down. All the "up" matrices of the clinic are stacked in
//     ONE tensor, {.expert, .hid, .d}, and all the "down" ones
//     in another, {.expert, .dout, .hid}: that is also how
//     they sit in the checkpoint. Choosing the experts of a
//     token is a `gather` on `.expert` (exercise 070). The ids
//     are {.b, .s, .topk}, so:
//
//         self.up.gather(.{ .expert = ids }, .{})
//             -> {.b, .s, .topk, .hid, .d}
//
//     every token now has its own k matrices. A `dot` over
//     `.d` with x {.b, .s, .d} contracts `.d` and treats the
//     axes the two sides share (`.b`, `.s`) as a batch: you get
//     {.b, .s, .topk, .hid}. Activation, then the same with
//     `down` over `.hid`.
//  5. BLEND. The k opinions {.b, .s, .topk, .dout} against the
//     k weights {.b, .s, .topk}: one last `dot`, over `.topk`.
//
// (Read the reference code of the model you port for step 3:
// some renormalize, some do not, some take the softmax over
// the k winners only. On two numbers that sum to 1, the last
// one and ours are the same thing.)
//
// THE PANTRY. Gathering one matrix per token and per choice is
// the clear way, not the fast one: it copies whole experts
// around. A serious backend sorts the tokens by expert and
// runs each expert once on its group, with quantized weights
// and a hand-written GPU kernel. ZML has a directory of those,
// `zml/zml/moe/`, behind one function:
//
//     pub fn forwardMoe(
//         input: zml.Tensor,          // {.b, .s, .d}
//         topk_ids: zml.Tensor,       // {.b, .s, .topk}, i32
//         topk_weights: zml.Tensor,   // {.b, .s, .topk}
//         gate_up: zml.nn.Linear,     // weight: [expert, out, in]
//         down: zml.nn.Linear,        // weight: [expert, out, in]
//         backend: Backend,
//         opts: Options,
//     ) zml.Tensor                    // {.b, .s, .d}
//
// The routing stays yours (steps 1 to 3); it does steps 4 and
// 5. The two Linears hold the STACKED rank 3 weights (it
// retags them itself, the order of the axes is what counts).
// `Options` has three fields and no defaults:
//
//     .activation                .relu, .gelu, .silu, .swiglu, ...
//     .quantize_input            false, unless you know
//     .routing_weight_placement  .after_down: blend the opinions,
//                                as in step 5
//
// ("gate_up" because with `.swiglu`, the gown of exercise 142,
// the first matrix holds the gate AND the up halves, side by
// side. With `.relu` it is a plain "up".)
//
// And `Backend`? `zml.moe.Backend.auto(platform, scheme, dtype)`
// picks a kernel for CUDA, ROCm, Metal... and for a CPU it
// returns an error: `main` prints which. One backend runs
// everywhere, `.stablehlo`: the file `moe/stablehlo.zig` begins
// with "Deliberately simple MoE reference". Open it: its
// `project` function is your step 4. It is what you compare a
// port against (ZML's Qwen 3.5 MoE example calls `forwardMoe`
// exactly this way), and today it checks YOU.
//
// The numbers. An animal is (limp, cough), a prescription is
// (bandages, syrup). The pony limps, the goose coughs, the
// goat mostly limps.
//
// To do:
//
//  1. `route`: steps 1, 2, 3.
//  2. `forward` is written, runs, and prescribes a NEGATIVE
//     amount of syrup. The reference disagrees with it. An
//     MLP is up, ACTIVATION, down.
//  3. `reference`: the same visit through `zml.moe.forwardMoe`
//     and its `.stablehlo` backend. One call.
//  4. In `main`: the weights of the clinic, as Buffers.
//
const std = @import("std");
const zml = @import("zml");

const Clinic = struct {
    /// weight {.expert, .d}: one row per vet.
    router: zml.nn.Linear,
    /// {.expert, .hid, .d}
    up: zml.Tensor,
    /// {.expert, .dout, .hid}
    down: zml.Tensor,
    top_k: u32,

    pub const Routing = struct {
        /// {.b, .s, .topk}, i32: the chosen vets, best first.
        ids: zml.Tensor,
        /// {.b, .s, .topk}: how much each one counts. Sums to 1.
        weights: zml.Tensor,
    };

    pub const Visit = struct {
        ids: zml.Tensor,
        weights: zml.Tensor,
        /// {.b, .s, .dout}
        prescription: zml.Tensor,
    };

    /// x: {.b, .s, .d}
    pub fn route(self: Clinic, x: zml.Tensor) Routing {
        return ???;
    }

    /// The clinic, by hand. x: {.b, .s, .d}
    pub fn forward(self: Clinic, x: zml.Tensor) Visit {
        const routing = self.route(x);

        const up = self.up.gather(.{ .expert = routing.ids }, .{});
        const down = self.down.gather(.{ .expert = routing.ids }, .{});

        const hidden = x.dot(up, .d);
        const opinions = hidden.dot(down, .hid);

        return .{
            .ids = routing.ids,
            .weights = routing.weights,
            .prescription = opinions.dot(routing.weights, .topk),
        };
    }

    /// The clinic, by the pantry. x: {.b, .s, .d} -> {.b, .s, .d}
    pub fn reference(self: Clinic, x: zml.Tensor) zml.Tensor {
        return ???;
    }
};

const vets = [_][]const u8{ "Dr. Hoof", "Dr. Lung", "Dr. Both", "Dr. Nope" };

/// {.expert = 4, .d = 2}: what each vet likes to see, (limp, cough).
const router_weight = [4][2]f32{
    .{ 2, 0 },
    .{ 0, 2 },
    .{ 1, 1 },
    .{ -1, -1 },
};

/// {.expert = 4, .hid = 2, .d = 2}: what each vet looks at.
const up_weight = [4][2][2]f32{
    .{ .{ 1, 0 }, .{ 0, 1 } },
    .{ .{ 1, 0 }, .{ 0, 1 } },
    // Dr. Both: how much worse the limp is than the cough, and
    // the other way round. One of the two is always negative.
    .{ .{ 1, -1 }, .{ -1, 1 } },
    .{ .{ -1, 0 }, .{ 0, -1 } },
};

/// {.expert = 4, .dout = 2, .hid = 2}: what each vet prescribes,
/// (bandages, syrup).
const down_weight = [4][2][2]f32{
    .{ .{ 2, 0 }, .{ 0, 0 } }, // bandages. Many.
    .{ .{ 0, 0 }, .{ 0, 2 } }, // syrup. A lot.
    .{ .{ 1, 0 }, .{ 0, 1 } },
    .{ .{ 0, 0 }, .{ 0, 0 } }, // "it will pass"
};

const patients = [_][]const u8{ "pony ", "goose", "goat " };

/// {.b = 1, .s = 3, .d = 2}: (limp, cough)
const symptoms = [1][3][2]f32{.{
    .{ 2, 0 },
    .{ 0, 2 },
    .{ 1.5, 0.5 },
}};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    const clinic: Clinic = .{
        .router = .init(.init(.{ .expert = 4, .d = 2 }, .f32), null, .d),
        .up = .init(.{ .expert = 4, .hid = 2, .d = 2 }, .f32),
        .down = .init(.{ .expert = 4, .dout = 2, .hid = 2 }, .f32),
        .top_k = 2,
    };
    const x: zml.Tensor = .init(.{ .b = 1, .s = 3, .d = 2 }, .f32);

    // The weights of the clinic, in a `zml.Bufferized(Clinic)` named
    // `weights`. Free them at the end.
    ???;

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), std.mem.sliceAsBytes(&symptoms));
    defer x_buffer.deinit();

    // By hand.
    var exe = try platform.compile(gpa, io, clinic, .forward, .{x}, .{});
    defer exe.deinit();
    var runner = try exe.runner(gpa);
    defer runner.deinit(gpa);

    var visit: zml.Bufferized(Clinic.Visit) = undefined;
    runner.run(io, .{ weights, x_buffer }, .{&visit}, .{ .wait = true });
    defer zml.Buffer.deinitAll(Clinic.Visit, &visit);

    try stdout.print("{d} vets, {d} opinions per animal: {f} -> ids {f}\n", .{ vets.len, clinic.top_k, x.shape(), visit.ids.shape() });
    const ids = try visit.ids.getValue([1][3][2]i32, io);
    const blend = try visit.weights.getValue([1][3][2]f32, io);
    const by_hand = try visit.prescription.getValue([1][3][2]f32, io);
    for (patients, ids[0], blend[0], by_hand[0]) |patient, id, w, p| {
        try stdout.print("  {s}  {d:.2} {s} + {d:.2} {s}  ->  {d:.3} bandages, {d:.3} syrup\n", .{
            patient,
            w[0],
            vets[@intCast(id[0])],
            w[1],
            vets[@intCast(id[1])],
            p[0],
            p[1],
        });
    }

    // By the pantry.
    if (zml.moe.Backend.auto(platform, null, .f32)) |backend| {
        try stdout.print("the backend ZML picks for this platform: {t}\n", .{backend});
    } else |err| {
        try stdout.print("the backend ZML picks for this platform: {t}\n", .{err});
    }
    try stdout.print(".stablehlo is available: {}\n", .{zml.moe.Backend.stablehlo.isAvailable(platform)});

    var reference_exe = try platform.compile(gpa, io, clinic, .reference, .{x}, .{});
    defer reference_exe.deinit();

    var reference: zml.Buffer = try reference_exe.eval(gpa, io, .{ weights, x_buffer });
    defer reference.deinit();

    var agrees = true;
    for (by_hand[0], (try reference.getValue([1][3][2]f32, io))[0]) |mine, theirs| {
        for (mine, theirs) |a, b| {
            if (@abs(a - b) > 1e-5) agrees = false;
        }
    }
    try stdout.print("the reference {f} agrees: {}\n", .{ reference.shape(), agrees });
    try stdout.flush();
}
