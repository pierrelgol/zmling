//
// Quiz: Special Delivery
//
// Friday, 16:40. Gus puts on his coat. "Four parcels came down
// the chute. I am going home. The machine does them, all four,
// and I read the report on Monday."
//
// This is the program of the chapter, whole: a small
// `examples/llm` for routers. For each label below it prints
// one report, and the reports are the specification:
//
//     parcel: overseas
//       entry point: index
//       slip: conveyor, 2 layers, tied head: false, float16
//       boxes: 5 tensors, 528 bytes
//       device: 528 bytes
//       route: dock > scale > x-ray > belt B > the basement
//
// What each line wants from you:
//
//  - THE DOCK (179). Every path of this program goes through
//    a `zml.io.VFS` with the `file` backend: the labels say
//    `file://`. The Platform gets that io too; stdout keeps
//    `init.io`.
//  - THE LABEL (176). `Label` below has its help text and
//    nothing else. Labels are parsed from the fixed lists in
//    `labels`. "parcel:" is the last word of the repository
//    path (`std.fs.path.basename`).
//  - THE ENTRY POINT (178): the kind of file
//    `resolveModelEntrypoint` finds. Then one registry for the whole repository.
//  - THE SLIP (177). `model_type`; `num_hidden_layers`, absent
//    for a chute; `eos_token_id`, a number or a list;
//    `max_length`, absent or null or a number, 20 unless it is
//    a number; `tie_word_embeddings`; `torch_dtype`. Printed
//    with
//
//        "  slip: {s}, {d} layers, tied head: {}, {s}\n"
//
//    One slip names an architecture this mail room does not
//    have. Its report ends there, with
//
//        "  slip: {s}: {}\n"      the model_type, the error
//
//    and the program goes on with the next label. Nothing was
//    compiled for it, nothing was loaded.
//  - THE SORTING TABLE (181). A chute and a conveyor, as you
//    built them: `model.embed_tokens.weight`,
//    `model.layers.N.fc.weight` and maybe `.bias`,
//    `lm_head.weight` or a tied head. Stored in f32, f16 or
//    bf16; computed in f32.
//  - THE FORKLIFT (180). Compile, then load, with the three
//    options of the label. "boxes:" is what the registry
//    holds, "device:" what the Loader moved:
//
//        "  boxes: {d} tensors, {d} bytes\n"
//        "  device: {d} bytes\n"
//
//  - THE ROUTE. From station `<start>` of the label, one call
//    of the model per hop, until a station the slip calls the
//    end, or until the route is `--max-steps` stations long
//    (the slip's `max_length` when the label says nothing):
//
//        "  route: {s}"   then   " > {s}"   for each hop
//        " ... (gave up after {d} stations)"   if it never ended
//        "\n"
//
// Free everything: the leak checker reads the report before
// Gus does. `std.Progress` stays in its drawer today.
//
const std = @import("std");
const zml = @import("zml");

const flags = zml.stdx.flags;
const json = zml.stdx.json;
const View = zml.io.TensorStore.View;

const parcels = "file://zmling/26_the_mail_room/parcels/";

const stations = [_][]const u8{ "dock", "scale", "x-ray", "belt A", "belt B", "the 14th floor", "return to sender", "the basement" };

const labels = [_][]const [*:0]const u8{
    &.{ "mailroom", parcels ++ "floor14" },
    &.{ "mailroom", "--parallelism=4", "--chunks=4", "--chunk-size=64KiB", parcels ++ "overseas" },
    &.{ "mailroom", parcels ++ "tube" },
    &.{ "mailroom", "--max-steps=5", parcels ++ "annex", "2" },
};

const Label = struct {
    pub const help =
        \\ mailroom [options] <repo> [<start>]
        \\
        \\   <repo>               the model repository (required)
        \\   <start>              number of the station to start from (default: 0)
        \\   --max-steps=<n>      longest route, in stations (default: ask the config)
        \\   --parallelism=<n>    tensors loaded at the same time (default: 1)
        \\   --chunks=<n>         number of loading buffers (default: 2)
        \\   --chunk-size=<size>  size of a loading buffer (default: 4KiB)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    // The slip, the models, the sorting table, the forklift, and Gus's
    // four reports.
    ???
}
