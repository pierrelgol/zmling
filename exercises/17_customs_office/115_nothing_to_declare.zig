//
// Chapter 17: The Customs Office
//
// Sixteen chapters of ZML, and every model you ran was one we
// built for you, with a script next to it explaining every
// weight. That is over. From here to the end of the course you
// PORT: somebody trained a model in PyTorch, saved the weights
// in a file, and you make it run in ZML and give the same
// numbers. It is most of what people do with ZML for a living.
//
// Your new job is at the customs office of a small port. Chief
// Inspector Odile has worked here for thirty years and has
// never believed a declaration in her life. Basile, the
// trainee, believes all of them. And at the counter this
// morning stands a traveller, M. Nist, with one suitcase:
//
//     mnist.safetensors        1.6 MB
//
// No passport. No papers. No source code. He says he can read
// handwritten digits, and that he would like a job.
//
// Basile is already typing `const Model = struct {`. Odile
// takes the keyboard away from him. Here is her first rule, and
// the first rule of porting:
//
//     INSPECT THE CHECKPOINT BEFORE YOU WRITE A LINE OF MODEL.
//
// A checkpoint, whatever its size, is a TABLE: for every
// tensor a name, a shape and a dtype, and then the bytes. You
// opened one by hand in exercise 097, and with the registry in
// exercise 098:
//
//     var registry: zml.safetensors.TensorRegistry =
//         try .fromPath(gpa, io, path);
//     defer registry.deinit();
//
//     registry.tensors.count()    how many tensors
//     registry.iterator()         all of them: `it.next()` gives
//                                 an entry, with `.key_ptr.*`
//                                 (the name) and `.value_ptr.*`
//     tensor.shape                a zml.Shape: `.rank()`,
//                                 `.dim(i)`, `.count()` (how
//                                 many numbers), `.dtype()`
//
// This costs nothing: the registry reads the header, a few
// hundred bytes, and not one weight. And the table answers the
// questions that decide everything you will write later:
//
//  - THE NAMES. PyTorch builds them from the attributes of the
//    Python class: `self.fc1 = nn.Linear(...)` saves
//    "fc1.weight" and "fc1.bias". So the names ARE the
//    structure of the model, and your Zig struct will follow
//    them (next exercise). No "bias" next to a "weight" means
//    a layer without bias. A name you cannot explain means a
//    part of the model you have not understood yet.
//
//  - THE SHAPES. A `nn.Linear` weight is a matrix stored
//    {out, in}: OUTPUT SIZE FIRST. Read them in order and the
//    data flow of the model appears: what goes in, how wide it
//    is inside, what comes out.
//
//  - THE DTYPES. f32 here. A big model is often f16 or bf16,
//    and sometimes a few tensors are not the dtype of the
//    others. Better to learn it now than from a panic.
//
//  - THE COUNT. Model cards say "0.4M parameters". If your sum
//    disagrees, this is not the file you think it is.
//
// The files of this chapter are not in the directory of the
// exercises. The ZML repository already knows where to
// download M. Nist and his test (it has a MNIST example of its
// own, `zml/examples/mnist/`), and the BUILD.bazel of this
// chapter copies them into the runfiles, at the usual place.
//
// Open the suitcase. Three things to write:
//
//  1. the registry;
//  2. `parameterCount`: every number of every tensor;
//  3. in `main`, the inventory: one line per tensor, and for
//     each MATRIX (rank 2) one more line that reads its shape
//     the way PyTorch means it. The format strings are given.
//
// Then look at what you printed, before the next exercise
// explains it: 784 is 28 x 28. And there are ten digits.
//
const std = @import("std");
const zml = @import("zml");

const checkpoint_path = "zmling/17_customs_office/mnist.safetensors";

/// How many numbers the checkpoint holds, all tensors together.
fn parameterCount(registry: *zml.safetensors.TensorRegistry) usize {
    return ???;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, checkpoint_path, gpa);
    defer gpa.free(path);

    // The header of the file. No weight is read.
    var registry: zml.safetensors.TensorRegistry = ???;

    try stdout.print("In the suitcase: {d} tensors\n", .{registry.tensors.count()});

    // One line per tensor, in the order of the file:
    //
    //     try stdout.print("  {s:<10} {f}\n", .{ name, shape });
    //
    // `all_f32` must end up false if one of them is not f32.
    var all_f32 = true;
    ???;
    try stdout.print("Every tensor is f32: {}\n", .{all_f32});
    try stdout.print("Parameters: {d}\n", .{parameterCount(&registry)});

    // One line per matrix (rank 2), in the order of the file:
    //
    //     try stdout.print("  {s}: {d} numbers in, {d} numbers out\n", .{ name, in, out });
    try stdout.print("What the matrices say:\n", .{});
    ???;
    try stdout.flush();
}
