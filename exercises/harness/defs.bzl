"""Build rules for zmling exercises."""

load("@rules_zig//zig:defs.bzl", "zig_binary", "zig_library")

def zmling_exercises(data = []):
    """Declares one binary per .zig file of the calling package.

    `NNN_name.zig` becomes the target `:NNN`, runnable with
    `bazel run //zmling/<chapter>:NNN`. The exercise only depends on ZML.

    The binary is deliberately named by number only. The name of the binary
    ends up in the `@typeName` of ZML's C types, and ZML's comptime reflection
    on those names runs out of Zig's default branch quota ("evaluation exceeded
    1000 backwards branches") when the name is longer than about 23 characters.

    Args:
      data: files the exercises need at run time (model weights, ...).
    """
    for f in native.glob(["*.zig"]):
        name = f.split("_")[0]
        zig_library(
            name = f[:-len(".zig")],
            main = f,
            import_name = "exercise",
            deps = ["//zml"],
        )
        zig_binary(
            name = name,
            main = "//zmling/harness:shim.zig",
            data = data,
            deps = [":" + f[:-len(".zig")], "//zml"],
        )
