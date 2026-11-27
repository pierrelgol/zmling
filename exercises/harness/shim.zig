//! The real root file of every exercise binary. You don't need to read or
//! change this to do the exercises, but nothing here is secret either.
//!
//! Each exercise is compiled as a module named "exercise"; this file imports
//! it and calls its `main`. Its main job is to keep ZML's (very chatty) debug
//! logs quiet, unless the exercise declares its own `pub const std_options`.
const std = @import("std");
const exercise = @import("exercise");
const zml = @import("zml");

comptime {
    // ZML always links a C constructor that calls into `zml.bazel`;
    // exercises that never touch ZML would fail to link without this.
    _ = zml.bazel;
}

pub const std_options: std.Options = if (@hasDecl(exercise, "std_options"))
    exercise.std_options
else
    .{ .log_level = .warn };

pub fn main(init: std.process.Init) !void {
    return exercise.main(init);
}
