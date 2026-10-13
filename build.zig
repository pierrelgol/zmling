const std = @import("std");

// ZML itself can only be built with Bazel, so this build.zig is a thin
// launcher (same idea as Ziglings): it builds runner/runner.zig, which owns
// the exercise list and drives Bazel inside the `zml` submodule.
pub fn build(b: *std.Build) !void {
    // Remove the standard install and uninstall steps.
    b.top_level_steps = .{};

    const healed = b.option(bool, "healed", "Run exercises from patches/healed") orelse false;
    const exno = b.option(usize, "n", "Select exercise");
    const start = b.option(usize, "s", "Start at exercise");
    const reset = b.option(bool, "reset", "Reset exercise progress") orelse false;
    const bazel = b.option([]const u8, "bazel", "Bazel (or bazelisk) executable") orelse "bazel";

    const zmling_step = b.step("zmling", "Run zmling");
    b.default_step = zmling_step;

    if (reset) {
        zmling_step.dependOn(&b.addSystemCommand(&.{ "rm", "-f", ".progress.txt" }).step);
        return;
    }

    const runner = b.addExecutable(.{
        .name = "runner",
        .root_module = b.createModule(.{
            .root_source_file = b.path("runner/runner.zig"),
            .target = b.graph.host,
        }),
    });

    const run = b.addRunArtifact(runner);
    run.addArg(b.fmt("--bazel={s}", .{bazel}));
    run.addArg(b.fmt("--work-path={s}", .{if (healed) "patches/healed" else "exercises"}));
    run.addArg(b.fmt("--root-path={s}", .{b.root.root_dir.path.?}));
    if (exno) |n| {
        run.addArg(b.fmt("--only={d}", .{n}));
    } else if (start) |s| {
        run.addArg(b.fmt("--start={d}", .{s}));
    }
    zmling_step.dependOn(&run.step);

    const tests = b.addTest(.{ .root_module = runner.root_module });
    b.step("test", "Test the runner itself").dependOn(&b.addRunArtifact(tests).step);
}
