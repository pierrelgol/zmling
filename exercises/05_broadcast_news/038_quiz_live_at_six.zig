//
// Quiz! It is 5:58 pm. The six o'clock news is live in two
// minutes, the camera (the one) is showing everybody slightly
// green, and Doug has gone to look for a sandwich.
//
// The camera delivers a batch of frames in the NCHW layout you
// met when moving the photographer's archive:
//
//     frames     {n=2,c=3,h=2,w=2}    2 frames, 3 color channels
//                                     (red, green, blue), and a
//                                     picture of 2 by 2 pixels.
//                                     It is not a good camera.
//
// Three corrections stand between this and something you can
// put on air:
//
//   1. TINT. One number per color channel, added to every pixel
//      of that channel in every frame. This is exactly how a
//      neural network adds a per-channel bias to a batch of
//      images, so you will write this line again in real life.
//
//          tint       {c=3}
//
//   2. VIGNETTE. The lens is darker in two corners. One factor
//      per pixel position, the same for every frame and channel,
//      multiplied in. It comes from the lens manufacturer's
//      spreadsheet. Kevin's cousin. No tags.
//
//          vignette   {2,2}
//
//   3. EXPOSURE. One number per frame; the whole frame is
//      divided by it. (Normalizing each item of a batch by a
//      value of its own: another line you will write again.)
//
//          exposure   {n=2}
//
//      Doug left a note about this one, taped to the camera:
//      "NO broadcast function for the exposure. Not `broad`, not
//      `broadcastAnything`. Read the fine print. - D."
//
// All together:
//
//     on_air = (frames + tint) * vignette / exposure
//
// Everything you need is in exercises 033 to 037. Mind which
// side of each operation the small tensor is on.
//
const std = @import("std");
const zml = @import("zml");

fn onAir(frames: zml.Tensor, tint: zml.Tensor, vignette: zml.Tensor, exposure: zml.Tensor) zml.Tensor {
    // 1. One value per channel.
    const tinted = frames.add(???);

    // 2. One value per pixel position, untagged.
    const shaded = tinted.mul(???);

    // 3. One value per frame. Mind Doug's note.
    return shaded.div(???);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &buffer);
    const stdout = &stdout_writer.interface;

    const platform: *zml.Platform = try .init(gpa, io, .cpu, .{});
    defer platform.deinit(gpa, io);

    // Four placeholders, with the shapes listed above. All f32.
    const frames: zml.Tensor = ???;
    const tint: zml.Tensor = ???;
    const vignette: zml.Tensor = ???;
    const exposure: zml.Tensor = ???;

    var exe = try platform.compileFn(gpa, io, onAir, .{ frames, tint, vignette, exposure }, .{});
    defer exe.deinit();

    // Frame 1 is frame 0 with the studio lights at full power.
    var frames_buffer: zml.Buffer = try .fromBytes(io, platform, frames.shape(), std.mem.sliceAsBytes(&[2][3][2][2]f32{
        .{
            .{ .{ 100, 120 }, .{ 140, 160 } },
            .{ .{ 110, 130 }, .{ 150, 170 } },
            .{ .{ 100, 120 }, .{ 140, 160 } },
        },
        .{
            .{ .{ 200, 240 }, .{ 280, 320 } },
            .{ .{ 220, 260 }, .{ 300, 340 } },
            .{ .{ 200, 240 }, .{ 280, 320 } },
        },
    }));
    defer frames_buffer.deinit();

    // Less green, please.
    var tint_buffer: zml.Buffer = try .fromBytes(io, platform, tint.shape(), std.mem.sliceAsBytes(&[_]f32{ 0, -10, 20 }));
    defer tint_buffer.deinit();

    var vignette_buffer: zml.Buffer = try .fromBytes(io, platform, vignette.shape(), std.mem.sliceAsBytes(&[2][2]f32{
        .{ 1, 0.5 },
        .{ 0.5, 1 },
    }));
    defer vignette_buffer.deinit();

    var exposure_buffer: zml.Buffer = try .fromBytes(io, platform, exposure.shape(), std.mem.sliceAsBytes(&[_]f32{ 1, 4 }));
    defer exposure_buffer.deinit();

    var result: zml.Buffer = try exe.eval(gpa, io, .{ frames_buffer, tint_buffer, vignette_buffer, exposure_buffer });
    defer result.deinit();

    const picture = try result.getValue([2][3][2][2]f32, io);

    try stdout.print("On air: {f}\n", .{result.shape()});
    for (picture, 0..) |frame, n| {
        for (frame, [_][]const u8{ "red", "green", "blue" }) |channel, name| {
            try stdout.print("frame {d} {s:<5}:", .{ n, name });
            for (channel) |line| {
                for (line) |pixel| try stdout.print(" {d:.2}", .{pixel});
            }
            try stdout.print("\n", .{});
        }
    }
    try stdout.flush();
}
