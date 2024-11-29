const std = @import("std");

pub fn build(b: *std.Build) void {
    const opencl = b.option(bool, "opencl", "Matrix ops opencl backend") orelse false;

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // const lib = b.addStaticLibrary(.{
    //     .name = "quark",
    //     .root_source_file = b.path("src/root.zig"),
    //     .target = target,
    //     .optimize = optimize,
    // });

    // b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "quark",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    if (opencl) {
        exe.linkLibC(); // Seems to need it.
        exe.linkSystemLibrary("OpenCL");
        exe.addIncludePath(b.path("backends/opencl/backend.h"));
        exe.addCSourceFile(.{ .file = b.path("backends/opencl/matrix_ops.c"), .flags = &.{} });
    } else {
        exe.linkLibC();
        exe.addIncludePath(b.path("backends/cpu/backend.h"));
        exe.addCSourceFile(.{ .file = b.path("backends/cpu/matrix_ops.c"), .flags = &.{} });
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
