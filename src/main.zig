const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const Backend = @import("backend.zig");
const NN = @import("neural_network.zig");

pub fn main() !void {
    var backend = Backend.Init();
    defer backend.Deinit();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        switch (gpa.deinit()) {
            .leak => std.debug.panic("Memory Leak Detected\n", .{}),
            .ok => std.debug.print("Cleaned up successfully\n", .{}),
        }
    }

    const allocator = gpa.allocator();

    var a = Matrix(f32).init(allocator, 2, 2) catch {
        return error.OutOfMemory;
    };
    defer a.deinit();
    a.fill(1);

    a.print();

    var b = Matrix(f32).init(allocator, 2, 2) catch {
        return error.OutOfMemory;
    };
    defer b.deinit();
    b.fill(5);
    b.print();

    try a.add(&b);
    a.print();
    a.fill(1);

    try a.subtract(&b);
    a.print();
    a.fill(1);

    try a.multiply(&b);
    a.print();
    a.fill(1);

    a.scale(20);
    a.print();

    var input = try Matrix(f32).initRandom(allocator, 5, 3);
    defer input.deinit();

    input.print();

    var layer = try NN.Linear(f32, 3, 1).init(allocator);
    defer layer.deinit();

    _ = try layer.forward(&input);

    input.print();

    var c = try Matrix(f32).init(allocator, 1, 5);
    defer c.deinit();

    var sig_inputs = [_]f32{ 0, 1, -1, 1000, -1000 };

    try c.set(@constCast(&sig_inputs));
    c.sigmoid();
    c.print();

    const sig_expect = [_]f32{ 0.5, 0.73105857863, 0.26894142137, 1.0, 0.0 };

    for (0..5) |i| {
        if (c.values[i] != sig_expect[i]) {
            std.debug.print("{} - {} - {}\n", .{ c.values[i], sig_expect[i], sig_inputs[i] });
        }
    }

    var d = try Matrix(f32).init(allocator, 1, 5);
    defer d.deinit();

    var relu_inputs = [_]f32{ 0, 1, -1, -2.5, 2.5 };

    try d.set(@constCast(&relu_inputs));
    d.relu();
    d.print();

    const relu_expect = [_]f32{ 0, 1, 0, 0, 2.5 };

    for (0..5) |i| {
        if (d.values[i] != relu_expect[i]) {
            std.debug.print("{} - {} - {}\n", .{ d.values[i], relu_expect[i], relu_inputs[i] });
        }
    }
}
