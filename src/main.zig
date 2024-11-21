const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const Backend = @import("backend.zig");

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
}
