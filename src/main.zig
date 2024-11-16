const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const MatrixOps = @import("matrix_ops.zig");
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

    var c = try MatrixOps.Add(allocator, a, b);
    defer c.deinit();
    c.print();

    var d = try MatrixOps.Subtract(allocator, a, b);
    defer d.deinit();
    d.print();

    var e = try MatrixOps.Multiply(allocator, a, b);
    defer e.deinit();
    e.print();

    var f = try MatrixOps.Scale(allocator, a, 20);
    defer f.deinit();
    f.print();
}
