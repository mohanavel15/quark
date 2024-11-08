const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
const MatrixOps = @import("matrix_ops.zig");

pub fn main() !void {
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

    var d = try MatrixOps.Sub(allocator, a, b);
    defer d.deinit();
    d.print();

    var e = try MatrixOps.Mul(allocator, a, b);
    defer e.deinit();
    e.print();
}
