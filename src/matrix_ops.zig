const std = @import("std");
const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;
const MatrixError = @import("matrix.zig").MatrixError;
const backend = @import("backend.zig");

extern fn matrix_add(context: *anyopaque, a: [*]const f32, b: [*]const f32, c: [*]const f32, n: c_uint) void;
extern fn matrix_sub(context: *anyopaque, a: [*]const f32, b: [*]const f32, c: [*]const f32, n: c_uint) void;
extern fn matrix_mul(context: *anyopaque, a: [*]const f32, b: [*]const f32, c: [*]const f32, a_rows: c_uint, a_cols: c_uint, b_rows: c_uint, b_cols: c_uint) void;
extern fn matrix_scale(context: *anyopaque, scale: f32, a: [*]const f32, b: [*]const f32, n: c_uint) void;

pub fn Add(allocator: Allocator, a: Matrix(f32), b: Matrix(f32)) MatrixError!Matrix(f32) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return MatrixError.MissMatchShape;
    }

    const result = Matrix(f32).init(allocator, a.rows, b.cols) catch {
        return MatrixError.FailAlloc;
    };

    const ctx = backend.GetInstance();
    matrix_add(ctx.context, a.values.ptr, b.values.ptr, result.values.ptr, a.rows * a.cols);

    return result;
}

pub fn Subtract(allocator: Allocator, a: Matrix(f32), b: Matrix(f32)) MatrixError!Matrix(f32) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return MatrixError.MissMatchShape;
    }

    const result = Matrix(f32).init(allocator, a.rows, a.cols) catch {
        return MatrixError.FailAlloc;
    };

    const ctx = backend.GetInstance();
    matrix_sub(ctx.context, a.values.ptr, b.values.ptr, result.values.ptr, a.rows * a.cols);

    return result;
}

pub fn Multiply(allocator: Allocator, a: Matrix(f32), b: Matrix(f32)) MatrixError!Matrix(f32) {
    if (a.cols != b.rows) {
        return MatrixError.MissMatchShape;
    }

    const result = Matrix(f32).init(allocator, a.rows, b.cols) catch {
        return MatrixError.FailAlloc;
    };

    const ctx = backend.GetInstance();
    matrix_mul(ctx.context, a.values.ptr, b.values.ptr, result.values.ptr, a.rows, a.cols, b.rows, b.cols);

    return result;
}

pub fn Scale(allocator: Allocator, a: Matrix(f32), scale: f32) MatrixError!Matrix(f32) {
    const result = Matrix(f32).init(allocator, a.rows, a.cols) catch {
        return MatrixError.FailAlloc;
    };

    const ctx = backend.GetInstance();
    matrix_scale(ctx.context, scale, a.values.ptr, result.values.ptr, a.rows * a.cols);

    return result;
}
