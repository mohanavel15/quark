const std = @import("std");
const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;
const MatrixError = @import("matrix.zig").MatrixError;

extern fn matrix_add(a: [*]const f32, b: [*]const f32, c: [*]const f32, n: c_uint) void;
extern fn matrix_sub(a: [*]const f32, b: [*]const f32, c: [*]const f32, n: c_uint) void;
extern fn matrix_mul(a: [*]const f32, b: [*]const f32, c: [*]const f32, a_rows: c_uint, a_cols: c_uint, b_rows: c_uint, b_cols: c_uint) void;

pub fn Add(allocator: Allocator, a: Matrix(f32), b: Matrix(f32)) MatrixError!Matrix(f32) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return MatrixError.MissMatchShape;
    }

    const result = Matrix(f32).init(allocator, a.rows, b.cols) catch {
        return MatrixError.FailAlloc;
    };

    matrix_add(a.values.ptr, b.values.ptr, result.values.ptr, a.rows * a.cols);

    return result;
}

pub fn Sub(allocator: Allocator, a: Matrix(f32), b: Matrix(f32)) MatrixError!Matrix(f32) {
    if (a.rows != b.rows or a.cols != b.cols) {
        return MatrixError.MissMatchShape;
    }

    const result = Matrix(f32).init(allocator, a.rows, a.cols) catch {
        return MatrixError.FailAlloc;
    };

    matrix_sub(a.values.ptr, b.values.ptr, result.values.ptr, a.rows * a.cols);

    return result;
}

pub fn Mul(allocator: Allocator, a: Matrix(f32), b: Matrix(f32)) MatrixError!Matrix(f32) {
    if (a.cols != b.rows) {
        return MatrixError.MissMatchShape;
    }

    const result = Matrix(f32).init(allocator, a.rows, b.cols) catch {
        return MatrixError.FailAlloc;
    };

    matrix_mul(a.values.ptr, b.values.ptr, result.values.ptr, a.rows, a.cols, b.rows, b.cols);

    return result;
}
