const std = @import("std");
const Allocator = std.mem.Allocator;
const backend = @import("backend.zig");

extern fn matrix_add(context: *anyopaque, a: [*]const f32, b: [*]const f32, n: c_uint) void;
extern fn matrix_sub(context: *anyopaque, a: [*]const f32, b: [*]const f32, n: c_uint) void;
extern fn matrix_mul(context: *anyopaque, a: [*]const f32, b: [*]const f32, c: [*]const f32, a_rows: c_uint, a_cols: c_uint, b_rows: c_uint, b_cols: c_uint) void;
extern fn matrix_scale(context: *anyopaque, scale: f32, a: [*]const f32, n: c_uint) void;

extern fn f_sigmoid(context: *anyopaque, a: [*]const f32, n: c_uint) void;
extern fn f_relu(context: *anyopaque, a: [*]const f32, n: c_uint) void;
extern fn f_softmax(context: *anyopaque, a: [*]const f32, n: c_uint) void;
extern fn f_tanh(context: *anyopaque, a: [*]const f32, n: c_uint) void;

pub const MatrixError = error{
    FailAlloc,
    MissMatchShape,
};

pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        rows: u32,
        cols: u32,
        allocator: Allocator,
        values: []T,

        pub fn init(allocator: Allocator, rows: u32, cols: u32) MatrixError!Self {
            const self = Self{
                .rows = rows,
                .cols = cols,
                .allocator = allocator,
                .values = allocator.alloc(T, rows * cols) catch return MatrixError.FailAlloc,
            };

            @memset(self.values, 0);

            return self;
        }

        pub fn initRandom(allocator: Allocator, rows: u32, cols: u32) MatrixError!Self {
            const self = Self{
                .rows = rows,
                .cols = cols,
                .allocator = allocator,
                .values = allocator.alloc(T, rows * cols) catch return MatrixError.FailAlloc,
            };

            var prng = std.rand.DefaultPrng.init(2423432);
            var random = prng.random();

            for (0..(rows * cols)) |idx| {
                self.values[idx] = random.float(T);
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.values);
        }

        pub fn print(self: *Self) void {
            std.debug.print("| ", .{});
            for (self.values, 0..) |v, i| {
                if (i != 0 and i % self.cols == 0) {
                    std.debug.print("|\n| ", .{});
                }
                std.debug.print("{d:7.4} ", .{v});
            }
            std.debug.print("|\n\n", .{});
        }

        pub fn size(self: *Self) usize {
            return self.rows * self.cols;
        }

        pub fn fill(self: *Self, fill_value: T) void {
            @memset(self.values, fill_value);
        }

        pub fn set(self: *Self, values: []T) MatrixError!void {
            if (self.values.len != values.len) {
                return MatrixError.MissMatchShape;
            }
            @memcpy(self.values, values);
        }

        pub fn set_at(self: *Self, row: usize, col: usize, value: T) void {
            self.values[(row * self.cols) + col] = value;
        }

        pub fn get(self: *Self, row: usize, col: usize) T {
            return self.values[(row * self.cols) + col];
        }

        pub fn scale(self: *Self, scalar: T) void {
            const ctx = backend.GetInstance();
            matrix_scale(ctx.context, scalar, self.values.ptr, @intCast(self.size()));
        }

        pub fn add(self: *Self, mat: *Self) !void {
            if (self.rows != mat.rows or self.cols != mat.cols) {
                return MatrixError.MissMatchShape;
            }

            const ctx = backend.GetInstance();
            matrix_add(ctx.context, self.values.ptr, mat.values.ptr, @intCast(self.size()));
        }

        pub fn subtract(self: *Self, mat: *Self) !void {
            if (self.rows != mat.rows or self.cols != mat.cols) {
                return MatrixError.MissMatchShape;
            }

            const ctx = backend.GetInstance();
            matrix_sub(ctx.context, self.values.ptr, mat.values.ptr, @intCast(self.size()));
        }

        pub fn multiply(self: *Self, mat: *Self) !void {
            if (self.cols != mat.rows) {
                return MatrixError.MissMatchShape;
            }

            const result = self.allocator.alloc(T, self.rows * mat.cols) catch return MatrixError.FailAlloc;

            const ctx = backend.GetInstance();
            matrix_mul(ctx.context, self.values.ptr, mat.values.ptr, result.ptr, self.rows, self.cols, mat.rows, mat.cols);

            self.allocator.free(self.values);
            self.values = result;
            self.cols = mat.cols;
        }

        pub fn sigmoid(self: *Self) void {
            const ctx = backend.GetInstance();
            f_sigmoid(ctx.context, self.values.ptr, @intCast(self.size()));
        }

        pub fn relu(self: *Self) void {
            const ctx = backend.GetInstance();
            f_relu(ctx.context, self.values.ptr, @intCast(self.size()));
        }

        pub fn softmax(self: *Self) void {
            const ctx = backend.GetInstance();
            f_softmax(ctx.context, self.values.ptr, @intCast(self.size()));
        }

        pub fn tanh(self: *Self) void {
            const ctx = backend.GetInstance();
            f_tanh(ctx.context, self.values.ptr, @intCast(self.size()));
        }
    };
}
