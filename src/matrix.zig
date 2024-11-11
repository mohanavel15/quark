const std = @import("std");
const Allocator = std.mem.Allocator;

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

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.values);
        }

        pub fn print(self: *Self) void {
            std.debug.print("| ", .{});
            for (self.values, 0..) |v, i| {
                if (i != 0 and i % self.cols == 0) {
                    std.debug.print("|\n| ", .{});
                }
                std.debug.print("{d:4.1}", .{v});
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
            for (0..self.size()) |idx| {
                self.values[idx] = self.values[idx] * scalar;
            }
        }
    };
}
