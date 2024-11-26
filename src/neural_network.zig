const std = @import("std");

const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;

pub fn Linear(dt: type, inputs: usize, outputs: usize) type {
    return struct {
        const Self = @This();

        weights: Matrix(dt),

        pub fn init(allocator: Allocator) !Self {
            const weights = try Matrix(dt).initRandom(allocator, inputs, outputs);
            return Self{
                .weights = weights,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weights.deinit();
        }

        pub fn forward(self: *Self, mat: *Matrix(dt)) !*Matrix(dt) {
            try mat.multiply(&self.weights);
            return mat;
        }
    };
}
