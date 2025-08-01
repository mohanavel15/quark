const std = @import("std");

const Allocator = std.mem.Allocator;
const Matrix = @import("matrix.zig").Matrix;
const Activation = @import("activation.zig").Activation;

pub fn Dense(T: type, inputs: usize, outputs: usize, activation: Activation(T)) type {
    return struct {
        const Self = @This();

        weights: Matrix(T),
        biases: Matrix(T),

        pub fn init(allocator: Allocator) !Self {
            const weights = try Matrix(T).initRandom(allocator, inputs, outputs);
            const biases = try Matrix(T).initRandom(allocator, 1, outputs);
            return Self{
                .weights = weights,
                .biases = biases,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weights.deinit();
            self.biases.deinit();
        }

        pub fn forward(self: *Self, mat: *Matrix(T)) !*Matrix(T) {
            try mat.multiply(&self.weights);
            try mat.add(&self.biases);
            activation.forward(T, mat.values);
            return mat;
        }
    };
}
