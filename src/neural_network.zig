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
            try mat.mat_mul(&self.weights);
            try mat.add(&self.biases);
            activation.forward(T, mat.values);
            return mat;
        }
    };
}

pub fn Embedding(T: type, vocab_size: usize, embed_dim: usize) type {
    return struct {
        const Self = @This();

        weights: Matrix(T),

        pub fn init(allocator: Allocator) !Self {
            const weights = try Matrix(T).initRandom(allocator, vocab_size, embed_dim);
            return Self{
                .weights = weights,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weights.deinit();
        }

        pub fn forward(self: *Self, allocator: Allocator, mat: []usize) !Matrix(T) {
            var out = try Matrix(T).init(allocator, mat.len, embed_dim);

            for (mat, 0..) |tok, tok_i| {
                if (tok >= vocab_size) {
                    @panic("oofff");
                }

                for (0..embed_dim) |idx| {
                    out.set_at(tok_i, idx, self.weights.get(tok, idx));
                }
            }

            return out;
        }
    };
}
