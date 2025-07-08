const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;

var rng: std.Random.DefaultPrng = undefined;
var rng_init: bool = false;

pub fn getRng() *std.Random.DefaultPrng {
    if (rng_init) return &rng;

    rng = std.Random.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch unreachable;
        break :blk seed;
    });

    return &rng;
}

pub const MatrixError = error{
    FailAlloc,
    MissMatchShape,
};

pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        rows: usize,
        cols: usize,
        allocator: Allocator,
        values: []T,

        pub fn init(allocator: Allocator, rows: usize, cols: usize) MatrixError!Self {
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

            var random = getRng().random();

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
            for (0..self.size()) |idx| {
                self.values[idx] *= scalar;
            }
        }

        pub fn add(self: *Self, mat: *Self) !void {
            if (self.rows != mat.rows or self.cols != mat.cols) {
                return MatrixError.MissMatchShape;
            }

            for (0..self.size()) |idx| {
                self.values[idx] += mat.values[idx];
            }
        }

        pub fn subtract(self: *Self, mat: *Self) !void {
            if (self.rows != mat.rows or self.cols != mat.cols) {
                return MatrixError.MissMatchShape;
            }

            for (0..self.size()) |idx| {
                self.values[idx] -= mat.values[idx];
            }
        }

        pub fn multiply(self: *Self, mat: *Self) !void {
            if (self.cols != mat.rows) {
                return MatrixError.MissMatchShape;
            }

            const result = self.allocator.alloc(T, self.rows * mat.cols) catch return MatrixError.FailAlloc;
            @memset(result, 0);

            for (0..self.rows) |i| {
                for (0..mat.cols) |j| {
                    for (0..self.cols) |k| {
                        result[mat.cols * i + j] += self.values[self.cols * i + k] * mat.values[mat.cols * k + j];
                    }
                }
            }

            self.allocator.free(self.values);
            self.values = result;
            self.cols = mat.cols;
        }

        pub fn sigmoid(self: *Self) void {
            for (0..self.size()) |idx| {
                self.values[idx] = 1 / (1 + math.exp(-self.values[idx]));
            }
        }

        pub fn relu(self: *Self) void {
            for (0..self.size()) |idx| {
                self.values[idx] = (self.values[idx] + @abs(self.values[idx])) / 2;
            }
        }

        pub fn softmax(self: *Self) void {
            var sum: T = 0;
            for (0..self.size()) |idx| {
                self.values[idx] = math.exp(self.values[idx]);
                sum += self.values[idx];
            }

            for (0..self.size()) |idx| {
                self.values[idx] /= sum;
            }
        }

        pub fn tanh(self: *Self) void {
            for (0..self.size()) |idx| {
                const ei = math.exp(self.values[idx]);
                const nei = math.exp(-self.values[idx]);
                self.values[idx] = (ei - nei) / (ei + nei);
            }
        }
    };
}
