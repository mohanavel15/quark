const std = @import("std");
const math = std.math;

pub fn Activation(T: type) type {
    return struct {
        forward: fn (T: type, values: []T) void,
        backward: fn (T: type, values: []T) void,
    };
}

const Sigmoid = struct {
    pub fn forward(T: type, values: []T) void {
        for (0..values.len) |idx| {
            values[idx] = 1 / (1 + math.exp(-values[idx]));
        }
    }

    pub fn backward(T: type, values: []T) void {
        for (0..values.len) |idx| {
            values[idx] = values[idx] * (1 - values[idx]);
        }
    }
};

pub fn sigmoid(T: type) Activation(T) {
    return Activation(T){
        .forward = Sigmoid.forward,
        .backward = Sigmoid.backward,
    };
}

const Noop = struct {
    pub fn forward(T: type, _: []T) void {}
    pub fn backward(T: type, _: []T) void {}
};

pub fn noop(T: type) Activation(T) {
    return Activation(T){
        .forward = Noop.forward,
        .backward = Noop.backward,
    };
}

const Tanh = struct {
    pub fn forward(T: type, values: []T) void {
        for (0..values.len) |idx| {
            const ei = math.exp(values[idx]);
            const nei = math.exp(-values[idx]);
            values[idx] = (ei - nei) / (ei + nei);
        }
    }

    pub fn backward(T: type, _: []T) void {}
};

pub fn tanh(T: type) Activation(T) {
    return Activation(T){
        .forward = Tanh.forward,
        .backward = Tanh.backward,
    };
}

const ReLU = struct {
    pub fn forward(T: type, values: []T) void {
        for (0..values.len) |idx| {
            values[idx] = (values[idx] + @abs(values[idx])) / 2;
        }
    }

    pub fn backward(T: type, _: []T) void {}
};

pub fn relu(T: type) Activation(T) {
    return Activation(T){
        .forward = ReLU.forward,
        .backward = ReLU.backward,
    };
}

const Softmax = struct {
    pub fn forward(T: type, values: []T) void {
        var sum: T = 0;
        for (0..values.len) |idx| {
            values[idx] = math.exp(values[idx]);
            sum += values[idx];
        }

        for (0..values.len) |idx| {
            values[idx] /= sum;
        }
    }

    pub fn backward(T: type, _: []T) void {}
};

pub fn softmax(T: type) Activation(T) {
    return Activation(T){
        .forward = Softmax.forward,
        .backward = Softmax.backward,
    };
}
