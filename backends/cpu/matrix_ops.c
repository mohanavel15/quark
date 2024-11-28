#include "backend.h"
#include <math.h>

void matrix_add(void* context, float* a, const float* b, unsigned int n) {
    for (int i = 0; i < n; ++i) {
        a[i] += b[i];
    }
}

void matrix_sub(void* context, float* a, const float* b, unsigned int n) {
    for (int i = 0; i < n; ++i) {
        a[i] -= b[i];
    }
}

void matrix_mul(void* context, const float* a, const float* b, float* c, unsigned int a_rows, unsigned int a_cols, unsigned int b_rows, unsigned int b_cols) {
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            float sum = 0;
            for (int k = 0; k < a_cols; ++k) {
                sum += a[a_cols * i + k] * b[b_cols * k + j];
            }
            c[b_cols * i + j] = sum;
        }
    }
}

void matrix_scale(void* context, float scale, float* a, unsigned int n) {
    for (int i = 0; i < n; ++i) {
        a[i] *= scale;
    }
}

void sigmoid(void* context, float* a, unsigned int n) {
    for (int i = 0; i < n; i++) {
        a[i] = 1.0 / (1.0 + expf(a[i]));
    }
}

void relu(void* context, float* a, unsigned int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (a[i] - fabsf(a[i])) / 2;
    }
}