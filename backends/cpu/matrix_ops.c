void matrix_add(const float* a, const float* b, float* c, unsigned int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void matrix_sub(const float* a, const float* b, float* c, unsigned int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

void matrix_mul(const float* a, const float* b, float* c, unsigned int a_rows, unsigned int a_cols, unsigned int b_rows, unsigned int b_cols) {
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

void matrix_scale(float scale, const float* a, float* b, unsigned int n) {
    for (int i = 0; i < n; ++i) {
        b[i] = a[i] * scale;
    }
}