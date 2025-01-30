#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <math.h>
#include <stdio.h>

#include "backend.h"

const char* cl_matrix_add = "\
__kernel void add(__global float *a, __global const float *b) { \
    int i = get_global_id(0); \
    a[i] += b[i]; \
}";

void matrix_add(void* context, float* a, const float* b, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bytes_len, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, bufferB, CL_TRUE, 0, bytes_len, b, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_add, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "add", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
}

const char* cl_matrix_sub = "\
__kernel void sub(__global float *a, __global const float *b) { \
    int i = get_global_id(0); \
    a[i] -= b[i]; \
}";

void matrix_sub(void* context, float* a, const float* b, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bytes_len, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, bufferB, CL_TRUE, 0, bytes_len, b, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_sub, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "sub", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
}

const char* cl_matrix_mul = "\
__kernel void mul(\
    __global const float *a, \
    __global const float *b, \
    __global float *c, \
    unsigned int a_cols, \
    unsigned int b_cols) { \
    int i = get_global_id(0); \
    int j = get_global_id(1); \
    float sum = 0; \
    for (int k = 0; k < a_cols; ++k) { \
        sum += a[a_cols * i + k] * b[b_cols * k + j]; \
    } \
    c[b_cols * i + j] = sum; \
}";

void matrix_mul(void* context, const float* a, const float* b, float* c, unsigned int a_rows, unsigned int a_cols, unsigned int b_rows, unsigned int b_cols) {
    Context* ctx = (Context*)context;
    int len_a = a_rows * a_cols * sizeof(float);
    int len_b = b_rows * b_cols * sizeof(float);
    int len_c = a_rows * b_cols * sizeof(float);

    cl_mem buf_a = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, len_a, NULL, NULL);
    cl_mem buf_b = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, len_b, NULL, NULL);
    cl_mem buf_c = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, len_c, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, buf_a, CL_TRUE, 0, len_a, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, buf_b, CL_TRUE, 0, len_b, b, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_mul, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);
    
    cl_kernel kernel = clCreateKernel(program, "mul", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_c);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &a_cols);
    clSetKernelArg(kernel, 4, sizeof(unsigned int), &b_cols);
    
    size_t globalSize[2] = {a_rows, b_cols};
    clEnqueueNDRangeKernel(ctx->queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, buf_c, CL_TRUE, 0, len_c, c, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_c);
}

const char* cl_matrix_scale = "\
__kernel void scale(float scale, __global float *a) { \
    int i = get_global_id(0); \
    a[i] *= scale; \
}";

void matrix_scale(void* context, float scale, float* a, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_scale, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "scale", NULL);

    clSetKernelArg(kernel, 0, sizeof(float), &scale);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferA);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
}

const char* cl_sigmoid = "\
__kernel void sigmoid(__global float *a) { \
    int i = get_global_id(0); \
    a[i] = 1.0 / (1.0 + exp(-a[i])); \
}";

void f_sigmoid(void* context, float* a, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_sigmoid, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "sigmoid", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
}

const char* cl_relu = "\
__kernel void relu(__global float *a) { \
    int i = get_global_id(0); \
    a[i] = (a[i] + fabs(a[i])) / 2; \
}";

void f_relu(void* context, float* a, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_relu, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "relu", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
}

void f_softmax(void* context, float* a, unsigned int n) {
    printf("Softmax is not computed parallely\n");

    float sum = 0;

    for (int i = 0; i < n; i++) {
        a[i] = expf(a[i]);
        sum +=  a[i];
    }

    for (int i = 0; i < n; i++) {
        a[i] /= sum;
    }
}

const char* cl_tanh = "\
__kernel void tanh(__global float *a) { \
    int i = get_global_id(0); \
    float ei = exp(a[i]); \
    float nei = exp(-a[i]); \
    a[i] = (ei - nei) / (ei + nei); \
}";

void f_tanh(void* context, float* a, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, bytes_len, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_tanh, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "tanh", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
}