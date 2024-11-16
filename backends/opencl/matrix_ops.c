#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include "backend.h"

/*
Following code is not great and can be improved.
*/

const char* cl_matrix_add = "\
__kernel void add(__global const float *a, __global const float *b, __global float *c) { \
    int i = get_global_id(0); \
    c[i] = a[i] + b[i]; \
}";

void matrix_add(void* context, const float* a, const float* b, float* c, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);
    cl_mem bufferC = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, bufferB, CL_TRUE, 0, bytes_len, b, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_add, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "add", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(ctx->queue, bufferC, CL_TRUE, 0, bytes_len, c, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
}

const char* cl_matrix_sub = "\
__kernel void sub(__global const float *a, __global const float *b, __global float *c) { \
    int i = get_global_id(0); \
    c[i] = a[i] - b[i]; \
}";

void matrix_sub(void* context, const float* a, const float* b, float* c, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);
    cl_mem bufferC = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(ctx->queue, bufferB, CL_TRUE, 0, bytes_len, b, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_sub, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "sub", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferC, CL_TRUE, 0, bytes_len, c, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
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
__kernel void scale(float scale, __global const float *a, __global float *b) { \
    int i = get_global_id(0); \
    b[i] = a[i] * scale; \
}";

void matrix_scale(void* context, float scale, const float* a, float* b, unsigned int n) {
    Context* ctx = (Context*)context;
    int bytes_len = n * sizeof(float);

    cl_mem bufferA = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY, bytes_len, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY, bytes_len, NULL, NULL);

    clEnqueueWriteBuffer(ctx->queue, bufferA, CL_TRUE, 0, bytes_len, a, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(ctx->context, 1, &cl_matrix_scale, NULL, NULL);
    clBuildProgram(program, 1, &ctx->device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "scale", NULL);

    clSetKernelArg(kernel, 0, sizeof(float), &scale);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferB);

    size_t globalSize = n;
    clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(ctx->queue, bufferB, CL_TRUE, 0, bytes_len, b, 0, NULL, NULL);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
}