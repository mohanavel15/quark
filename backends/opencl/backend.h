#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
} Context;

void* initialize() {
    Context* context = (Context*)malloc(sizeof(Context));

    clGetPlatformIDs(1, &context->platform, NULL);
    clGetDeviceIDs(context->platform, CL_DEVICE_TYPE_GPU, 1, &context->device, NULL);
    context->context = clCreateContext(NULL, 1, &context->device, NULL, NULL, NULL);
    context->queue = clCreateCommandQueueWithProperties(context->context, context->device, 0, NULL);

    return context;
}

void deinitialize(void* context) {
    Context* ctx = (Context*)context;

    clReleaseCommandQueue(ctx->queue);
    clReleaseContext(ctx->context);

    free(ctx);
}