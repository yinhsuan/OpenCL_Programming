#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
// #include <CL/cl.h>
// #include <time.h>


void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    cl_int cirErrNum;
    cl_int errcode_ret;

    int imageSize = imageHeight * imageWidth * sizeof(float);
    int filterSize = filterWidth * filterWidth * sizeof(float);

    // printf("imageWidth: %d\n", imageWidth);
    // printf("imageHeight: %d\n", imageHeight);
    // printf("filterWidth: %d\n", filterWidth);

    // Creates a kernal object
    cl_kernel kernel = clCreateKernel(*program, "convolution", &errcode_ret);

    // Create a command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &cirErrNum);

    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize, inputImage, &cirErrNum); // context: the execute env. for OpenCL to run the kernel
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, imageSize, outputImage, &cirErrNum);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize, filter, &cirErrNum);

    // Set Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &imageHeight);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &filterWidth);

    // Set global size & local size
    size_t localws[2] = {8, 8};
    size_t globalws[2] = {imageWidth, imageHeight};
    printf("OpenCL\n");

    // Execute kernel
    cirErrNum = clEnqueueNDRangeKernel(queue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);
    // CHECK(cirErrNum, "clEnqueueNDRangeKernel");
    cirErrNum = clFinish(queue);
    // CHECK(cirErrNum, "clFinish");
    cirErrNum = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize, outputImage, NULL, NULL, NULL);
    // CHECK(cirErrNum, "clEnqueueReadBuffer");
    
}