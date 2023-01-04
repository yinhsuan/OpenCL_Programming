#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "helper.h"
extern "C"{
#include "hostFE.h"
}

__global__ void convKernel(float *inputImage,
                           float *outputImage,
                           float *filter, 
                           int imageWidth, 
                           int imageHeight, 
                           int filterWidth) {
    int thisX = blockIdx.y * blockDim.y + threadIdx.y;
    int thisY = blockIdx.x * blockDim.x + threadIdx.x;
    int hf = filterWidth / 2;
    float sum = 0;
    int idx = 0;

    for (int i = -hf; i <= hf; i++) {
        if (0 <= (thisX+i) && (thisX+i) < imageHeight) {
            for (int j = -hf; j <= hf; j++){
                if (0 <= (thisY+j) && (thisY+j) < imageWidth) {
                    float filterVal = filter[idx];
                    if (filterVal != 0) {
                        sum += inputImage[(i+thisX)*imageWidth + (j+thisY)] * filterVal;
                    }
                }
                idx++;
            }
        }
    }
    outputImage[thisX*imageWidth + thisY] = sum;
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    // int filterSize = filterWidth * filterWidth * sizeof(float);
    // int mem_size = imageHeight * imageWidth * sizeof(float);
    // int threadsPerBlock = 16;
    
    int imageSize = imageHeight * imageWidth * sizeof(float);
    int filterSize = filterWidth * filterWidth * sizeof(float);

    float *inputBuffer, *outputBuffer, *filterBuffer;
    cudaMalloc(&inputBuffer, imageSize);
    cudaMalloc(&outputBuffer, imageSize);
    cudaMalloc(&filterBuffer, filterSize);
    printf("CUDA\n");

    cudaMemcpy(inputBuffer, inputImage, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(filterBuffer, filter, filterSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
    convKernel<<<numBlocks, threadsPerBlock>>>(inputBuffer, outputBuffer, filterBuffer, imageWidth, imageHeight, filterWidth);

    cudaMemcpy(outputImage, outputBuffer, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(inputBuffer);
    cudaFree(outputBuffer);
    cudaFree(filterBuffer);
}









// #include <stdio.h>
// #include <stdlib.h>
// #include "hostFE.h"
// #include "helper.h"
// // #include <CL/cl.h>
// // #include <time.h>

// __global__ void convKernel(float *inputImage,
//                            float *outputImage,
//                            float *filter, 
//                            int imageWidth,
//                            int imageHeight, 
//                            int hf,
//                            int filterWidth) {
//     int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
//     int thisY = blockIdx.y * blockDim.y + threadIdx.y;
//     if (thisX >= imageWidth || thisY >= imageHeight) return;

//     float4 sum = make_float4(0.0, 0.0, 0.0, 0.0), f, cal;
//     int yy, xx, k, l, idx;
//     int fidx = -1;


//     for (k = -hf; k <= hf; k++) {
//         if(thisY+k >= 0 && thisY+k < imageHeight) {
// 		yy = now_y+k;
//         for (l = -hf; l <= hf; l++) {
//             fidx++;
//             if(filter[fidx] == 0) continue;
//             xx = thisX + l;
//             if (xx >= 0 && xx < imageWidth) {
//                 cal = make_float4(inputImage[yy+xx], inputImage[yy+xx+1], inputImage[yy+xx+2], inputImage[yy+xx+3]);
//                 f = make_float4(filter[fidx], filter[fidx], filter[fidx], filter[fidx]);
//                 sum.x += cal.x * f.x;
//                 sum.y += cal.y * f.y;
//                 sum.z += cal.z * f.z;
//                 sum.w += cal.w * f.w;
//             }
            
//             }
//         }
//     }
//     idx = thisX * imageWidth + thisY;
//     outputImage[idx] = sum.x;
//     outputImage[idx+1] = sum.y;
//     outputImage[idx+2] = sum.z;
//     outputImage[idx+3] = sum.w;

// }

// void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
//             float *inputImage, float *outputImage, cl_device_id *device,
//             cl_context *context, cl_program *program)
// {
//     int threadsPerBlock = 16;
//     dim3 block(4, 16);
//     dim3 grid((int) ceil(imageWidth / threadsPerBlock), (int) ceil(imageHeight / threadsPerBlock));

//     int imageSize = imageHeight * imageWidth * sizeof(float);
//     int filterSize = filterWidth * filterWidth * sizeof(float);

//     float *inputBuffer, *outputBuffer, *filterBuffer;
//     cudaMalloc((void**)&inputBuffer, imageSize);
//     cudaMalloc((void**)&outputBuffer, imageSize);
//     cudaMalloc((void**)&filterBuffer, filterSize);

//     cudaMemcpy(inputBuffer, inputImage, imageSize, cudaMemcpyHostToDevice);
//     cudaMemcpy(filterBuffer, filter, filterSize, cudaMemcpyHostToDevice);

//     int hf = (filterWidth>>1);
//     convKernel<<<grid, block>>>(inputBuffer, outputBuffer, filterBuffer, imageWidth, imageHeight, hf, filterWidth);

// }