# OpenCL_Programming

### Q1: Explain your implementation. How do you optimize the performance of convolution?

1. Reduce redundant calculation:
    ```cpp=
        for (int i=-hf; i<=hf; i++) {
            if (0 <= (thisX+i) && (thisX+i) < imageHeight) {
                for (int j=-hf; j<=hf; j++) {
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
    ```
    -  Only calculate the value within the boundary of the inputImage:
        - Because we need to do the zero-padding, some of the value will be multiplied 0 (if the index is out of the boundary of the original image), which is unnecessary. Therefore, I will check the index of the inputImage. If the index is out of the boundary of the inputImage, I will skip the calculation and continue.
        - In this way, if we check the index of the inputImage before entering the next for loop, we can reduce the computation of the sum. Because in the for loop, we will check whether the position of y is within the boundary, and then enter another for loop to check whether the position of x is within the boundary. In this way, we can save much computation than checking the position of x and y together in the inner for loop. <font color="green">(Line 2 & Line 4)</font>
    - Use a simple variable to get the value of the filter:
        - Each kernel will handle the calculation of different part of the inputImage, so the index to get the element in each kernel varies. However, each kernel will use the same filter, so we can simply get the value of the filter with a simple variable by just +1 in the iteration. In this way, we don't have the calculate the index of the filter. <font color="green">(Line 5 & Line 10)</font>
    - Skip the calculation if the element of the filter is 0:
        - If we multiply any number by 0, the result is still 0. Therefore, if the element of the filter is 0, we do not need to do the calculation. <font color="green">(Line 6)</font>
2. Use the cache memory:
    - I use the flag `CL_MEM_USE_HOST_PTR` when allocating the device memory using `clCreateBuffer`. OpenCL implementations are allowed to cache the buffer contents pointed to by host_ptr in device memory. This cached copy can be used when kernels are executed on a device. Therefore, we do not have to explicitily call `clEnqueueWriteBuffer` to write the data to the device.
    - [Document of clCreateBuffer](https://registry.khronos.org/OpenCL/sdk/1.0/docs/man/xhtml/clCreateBuffer.html)

### [Bonus] Q2: Rewrite the program using CUDA. (1) Explain your CUDA implementation, (2) plot a chart to show the performance difference between using OpenCL and CUDA, and (3) explain the result.

1. The following is the implementation of convolution using CUDA:
    - First, I allocate the memory for the inputImage, outputImage, and the filter on the device. And then, I copy the inputImage and the filter to the memory space on the device.
    - Second, I call the `convKernel()` and compute on the kernel.
    - In the end, I copy the result from `outputBuffer` to `outputImage`, and then I free the memory space on the device.

```cpp=
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
    
    int imageSize = imageHeight * imageWidth * sizeof(float);
    int filterSize = filterWidth * filterWidth * sizeof(float);

    float *inputBuffer, *outputBuffer, *filterBuffer;
    cudaMalloc(&inputBuffer, imageSize);
    cudaMalloc(&outputBuffer, imageSize);
    cudaMalloc(&filterBuffer, filterSize);

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
```

- The computation is similar to OpenCL. However, CUDA cannot get global ID directly, so I get the thread ID by using `blockIdx.x` and `blockIdx.y`

```cpp=
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
```

2. The performance difference between using OpenCL and CUDA:
![](https://i.imgur.com/18VSvfs.png)

3. Explain the result:
    - We can see that the performances of OpenCL are better than CUDA in all filter cases. We can say that it might because OpenCL use the flag `CL_MEM_USE_HOST_PTR` when using `clCreateBuffer` to get the memory. Therefore, we do not have to copy the data to the device once again. However, in CUDA, we need to allocate the memory on the host and copy the data to the device, which is time consuming.

#### Reference:
- [OpenCL介紹](https://www.readfog.com/a/1642960408086155264)
