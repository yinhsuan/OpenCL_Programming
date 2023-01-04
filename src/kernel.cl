__kernel void convolution(__global __read_only float *inputImage,
                          __global __write_only float *outputImage,
                          __constant float *filter, 
                          const int imageWidth,
                          const int imageHeight, 
                          const int filterWidth)
{
    int thisX = get_global_id(1);
    int thisY = get_global_id(0);
    int hf = filterWidth / 2;
    float sum = 0;
    int idx = 0;

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
    outputImage[thisX*imageWidth + thisY] = sum;
}