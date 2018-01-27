#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t cvtWithCuda(float *c,unsigned char *a, int w, int h, int tile_w, int tile_h, int idx, cudaStream_t stream);

__global__ void cvtKernel(float *c,unsigned char *a,int w, int h, int tile_w, int tile_h)  
{ 
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < w * tile_w && y < h * tile_h ){
          int offset = x + y * w * tile_w;
          float b = (float)a[offset * 3 + 0]/255.0;
          float g = (float)a[offset * 3 + 1]/255.0;
          float r = (float)a[offset * 3 + 2]/255.0;
          int n = x/w + y/h * tile_w;
          int x1 = x - (x / w) * w;
          int y1 = y - (y / h) * h;
          int idx = x1 + y1 * w;
          c[n*w*h*3+idx] = r;
          c[n*w*h*3+w*h+idx] = g;
          c[n*w*h*3+w*h*2+idx] = b; 
        }

} 

cudaError_t cvtWithCuda(float *c,unsigned char *a, int w, int h,int tile_w, int tile_h, int idx,cudaStream_t stream)  
{  
    unsigned char *dev_a = 0;
    float *dev_c = c;
    int size = 3*w*h*tile_w*tile_h;  
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system. 
     cudaStatus = cudaSetDevice(idx);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");  
        return cudaStatus; 
   }  
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(unsigned char));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        cudaFree(dev_a);
        return cudaStatus;  
       
    }  
    cudaStatus = cudaMemcpyAsync(dev_a, a, size * sizeof(unsigned char), cudaMemcpyHostToDevice,stream);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        cudaFree(dev_a);  
        return cudaStatus;  
    } 
    // Launch a kernel on the GPU with one thread for each element.
    dim3 block(32,32,1);
    int gridx = (w * tile_w - 1 + 32)/32;
    int gridy = (h * tile_h - 1 + 32)/32;
    dim3 grid(gridx,gridy,1); 
    cvtKernel<<<grid,block,0,stream>>>(dev_c,dev_a,w,h,tile_w,tile_h);  
    cudaStatus = cudaStreamSynchronize(stream);
    if(cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaStreamSynchronize returned error code %d after launching addKernel!\n", cudaStatus);  
        cudaFree(dev_a);
        return cudaStatus;  
    }  
    cudaFree(dev_a);
    return cudaStatus;  
}

extern "C" int Cudacvt(float *c,unsigned char *a, int w, int h, int tile_w, int tile_h, int idx, cudaStream_t stream){
    cudaError_t cudaStatus;  
    cudaStatus = cvtWithCuda(c,a, w, h, tile_w, tile_h, idx,stream);  
    if (cudaStatus != cudaSuccess)   
    {  
        fprintf(stderr, "addWithCuda failed!");  
        return -1;  
    }  
    return 0;
}

