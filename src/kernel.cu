#include <math_functions.h>
#include "pso_cluster.h"

/*
 * Get euclidean distance between 2 pixels
 */
__host__ __device__ 
float devGetDistance(int *first, int *second, int channel)
{
    float total = 0.0f;

    for (int i = 0; i < channel; i++)
    {
        int res = (first[i] - second[i]);
        total += res * res;
    }

    return sqrt(total);
}

/*
 * Get error for given centroids
 */
__host__ __device__ 
float devFitness(short* assignMat, int* datas, int* centroids, int data_size, 
                 int cluster_size, int channel)
{
    float total = 0.0f;

    for (int i = 0; i < cluster_size; i++)
    {
        float subtotal = 0.0f;

        for (int j = 0; j < data_size; j++)
        {
            if (assignMat[j] == i)
                subtotal += devGetDistance(&datas[j * channel],
                                           &centroids[i * channel], channel);
        }

        total += subtotal / data_size;
    }

    return total / cluster_size;
}

/*
 * Assign pixels to centroids
 */
__host__ __device__ 
void devAssignDataToCentroid(short *assignMat, int *datas, int *centroids, 
                             int data_size, int cluster_size, int channel)
{
    for (int i = 0; i < data_size; i++)
    {
        int nearestCentroidIdx = 0;
        float nearestCentroidDist = INF;

        for (int j = 0; j < cluster_size; j++)
        {
            float nearestDist = devGetDistance(&datas[i * channel],
                                               &centroids[j * channel], channel);

            if (nearestDist < nearestCentroidDist)
            {
                nearestCentroidDist = nearestDist;
                nearestCentroidIdx = j;
            }
        }

        assignMat[i] = nearestCentroidIdx;
    }
}

/*
 * Initialize necessary variables for PSO
 */
void initialize(int *positions, int *velocities, int *pBests, int *gBest, 
                const Data* datas, int data_size, int particle_size,
                int cluster_size, int channel)
{
    for (int i = 0; i < particle_size * cluster_size * channel; i+= channel)
    {
        int rand = round(getRandom(0, data_size - 1));

        for(int j = 0; j < channel; j++)
        {
            positions[i + j] = datas[rand].info[j];
            pBests[i + j] = datas[rand].info[j];
            velocities[i + j] = 0;
        }
    }

    for(int i = 0; i < cluster_size * channel; i++)
        gBest[i] = pBests[i];
}

/*
 * Kernel to update particle
 */
__global__ void kernelUpdateParticle(int *positions, int *velocities, 
                                     int *pBests, int *gBest, short *posAssign, 
                                     int* datas, float rp, float rg, 
                                     int data_size, int particle_size, 
                                     int cluster_size, int channel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= particle_size * cluster_size * channel)
        return;

    // Update particle velocity and position
    velocities[i] = (int)lroundf(OMEGA * velocities[i]
            + c1 * rp * (pBests[i] - positions[i])
            + c2 * rg * (gBest[i % (cluster_size * channel)] - positions[i]));

    positions[i] += velocities[i];
}

/*
 * Kernel to update particle
 */
__global__ void kernelUpdatePBest(int *positions, int *pBests, short *posAssign, 
                                  short *pBestAssign, int* datas, int data_size,
                                  int particle_size, int cluster_size,
                                  int channel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetParticle = i * cluster_size * channel;
    int offsetAssign = i * data_size;

    if(i >= particle_size)
        return;

    devAssignDataToCentroid(&posAssign[offsetAssign], datas, 
                            &positions[offsetParticle], data_size, 
                            cluster_size, channel);

    // Update pBest
    if (devFitness(&posAssign[offsetAssign], datas, &positions[offsetParticle], 
                   data_size, cluster_size, channel)
            < devFitness(&pBestAssign[offsetAssign], datas, 
                         &pBests[offsetParticle], data_size, cluster_size,
                         channel))
    {
        // Update pBest position
        for (int k = 0; k < cluster_size * channel; k++)
            pBests[offsetParticle + k] = positions[offsetParticle + k];

        // Update pBest assignment matrix
        for(int k = 0; k < data_size; k++)
            pBestAssign[offsetAssign + k] = posAssign[offsetAssign + k];
    }
}

/*
 * Wrapper to initialize and running PSO on device
 */
GBest devicePsoClustering(Data *datas, int *flatDatas, int data_size,
                          int channel, int particle_size, int cluster_size,
                          int max_iter)
{
    // Initialize host memory
    int *positions = new int[particle_size * cluster_size * channel];
    int *velocities = new int[particle_size * cluster_size * channel];
    short *posAssign = new short[particle_size * data_size];
    short *gBestAssign = new short[data_size];  

    size_t size = sizeof(int) * particle_size * cluster_size * channel;
    size_t assign_size = sizeof(short) * particle_size * data_size;
        
    int *pBests;
    int *gBest;
    short *pBestAssign;

    cudaHostAlloc((void**)&pBests, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&gBest, sizeof(int) * cluster_size * channel,
                  cudaHostAllocDefault);
    cudaHostAlloc((void**)&pBestAssign, assign_size, cudaHostAllocDefault);

    // Initialize assignment matrix to cluster 0
    for(int i = 0; i < particle_size * data_size; i++)
    {
        posAssign[i] = 0;
        pBestAssign[i] = 0;

        if(i < data_size)
            gBestAssign[i] = 0;
    }

    initialize(positions, velocities, pBests, gBest, datas, data_size, 
               particle_size, cluster_size, channel);

    // Initialize device memory
    int *devPositions, *devVelocities, *devPBests, *devGBest;
    short *devPosAssign, *devPBestAssign;
    int *devDatas;

    cudaMalloc((void**)&devPositions, size);
    cudaMalloc((void**)&devVelocities, size);
    cudaMalloc((void**)&devPBests, size);
    cudaMalloc((void**)&devGBest, sizeof(int) * cluster_size * channel);
    cudaMalloc((void**)&devPosAssign, assign_size);
    cudaMalloc((void**)&devPBestAssign, assign_size);
    cudaMalloc((void**)&devDatas, sizeof(int) * data_size * channel);

    // Copy Data from host to device
    cudaMemcpy(devPositions, positions, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocities, velocities, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBests, pBests, size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(int) * cluster_size * channel,
               cudaMemcpyHostToDevice);
    cudaMemcpy(devPosAssign, posAssign, assign_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBestAssign, pBestAssign, assign_size, 
               cudaMemcpyHostToDevice);
    cudaMemcpy(devDatas, flatDatas, sizeof(int) * data_size * channel,
               cudaMemcpyHostToDevice);

    // Threads and blocks number
    int threads = 32;
    int blocksPart = (particle_size / threads) + 1;
    int blocksFull = (particle_size * cluster_size * channel / threads) + 1;

    // Create event for asynchronous processing between memcpy and kernel
    cudaStream_t stream0, stream1;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Run kernel for the first time to populate memory
    kernelUpdateParticle<<<blocksFull, threads>>>
        (devPositions, devVelocities, devPBests, devGBest, devPosAssign, 
         devDatas, getRandomClamped(), getRandomClamped(), data_size, 
         particle_size, cluster_size, channel);

    kernelUpdatePBest<<<blocksPart, threads>>>
        (devPositions, devPBests, devPosAssign, devPBestAssign, devDatas, 
         data_size, particle_size, cluster_size, channel);

    cudaThreadSynchronize();

    // Iteration
    for (int iter = 0; iter < max_iter; iter++)
    {
        float rp = getRandomClamped();
        float rg = getRandomClamped();

        // Copy result of last iteration to host memory, async using stream-0
        cudaMemcpyAsync(pBests, devPBests, size, cudaMemcpyDeviceToHost, 
                        stream0);
        cudaMemcpyAsync(pBestAssign, devPBestAssign, assign_size, 
                        cudaMemcpyDeviceToHost, stream0);
        
        // Run kernel, async using stream-1
        kernelUpdateParticle<<<blocksFull, threads, 0, stream1>>>
            (devPositions, devVelocities, devPBests, devGBest, devPosAssign, 
             devDatas, rp, rg, data_size, particle_size, cluster_size, channel);

        kernelUpdatePBest<<<blocksPart, threads, 0, stream1>>>
            (devPositions, devPBests, devPosAssign, devPBestAssign, devDatas, 
             data_size, particle_size, cluster_size, channel);

        // Calculate gBest async between GPU and CPU
        for(int i = 0; i < particle_size; i++)
        {
            // Get slice of array
            int offsetParticle = i * cluster_size * channel;
            int offsetAssign = i * data_size;

            // Compare pBest and gBest
            if (devFitness(&pBestAssign[offsetAssign], flatDatas, 
                           &pBests[offsetParticle], data_size, cluster_size,
                           channel)
                < devFitness(gBestAssign, flatDatas, gBest, data_size, 
                             cluster_size, channel))
            {
                // Update gBest position
                for (int k = 0; k < cluster_size * channel; k++)
                    gBest[k] = pBests[offsetParticle + k];

                // Update gBest assignment matrix
                for(int k = 0; k < data_size; k++)
                    gBestAssign[k] = pBestAssign[offsetAssign + k];
            }
        }

        // Copy gBest Data back to GPU
        cudaMemcpyAsync(devGBest, gBest, sizeof(int) * cluster_size * channel,
                        cudaMemcpyHostToDevice, stream0);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    // Copy gBest from device to host
    cudaMemcpy(gBest, devGBest, sizeof(int) * cluster_size * channel,
               cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] positions;
    delete[] velocities;
    delete[] posAssign;

    cudaFree(devPositions);
    cudaFree(devVelocities);
    cudaFree(devDatas); 
    cudaFree(devPosAssign);
    cudaFree(devPBests);
    cudaFree(devGBest);
    cudaFree(devPBestAssign);

    cudaFreeHost(pBests);
    cudaFreeHost(pBestAssign);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    GBest gBestReturn;
    gBestReturn.gBestAssign = gBestAssign;
    gBestReturn.arrCentroids = gBest;
    gBestReturn.quantError = devFitness(gBestReturn.gBestAssign, flatDatas,
                                        gBestReturn.arrCentroids, data_size,
                                        cluster_size, channel);

    return gBestReturn;
}
