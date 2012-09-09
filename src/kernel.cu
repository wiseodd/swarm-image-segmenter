#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>
#include "kernel.h"

/*
 * Get euclidean distance between 2 pixels
 */
__device__ float getDistance(int *first, int *second)
{
	float total = 0.0f;

	for (int i = 0; i < DATA_DIM; i++)
	{
		int res = (first[i] - second[i]);
		total += res * res;
	}

	return sqrt(total);
}

/*
 * Get error for given centroids
 */
__device__ float devFitness(const short* assignMat, const int* datas, const int* centroids, 
	int data_size, int cluster_size)
{
	float total = 0.0f;
	int tempCentroid[DATA_DIM], tempData[DATA_DIM];
	int iter = 0;

	for (int i = 0; i < cluster_size * DATA_DIM; i += DATA_DIM)
	{
		float subtotal = 0.0f;

		tempCentroid[0] = centroids[i + 0];
		tempCentroid[1] = centroids[i + 1];
		tempCentroid[2] = centroids[i + 2];

		for (int j = 0; j < data_size * DATA_DIM; j += DATA_DIM)
		{
			if (assignMat[j] == iter)
			{
				tempData[0] = datas[j + 0];
				tempData[1] = datas[j + 1];
				tempData[2] = datas[j + 2];

				subtotal += getDistance(tempData, tempCentroid);
			}
		}

		total += subtotal / data_size;

		iter++;
	}

	return total / cluster_size;
}

/*
 * Assign pixels to centroids
 */
__device__ void devAssignDataToCentroid(short *assignMat, const int *datas, const int *centroids, 
	int data_size, int cluster_size)
{
	int tempCentroid[DATA_DIM], tempData[DATA_DIM];
	int iter = 0;

	for (int i = 0; i < data_size * DATA_DIM; i += DATA_DIM)
	{
		int nearestCentroidIdx = 0;
		double nearestCentroidDist = INF;

		tempData[0] = datas[i + 0];
		tempData[1] = datas[i + 1];
		tempData[2] = datas[i + 2];

		int centroidNum = 0;

		for (int j = 0; j < cluster_size * DATA_DIM; j += DATA_DIM)
		{
			tempCentroid[0] = centroids[j + 0];
			tempCentroid[1] = centroids[j + 1];
			tempCentroid[2] = centroids[j + 2];

			double nearestDist = getDistance(tempData, tempCentroid);

			if (nearestDist < nearestCentroidDist)
			{
				nearestCentroidDist = nearestDist;
				nearestCentroidIdx = centroidNum;
			}

			centroidNum++;
		}

		assignMat[iter++] = nearestCentroidIdx;
	}
}

/*
 * Initialize necessary variables for PSO
 */
void initialize(int *positions, int *velocities, int *pBests, int *gBest, const data* datas, int data_size
	int particle_size, int cluster_size)
{
	for (int i = 0; i < particle_size * cluster_size * DATA_DIM; i+= DATA_DIM)
	{
		int rand = round(getRandom(0, data_size - 1));

		for(int j = 0; j < DATA_DIM; j++)
		{
			positions[i + j] = datas[rand].info[j];
			pBests[i + j] = datas[rand].info[j];
			velocities[i + j] = 0;
		}
	}

	for(int i = 0; i < cluster_size * DATA_DIM; i++)
		gBest[i] = pBests[i];
}

/*
 * Kernel to update particle
 */
__global__ void kernelUpdateParticle(int *positions, int *velocities, int *pBests, int *gBest, int *posAssign,
	const data* datas, float rp, float rg, int data_size, int particle_size, int cluster_size)
{
	size_t size = particle_size * cluster_size * DATA_DIM;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= size || i % (cluster_size * DATA_DIM) != 0)
		return;

	for (int j = 0; j < cluster_size * DATA_DIM; j += DATA_DIM)
	{
		for (int k = 0; k < DATA_DIM; k++)
		{
			// Update particle velocity and position
			velocities[i + j + k] = (int)lroundf(OMEGA * velocities[i + j + k]
					+ c1 * rp * (pBests[i + j + k] - positions[i + j + k])
					+ c2 * rg * (gBest[j + k] - positions[i + j + k]);

			positions[i + j + k] += velocities[i + j + k];
		}
	}

	int centroids[cluster_size * DATA_DIM];

	for(int j = 0; j < cluster_size * DATA_DIM; j++)
		centroids[j] = positions[i + j];

	devAssignDataToCentroid(posAssign, datas, particles[i].position, data_size, cluster_size);
}

/*
 * Kernel to update pBests
 */
 __global__ void kernelUpdatePBest(int *positions, int *pBests, int *posAssign, int *pBestAssign, 
 	int *datas, int data_size, int particle_size, int cluster_size)
{
	size_t size = particle_size * data_size * DATA_DIM;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= size || i % (data_size * DATA_DIM) != 0)
		return;

	int particleNum = i / (data_size * DATA_DIM);
	int tempAssign1[data_size * DATA_DIM], tempAssign2[data_size * DATA_DIM];
	int tempPos1[cluster_size * DATA_DIM], tempPos2[cluster_size * DATA_DIM]

	// Get slices of assignment array
	for(int j = 0; j < data_size * DATA_DIM; j++)
	{
		tempAssign1[j] = posAssign[i + j];
		tempAssign2[j] = pBestAssign[i + j];
	}

	// Get slices of particle array
	int stride = cluster_size * DATA_DIM;

	for(int j = 0; j < stride; j++)
	{
		tempPos1[j] = positions[particleNum * stride + j];
		tempPos2[j] = pBests[particleNum * stride + j];
	}

	// Update pBest
	if (devFitness(tempAssign1, datas,tempPos1, data_size, cluster_size)
			< devFitness(tempAssign2, datas, tempPos2, data_size, cluster_size))
	{
		// Update pBest position
		for (int k = 0; k < stride; k++)
			pBests[particleNum * stride + k] = positions[particleNum * stride + k];

		// Update pBest assignment matrix
		for(int k = 0; k < data_size * DATA_DIM; k++)
			pBestAssign[i + k] = posAssign[i + k]	
	}
}

/*
 * Wrapper to initialize and running PSO on device
 */
extern "C" GBest devicePsoClustering(data *datas, int *flatDatas, int data_size, int particle_size, 
	int cluster_size, int max_iter)
{
	// Initialize host memory
	int *positions = new int[particle_size * cluster_size * DATA_DIM];
	int *velocities; = new int[particle_size * cluster_size * DATA_DIM];
	int *pBests = new int[particle_size * cluster_size * DATA_DIM];
	int *gBest = new int[cluster_size * DATA_DIM];
	short *posAssign = new short[particle_size * data_size * DATA_DIM];
	short *pBestAssign = new short[particle_size * data_size * DATA_DIM];
	short *gBestAssign = new short[data_size * DATA_DIM];

	// Initialize assignment matrix to cluster 0
	for(int i = 0; i < particle_size * data_size; i++)
	{
		posAssign[i] = 0;
		pBestAssign[i] = 0;

		if(i < data_size)
			gBestAssign[i] = 0;
	}

	initialize(positions, velocities, pBests, gBest, datas, data_size, particle_size, cluster_size);

	// Initialize device memory
	int *devPositions;
	int *devVelocities;
	int *devPBests;
	int *devGBest;
	short *devPosAssign;
	short *devPBestAssign;
	short *devGBestAssign;
	int *devDatas;

	size_t size = sizeof(int) * particle_size * cluster_size * DATA_DIM;
	size_t assign_size = sizeof(int) * particle_size * data_size * DATA_DIM;

	cudaMalloc((void**)&devPositions, size);
	cudaMalloc((void**)&devVelocities, size);
	cudaMalloc((void**)&devPBests, size);
	cudaMalloc((void**)&devGBest, sizeof(int) * cluster_size * DATA_DIM);
	cudaMalloc((void**)&devPosAssign, assign_size);
	cudaMalloc((void**)&devPBestAssign, assign_size);
	cudaMalloc((void**)&devGBestAssign, assign_size);
	cudaMalloc((void**)&devDatas, sizeof(int) * data_size * DATA_DIM)

	// Copy data from host to device
	cudaMemcpy(devPositions, positions, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVelocities, velocities, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devPBests, pBests, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devGBest, gBest, sizeof(int) * cluster_size * DATA_DIM, cudaMemcpyHostToDevice);
	cudaMemcpy(devPosAssign, posAssign, assign_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devPBestAssign, pBestAssign, assign_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devGBestAssign, gBestAssign, assign_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devDatas, flatDatas, sizeof(int) * data_size * DATA_DIM, cudaMemcpyHostToDevice);

	// Threads and blocks number
	int threads = 32;
	int blocks = particles_size <= 32 ? 1 : particle_size / threads;

	// Iteration
	for (int iter = 0; iter < max_iter; iter++)
	{
		float rp = getRandomClamped();
		float rg = getRandomClamped();

		kernelUpdateParticle<<<blocks, threads>>>(devPositions, devVelocities, devPBests, devGBest, 
				devPosAssign, devDatas, rp, rg, data_size, particle_size, cluster_size);

		kernelUpdatePBest<<<blocks, threads>>>(devPositions, devPBests, devPosAssign, devPBestAssign, 
 				devDatas, data_size, particle_size, cluster_size);

		// Compute gBest on host
		cudaMemcpy(pBests, devPBests, size, cudaMemcpyDeviceToHost);
		cudaMemcpy(pBestAssign, devPBestAssign, assign_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(gBest, devGBest, sizeof(int) * cluster_size * DATA_DIM, cudaMemcpyDeviceToHost);
		cudaMemcpy(gBestAssign, devGBestAssign, assign_size, cudaMemcpyDeviceToHost);

		for(int i = 0; i < particle_size; i++)
		{
			// Get slice of array
			int strideParticle = cluster_size * DATA_DIM;
			int strideAssign = data_size * DATA_DIM;

			int tempAssign[strideAssign], tempPos[strideParticle];

			for(int j = 0; j < strideAssign; j++)
				tempAssign[j] = pBestAssign[i * strideAssign + j]; 

			for(int j = 0; j < strideParticle; j++)
				tempPos[j] = pBests[i * strideParticle + j]; 

			// Compare pBest and gBest
			if (devFitness(tempAssign, flatDatas, tempPos, data_size, cluster_size)
					< devFitness(gBestAssign, flatDatas, gBest, data_size, cluster_size))
			{
				// Update gBest position
				for (int k = 0; k < strideParticle; k++)
					gBest[k] = pBests[i * strideParticle + k];

				// Update gBest assignment matrix
				for(int k = 0; k < data_size * DATA_DIM; k++)
					gBestAssign[k] = pBestAssign[i * strideAssign + k]
			}
		}

		cudaMemcpy(devPBests, pBests, size, cudaMemcpyHostToDevice);
		cudaMemcpy(devPBestAssign, pBestAssign, assign_size, cudaMemcpyHostToDevice);
		cudaMemcpy(devGBest, gBest, sizeof(int) * cluster_size * DATA_DIM, cudaMemcpyHostToDevice);
		cudaMemcpy(devGBestAssign, gBestAssign, assign_size, cudaMemcpyHostToDevice);
	}

	// Copy gBest from device to host
	cudaMemcpy(gBest, devGBest, sizeof(int) * cluster_size * DATA_DIM, cudaMemcpyDe
	cudaMemcpy(gBestAssign, devGBestAssign, assign_size, cudaMemcpyDeviceToHost);
}