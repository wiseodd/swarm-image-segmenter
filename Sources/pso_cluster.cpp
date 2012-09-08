#include "pso_cluster.h"

float getRandom(float low, float high)
{
	return low + float(((high - low) + 1) * rand() / ((float) RAND_MAX + 1));
}

// Get random between 0.0f and 1.0f inclusive
float getRandomClamped()
{
	return (float) rand() / (double) RAND_MAX;
}

int round(float x)
{
	if (x > 0.0)
		return (int) floor(x + 0.5);
	else
		return (int) ceil(x - 0.5);
}

// euclidean distance of two datas
float getDistance(data first, data second)
{
	double total = 0.0;

	for (int i = 0; i < DATA_DIM; i++)
	{
		int res = (first.info[i] - second.info[i]);
		total += res * res;
	}

	return sqrt(total);
}

float quantizationError(const short* assignMat, const data* datas, const data* centroids, int data_size,
		int cluster_size)
{
	double total = 0.0;

	for (int i = 0; i < cluster_size; i++)
	{
		double subtotal = 0.0;

		for (int j = 0; j < data_size; j++)
		{
			if (assignMat[j] == i)
				subtotal += getDistance(datas[j], centroids[i]);
		}

		total += subtotal / data_size;
	}

	return total / cluster_size;
}

void assignDataToCentroid(short* assignMat, const data* datas, const data* centroids, int data_size, int cluster_size)
{
	for (int i = 0; i < data_size; i++)
	{
		int nearestCentroidIdx = 0;
		double nearestCentroidDist = INF;

		for (int j = 0; j < cluster_size; j++)
		{
			double nearestDist = getDistance(datas[i], centroids[j]);

			if (nearestDist < nearestCentroidDist)
			{
				nearestCentroidDist = nearestDist;
				nearestCentroidIdx = j;
			}
		}

		assignMat[i] = nearestCentroidIdx;
	}
}

void initializePSO(int particle_size, int cluster_size, particle* particles, GBest& gBest, const data* datas, int size)
{
	for (int i = 0; i < particle_size; i++)
	{
		particle p;

		p.pBest = new data[cluster_size];
		p.position = new data[cluster_size];
		p.velocity = new data[cluster_size];

		particles[i] = p;

		for (int j = 0; j < cluster_size; j++)
		{
			data d;

			for (int k = 0; k < DATA_DIM; k++)
				d.info[k] = 0;

			particles[i].velocity[j] = d;

			int rand = round(getRandom(0, size - 1));

			particles[i].position[j] = datas[rand];
			particles[i].pBest[j] = datas[rand];
		}
	}

	gBest.centroids = new data[cluster_size];

	for (int j = 0; j < cluster_size; j++)
	{
		data d;

		for (int k = 0; k < DATA_DIM; k++)
			d.info[k] = round(abs(getRandom(RANGE_MIN, RANGE_MAX)));

		gBest.centroids[j] = d;
	}
}

GBest psoClustering(int particle_size, int cluster_size, data* datas, int size)
{
	// initialize
	GBest gBest;
	particle* particles = new particle[particle_size];
	short** assignMatrix = new short*[particle_size];
	short** pBestAssign = new short*[particle_size];
	gBest.gBestAssign = new short[size];

	for (int i = 0; i < particle_size; i++)
	{
		assignMatrix[i] = new short[size];
		pBestAssign[i] = new short[size];

		for (int j = 0; j < size; j++)
		{
			assignMatrix[i][j] = 0;
			pBestAssign[i][j] = 0;
		}
	}

	for (int i = 0; i < size; i++)
		gBest.gBestAssign[i] = 0;

	initializePSO(particle_size, cluster_size, particles, gBest, datas, size);

	// PSO main function
	for (int i = 0; i < MAX_ITER; i++)
	{
		cout << "Iteration-" << i + 1 << endl;

		float rp = getRandomClamped();
		float rg = getRandomClamped();

		// foreach particle
		for (int j = 0; j < particle_size; j++)
		{
			// foreach dimension
			for (int k = 0; k < cluster_size; k++)
			{
				// foreach data dimension
				for (int l = 0; l < DATA_DIM; l++)
				{
					particles[j].velocity[k].info[l] = round(
							OMEGA * particles[j].velocity[k].info[l]
									+ c1 * rp * (particles[j].pBest[k].info[l] - particles[j].position[k].info[l])
									+ c2 * rg * (gBest.centroids[k].info[l] - particles[j].position[k].info[l]));

					particles[j].position[k].info[l] += particles[j].velocity[k].info[l];
				}
			}

			assignDataToCentroid(assignMatrix[j], datas, particles[j].position, size, cluster_size);
		}

		for (int j = 0; j < particle_size; j++)
		{
			if (quantizationError(assignMatrix[j], datas, particles[j].position, size, cluster_size)
					< quantizationError(assignMatrix[j], datas, particles[j].pBest, size, cluster_size))
			{
				for (int k = 0; k < cluster_size; k++)
					particles[j].pBest[k] = particles[j].position[k];

				assignDataToCentroid(pBestAssign[j], datas, particles[j].pBest, size, cluster_size);

				if (quantizationError(pBestAssign[j], datas, particles[j].pBest, size, cluster_size)
						< quantizationError(gBest.gBestAssign, datas, gBest.centroids, size, cluster_size))
				{
					for (int k = 0; k < cluster_size; k++)
						gBest.centroids[k] = particles[j].pBest[k];

					assignDataToCentroid(gBest.gBestAssign, datas, gBest.centroids, size, cluster_size);
				}
			}
		}
	}

	// cleanup
	for (int i = 0; i < particle_size; i++)
	{
		delete[] assignMatrix[i];
		delete[] pBestAssign[i];
		delete[] particles[i].pBest;
		delete[] particles[i].position;
		delete[] particles[i].velocity;
	}

	delete particles;
	delete[] assignMatrix;
	delete[] pBestAssign;

	return gBest;
}
