/*
 * pso_header.hpp
 *
 *  Created on: Jun 7, 2012
 *      Author: adi
 */

#ifndef PSO_CLUSTER_H
#define PSO_CLUSTER_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>

using namespace std;

const int RANGE_MAX = 255;
const int RANGE_MIN = 0;
const double INF = 9999.0;
const int MAX_ITER = 50;
const float OMEGA = 0.72;
const float EPSILON = 0.0005;
const float c1 = 1.49;
const float c2 = 1.49;
const int DATA_DIM = 3;

struct data
{
	int info[DATA_DIM];
};

struct particle
{
	data* position;
	data* pBest;
	data* velocity;
};

struct GBest
{
	short* gBestAssign;
	data* centroids;
};

float getRandom(float low, float high);
float getRandomClamped();
float getDistance(data first, data second);
float fitness(const short* assignMat, const data* datas, const data* centroids, 
	int data_size, int centroid_size);
void assignDataToCentroid(short* assignMat, const data* datas, const data* centroids, 
	int data_size, int centroid_size);
void initializePSO(int numOfParticle, int numOfCluster, particle* particles, GBest& gBest, 
	const data* datas);
GBest psoClustering(int numOfParticle, int numOfCluster, data* datas, int size);

#endif /* PSO_CLUSTER_H */
