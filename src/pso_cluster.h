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
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

const int RANGE_MAX = 255;
const int RANGE_MIN = 0;
const double INF = 9999.0;
const int MAX_ITER = 20;
const float OMEGA = 0.72;
const float EPSILON = 0.0005;
const float c1 = 1.49;
const float c2 = 1.49;
const int DATA_DIM = 3;

struct Data
{
    int info[DATA_DIM];
};

struct Particle
{
    Data *position;
    Data *pBest;
    Data *velocity;
};

struct GBest
{
    short *gBestAssign;
    Data *centroids;
    int *arrCentroids;
    float quantError;
};

float getRandom(float low, float high);
float getRandomClamped();
float getDistance(Data first, Data second);
float fitness(const short *assignMat, const Data *datas, const Data *centroids,
              int data_size, int cluster_size);
void assignDataToCentroid(short *assignMat, const Data *datas,
                          const Data *centroids, int data_size,
                          int cluster_size);
void initializePSO(Particle *particles, GBest& gBest, const Data *datas,
                   int data_size, int particle_size, int cluster_size);
GBest hostPsoClustering(Data *datas, int data_size, int channel,
                        int particle_size, int cluster_size, int max_iter);
extern "C" float devFitness(short *assignMat, int *Datas, int *centroids,
                            int data_size, int cluster_size, int channel);
extern "C" GBest devicePsoClustering(Data *datas, int *flatDatas, int data_size,
                                     int channel, int particle_size,
                                     int cluster_size, int max_iter);
#endif /* PSO_CLUSTER_H */
