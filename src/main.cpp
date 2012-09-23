/*
 * main.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: adi
 */

#include "pso_cluster.h"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    srand(time(NULL));
    IplImage* inputImage = NULL;
    inputImage = cvLoadImage("/home/adi/Pictures/DSC00156.resized.resized.jpg", 
                             -1);

    int width = inputImage->width;
    int height = inputImage->height;
    int channel = inputImage->nChannels;

    char *arrImage = new char[width * height * channel];
    int *flatDatas = new int[width * height * channel];
    Data *datas = new Data[width * height];

    // Load image to array
    for (int i = 0; i < width * height; i++)
    {
        Data d;

        for(int j = 0; j < channel; j++)
        {
            arrImage[i * channel + j] = 
                (unsigned char) inputImage->imageData[i * channel + j];
            flatDatas[i * channel + j] = 
                (unsigned char) inputImage->imageData[i * channel + j];
            d.info[j] = (unsigned char) inputImage->imageData[i * channel + j];
        }

        datas[i] = d;
    }

    // PSO parameters
    int particle_num, cluster_num, max_iter;
    char comp;

    cout << "Number of cluster : ";
    cin >> cluster_num;
    cout << "Number of particle : ";
    cin >> particle_num;
    cout << "Number of iteration : ";
    cin >> max_iter;

    do
    {
        cout << "CPU or GPU (C / G) ? : ";
        cin >> comp;
    }
    while(comp != 'C' && comp != 'G');

    cout << endl;

    clock_t begin = clock();

    GBest gBest;

    // Check if use host code or device code
    if(comp == 'C')
        gBest = hostPsoClustering(datas, width * height, channel, particle_num, 
                                  cluster_num, max_iter);
    else
        gBest = devicePsoClustering(datas, flatDatas, width * height, channel, 
                                    particle_num, cluster_num, max_iter);

    clock_t end = clock();

    cout << "Time elapsed : " << (double)(end - begin) / CLOCKS_PER_SEC << "s" 
         << endl;

    // Compute quantization error of clusters, less is better
    cout << "Error : " << gBest.quantError << endl;

    // List for cluster color
    unsigned char colorList[9][3] = { { 0, 0, 255 }, { 255, 0, 0 }, 
                                      { 0, 255, 0 }, { 255, 255, 0 }, 
                                      { 255, 0, 255 }, { 255, 128, 128 }, 
                                      { 128, 128, 128 }, { 128, 0, 0 }, 
                                      { 255, 128, 0 } };

    channel = 3;
    char *res_image = new char[width * height * channel]; 

    // Coloring clusters
    for (int i = 0; i < width * height; i++)
    {
        for (int j = 0; j < cluster_num; j++)
        {
            if (gBest.gBestAssign[i] == j)
            {
                res_image[i * channel + 0] = colorList[j][0];
                res_image[i * channel + 1] = colorList[j][1];
                res_image[i * channel + 2] = colorList[j][2];
            }
        }
    }

    if(comp == 'C')
        delete[] gBest.centroids;
    else
        cudaFreeHost(gBest.arrCentroids);

    delete[] gBest.gBestAssign;

    // Write array to image
    IplImage* outImage = cvCreateImage(cvSize(width, height), inputImage->depth, 
                                       channel);
    outImage->imageData = res_image;

    cvSaveImage("out1.jpg", outImage);

    cvNamedWindow("Result");
    cvShowImage("Result", outImage);
    cvWaitKey(0);

    // Cleanup
    cvReleaseImage(&inputImage);
    cvReleaseImage(&outImage);

    delete[] res_image;
    delete[] arrImage;

    return 0;
}

