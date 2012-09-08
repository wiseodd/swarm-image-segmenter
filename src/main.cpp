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
	inputImage = cvLoadImage("/home/adi/Pictures/DSC00156.resized.resized.jpg", -1);

	int width = inputImage->width;
	int height = inputImage->height;
	int channel = inputImage->nChannels;

	char* arrImage = new char[width * height * channel];
	data* datas = new data[width * height];

	for (int i = 0; i < width * height; i++)
	{
		arrImage[i * channel + 0] = (unsigned char) inputImage->imageData[i * channel + 0];
		arrImage[i * channel + 1] = (unsigned char) inputImage->imageData[i * channel + 1];
		arrImage[i * channel + 2] = (unsigned char) inputImage->imageData[i * channel + 2];

		data d;

		d.info[0] = (unsigned char) inputImage->imageData[i * channel + 0];
		d.info[1] = (unsigned char) inputImage->imageData[i * channel + 1];
		d.info[2] = (unsigned char) inputImage->imageData[i * channel + 2];

		datas[i] = d;
	}

	cout << sizeof(data) << endl;
	cout << sizeof(arrImage) << endl;

	int particle_num, cluster_num;

	cout << "Number of cluster : ";
	cin >> cluster_num;
	cout << "Number of particle : ";
	cin >> particle_num;

	cout << endl;

	GBest gBest = psoClustering(particle_num, cluster_num, datas, width * height);

	cout << "QuantError : "
			<< quantizationError(gBest.gBestAssign, datas, gBest.centroids, width * height,
					cluster_num) << endl;

	unsigned char colorList[9][3] =
	{
	{ 0, 0, 255 },
	{ 255, 0, 0 },
	{ 0, 255, 0 },
	{ 255, 255, 0 },
	{ 255, 0, 255 },
	{ 255, 128, 128 },
	{ 128, 128, 128 },
	{ 128, 0, 0 },
	{ 255, 128, 0 } };

	for (int i = 0; i < width * height; i++)
	{
		for (int j = 0; j < cluster_num; j++)
		{
			if (gBest.gBestAssign[i] == j)
			{
				arrImage[i * channel + 0] = colorList[j][0];
				arrImage[i * channel + 1] = colorList[j][1];
				arrImage[i * channel + 2] = colorList[j][2];
			}
		}
	}

	delete[] gBest.centroids;
	delete[] gBest.gBestAssign;

	IplImage* outImage = cvCreateImage(cvSize(width, height), inputImage->depth, channel);
	outImage->imageData = arrImage;

	cvSaveImage("out1.jpg", outImage);

	cvNamedWindow("Result");
	cvShowImage("Result", outImage);
	cvWaitKey(0);

	cvReleaseImage(&inputImage);
	cvReleaseImage(&outImage);

	delete[] arrImage;

	return 0;
}

