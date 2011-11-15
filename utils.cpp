/*
 *  utils.cpp
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/8/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "utils.hpp"

void loadImage(const cv::Mat& mat, IplImage *image)
{
	int widthStep = image->widthStep;
	int N = image->width; // width
	int M = image->height; // height
	
	if (N == 0 || M == 0)
	{
		printf("Input image error\n");
		return;
	}
	
	for (int i=0; i<M; i++)
		for (int j=0; j<N; j++) 
			(image->imageData + i*widthStep)[j] = mat.at<unsigned char>(i,j);
	
}