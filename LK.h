/*
 *  LK.h
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/23/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef LK_H_
#define LK_H_

//From OpenCV library
#include <opencv2/opencv.hpp>


class Lk
{
	
public:
	Lk();
	~Lk();
	void init();
	void track(cv::Mat& imgI, cv::Mat& imgJ, const cv::Mat_<double>& ptsIMat, const cv::Mat_<double>& ptsJMat, cv::Mat_<double>& outMat, int level = 5);
	

	
protected:
	void euclideanDistance(CvPoint2D32f *point1, CvPoint2D32f *point2, float *match, int nPts);
	void normCrossCorrelation(IplImage *imgI, IplImage *imgJ, CvPoint2D32f *points0, CvPoint2D32f *points1, int nPts, char *status, float *match,int winsize, int method);
};

#endif /* LK_H_ */