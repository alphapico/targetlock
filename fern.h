/*
 *  fern.h
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FERNH_
#define FERNH_

#include <vector>
#include <opencv/cv.h>
#include <iostream>
//For stream data
#include <sstream>
#include <fstream>
#include <iomanip>


using namespace cv;

typedef std::pair<cv::Mat_<unsigned char>* /*img*/, cv::Mat_<unsigned char>* /*blur*/ >ImagePairType;

class Fern
{
	
public:
	Fern();
	~Fern();
	void reset();
	bool init(const IplImage& image, const Mat_<double>& grid, const Mat_<double>& features, const Mat_<double>& scales);
	void getPatterns(const ImagePairType& input, const cv::Mat_<unsigned>& idx, double var, cv::Mat_<double>& patt, cv::Mat_<double>& status);
	void update(const cv::Mat_<double>& x, const cv::Mat_<double>& y, double thr_fern, int bootstrap, const cv::Mat_<double>* idx = NULL);
	void evaluate(const cv::Mat_<double>& X, cv::Mat_<double>& resp0);
	void detect(const ImagePairType& img, double maxBBox, double minVar, const cv::Mat_<double>& conf, const cv::Mat_<double>& patt);
	//test correcness
	void getPatterns(const ImagePairType& input, const cv::Mat_<double>& grid, const cv::Mat_<unsigned>& idx, double var, cv::Mat_<double>& patt, cv::Mat_<double>& status);
	void printWeight();
	//
	
protected:
	int* create_offsets(double *scale0, double *x0);
	int* create_offsets_bbox(double *bb0);
	void iimg(unsigned char *in, double *ii, int imH, int imW);
	void iimg2(unsigned char *in, double *ii2, int imH, int imW);
	double bbox_var_offset(double *ii,double *ii2, int *off);
	int measure_tree_offset(unsigned char *img, int idx_bbox, int idx_tree);
	//test correctness
	int measure_tree_offset(const Mat& image, const cv::Mat_<double>& grid, int idx_bbox, int idx_tree);
	//
	double measure_bbox_offset(unsigned char *blur, int idx_bbox, double minVar, double *tPatt);
	int row2col(int ci);
	void update(double *x, int C, int N, int offset);
	double measure_forest(double *idx, int offset);
	double randdouble();

private:
	double thrN;
	int nBBOX;
	int mBBOX;
	int nTREES;
	int nFEAT;
	int nSCALE;
	int iHEIGHT;
	int iWIDTH;
	int *BBOX;
	int *OFF;
	double *IIMG;
	double *IIMG2;
	std::vector<std::vector <double> > WEIGHT;
	std::vector<std::vector <int> > nP;
	std::vector<std::vector <int> > nN;
	int BBOX_STEP;
	int nBIT; // number of bits per feature
	int m_inc; // just to globally incerment to debug fern->getPattern
};

inline int Fern::row2col(int ci)
{
	int ri = floor(((float) ci )/ iHEIGHT)+ ((ci % iHEIGHT) * iWIDTH);
	return ri;
}

#endif /* FERNH_ */