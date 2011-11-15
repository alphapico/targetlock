/*
 *  targetLock.h
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/8/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TARGETLOCK_H_
#define TARGETLOCK_H_


//From OpenCV library
#include <opencv2/opencv.hpp>

//From STL
#include <map>
#include <list>
#include <vector>
#include <algorithm>
#include <iterator>

//From SL
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <numeric>

#include "qrectf.h"
#include "fern.h"
#include "LK.h"

class Lk;
class Fern;


typedef std::pair<cv::Mat_<unsigned char>* /*img*/, cv::Mat_<unsigned char>* /*blur*/ > ImagePairType;
// rowwise access used in warp
/*#define coord(x, y, width, height) (y+x*img_step)
#define nextrow(tmp, width, height) ((tmp)+img_step)
#define nextcol(tmp, width, height) ((tmp)+1)
#define nextr_c(tmp, width, height) ((tmp)+img_step+1)*/
#define M(r, c) H[c+r*3]

class TargetLock {
public:
	TargetLock();
	~TargetLock();
	//void run();
	bool init(IplImage& image, const QRectF& region);
	bool initialised(void) { return m_initialised; };
	void display(int runtime, int frameNumber = 0);
	//void processFrame(IplImage* frame);
	void processFrames(IplImage& image);
	void learning(int I);
	bool myIsNaN(double var); //declare own isnan
	
	int m_key; //check key press
	
	//for experiment purposed (sydney)
	cv::Mat_<float> m_result;
	cv::Mat_<float> m_resultConfidence;
	
	//should be at protected, but because i want to print the result in main..
	Lk* m_lk;
	Fern* m_fern;

	
protected:
	typedef enum {FORREST=0} FeatureType;
	
	typedef struct {
		unsigned num_closest;
		unsigned num_warps;
		unsigned noise;
		unsigned angle;
		double shift;
		double scale;
	} ParOptionsType;
	
	typedef struct {
		double overlap;
		unsigned num_patches;
	} NParOptionsType;
	
	typedef struct {
		cv::Mat_<double> bb;
		cv::Mat_<double> idx;
		cv::Mat_<double> conf1;
		cv::Mat_<double> conf2;
		cv::Mat_<double> isin;
		cv::Mat_<double> patt;
		cv::Mat_<double> patch;
	} DtType;

//Mostly used in tldInit
	void reset();
	IplImage* blur(IplImage& image, double sigma);
	void blur(IplImage& image, cv::Mat_<unsigned char>& outMat, double sigma = 1.5);
	bool bb_scan(const cv::Mat_<double>& bb, const CvSize& imsize, double min_win, cv::Mat_<double>& outGrid, cv::Mat_<double>& outScales);
	double bbHeight(const cv::Mat_<double>& bb);
	double bbWidth(const cv::Mat_<double>& bb);
	void bbSize(const cv::Mat_<double>& bb, cv::Mat_<double>& s);
	void bbCenter(const cv::Mat_<double>& bb, cv::Mat_<double>& center);
	void bbHull(const cv::Mat_<double>& bb, cv::Mat_<double>& hull);
	void generateFeatures(unsigned nTrees, unsigned nFeatures, cv::Mat_<double>& features, bool show = false);
	double bb_overlap(double *bb1, double *bb2);
	void bb_overlap(const cv::Mat_<double>& bb, cv::Mat_<double>& overlap);
	void bb_overlap(const cv::Mat_<double>& bb1, const cv::Mat_<double>& bb2, cv::Mat_<double>& overlap);
	void imagePatch(const cv::Mat_<unsigned char>& inMat, const cv::Mat_<double>& bb, cv::Mat_<unsigned char>& outMat, const ParOptionsType& init, double randomise = -1);
	void warp(const cv::Mat_<unsigned char>& im, const cv::Mat_<double>& H, const cv::Mat_<double>& B, cv::Mat_<double>& outMat);
	void tldGeneratePositiveData(const cv::Mat_<double>& overlap, const ImagePairType& img, const ParOptionsType& init, cv::Mat_<double>& pX, cv::Mat_<double>& pEx, cv::Mat_<double>& bbP0);
	void tldGenerateNegativeData(const cv::Mat_<double>& overlap, const ImagePairType& img, cv::Mat_<double>& nX, cv::Mat_<double>& nEx);
	void tldSplitNegativeData(const cv::Mat_<double>& nX, const cv::Mat_<double>& nEx, cv::Mat_<double>& nX1, cv::Mat_<double>& nX2, cv::Mat_<double>& nEx1, cv::Mat_<double>& nEx2);
	void randValues(const cv::Mat_<double>& in, double k, cv::Mat_<double>& out);
	void getPattern(const ImagePairType& img, const cv::Mat_<double>& bb, const unsigned patchSize, cv::Mat_<double>& pattern, bool flip= false);
	void patch2Pattern(const cv::Mat_<unsigned char>& patch, const unsigned patchSize, cv::Mat_<double>& pattern);
	template<class T> bool randperm(int n, cv::Mat_<T>& outMat);
	void var(const cv::Mat_<double>& inMat, cv::Mat_<double>& outMat);
	void tldTrainNN(const cv::Mat_<double>& pEx, const cv::Mat_<double>& nEx);
	void tldNN(const cv::Mat_<double>& x, cv::Mat_<double>& conf1, cv::Mat_<double>& conf2, cv::Mat_<double>& isin);
	void distance(const cv::Mat_<double>& x1, const cv::Mat_<double>& x2, int flag, cv::Mat_<double>& resp);
//tldProcessFrame	
	void tracking(const cv::Mat_<double>& bb, int I, int J, cv::Mat_<double>& BB2, double& tConf, double& tValid);
	void bbPoints(const cv::Mat_<double>& bb, double numM, double numN, double margin, cv::Mat_<double>& pt);
	template<class T> bool bbIsDef(const cv::Mat_<T>& bb);
	template<class T> void isFinite(const cv::Mat_<T>& bb, cv::Mat_<int>& outMat);
	template<class T> bool bbIsOut(const cv::Mat_<T>& bb, const cv::Mat_<T>& imsize);
	template<class T> double median(std::vector<T> v);
	template<class T> double median2(std::vector<T> v);
	void bbPredict(const cv::Mat_<double>& BB0, const cv::Mat_<double>& pt0, const cv::Mat_<double>& pt1, cv::Mat_<double>& BB2, cv::Mat_<double>& shift);
	void pdist(const cv::Mat_<double>& inMat, cv::Mat_<double>& outMat);
	void detection(int I, cv::Mat_<double>& BB,  cv::Mat_<double>& Conf);
	void bb_cluster_confidence(const cv::Mat_<double>& iBB, const cv::Mat_<double> iConf, cv::Mat_<double>& oBB, cv::Mat_<double>& oConf, cv::Mat_<double>& oSize);
	double bb_distance(const cv::Mat_<double>& bb1, const cv::Mat_<double>* bb2 = NULL);
//for bb_clustering
	
	
	
private:
	double shift;
	double min_bb;
	const unsigned m_patchSize;
	QRectF m_bbRect;
	cv::Mat_<double> m_source_bb;
	double m_min_win;
	double m_model_thr_fern;
	double m_model_thr_nn;
	double m_num_trees;
	unsigned m_num_init;
	double m_model_valid;
	double m_model_thr_nn_valid;
	double m_ncc_thesame;
	unsigned m_model_patchsize;
	bool m_plot_pex;
	bool m_plot_nex;
	bool m_plot_dt;
	bool m_plot_confidence;
	bool m_plot_target;
	int m_plot_drawoutput;
	bool m_plot_draw;
	bool m_plot_pts;
	int m_plot_patch_rescale;
	
	int m_control_maxbbox;
	bool m_control_update_detector;
	bool m_control_drop_img;
	
	IplImage* m_image;
	cv::Mat_<double> m_scale;
	cv::Mat_<double> m_grid;
	cv::Mat_<double> m_scales;
	int m_nGrid;
	FeatureType m_featureType;
	cv::Mat_<double> m_features;
	unsigned m_nTrees;
	unsigned m_nFeatures;
	
	cv::Mat_<double> m_tmpConf;
	cv::Mat_<double> m_tmpPatt;
	
	std::vector<ImagePairType> m_img;
	std::list<cv::Mat_<double>*> m_snapshot;
	std::map<int, DtType> m_dt;
	
	cv::Mat_<double> m_bb;
	cv::Mat_<double> m_conf;
	cv::Mat_<double> m_valid;
	cv::Mat_<double> m_size;
	std::list<int *> m_trackerFailure;
	std::vector<cv::Mat_<double> > m_draw;
	cv::Mat_<double> m_pts;
	
	cv::Mat_<double> m_imgSize; // originally unsigned int
	std::vector<cv::Mat_<double> > m_X;
	std::vector<cv::Mat_<double> > m_Y;
	std::vector<cv::Mat_<double> > m_pEx;
	std::vector<cv::Mat_<double> > m_nEx;
	double m_var;
	
	cv::Mat_<double> m_overlap;
	cv::Mat_<unsigned char> m_target;
	ParOptionsType m_p_par_init;
	ParOptionsType m_p_par_update;
	NParOptionsType m_n_par;
	bool m_fliplr;
	
	cv::Mat_<double> m_pex;
	cv::Mat_<double> m_nex;
	int m_pexCol;
	int m_nexCol;
	
	bool m_initialised;
	int m_frameNumber;
	
	cv::Mat_<double> m_xFJ;
	cv::Mat_<float> m_gaussMat;
	
	cv::RNG* m_rng;
	uint64 m_rngState;
	
	cv::Mat m_imgDisplay; //for color display purposed
	int m_noFrameForWarp;
	
	
	
	
};

#endif //TARGETLOCK_H_