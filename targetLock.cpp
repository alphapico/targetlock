/*
 *  targetLock.cpp
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/8/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "targetLock.h"
#include "utils.hpp"

const int MAX_FRAMES = 10000;

// Options
#define USE_CVSMOOTH_BLUR

template <class T>
bool pairValueSortPredicate(T i, T j) {return (i.second > j.second);}

TargetLock::TargetLock()
: shift(0.1)
, min_bb(24.0) //Matlab 24.0, resize factor 2, so take 12.0
, m_patchSize(15)
, m_min_win(24) //change 24 here as well if you want to change to 12
, m_model_thr_fern(0.5) //0.5 in earlier experiment
, m_model_thr_nn(0.65) //0.65 earlier
, m_num_trees(10)
, m_ncc_thesame(0.95)
, m_model_patchsize(m_patchSize)
, m_plot_pex(true)
, m_plot_nex(true)
, m_plot_dt(true)
, m_plot_confidence(true)
, m_plot_target(true)
, m_plot_drawoutput(3)
, m_plot_draw(true)
, m_plot_pts(true)
, m_control_maxbbox(1)
, m_control_update_detector(true)
, m_control_drop_img(true)
, m_image(NULL)
, m_model_valid(0.5)
, m_model_thr_nn_valid(0.7)
, m_plot_patch_rescale(1)
, m_nGrid(0)
, m_nTrees(10)
, m_nFeatures(13)
, m_fliplr(false)
, m_var(0)
, m_pexCol(0)
, m_nexCol(0)
, m_initialised(false)
, m_frameNumber(0)
, m_noFrameForWarp(0)
{
	//======for experiment purposed======//
	/*
	m_result.create(2000,5);
	m_resultConfidence.create(1400,2);
	 */
	//===================================//

	
	m_rng = new cv::RNG(0);
	m_rngState = m_rng->state;
	
	m_source_bb.create(4, 1);
	m_scale.create(1, 21); //1.2^[-10,10]
	//m_scale.create(1, 11); //Test from Matlab for scale = 1.2^[-5,5]
	//m_scale.create(1, 9); //1.2^[-4,4]
	//m_scale.create(1, 7); //1.2^[-3,3]
	//m_scale.create(1, 5); //1.2^[-2,2]
	//m_scale.create(1, 3); //1.2^[-1,1]
	
	//use in bb_scan
	for(int i=0; i < m_scale.cols; i++)
		m_scale.at<double>(0,i) = pow(1.2, i-10); //decrease speed i-10
	
	m_p_par_init.num_closest = 10;
	m_p_par_init.num_warps = 20;
	m_p_par_init.noise = 5;
	m_p_par_init.angle = 20;
	m_p_par_init.shift = 0.02;
	m_p_par_init.scale = 0.02;
	
	m_p_par_update.num_closest = 10;
	m_p_par_update.num_warps = 10;
	m_p_par_update.noise = 5;
	m_p_par_update.angle = 10;
	m_p_par_update.shift = 0.02;
	m_p_par_update.scale = 0.02;
	
	m_n_par.overlap = 0.2;
	m_n_par.num_patches = 100;
	
	m_fern = new Fern();
	m_lk = new Lk();
}


TargetLock::~TargetLock()
{
	
	reset();
	delete m_lk;
	delete m_fern;
}

void TargetLock::reset()
{

	m_rng->state = m_rngState;
	m_gaussMat.create(0,0);
	//m_source_bb.create(0,0);
	//m_scale.create(0,0);
	m_grid.create(0,0);
	m_scales.create(0,0);
	m_features.create(0,0);
	m_tmpConf.create(0,0);
	m_tmpPatt.create(0,0);
	//m_bb.create(0,0);
	m_draw.clear();
	m_pts.create(0,0);
	m_imgSize.create(0,0);
	m_overlap.create(0,0);
	m_target.create(0,0);
	m_pex.create(0,0);
	m_nex.create(0,0);
	m_xFJ.create(0,0);
	m_bb.create(0,0);
	m_conf.create(0,0);
	m_valid.create(0,0);
	m_size.create(0,0);
	
	if (m_image) cvReleaseImage(&m_image);
	
	//clear pointer if exist
	std::vector<ImagePairType>::iterator iptlit;
	for (iptlit = m_img.begin(); iptlit != m_img.end(); ++iptlit)
	{
		ImagePairType& ipt = *iptlit;
		delete ipt.first;
		delete ipt.second;
	}
	m_img.clear();
	
	std::list<cv::Mat_<double>*>::iterator dmlit;
	for (dmlit = m_snapshot.begin(); dmlit != m_snapshot.end(); ++dmlit)
	{
		cv::Mat_<double>* dm = *dmlit;
		delete dm;
	}
	m_snapshot.clear();
	
	m_dt.clear();
	
	//std::vector<DtType>::iterator dtvit;
	//for (dtvit = m_dt.begin(); dtvit != m_dt.end(); ++dtvit)
	//{
	//	DtType& dm = *dtvit;
	//	delete dm;
	//}
	m_trackerFailure.clear();
	
	m_X.clear();
	m_Y.clear();
	m_pEx.clear();
	m_nEx.clear();
	
	m_fern->reset();
}


bool TargetLock::init(IplImage& image, const QRectF& region)
{
	
	

	reset();
	
	QRectF regionRect(0, 0, image.width, image.height);
	if (!region.intersects(regionRect))
	{
		std::cout << "Region selected is outside frame bounds\n";
		return false;
	}
	
	
	m_bbRect = region.intersect(regionRect);
	if ((m_bbRect.width() < min_bb) || (m_bbRect.height() < min_bb))
	{
		std::cout << "Region selected is too small\n";
		return false;
	}
	
	
	m_bb.create(4, MAX_FRAMES);
	m_conf.create(1, MAX_FRAMES);
	m_valid.create(1, MAX_FRAMES);
	m_size.create(1, MAX_FRAMES);
	
	for (int i = 0; i< MAX_FRAMES; i++) {
		m_bb(0,i) = NaN;
		m_bb(1,i) = NaN;
		m_bb(2,i) = NaN;
		m_bb(3,i) = NaN;
		m_conf(0,i) = NaN;
		m_valid(0,i) = NaN;
		m_size(0,i) = NaN;
	}
	
	
	// Initialise LK tracker
	m_lk->init();
	
	m_source_bb(0,0) = m_bbRect.x();//left
	m_source_bb(1,0) = m_bbRect.y();//top
	m_source_bb(2,0) = m_bbRect.x() + m_bbRect.width();//right
	m_source_bb(3,0) = m_bbRect.y() + m_bbRect.height();//bottom
	
	cv::Mat mono(&image);
	if (image.nChannels != 1) {
		cv::cvtColor(mono, mono, CV_BGR2GRAY);
	}
	
	//debug
	//cv::imshow("Debug", mono);
	
	IplImage monoImage = mono;
	CvSize imageSize = cvSize(monoImage.width, monoImage.height);
	m_image = cvCreateImage(imageSize, monoImage.depth, monoImage.nChannels);
	cvCopy(&monoImage, m_image);


	cv::Mat_<unsigned char> imageMat;
	cv::Mat imageMatTemp = cvarrToMat(m_image);
	imageMatTemp.convertTo(imageMat, CV_8UC1); //ori CV_8U
	
	
	
	cv::Mat_<unsigned char> blurMat;
	blur(*m_image, blurMat, 7); //only 2? ..seems like ozuysal use 7x7 block for blur
	
	//cv::namedWindow("blur image", 1);
	//cv::imshow("blur image", blurMat);
	
	if (!bb_scan(m_source_bb, imageSize, m_min_win, m_grid, m_scales))
		return false;
	
	//Features//
	m_nGrid = m_grid.cols;
	generateFeatures(m_nTrees, m_nFeatures, m_features);
	//printMatrix(m_features, "m_features");
	
	//initialise detector//
	if (!m_fern->init(*m_image, m_grid, m_features, m_scales))
		return false;
	
	//Temporal Structure//
	m_tmpConf = cv::Mat_<double>::zeros(1, m_nGrid);
	m_tmpPatt = cv::Mat_<double>::zeros(m_nTrees, m_nGrid);
	
	//=================RESULTS==================//
	m_pts = cv::Mat_<double>::zeros(2, 0);
	ImagePairType img;
	
	img.first = new cv::Mat_<unsigned char>(imageMat);
	img.second = new cv::Mat_<unsigned char>(blurMat);
	m_img.push_back(img);
	//int size = m_img.size();
	
	m_bb(0,0) = m_bbRect.x(); //Left
	m_bb(1,0) = m_bbRect.y(); //Top
	m_bb(2,0) = m_bbRect.x() + m_bbRect.width(); //Right
	m_bb(3,0) = m_bbRect.y() + m_bbRect.height(); //Bottom
	
	m_conf(0,0) = 1;
	m_valid(0,0) = 1;
	m_size(0,0) = 1;
	
	//=============TRAIN DETECTOR===============//
	//Initialize structures//
	m_imgSize.create(1, 2);
	m_imgSize(0,0) = (double)m_image->height;
	m_imgSize(0,1) = (double)m_image->width;
	
	bb_overlap(m_source_bb, m_grid, m_overlap);
	
	
	//Target (display only)//
	imagePatch(imageMat, m_source_bb, m_target, m_p_par_init);
	
	
	// Generate Positive Examples
	cv::Mat_<double> pX;
	cv::Mat_<double> pEx;
	cv::Mat_<double> bbP;
	tldGeneratePositiveData(m_overlap, m_img.at(0), m_p_par_init, pX, pEx, bbP);

	
	cv::Mat_<double> pY(1, pX.cols);
	pY = cv::Mat_<double>::ones(1, pX.cols);
	
	Mat col = m_bb.col(0);
	bbP.col(0).copyTo(col);
	
	cv::Mat_<double> varMat;
	var(pEx.col(0), varMat);
	m_var = varMat(0,0)/2;
	
	std::cout << "variance: " << m_var << std::endl;
	
	//Generate Negative Examples

	cv::Mat_<double> nX;
	cv::Mat_<double> nEx;
	tldGenerateNegativeData(m_overlap, m_img.at(0), nX, nEx);

	
	cv::Mat_<double> nX1;
	cv::Mat_<double> nX2;
	cv::Mat_<double> nEx1;
	cv::Mat_<double> nEx2;
	
	
	cv::Mat_<double> nY1 = cv::Mat_<double>::zeros(1, nX1.cols);
	
	//save positive patches for later
	m_pEx.push_back(pEx);
	
	//save negative patches for later
	m_nEx.push_back(nEx);
	
	cv::Mat_<double> X(pX.rows, pX.cols + nX1.cols);
	for (int c=0; c < pX.cols; ++c)
	{
		Mat col = X.col(c);
		pX.col(c).copyTo(col);
	}
	for (int c=0; c < nX1.cols; ++c)
	{
		Mat col = X.col(c+pX.cols);
		nX1.col(c).copyTo(col);
	}
	m_X.push_back(X);

	
	cv::Mat_<double> Y(pY.rows, pY.cols + nY1.cols);
	for (int c=0; c < pY.cols; ++c)
	{
		Mat col = Y.col(c);
		pY.col(c).copyTo(col);
	}
	for (int c=0; c < nY1.cols; ++c)
	{
		Mat col = Y.col(c+pY.cols);
		nY1.col(c).copyTo(col);
	}
	m_Y.push_back(Y);

	//printMatrix(Y, "Y");

	cv::Mat_<int> idx; //double
	randperm<int>((int) m_X.at(0).cols, idx); //double

	
	cv::Mat_<double> tmp(X.rows, idx.cols);
	for (int c=0; c < idx.cols; ++c)
	{
		Mat col = tmp.col(c);
		X.col(idx(0, c)).copyTo(col);
	}
	tmp.copyTo(X);
	

	tmp.create(Y.rows, idx.cols);
	for (int c=0; c < idx.cols; ++c)
	{
		Mat col = tmp.col(c);
		Y.col(idx(0, c)).copyTo(col);
	}
	tmp.copyTo(Y);
	
	
	//printMatrix(X, "X");
	//printMatrix(Y, "Y");

	
	//=============Train using training set=============//
	
	//Fern
	int bootstrap = 2; //earlier experiment boostrap = 2
	m_fern->update(X, Y, m_model_thr_fern, bootstrap);
	
	//Nearest Neightbour
	tldTrainNN(pEx, nEx1);

	m_num_init = m_pex.cols;
	
	//========Estimate thresholds on validation set==========//
	
	//Fern
	cv::Mat_<double> conf_fern;
	m_fern->evaluate(nX2, conf_fern);

	cv::Mat_<double>::iterator pos = std::max_element(conf_fern.begin(), conf_fern.end());
	double conf_fern_max = *pos;
	m_model_thr_fern = max(conf_fern_max/m_num_trees, m_model_thr_fern);
	
	//Nearest neighbor
	cv::Mat_<double> conf_nn;
	cv::Mat_<double> dummy;
	cv::Mat_<double> isin;
	
	tldNN(nEx2, conf_nn, dummy, isin); //isin -> nan 

	
	int index;
	double max_conf_nn;
	maxMat<double>(conf_nn, index, max_conf_nn);
	m_model_thr_nn = max(m_model_thr_nn, max_conf_nn); 
	
	m_model_thr_nn_valid = max(m_model_thr_nn_valid, m_model_thr_nn);
	m_initialised = true;
	
	
	//cv::namedWindow("warp image", 1);
	//cv::imshow("warp image", m_target);
	
	return true;
	
}


void TargetLock::processFrames(IplImage& image)
{

	
		int I = ++m_frameNumber;
		
		cv::Mat mono(&image);
		m_imgDisplay = mono; //<--store original here for color display later
		
		if (image.nChannels != 1) {
			cv::cvtColor(mono, mono, CV_BGR2GRAY);
			
		}
	
		//debug
		//cv::imshow("Debug", mono);
	
		
		IplImage monoImage = mono;
		CvSize imageSize = cvSize(monoImage.width, monoImage.height);
		cvReleaseImage(&m_image);
		m_image = cvCreateImage(imageSize, monoImage.depth, monoImage.nChannels);
		cvCopy(&monoImage, m_image);

		
		cv::Mat_<unsigned char> imageMat;
		cv::Mat imageMatTemp = cvarrToMat(m_image);
		imageMatTemp.convertTo(imageMat, CV_8UC1); //ori CV_8U
		
		cv::Mat_<unsigned char> blurMat;
		blur(*m_image, blurMat, 7);  //only 2? ..seems like ozuysal use 7x7 block for blur
		ImagePairType img;
		img.first = new cv::Mat_<unsigned char>(imageMat);
		img.second = new cv::Mat_<unsigned char>(blurMat);
		m_img.push_back(img);
	
		//debug
		//cv::imshow("Debug2", m_imgDisplay);
		
		//=============TRACKER=============//
		
		cv::Mat_<double> tBB;
		double tConf;
		double tValid = 0;
		tracking(m_bb.col(I-1), I-1, I, tBB, tConf, tValid);
		
		//============DETECTION=============//
		
		cv::Mat_<double> dBB;
		cv::Mat_<double> dConf;
		detection(I, dBB, dConf);
		
		//============INTEGRATOR=============//
		
		bool DT = !dBB.empty();
		bool TR = !tBB.empty();

	
		cout << "TR: " << TR << endl;
		cout << "DT: " << DT << endl;
		cout << "tConf: " << tConf << endl;
	
		
		if (TR)
		{
			Mat col = m_bb.col(I);
			tBB.col(0).copyTo(col);

			m_conf(0,I) = tConf;
			m_size(0,I) = 1;
			m_valid(0,I) = tValid;


			if (DT)
			{ 
				cv::Mat_<double> cBB;
				cv::Mat_<double> cConf;
				cv::Mat_<double> cSize;
				bb_cluster_confidence(dBB, dConf, cBB, cConf, cSize);
				cv::Mat_<double> overlap;
				
				bb_overlap(m_bb.col(I), cBB, overlap);
			
				cv::Mat_<int> id(1, overlap.cols); //put 1 intead of overlap.rows
				 //overlap.rows always == 1
				
				for (int c = 0; c <overlap.cols; c++) 
				{
					if ((overlap(0,c) < 0.5) && ( cConf(0,c) > m_conf(0,I) )) 
					{
						id(0,c) = 1;
					}else {
						id(0,c) = 0;
					}
				}
					
				
				
		
				int sum = cv::sum(id)(0);
				
				if (sum == 1) //only 1 result
				{
					for (int r=0; r < id.cols; ++r) //ori id.rows
					{
						
						if (id(0,r)) // intead of id(r,0)
						{
							Mat col = m_bb.col(I);
							cBB.col(r).copyTo(col);
							col = m_conf.col(I);
							cConf.col(r).copyTo(col);
							col = m_size.col(I);
							cSize.col(r).copyTo(col);
						}
						m_valid(0,I) = 0;
					}
				} //if sum == 1;
				else // otherwise adjust the tracker's trajectory
				{        
					// get indexes of close detections
					cv::Mat_<double> overlap;
					bb_overlap(tBB, m_dt[I].bb, overlap);
			
					cv::Mat_<int> idTr(1, overlap.cols); //overlap.rows always == 1
					for (int c=0; c < overlap.cols; ++c)
					{
						if (overlap(0,c) > 0.7) 
						{
							idTr(0,c) = 1;
						}else{
							idTr(0,c) = 0;
						}
					}
	
					//weighted average trackers trajectory with the close detections
					cv::Mat_<double> rtBB;
					repmat(tBB, 1, rtBB, 10);
	
					cv::Mat_<double> bbC(m_dt[I].bb.rows, cv::sum(idTr)(0));
					int idx = 0;
					for (int c=0; c < idTr.cols; ++c)
					{
						if (idTr(0,c))
						{
							Mat col = bbC.col(idx++);
							m_dt[I].bb.col(c).copyTo(col);
						}
					}
			
					cv::Mat_<double> aBB(rtBB.rows, rtBB.cols + bbC.cols);
					for (int c=0; c < rtBB.cols; ++c)
					{
						Mat col = aBB.col(c);
						rtBB.col(c).copyTo(col);
					}
					for (int c=0; c < bbC.cols; ++c)
					{
						Mat col = aBB.col(rtBB.cols+c);
						bbC.col(c).copyTo(col);
					}
		
					for (int r=0; r < aBB.rows; ++r) //aBB.rows always == 4
						m_bb(r, I) = cv::mean(aBB.row(r))(0);

				} //end of else
				
			}
			
		}
		else 
		{
			// if DT  and detector is defined
			if (DT)
			{
				//cluster detections
				cv::Mat_<double> cBB;
				cv::Mat_<double> cConf;
				cv::Mat_<double> cSize;
				bb_cluster_confidence(dBB, dConf, cBB, cConf, cSize);
				//and if there is just a single cluster, re-initalize the tracker
				if (cConf.cols == 1)
				{

					Mat col = m_bb.col(I);
					cBB.copyTo(col);
					m_conf(0, I) = cConf(0,0);
					m_size(0,I) = cSize(0,0);
					m_valid(0,I) = 0;
					
					cout << "cConf: " << cConf << endl;
					
				}
			}
		}
	
	
	
		
		
		//=====================LEARNING=======================//
		//cout << "mkey: "<< m_key << endl;
	
		if (m_key == 's') {
			
			cout << "Stop Learning\n";
		}
		else 
		{
			if (m_control_update_detector && (m_valid(0, I) == 1))
			{
				cout << "Learning...\n";
				learning(I);
			}
		}

		
		
		
		if (!isnan(m_bb(0,I))) //<-- should change this?
		{
	
			cv::Mat_<double> center;
			bbCenter(m_bb.col(I), center);
			m_draw.push_back(center);
	
			if (m_plot_draw)
			{
				cv::Mat_<double> center(2,1);
				center(0,0) = NaN;
				center(1,0) = NaN;
				m_draw.push_back(center);
			}
		}
		else
		{
			m_draw.clear();
		}
		
		//forget previous image
		if (m_control_drop_img && (I > 2))
		{
			delete m_img[I-1].first; m_img[I-1].first = NULL;
			delete m_img[I-1].second; m_img[I-1].second = NULL;
			
		}
		
		//display results on frame i
		display(1, I);

		

}


void TargetLock::learning(int I)
{

	cv::Mat_<double> bb(m_bb.col(I));
	
	//current image
	const ImagePairType& img = m_img.at(I);
	
	//=================== Check consistency ===================//
	//
	//get current patch
	cv::Mat_<double> pPatt;
	getPattern(img, bb, m_model_patchsize, pPatt);
	
	// measure similarity to model
	cv::Mat_<double> pConf1;
	cv::Mat_<double> dummy1;
	cv::Mat_<double> pIsin;
	tldNN(pPatt, pConf1, dummy1, pIsin);

	
	// too fast change of appearance
	if (pConf1(0,0) < 0.5)
	{
		cout << "Fast change\n";
		m_valid(0,I) = 0;
		return;
	}
	
	// too low variance of the patch
	cv::Mat_<double> patVar;
	var(pPatt, patVar);

	if (patVar(0,0) < m_var)
	{
		cout << "Low variance\n";
		m_valid(0,I) = 0;
		return;
	}
	
	//patch is in negative data
	if (pIsin(2,0) == 1)
	{
		cout << "In negative data\n";
		m_valid(0,I) = 0;
		return;
	}
	
	//========================= Update ============================//

	//generate positive data
	//measure overlap of the current bounding box with the bounding boxes on the grid
	cv::Mat_<double> overlap;

	bb_overlap(bb, m_grid, overlap);
	
	
	//generate positive examples from all bounding boxes that are highly overlappipng with current bounding box
	cv::Mat_<double> pX;
	cv::Mat_<double> pEx;
	cv::Mat_<double> bbP;
	tldGeneratePositiveData(overlap, img, m_p_par_update, pX, pEx, bbP);
	//===check to avoid EXEC_BAD_ACCESS during learning======//
	if (pX.empty()) {
		std::cout<<"Cancel Learning!\n";
		return;
	}
	//=======================================================//
	
	//labels of the positive patches
	cv::Mat_<double> pY = cv::Mat_<double>::ones(1, pX.cols);
	
	//generate negative data
	//get indexes of negative bounding boxes on the grid (bounding boxes on the grid that are far from current bounding box and which confidence was larger than 0)
	cv::Mat_<int> idx(1, overlap.cols, 0);
	for (int c=0; c < overlap.cols; ++c)
	{
		if ((overlap(0,c) < m_n_par.overlap) && (m_tmpConf(0,c) >= 1))
		{
			idx(0,c) = 1;
		}
	}

	
	//measure overlap of the current bounding box with detections
	bb_overlap(bb, m_dt[I].bb, overlap);
	
	
	//nEx get negative patches that are far from current bounding box
	std::list<int> oidx;
	for (int c=0; c < overlap.cols; ++c)
	{
		if (overlap(0,c) < m_n_par.overlap)
			oidx.push_back(c);
	}
	cv::Mat_<double> nEx(m_dt[I].patch.rows, 0);
	if (!oidx.empty())
	{
		int c=0;
		nEx.create(m_dt[I].patch.rows, oidx.size());
		for (std::list<int>::iterator oidxit = oidx.begin(); oidxit != oidx.end(); ++oidxit)
		{
			Mat col = nEx.col(c++);
			m_dt[I].patch.col(*oidxit).copyTo(col);
		}
	}
	
	
	//update the Ensemble Classifier (reuses the computation made by detector)
	cv::Mat_<double> x(pX.rows, pX.cols + cv::sum(idx)(0));
	for (int c =0; c < pX.cols; ++c)
	{
		Mat col = x.col(c);
		pX.col(c).copyTo(col);
	}
	int j=0;
	for (int c=0; c < idx.cols; ++c)
	{
		if (idx(0,c) == 1)
		{
			Mat col = x.col(pX.cols + j++);
			m_tmpPatt.col(c).copyTo(col);
		}
	}
	

	int sumIdx = sum(idx)(0);
	cv::Mat_<double> y(pY.rows, pY.cols + sumIdx, 0.0);
	for (int c =0; c < pY.cols; ++c)
	{
		Mat col = y.col(c);
		pY.col(c).copyTo(col);
	}
	
	int bootstrap = 2;

	m_fern->update(x, y, m_model_thr_fern, bootstrap);
	
	//update nearest neighbour
	tldTrainNN(pEx, nEx);
}


void TargetLock::tracking(const cv::Mat_<double>& BB1, int I, int J, cv::Mat_<double>& BB2, double& Conf, double& Valid)
{
	
	BB2.create(0,0);
	Conf = 0;
	Valid = 0;
	
	if ((BB1.cols == 0) || (BB1.rows == 0) || !bbIsDef(BB1))
		return;
	
	//estimate BB2
	//xFI generate 10x10 grid of points within BB1 with margin 5 px
	cv::Mat_<double> xFI;
	bbPoints(BB1, 10.0, 10.0, 5.0, xFI);
	//bbPoints(BB1, 30.0, 30.0, 3.0, xFI);

	
	
	cv::Mat_<double> xFJ;
	m_lk->track(*m_img.at(I).first, *m_img.at(J).first, xFI, xFI, xFJ);

	
	
	
	//medFB get median of Forward-Backward error
	cv::Mat_<double> xFKr2 = xFJ.row(2);
	std::vector<double> xFKr2V(xFKr2.begin(), xFKr2.end());
	double medFB = median2<double>(xFKr2V);
	//medNCC get median for NCC
	cv::Mat_<double> xFKr4 = xFJ.row(3);
	std::vector<double> xFKr4V(xFKr4.begin(), xFKr4.end());
	double medNCC = median2<double>(xFKr4V);
	//idxF get indexes of reliable points
	int idxFt = 0; // Number of true elements in idxF
	
	
	cv::Mat_<bool> idxF(1, xFJ.cols, false);
	for (int c=0; c < xFJ.cols; ++c)
	{
		if ((xFJ.row(2)(c) <= medFB) && (xFJ.row(3)(c) >= medNCC))
		{
			idxF(c) = true;
			idxFt++;
		}
	}

	
	//BB2  estimate BB2 using the reliable points only
	
	int idx=0;	
	
	cv::Mat_<double> tmp1(xFI.rows, idxFt);
	cv::Mat_<double> tmp2(2, idxFt);
	for (int c=0; c < xFJ.cols; ++c)
	{
		if (idxF(0, c))
		{
			Mat col = tmp1.col(idx);
			xFI.col(c).copyTo(col);
			tmp2(0, idx) = xFJ(0, c);
			tmp2(1, idx) = xFJ(1, c);
			idx++;
		}
	}

	cv::Mat_<double> shift;
	
	
	bbPredict(BB1, tmp1, tmp2, BB2, shift);

	
	
	//m_xFJ save selected points (only for display purposes)
	cv::Mat_<double> tmp = xFJ;
	m_xFJ.create(tmp.rows, idx);
	int i=0;
	for (int c=0; c < xFJ.cols; ++c)
	{
		if (idxF(0,c))
		{
			Mat col = m_xFJ.col(i++);
			tmp.col(c).copyTo(col);
		}
	}
	
	
	//detect failures
	if (!bbIsDef<double>(BB2) || bbIsOut<double>(BB2, m_imgSize))
	{
		BB2.create(0,0);
		return; //bounding box out of image
	}
	
	
	//too unstable predictions
	if ((m_control_maxbbox > 0) && (medFB > 10))  //earlier exp put medFB > 10
	{
		BB2.create(0,0);
		return;
	}
	
	//estimate confidence and validity
	//patchJ sample patch in current image
	cv::Mat_<double> patchJ;
	getPattern(m_img.at(J), BB2, m_model_patchsize, patchJ);

	
	//estimate its Conservative Similarity (considering 50% of positive patches only)
	cv::Mat_<double> dummy1;
	cv::Mat_<double> confMat;
	cv::Mat_<double> isin;
	

	tldNN(patchJ, dummy1, confMat, isin);
	Conf = confMat(0,0);
	
	//Validity
	//Valid copy validity from previous frame
	Valid = m_valid(0, I);
	
	if (Conf > m_model_thr_nn_valid)
		Valid = 1;//tracker is inside the 'core'
	
}


void TargetLock::detection(int I, cv::Mat_<double>& BB,  cv::Mat_<double>& Conf)
{

	//scanns the image(I) with a sliding window, returns a list of bounding
	//boxes and their confidences that match the object description

	BB.create(0,0);
	Conf.create(0,0);

	DtType dt;
	dt.isin.create(3,1);
	dt.isin(0,0) = NaN; //declared like this will make it works!
	dt.isin(1,0) = NaN;
	dt.isin(2,0) = NaN;

	const ImagePairType& img = m_img.at(I);
	
	
	
	//evaluates Ensemble Classifier: saves sum of posteriors to 'm_tmpConf', saves measured codes to 'm_tmpPatt', does not considers patches with variance < m_var
	m_fern->detect(img, m_control_maxbbox, m_var, m_tmpConf, m_tmpPatt);

	
	//get indexes of bounding boxes that passed throu the Ensemble Classifier
	double thr = m_num_trees * m_model_thr_fern;
	
	std::vector<double> idx_dt;
	std::vector<std::pair<int, double> > val_dt;
	for (int c=0; c < m_tmpConf.cols; ++c)
	{
		if (m_tmpConf(0,c) > thr)
		{
			idx_dt.push_back(c);
			val_dt.push_back(std::make_pair(c, m_tmpConf(0,c)));
		}
	}
	
	//if there are more than 100 detections, pick 100 of the most confident only
	if (idx_dt.size() > 100) 
	{
		std::sort(val_dt.begin(), val_dt.end(), pairValueSortPredicate<std::pair<int, double> >);
		val_dt.resize(100);
		idx_dt.clear();
		std::vector<std::pair<int, double> >::reverse_iterator vit;
		for (vit = val_dt.rbegin(); vit != val_dt.rend(); ++vit)
			idx_dt.push_back((*vit).first);
	}
	
	
	//get the number detected bounding boxes so-far
	int num_dt = (int) idx_dt.size();
	//if nothing detected, return
	cout << "num_dt at detection(): " << num_dt << endl;
	if (num_dt == 0)
	{
		m_dt[I] = dt;
		return;
	}
	
	// initialize detection structure
	dt.bb.create(4, num_dt);
	dt.patt.create(m_tmpPatt.rows, num_dt);
	dt.idx.create(1, num_dt);
	dt.conf1.create(1, num_dt);
	dt.conf2.create(1, num_dt);
	dt.isin.create(3, num_dt);
	dt.patch.create(m_model_patchsize*m_model_patchsize, num_dt);
	
	//======test correctness======//
	//Mat img_disp = (*m_img.at(I).first).clone();
	//namedWindow("Detection", 1);
	
	for (int c=0; c < num_dt; ++c)
	{
		int idx = idx_dt.at(c);
		dt.bb(0, c) = m_grid(0, idx);
		dt.bb(1, c) = m_grid(1, idx);
		dt.bb(2, c) = m_grid(2, idx);
		dt.bb(3, c) = m_grid(3, idx);
		
		//cv::rectangle(img_disp, Point(dt.bb(0, c),dt.bb(1, c)), Point(dt.bb(2, c) , dt.bb(3, c)), Scalar(255,255,255), 1, CV_AA);
		
		//imshow("Detection", img_disp);
		//waitKey(10);
		
		Mat col = dt.patt.col(c);
		m_tmpPatt.col(idx).copyTo(col);
		
		if (idx > 0) dt.idx(c) = c;
		
		dt.conf1(0,c) = NaN;
		dt.conf2(0,c) = NaN;
		dt.isin(0,c) = NaN;
		dt.isin(1,c) = NaN;
		dt.isin(2,c) = NaN;
		for (int r=0; r < m_model_patchsize*m_model_patchsize; ++r)
			dt.patch(r,c) = NaN;
	}
	
	
	
	//for every remaining detection
	for (int i=0; i < num_dt; ++i)
	{
		//measure patch
		cv::Mat_<double> ex;
		getPattern(img, dt.bb.col(i), m_model_patchsize, ex);
	
		//evaluate nearest neighbour classifier
		cv::Mat_<double> conf1;
		cv::Mat_<double> conf2;
		cv::Mat_<double> isin;
		tldNN(ex, conf1, conf2, isin); 
		
		//fill detection structure
		dt.conf1(0,i) = conf1(0,0); 
		dt.conf2(0,i) = conf2(0,0);
		Mat col = dt.isin.col(i);
		isin.col(0).copyTo(col);
		col = dt.patch.col(i);
		ex.col(0).copyTo(col);
	}
	
	
	
	//get all indexes that made it through the nearest neighbour
	int idxCount = 0;
	std::vector<bool> idx;
	for (int c=0; c < dt.conf1.cols; ++c)
	{
		if (dt.conf1(0,c) > m_model_thr_nn)
		{
			idx.push_back(true);
			idxCount++;
		}
		else
		{
			idx.push_back(false);
		}
	}
	
	
	//output
	//BB bounding boxes
	//Conf  conservative confidences
	
	//======test correctness======//
	//Mat img_disp = (*m_img.at(I).first).clone();
	//namedWindow("DetectionNN", 1);
	
	if (idxCount)
	{
		Conf.create(dt.conf2.rows, idxCount);
		BB.create(dt.bb.rows, idxCount);
		
		int colIdx = 0;
		for (int c = 0; c < num_dt; ++c) 
		{
			if (idx.at(c))
			{
				//cv::rectangle(img_disp, Point(dt.bb(0, c),dt.bb(1, c)), Point(dt.bb(2, c) , dt.bb(3, c)), Scalar(255,255,255), 1, CV_AA);
				
				//imshow("DetectionNN", img_disp);
				//waitKey(10);
				
				Mat col = BB.col(colIdx);
				dt.bb.col(c).copyTo(col);
				col = Conf.col(colIdx++);
				dt.conf2.col(c).copyTo(col);
			}
		}
	}
	
	//save the whole detection structure
	m_dt[I] = dt;
}


template<class T>
bool equalsCompPredicate(T i, T j)
{
	return (i==j);
}

template<class T>
void unique(const std::vector<T>& inv, std::vector<T>& outv)
{
	if (!inv.empty())
	{
		if (inv.size() == 1)
		{
			outv.push_back(inv.front());
			return;
		}
		std::list<T> l(inv.begin(), inv.end());
		l.sort();
		l.unique();
		outv.insert(outv.begin(), l.begin(), l.end());
	}
}

double bbOverlap(const QRectF& box1,const QRectF& box2){
	if (box1.xp > box2.xp+box2.w) { return 0.0; }
	if (box1.yp > box2.yp+box2.h) { return 0.0; }
	if (box1.xp+box1.w < box2.xp) { return 0.0; }
	if (box1.yp+box1.h < box2.yp) { return 0.0; }
	
	double colInt =  std::min(box1.xp+box1.w,box2.xp+box2.w) - std::max(box1.xp, box2.xp);
	double rowInt =  std::min(box1.yp+box1.h,box2.yp+box2.h) - std::max(box1.yp,box2.yp);
	
	double intersection = colInt * rowInt;
	double area1 = box1.w*box1.h;
	double area2 = box2.w*box2.h;
	return intersection / (area1 + area2 - intersection);
}

bool bbcomp(const QRectF& b1,const QRectF& b2){
	
    if (bbOverlap(b1,b2)<0.5)
		return false;
    else
		return true;
}


void TargetLock::bb_cluster_confidence(const cv::Mat_<double>& iBB, const cv::Mat_<double> iConf, cv::Mat_<double>& oBB, cv::Mat_<double>& oConf, cv::Mat_<double>& oSize)
{

	//Clusterering of tracker and detector responses
	//First cluster returned corresponds to the tracker
	double SPACE_THR = 0.5;
	
	
	if (iBB.empty())
		return;
	
	cv::Mat_<double> T;
	cv::Mat_<double> idx_cluster;

	
	switch (iBB.cols)
	{
		case 0:
			cout << "case 0\n";
			T.create(0,0);
			break;
		case 1:
			cout << "case 1\n";
			T.create(1,1);
			T(0,0) = 1;
			break;
		case 2:
			cout << "case 2\n";
			T = cv::Mat_<double>::ones(2,1);

			if (bb_distance(iBB) > SPACE_THR)
				T(1,0) = 2;
			break;
		default:
			cout << "case "<<iBB.cols <<endl;
			//using Alantar method
			vector<QRectF> dBB;
			vector<int> T_tmp(iBB.cols);
			
			for (int i = 0; i < iBB.cols; i++) {
				dBB.push_back(QRectF( iBB(0,i) , iBB(1,i) , iBB(2,i)-iBB(0,i) ,iBB(3,i)-iBB(1,i)  ));
			}
			
			cv::partition(dBB,T_tmp,(*bbcomp));
			
			T = cv::Mat_<double>::zeros(T_tmp.size(),1); 
			
			cout << "T [ ";
			for (int i = 0; i < T_tmp.size(); i++) {
				T(i,0) = T_tmp[i] + 1; //should be + 1 like matlab, or else bcoz of unique() ->BAD_ACCESS
				cout << T_tmp[i] + 1 << " ";
				
			}
			cout << "] \n";
			
			break;
		
	}
	

	std::vector<double> v;
	if (T.cols)
		::unique((std::vector<double>) T.col(0), v);
	if (!v.empty())
	{
		cv::Mat_<double> tmp(v, true);
		idx_cluster = tmp;
	}

	int num_clusters = (int) v.size();
	cout << "num_clusters: " << num_clusters << endl ;
	
	
	oBB.create(4, num_clusters);
	oConf.create(1,num_clusters);
	oSize.create(1,num_clusters);
	
	for (int i = 0; i < num_clusters; i++) {
		oBB(0,i) = NaN;
		oBB(1,i) = NaN;
		oBB(2,i) = NaN;
		oBB(3,i) = NaN;
		oConf(0,i) = NaN;
		oSize(0,i) = NaN;
	}
	

	for (int i=0; i < num_clusters; ++i)
	{

		cv::Mat_<int> idx(T.rows, T.cols);
		for (int r=0; r < T.rows; ++r)
		{
			if (T(r,0) == idx_cluster(r,0)) 
			{
				idx(r,0) = 1; // TBR
			}
			else 
			{
				idx(r,0) = 0;
			}
		}

		int nrows = 0;
		for (int r=0; r < idx.rows; ++r)
		{
			if (idx(r, 0))
				nrows++;
		}
		cv::Mat_<double> iBBc(4, nrows);
		int ir = 0;
		for (int r=0; r < idx.rows; ++r)
		{
			if (idx(r,0))
			{
				Mat col = iBBc.col(ir++);
				iBB.col(r).copyTo(col);
			}
		}

		
		cv::Mat_<double> meanMat;
		mean<double>(iBBc, 2, meanMat);

		Mat col = oBB.col(i);
		meanMat.col(0).copyTo(col);
	
		cv::Scalar m = cv::mean(iConf.col(i));
		oConf(0, i) = m(0);
	
		oSize(0,i) = cv::sum(idx)(0);
	}
}


double TargetLock::bb_distance(const cv::Mat_<double>& bb1, const cv::Mat_<double>* bb2)
{
	cv::Mat_<double> dMat;
	if (!bb2)
		bb_overlap(bb1, dMat);
	else
		bb_overlap(bb1, *bb2, dMat);
	double d = 1 - dMat(0,0);
	return d;
}


void TargetLock::display(int runtime, int frameNumber)
{
	if (!runtime)
	{
		
	}
	else
	{
	
		int i = frameNumber;
	
		/*
		if (m_plot_dt && (m_dt.find(frameNumber) != m_dt.end()))
		{
			cv::Mat_<double> cp;
			bbCenter(m_dt[i].bb, cp);
			
			if (! cp.empty())
			{
				//plot points
			}
			
		
			cv::Mat_<double>& conf1 = m_dt[i].conf1;
			std::vector<int> idx;
			for (int c=0; c < conf1.cols; ++c)
			{
				if (conf1(0,c) > m_model_thr_nn)
				{
					idx.push_back(c);
				}
			}
		
			cv::Mat_<double>& bb = m_dt[i].bb;
			cv::Mat_<double> bbCol(bb.rows, idx.size());
			for (int c=0; c < idx.size(); ++c)
			{
				Mat col = bbCol.col(c);
				m_dt[i].bb.col(idx.at(c)).copyTo(col);
			}
			bbCenter(bbCol, cp);
	
			if (!cp.empty())
			{
				//plot point
			}
		}*/
		
		//=============Draw Track===============//
		
		//cv::Mat_<double>& bb = m_dt[i].bb; <--WTH??
		cv::Mat_<double> bb = cv::Mat_<double>::zeros(4,1);
		bb += m_bb.col(i);

		switch (m_plot_drawoutput)
		{   
			
			case 3:
			{
				if (!bb.empty())
				{
					QRectF bbRect;
					bbRect.setX(bb(0,0));
					bbRect.setY(bb(1,0));
					bbRect.setWidth(bb(2,0) - bb(0,0));
					bbRect.setHeight(bb(3,0) - bb(1,0));
					
					//i = frameNumber
					
					//cv::Mat_<unsigned char> tmp_Draw = *m_img[i].first;
					//cv::Mat_<unsigned char> tmpImgDraw = tmp_Draw.clone();
					std:stringstream str_conf;
					
					//namedWindow("TargetLock", 1); <---I delcared that at predator::init()
					
					
					
					if (m_plot_confidence)
					{
						str_conf << m_conf(0,i) ; //<-- i = frameNumber
				
					}
					else {
						str_conf << "Null";
					}
					
					//=======experiment purposed=======//
					/*
					m_result(i,0) = i;
					m_resultConfidence(i,0) = i;
					if (isnan( m_conf(0,i) )) 
					{
						m_resultConfidence(i,1) = 0;
						
						m_result(i,1) = 0;
						m_result(i,2) = 0;
						m_result(i,3) = 0;
						m_result(i,4) = 0;
						 
					}
					else {
						m_resultConfidence(i,1) = m_conf(0,i);
						
						m_result(i,1) = bb(0,0)/320; //left relative
						m_result(i,2) = bb(1,0)/240; //top relative
						m_result(i,3) = (bb(2,0) - bb(0,0))/320; //width relative
						m_result(i,4) = (bb(3,0) - bb(1,0))/240; //height relative
						 
					}
					*/
					//==============end=================//
					
					cout << "m_conf(0,i): " << m_conf(0,i) << endl;
					cout << "bb(0,0): " << bb(0,0) << endl;
					cout << "bb(1,0): " << bb(1,0) << endl;
					cout << "bb(2,0): " << bb(2,0) << endl;
					cout << "bb(3,0): " << bb(3,0) << endl;
					cout << endl;
					
					if (!isnan(m_conf(0,i) )) {
						if ( (bb(2,0) - bb(0,0) > 0) && (bb(3,0) - bb(1,0) > 0) 
							/*&& bb(0,0)>0 && bb(1,0)>0 && bb(2,0)>0 && bb(3,0)>0 
							&& bb(0,0) < m_imgDisplay.rows && bb(1,0) < m_imgDisplay.cols
							&& bb(2,0) < m_imgDisplay.rows && bb(3,0) < m_imgDisplay.cols*/) 
						{
							cv::rectangle(m_imgDisplay, Point(bb(0,0), bb(1,0)), Point(bb(2,0), bb(3,0)), Scalar(0,255,255), 2, CV_AA);
						}
						cv::putText(m_imgDisplay, str_conf.str(), Point(bbRect.xp + bbRect.w/2, bbRect.yp + bbRect.h/2), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255,255,255), 1, CV_AA);
					}
					
					Mat imgDisplay_resize;
					cv::resize(m_imgDisplay, imgDisplay_resize, Size(),2, 2, INTER_LANCZOS4); //Lanczos4
					cv::imshow("TargetLock", imgDisplay_resize); //put tmpImgDraw  for  debug Matlab
				}
				else
				{
					Mat imgDisplay_resize;
					cv::resize(m_imgDisplay, imgDisplay_resize, Size(),2, 2, INTER_LANCZOS4); //Lanczos4
					cv::imshow("TargetLock", imgDisplay_resize);
			
				}
			}
				break;
		}
		    
		//Info
		if (m_plot_pts)
		{
			//plot points
		}
	}
}


IplImage* TargetLock::blur(IplImage& image, double sigma)
{
	IplImage *blurred = cvCreateImage(cvGetSize(&image), image.depth, image.nChannels);
	cvSmooth(&image, blurred, CV_GAUSSIAN, 0, 0, sigma, sigma);
	return blurred;	
}

void TargetLock::blur(IplImage& image, cv::Mat_<unsigned char>& outMat, double sigma)
{
	
#ifdef USE_CVSMOOTH_BLUR
	IplImage *blurred = cvCreateImage(cvGetSize(&image), image.depth, image.nChannels);
	cvSmooth(&image, blurred, CV_GAUSSIAN, 0, 0, sigma, sigma);
	//outMat = blurred;
	cv::Mat outMatTmp = cvarrToMat(blurred);
	//outMatTmp.convertTo(outMat, CV_16U); 
	outMatTmp.convertTo(outMat, CV_8UC1); //ori CV_8U
	cvReleaseImage(&blurred);
#else
	//csize = 6*sigma;
	float csize = 6 * sigma;
	//
	//shift = (csize - 1)/2;
	int shift = (csize - 1)/2;
	//
	//h = FSPECIAL('gaussian',csize,sigma);
	if (m_gaussMat.empty())
	{
		//cv::Mat_<double> gaussMat;
		gaussMatrix<float>(csize, sigma, m_gaussMat);
	}
	
	cv::SVD svd;
	cv::Mat_<float> S;
	cv::Mat_<float> U;
	cv::Mat_<float> V;
	svd.compute(m_gaussMat, S, U , V, cv::SVD::FULL_UV);
	
	//cv::Mat_<double> g;
	//g = U * cv::Mat::diag(S) * V;
	
	cv::Mat_<float> v;
	U.col(0).copyTo(v);
	v = v * sqrt(S(0,0));
	
	cv::Mat_<float> h;
	V.row(0).copyTo(h);
	h = h * sqrt(S(0,0));

	cv::Mat_<float> tmpMat1;
	cv::Mat_<float> tmpMat2;
	cv::Mat_<unsigned char> inMat(&image); 
	convolve2D<unsigned char, float>(inMat, v, tmpMat1); 
	convolve2D<float, float>(tmpMat1, h, tmpMat2);
	
	
	outMat.create(inMat.rows, inMat.cols);
	for (int r = 0; r < outMat.rows; ++r)
		for (int c = 0; c < outMat.cols; ++c)
			outMat(r,c) = (unsigned char) tmpMat2(r+shift+1, c+shift+1);
	
	
#endif
	return;
}



bool TargetLock::bb_scan(const cv::Mat_<double>& bb, const CvSize& imSize, double min_win, Mat_<double>& bb_out, Mat_<double>& sca)
{

	
	const double minbb = min_win;
	if ((imSize.width < min_win) || (imSize.height < min_win))
		return false;
	
	static const double shift = 0.1;
	cv::Mat_<double> ident = cv::Mat_<double>::ones(m_scale.rows, m_scale.cols);
	

	cv::Mat_<double> bbW(m_scale.rows, m_scale.cols);
	cv::multiply(m_scale, ident, bbW, bbWidth(bb));
	for (int r=0; r < bbW.rows; ++r)
		for (int c=0; c < bbW.cols; ++c)
			bbW(r, c) = cvRound(bbW(r, c));
	
	
	cv::Mat_<double> bbH(m_scale.rows, m_scale.cols);
	cv::multiply(m_scale, ident, bbH, bbHeight(bb));
	for (int r=0; r < bbH.rows; ++r)
		for (int c=0; c < bbH.cols; ++c)
			bbH(r, c) = cvRound(bbH(r, c));
	
	cv::Mat_<double> bbSHH(m_scale.rows, m_scale.cols);
	for (int r=0; r < bbSHH.rows; ++r)
		for (int c=0; c < bbSHH.cols; ++c)
			bbSHH(r, c) = shift * bbH(r, c);
	
	cv::Mat_<double> bbSHW(m_scale.rows, m_scale.cols);
	for (int r=0; r < bbSHW.rows; ++r)
		for (int c=0; c < bbSHW.cols; ++c)
			bbSHW(r, c) = shift * ((bbH(r, c) < bbW(r, c)) ? bbH(r, c) : bbW(r, c));
	
	Mat_<double> bbF(4,1);
	bbF(0,0) = 2;
	bbF(1,0) = 2;
	bbF(2,0) = imSize.width;
	bbF(3,0) = imSize.height;

	
	int idx = 1;
	vector<cv::Mat_<double> > bbs;
	for (int i=0; i < m_scale.cols; ++i)
	{
	
		if ((bbW(0,i) < minbb) || (bbH(0,i) < minbb))
			continue;
		
		vector<double> leftV;
		colon(bbF(0,0), bbSHW(0, i), bbF(2, 0)-bbW(0, i)-1, leftV);
		cv::Mat_<double> left(1, leftV.size());
		for (int j=0; j < leftV.size(); ++j)
			left(0,j) = cvRound(leftV.at(j));
	
		vector<double> topV;
		colon(bbF(1,0), bbSHH(0, i), bbF(3, 0)-bbH(0, i)-1, topV);
		cv::Mat_<double> top(1, topV.size());
		for (int j=0; j < topV.size(); ++j)
			top(0,j) = cvRound(topV.at(j));
	
		cv::Mat_<double> grid;
		ntuples<cv::Mat_<double> >(top, left, grid);
		if (grid.cols == 0)
			continue;
		
		cv::Mat_<double> bbsMat = cv::Mat_<double>::zeros(6 , grid.cols);
		
		
		
		bbsMat.row(0) += grid.row(1);
		bbsMat.row(1) += grid.row(0);
		bbsMat.row(2) += (cv::Mat_<double>)(grid.row(1)+bbW(0,i)-1);
		bbsMat.row(3) += (cv::Mat_<double>)(grid.row(0)+bbH(0,i)-1);
		bbsMat.row(4) += (cv::Mat_<double>)(cv::Mat_<double>::ones(1, grid.cols)*idx);
		bbsMat.row(5) += (cv::Mat_<double>)(cv::Mat_<double>::ones(1, grid.cols)*left.cols);
		
		
		bbs.push_back(bbsMat);
	
		cv::Mat_<double> tmp = cv::Mat_<double>::zeros(2, sca.cols+1);
		tmp(0, tmp.cols-1) = bbH(0,i);
		tmp(1, tmp.cols-1) = bbW(0,i);
		for (int c=0; c < sca.cols; ++c)
			tmp.col(c) += sca.col(c);
		tmp.copyTo(sca);

		idx++;
	}
	
	bb_out.create(0, 0);
	for (int i=0; i < bbs.size(); ++i)
	{
		cv::Mat_<double>& bbsi = bbs.at(i);
		cv::Mat_<double> tmp = cv::Mat_<double>::zeros(bbsi.rows, bb_out.cols + bbsi.cols);
		for (int c=0; c < bb_out.cols; ++c)
			tmp.col(c) += bb_out.col(c);
		for (int c=0; c < bbsi.cols; ++c)
			tmp.col(c+bb_out.cols) += bbsi.col(c);
		tmp.copyTo(bb_out);
	}
	
	return true;
}


void TargetLock::generateFeatures(unsigned nTrees, unsigned nFeatures, cv::Mat_<double>& features, bool)
{

	m_featureType = TargetLock::FORREST;
	static const double shift = 0.2;
	static const double sca = 1.0;
	static const double offset = shift;
	
	
	std::vector<double> v;
	colon<double>(0, shift, 1, v);  
	cv::Mat_<double> tup(v); 

	tup = tup.t(); 
	
	cv::Mat_<double> n;
	ntuples<cv::Mat_<double> >(tup, tup, n); 

	cv::Mat_<double> x;
	repmat<cv::Mat_<double> >(n, 2, x, 1); 
	
	cv::Mat_<double> tmp = cv::Mat_<double>::zeros(x.rows, 2 * x.cols);
	for (int c=0; c < x.cols; ++c)
		tmp.col(c) += x.col(c);
	for (int c=0; c < x.cols; ++c)
		tmp.col(c+x.cols) += (x.col(c) + (shift/2));
	tmp.copyTo(x);
	
	int k = x.cols;
	

	cv::Mat_<double> rnd(1, k);
	cv::Mat_<double> r(x.rows, x.cols); x.copyTo(r);

	cv::randu(rnd, 0, 1);
	r.row(2) += ((sca*rnd) + offset);
	
	cv::Mat_<double> l(x.rows, x.cols); x.copyTo(l);
	cv::randu(rnd, 0, 1);
	l.row(2) -= ((sca*rnd) + offset);

	cv::Mat_<double> t(x.rows, x.cols); x.copyTo(t);
	cv::randu(rnd, 0, 1);
	t.row(3) -= ((sca*rnd) + offset);

	cv::Mat_<double> b(x.rows, x.cols); x.copyTo(b);
	cv::randu(rnd, 0, 1);
	b.row(3) += ((sca*rnd) + offset);
	
	
	tmp = cv::Mat_<double>::zeros(x.rows, r.cols+l.cols+t.cols+b.cols);
	for (int c=0; c < r.cols; ++c)
		tmp.col(c) += r.col(c);
	for (int c=0; c < l.cols; ++c)
		tmp.col(c+r.cols) += l.col(c);
	for (int c=0; c < t.cols; ++c)
		tmp.col(c+r.cols+l.cols) += t.col(c);
	for (int c=0; c < b.cols; ++c)
		tmp.col(c+r.cols+l.cols+t.cols) += b.col(c);
	tmp.copyTo(x);

	cv::Mat_<double> xt = x.t();
	
	
	// could use cvReduce?
	cv::Mat_<int> idx = cv::Mat_<int>::zeros(1, x.cols);
	for (int c=0; c < x.cols; ++c)
	{
		if ((x(0,c) < 1) && (x(1,c) < 1) && (x(0,c) >= 0.1) && (x(1,c) >= 0.1))
			idx(0, c) = 1;
	}

	int i=0;
	for (int c=0; c < idx.cols; ++c)
	{
		if (idx(c))
		{
			Mat col = tmp.col(i++);
			x.col(c).copyTo(col);
		}
	}
	if (i > 0)
	{
		x.create(x.rows, i);
		for (int c=0; c < i; ++c)
		{
			Mat col = x.col(c);
			(tmp.col(c)).copyTo(col);
		}
	}
	
	for (int r=0; r < x.rows; ++r)
		for (int c=0; c < x.cols; ++c)
		{
			if (x(r,c) > 1) x(r,c) = 1;
			if (x(r,c) < 0) x(r,c) = 0;
		}
	
	int numF = x.cols; 
	
	i=0;
	tmp.create(x.rows, x.cols);
	tmp = 0;
	
	vector<int> indexes;
	colon<int>(0, 1, numF-1, indexes);
	std::random_shuffle(indexes.begin(), indexes.end());
	for (vector<int>::const_iterator vit=indexes.begin(); vit != indexes.end(); ++vit)
	{
		tmp.col(i++) += x.col(*vit);
	}
	tmp.copyTo(x);

	tmp.create(x.rows, nTrees * nFeatures);
	for (int c=0; c < tmp.cols; ++c)
	{
		Mat col = tmp.col(c);
		(x.col(c)).copyTo(col);
	}
	
	features = reshape<double>(tmp, 4*nFeatures, nTrees);

}


double TargetLock::bb_overlap(double *bb1, double *bb2)
{
	if (bb1[0] > bb2[2]) { return 0.0; }
	if (bb1[1] > bb2[3]) { return 0.0; }
	if (bb1[2] < bb2[0]) { return 0.0; }
	if (bb1[3] < bb2[1]) { return 0.0; }
	
	double colInt = std::min(bb1[2], bb2[2]) - std::max(bb1[0], bb2[0]) + 1;
	double rowInt = std::min(bb1[3], bb2[3]) - std::max(bb1[1], bb2[1]) + 1;
	
	double intersection = colInt * rowInt;
	double area1 = (bb1[2]-bb1[0]+1)*(bb1[3]-bb1[1]+1);
	double area2 = (bb2[2]-bb2[0]+1)*(bb2[3]-bb2[1]+1);
	return intersection / (area1 + area2 - intersection);
}

void TargetLock::bb_overlap(const cv::Mat_<double>& bb, cv::Mat_<double>& overlap)
{
	double *bbData = (double*) bb.data;
	int nBB = bb.cols;
	
	// Output
	overlap = cv::Mat_<double>::zeros(1, nBB*(nBB-1)/2);
	double *out = (double*) overlap.data;
	
	for (int i = 0; i < nBB-1; i++) {
		for (int j = i+1; j < nBB; j++) {
			*out++ = bb_overlap(bbData + i, bbData + j);
		}
	}
}

void TargetLock::bb_overlap(const cv::Mat_<double>& bb1, const cv::Mat_<double>& bb2, cv::Mat_<double>& overlap)
{
	int N1 = bb1.cols; // bb1 cols
	int N2 = bb2.cols; // bb2 cols
	
	if (N1 == 0 || N2 == 0) {
		N1 = 0; N2 = 0;
	}
	
	overlap.create(N1, N2);
	double *data = (double *) overlap.data;
	
	for (int j = 0; j < N2; j++)
	{
		for (int i = 0; i < N1; i++)
		{
			cv::Mat_<double> bb1Col(bb1.rows, 1);
			cv::Mat_<double> bb2Col(bb2.rows, 1);
			bb1.col(i).copyTo(bb1Col);
			bb2.col(j).copyTo(bb2Col);
			*data++ = bb_overlap((double*) bb1Col.data, (double*) bb2Col.data);
		}
	}
}

void TargetLock::imagePatch(const Mat_<unsigned char>& inMat, const cv::Mat_<double>& bb, cv::Mat_<unsigned char>& outMat, const ParOptionsType& p_par, double randomize)
{
	if (randomize > 0)
	{
		double randNumber = 0.9501; //from Matlab Debug
		
		int noise = p_par.noise;
		int angle = p_par.angle;
		double scale = p_par.scale;
		double shift = p_par.shift;
		
		cv::Mat_<double> cp;
		bbCenter(bb, cp);
	
		cp -= 1;
		
		cv::Mat_<double> Sh1 = (cv::Mat_<double>(3, 3) << 1, 0, -cp(0, 0), 0, 1, -cp(1, 0), 0, 0, 1);
		
		
		double sca = 1 - scale*((double)(randNumber) - 0.5); //*m_rng
		cv::Mat_<double> Sca = cv::Mat_<double>::zeros(3,3);
		Sca(0,0) = sca;
		Sca(1,1) = sca;
		Sca(2,2) = 1;
	
		
		double ang = 2*pi/360*angle*((double)(randNumber)-0.5); //*m_rng
		double ca = cos(ang);
		double sa = sin(ang);
		cv::Mat_<double> Ang = (cv::Mat_<double>(3, 3) << ca, -sa, 0, sa, ca, 0, 0, 0, 1);
		
		
		double shR = shift*bbHeight(bb)*((double)(randNumber)-0.5); //*m_rng
		double shC = shift*bbWidth(bb)*((double)(randNumber)-0.5); //*m_rng
		cv::Mat_<double> Sh2 = (cv::Mat_<double>(3, 3) << 1, 0, shC, 0, 1, shR, 0, 0, 1);
		
		
		double bbW = bbWidth(bb)-1;
		double bbH = bbHeight(bb)-1;
		cv::Mat_<double> box = (cv::Mat_<double>(1, 4) << -bbW/2, bbW/2, -bbH/2, bbH/2);
		
		
		cv::Mat_<double> H = Sh2*Ang*Sca*Sh1;
		cv::Mat_<double> bbsize;
		bbSize(bb, bbsize);
		
		cv::Mat_<double> HInv = H.inv();
		cv::Mat_<double> warpMat;
		warp(inMat, HInv, box, warpMat);
		
		
		cv::Mat_<double> randnMat;
		randnMat = cv::Mat_<double>::zeros(bbsize(0,0),bbsize(1,0));
		m_rng->fill(randnMat, RNG::NORMAL, Scalar(0), Scalar(1));
		
		warpMat = warpMat + (noise * randnMat);

		outMat.create(bbsize(0,0),bbsize(1,0));
		warpMat.convertTo(outMat, CV_8UC1); //<--this is correct way to convert
		/*
		for (int i=0; i < warpMat.rows ;i++)
		{
			for (int j=0; j < warpMat.cols; j++)
				outMat(i,j) = (unsigned char) warpMat(i,j); // <-- this is wrong way
		}
		 */
		
		/*
		std::stringstream kkk;
		kkk << "warp ";
		kkk << m_noFrameForWarp++;
		cv::namedWindow(kkk.str(), 1);
		cv::imshow(kkk.str(), outMat);
		 */
		
	}
	else
	{
		bool isint = true;
		for (int r=0; r < bb.rows; ++r)
		{
			for (int c=0; c < bb.cols; ++c)
			{
				if ((floor(bb(r,c)) - bb(r,c)) != 0.00000000000)
				{
					isint = false;
					break;
				}
			}
			if (!isint) break;
		}
		if (isint)
		{
			
			
		
			int L = max(0, (int) bb(0,0) - 1); // x0
			int T = max(0, (int) bb(1,0) - 1); // y0
			int R = min(inMat.cols, (int) bb(2,0)); // x1
			int B = min(inMat.rows, (int) bb(3,0)); // y1
			
			outMat.create(B-T, R-L);
			for (int i=T; i < B; ++i)
				for (int j=L; j < R; ++j)
					outMat(i-T, j-L) = inMat(i, j);
			
			//cv::namedWindow("warp image2", 1);
			//cv::imshow("warp image2", outMat);
		}
		else
		{
			
			cv::Mat_<double> cp(2,1);
			cp(0,0) = 0.5 * (bb(0) + bb(2)) - 1;
			cp(1,0) = 0.5 * (bb(1) + bb(3)) - 1;
		
		
			cv::Mat_<double> H(3,3);
			H(0,0) = 1; H(0,1) = 0; H(0,2) = -cp(0,0);
			H(1,0) = 0; H(1,1) = 1; H(1,2) = -cp(1,0);
			H(2,0) = 0; H(2,1) = 0; H(2,2) = 1;
		
			
			double bbW = bb(2,0) - bb(0,0);
			
			double bbH = bb(3,0) - bb(1,0);
			
			if ((bbW <= 0) || (bbH <= 0))
			{
				outMat.create(0,0);
				return;
			}
			
			cv::Mat_<double> box(1,4);
			box(0,0) = -bbW/2; box(0,1) = bbW/2;
			box(0,2) = -bbH/2; box(0,3) = bbH/2;
		
			
			cv::Mat_<double> patch;
			cv::Mat_<double> Hinv = H.inv();
		
			warp(inMat, Hinv, box, patch);
			
			patch.convertTo(outMat, CV_8UC1);
			
			
			
			//cv::namedWindow("warp image", 1);
			//cv::imshow("warp image", outMat);
		
		}
	}
	
}




void warp_image_roi(const cv::Mat_<unsigned char>& image , int w, int h, double *H,
                    double xmin, double xmax, double ymin, double ymax,
                    double fill, double *result)
{
	
	double curx, cury, curz, wx, wy, wz, ox, oy, oz;
	int x, y;
	unsigned char *tmp;
	double *output=result;
	double i, j, xx, yy;
	//precalulate necessary constant with respect to i,j offset 
	//translation, H is column oriented (transposed)  
	ox = M(0,2); //95.000
	oy = M(1,2); //90.49999
	oz = M(2,2); //1.000

	
	yy = ymin;  // -42.49999
	for (j=0; j<(int)(ymax-ymin+1); j++)
	{
		// calculate x, y for current row 
		curx = M(0,1)*yy + ox;
		cury = M(1,1)*yy + oy;
		curz = M(2,1)*yy + oz;
		xx = xmin; 
		yy = yy + 1;
		for (i=0; i<(int)(xmax-xmin+1); i++)
		{
			// calculate x, y in current column 
			wx = M(0,0)*xx + curx;
			wy = M(1,0)*xx + cury;
			wz = M(2,0)*xx + curz;
			//       printf("%g %g, %g %g %g\n", xx, yy, wx, wy, wz);
			wx /= wz; wy /= wz;
			xx = xx + 1;
			
			x = (int)floor(wx);
			y = (int)floor(wy);
			
			if (x>=0 && y>=0)
			{
				wx -= x; wy -= y; 
				if (x+1==w && wx==1)
					x--;
				if (y+1==h && wy==1)
					y--;
				if ((x+1)<w && (y+1)<h)
				{
					
					tmp = image.data + y*image.step + x*image.elemSize();
					
					*output++ = 
					(*(tmp) * (1-wx) + *(tmp + image.elemSize()) * wx) * (1-wy) +
					(*(tmp + image.step) * (1-wx) + *(tmp + image.step + image.elemSize()) * wx) * wy;
					
				} else 
					*output++ = fill;
			} else 
				*output++ = fill;
		}
	}
}

void TargetLock::warp(const cv::Mat_<unsigned char>& imMat, const cv::Mat_<double>& HMat, const cv::Mat_<double>& B, cv::Mat_<double>& outMat)
{
	//Note opencv read in row order, while matlab in column order
	int w, h;

	double *result;
	double *H = (double*) HMat.data;
	double xmin, xmax, ymin, ymax, fill;
	
	w = imMat.cols;
	h = imMat.rows;
	
	xmin = B(0,0); xmax = B(0,1);
	ymin = B(0,2); ymax = B(0,3);
	
	fill=0;
	
	result = new double[((int)(xmax-xmin+1)*(int)(ymax-ymin+1))];
	{
		warp_image_roi(imMat , w, h, H, xmin, xmax, ymin, ymax, fill, result);
	}
	
	//to_matlab//
	int num_cols = (int)(xmax-xmin+1);
	int num_rows = (int)(ymax-ymin+1);
	
	const double* s_ptr = result;
	outMat.create(num_rows, num_cols);
	outMat = cv::Mat_<double>::zeros(num_rows, num_cols);
	for (int i=0;i<num_rows;i++)
	{
		for (int j=0; j<num_cols; j++)
		{
			outMat(i,j) = *s_ptr++ ;
		}
	}
	
	delete [] result;
}

void to_matlab(const double *image, int num_cols, int num_rows, cv::Mat_<double>& outMat)
{
	int i, j;
	const double* s_ptr = image;
	outMat.create(num_rows, num_cols);
	outMat = cv::Mat_<double>::zeros(num_rows, num_cols);
	for (i=0;i<num_rows;i++)
	{
		for (j=0; j<num_cols; j++)
			outMat(i,j) = *s_ptr++;
	}
}




void TargetLock::tldGeneratePositiveData(const cv::Mat_<double>& overlap, const ImagePairType& img0, const ParOptionsType& init, cv::Mat_<double>& pX, cv::Mat_<double>& pEx, cv::Mat_<double>& bbP0)
{

	//Get closest bbox
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(overlap, &minVal, &maxVal, &minLoc, &maxLoc);
	bbP0.create(4, 1);
	for (int r=0; r < 4; ++r)
		bbP0(r, 0) = m_grid(r, maxLoc.x);
	
	//Get overlapping bboxes
	std::vector<std::pair<unsigned, double> > idx;
	for (int c=0; c < overlap.cols; ++c)
	{
		if (overlap(0, c) > 0.8) //originally 0.6
			idx.push_back(std::make_pair(c, overlap(0, c)));
	}
	cv::Mat_<unsigned> idxP(1, idx.size());
	unsigned *data = (unsigned *)idxP.data;
	std::vector<std::pair<unsigned, double> >::iterator it;
	for (it = idx.begin(); it != idx.end(); ++it)
		*data++ = it->first;
	
	if ((unsigned)idxP.cols > init.num_closest)
	{
		idxP.create(1, init.num_closest);
		std::sort(idx.begin(), idx.end(), pairValueSortPredicate<std::pair<unsigned, double> >);
		idx.resize(init.num_closest);
		data = (unsigned *)idxP.data;
		for (it = idx.begin(); it != idx.end(); ++it)
			*data++ = it->first;
	}
	
	cv::Mat_<double> bbP(m_grid.rows, idxP.cols);
	for (int c=0; c < idxP.cols; ++c)
	{
		Mat col = bbP.col(c);
		m_grid.col(idxP(0, c)).copyTo(col);
	}

	if (bbP.empty())
	{
		std::cout<< "return bbP empty@tldGeneratePositiveData, not suitable for learning!" << std::endl;
		return;
	}
	
	// Get hull
	cv::Mat_<double> bbH;
	bbHull(bbP, bbH);

	std::vector<double> cols;
	colon<double>(bbH(0,0), 1, bbH(2,0), cols);
	std::vector<double> rows;
	colon<double>(bbH(1,0), 1, bbH(3,0), rows);
	
	
	ImagePairType im1;
	im1.first = new cv::Mat_<unsigned char>(img0.first->rows, img0.first->cols);
	img0.first->copyTo(*im1.first);
	im1.second = new cv::Mat_<unsigned char>(img0.second->rows, img0.second->cols);
	img0.second->copyTo(*im1.second);

	
	getPattern(im1, bbP0, m_patchSize, pEx);
	
	 
	if (m_fliplr)
	{
		cv::Mat_<double> pExFlip = cv::Mat_<double>::zeros(pEx.rows, pEx.cols + 1);
		cv::Mat_<double> pExFlip_;
		getPattern(im1, bbP0, m_patchSize, pExFlip_, m_fliplr);
		
		pExFlip.col(1) += pExFlip_;
		Mat col = pExFlip.col(0);
		pEx.col(0).copyTo(col);
		pEx.create(pExFlip.rows, pExFlip.cols);
		pExFlip.copyTo(pEx);
	}
	
	//=============Test Correctness===========//
	
	//Mat_<double> testGrid = m_grid.t();
	
	//========================================//
	

	for (int i=1; i <= init.num_warps; ++i)
	{
		if (i > 1)
		{
			static uint64 rngState = m_rng->state;
			m_rng->state = rngState;
			double randomize = (double)(*m_rng);
			cv::Mat_<unsigned char> patch_blur;
			//randomize(0,0) = 0.779079247486771;
			imagePatch(*(img0.second), bbH, patch_blur, init, randomize);
			for (int r=0; r < rows.size(); ++r)
			{
				for (int c=0; c < cols.size(); ++c)
					(*im1.second)(rows.at(r), cols.at(c)) = patch_blur(r, c);
			}
		}
		
		cv::Mat_<double> patt;
		cv::Mat_<double> status;
		
		m_fern->getPatterns(im1, idxP, 0, patt, status);
		
		//Test Correctness
		//m_fern->getPatterns(im1, testGrid, idxP, 0, patt, status);
		
		//std::cout<<"patt.rows: " << patt.rows << "  patt.cols:" << patt.cols << std::endl;
		cv::Mat_<double> tmp(patt.rows, pX.cols + patt.cols);
		if (i>1)
			for (int c=0; c < pX.cols; ++c)
			{
				Mat col = tmp.col(c);
				pX.col(c).copyTo(col);
			}
		for (int c=0; c < patt.cols; ++c)
		{
			Mat col = tmp.col(c+pX.cols);
			patt.col(c).copyTo(col);
		}
		
		//std::cout<<"tmp.rows: " << tmp.rows << "  tmp.cols:" << tmp.cols << std::endl;
		pX.create(tmp.rows, tmp.cols);
		tmp.copyTo(pX);
	}
	
	//std::cout<<"pX.rows: " << pX.rows << "  pX.cols:" << pX.cols << std::endl;
	
	 
}


void TargetLock::tldGenerateNegativeData(const cv::Mat_<double>& overlap, const ImagePairType& img, cv::Mat_<double>& nX, cv::Mat_<double>& nEx)
{

	// Measure patterns on all bboxes that are far from initial bbox
	std::vector<std::pair<unsigned, double> > idx;
	for (int c=0; c < overlap.cols; ++c)
	{
		if (overlap(0, c) < m_n_par.overlap)
			idx.push_back(std::make_pair(c, overlap(0, c)));
	}
	cv::Mat_<unsigned> idxN(1, idx.size());
	unsigned *data = (unsigned *)idxN.data;
	std::vector<std::pair<unsigned, double> >::iterator it;
	for (it = idx.begin(); it != idx.end(); ++it)
		*data++ = it->first;
	
	cv::Mat_<double> status;
	m_fern->getPatterns(img, idxN, m_var/2, nX, status);
	
	//bboxes far and with big variance
	std::vector<std::pair<unsigned,unsigned> >tmp;
	for (int c=0; c < status.cols; ++c)
	{
		if (status(0, c) == 1)
			tmp.push_back(std::make_pair(c, idxN(0, c)));
	}
	idxN.create(1, tmp.size());
	data = (unsigned *)idxN.data;
	std::vector<std::pair<unsigned, unsigned> >::iterator tmpit;
	for (tmpit = tmp.begin(); tmpit != tmp.end(); ++tmpit)
		*data++ = tmpit->second;

	
	cv::Mat_<double> nX_(nX);
	nX.create(nX.rows, tmp.size()); //<-- nX_ or nX?
	int i=0;
	for (tmpit = tmp.begin(); tmpit != tmp.end(); ++tmpit)
	{
		Mat col = nX.col(i++);
		nX_.col(tmpit->first).copyTo(col);
	}
	
	
	//Randomly select 'num_patches' bboxes and measure patches
	cv::Mat_<double> idx1;
	std::vector<double> indexes;
	colon<double>(0, 1, idxN.cols-1, indexes);
	cv::Mat_<double> inTmp(indexes, true);
	cv::Mat_<double> in = inTmp.t();
	randValues(in, m_n_par.num_patches, idx1);
	cv::Mat_<double> bb(m_grid.rows, idx1.cols); // 6 x 100
	for (int c=0; c < idx1.cols; ++c)
	{
		Mat col = bb.col(c);
		m_grid.col(idxN(0, idx1(0, c))).copyTo(col);
	}

	getPattern(img, bb, m_patchSize, nEx);
	
}


void TargetLock::getPattern(const ImagePairType& img, const cv::Mat_<double>& bb, const unsigned patchSize, cv::Mat_<double>& pattern, bool flip)
{
	
	unsigned nBB = (unsigned) bb.cols;
	pattern.create(patchSize*patchSize, nBB);
	pattern = cv::Mat_<double>::zeros(pattern.size());
	
	
	
	for (int c=0; c < nBB; ++c) 
	{
		cv::Mat_<unsigned char> patch;
		imagePatch(*img.first, bb.col(c), patch, m_p_par_init);
		
		if (flip)
			cv::flip(patch, patch, 1);
		
		//pattern temporary holder
		Mat_<double> pattern_ ;
		
		patch2Pattern(patch, patchSize, pattern_);
		pattern.col(c) += pattern_;
	}
	
}


void TargetLock::patch2Pattern(const cv::Mat_<unsigned char>& patch, const unsigned patchSize, cv::Mat_<double>& pattern)
{
	
	//cv::namedWindow("patch image", 1);
	//cv::imshow("patch image", patch);
	
	cv::Mat_<unsigned char> patch_;
	cv::resize(patch, patch_, Size(patchSize,patchSize),0, 0, INTER_LINEAR); //Bilinear
	
	//cv::namedWindow("resize image", 1);
	//cv::imshow("resize image", patch_);
	
	
	std::vector<double> patchVector;
	pattern.create(patchSize*patchSize, 1);
	
	for (int c=0; c < patch_.cols; ++c)
	{
		for (int r=0; r < patch_.rows; ++r)
		{
			pattern((r+(patch_.rows*c)), 0) = (double) patch_.at<uchar>(r,c);
			patchVector.push_back(patch_.at<uchar>(r,c));
		}
	}
	double mean = std::accumulate(patchVector.begin(), patchVector.end(), 0.0) / patchVector.size();

	pattern = pattern - mean;
}

void TargetLock::randValues(const cv::Mat_<double>& in, double k, cv::Mat_<double>& out)
{
	
	int N = in.cols;
	if (k == 0) return;
	if (k > N) k = N;
	if (k/N < 0.0001)
	{
		cv::Mat_<double> i1(1, k);
		cv::randu(i1, Scalar(0), Scalar(1));
		for (int i=0; i < k; ++i)
			i1(0, i) = ceil(N * i1(0, i));
	
		std::sort(i1.begin(), i1.end());
	
		std::vector<double> si1(i1.cols);
		std::vector<double>::iterator it;
		it=std::unique_copy(i1.begin(),i1.end(),si1.begin());
		si1.resize(it - si1.begin());
		out.create(1, si1.size());
		
		for (int c=0; c < si1.size(); ++c)
		{
			Mat col = out.col(c);
			in.col(si1.at(c)).copyTo(col);
		}
	}
	else
	{
		std::vector<double> indexes;
		colon<double>(0, 1, N-1, indexes);
		std::random_shuffle(indexes.begin(), indexes.end());
		//printVector(indexes, "indexes");
		
		indexes.resize(k);
		std::sort(indexes.begin(), indexes.end());
		out.create(1, k);
		for (int c=0; c < indexes.size(); ++c)
		{
			Mat col = out.col(c);
			in.col(indexes.at(c)).copyTo(col);
		}
	}
}





void TargetLock::tldTrainNN(const cv::Mat_<double>& pEx, const cv::Mat_<double>& nEx)
{
	//nP get the number of positive example 
	//nN get the number of negative examples
	int nP = pEx.cols;
	int nN = nEx.cols;
	
	
	cv::Mat_<double> x = cv::Mat_<double>::zeros(pEx.rows, nP + nN);
	x.colRange(0, nP) += pEx.colRange(0, nP);
	
	//x.colRange(nP, x.cols) += nEx.colRange(0, nN).clone();
	x.colRange(nP, x.cols) += nEx.colRange(0, nN);
	
	cv::Mat_<double> y = cv::Mat_<double>::zeros(1, nP + nN);
	cv::Mat_<double> ones = cv::Mat_<double>::ones(1, nP);
	cv::Mat_<double> zeros = cv::Mat_<double>::zeros(1, nN);
	
	y.colRange(0, nP) += ones.colRange(0, nP);
	
	y.colRange(nP, nP + nN) += zeros.colRange(0, nN);
	
	
	//Permutate the order of examples
	cv::Mat_<int> idx;
	randperm<int>(nP+nN, idx);
	
	//always add the first positive patch as the first (important in initialization)
	if ((pEx.rows != 0) && (pEx.cols != 0))
	{
		cv::Mat_<double> x_(x.rows, x.cols + 1);
		Mat col = x_.col(0);
		pEx.col(0).copyTo(col);
	
		for (int i=0; i < idx.cols; ++i)
		{
			col = x_.col(i+1);
			x.col(idx(0,i)).copyTo(col);
		}
	
		x.create(x_.rows, x_.cols);
		x_.copyTo(x);
		
		cv::Mat_<double> y_(1, y.cols + 1);
		y_(0,0) = 1;
		for (int i=0; i < idx.cols; ++i)
		{
			col = y_.col(i+1);
			y.col(idx(0,i)).copyTo(col);
		}

		y.create(y_.rows, y_.cols);
		y_.copyTo(y);

	}
	
	//Bootstrap // not used?
	for (int i=0; i != y.cols; ++i)
	{
		//measure Relative similarity
		cv::MatExpr r;
	
		cv::Mat_<double> conf1;
		cv::Mat_<double> dummy1;
		cv::Mat_<double> isin;
		
		tldNN(x.col(i), conf1, dummy1, isin); 	
		
		//Positive
		// m_model.thr_nn  0.65
		if ((y(0,i) == 1) && (conf1(0,0) <= m_model_thr_nn))
		{
			
			if (isnan(isin(1,0)))
			{
				m_pex.create(x.rows, 1);
				x.col(i).copyTo(m_pex);
				continue;
			}
			
			
			//add to model
			//
			// a = m_pex(:,1:isin(2));
            // b = x(:,i);
            // c = m_pex(:,isin(2)+1:end);
			// m_pex = [a b c]
	
			cv::Mat_<double> a(x.rows, isin(1, 0) + 1);
			m_pex.colRange(0, isin(1, 0) + 1).copyTo(a);
			
			cv::Mat_<double> b(x.rows, 1);
			Mat col = b.col(0);
			x.col(i).copyTo(col);
			
			cv::Mat_<double> c(0,0);
			if (m_pex.cols >= (isin(1, 0) + 1))
			{
				c.create(x.rows, m_pex.cols - isin(1, 0));
				m_pex.colRange(isin(1, 0), m_pex.cols).copyTo(c);
			}
			
			m_pex.create(x.rows, a.cols + b.cols + c.cols);
			if (a.cols > 0)
			{
				for (int c=0; c < a.cols; c++)
				{
					Mat col = m_pex.col(c);
					a.col(c).copyTo(col);
				}
			}
		
			if (b.cols > 0)
			{
				for (int c=0; c < b.cols; c++)
				{
					Mat col = m_pex.col(c + a.cols);
					b.col(c).copyTo(col);
				}
			}
		
			if (c.cols > 0)
			{
				for (int ci=0; ci < c.cols; ci++)
				{
					Mat col = m_pex.col(ci + a.cols + b.cols);
					c.col(ci).copyTo(col);
				}
			}
			
		}
		
		//Negative
		if ((y(0,i) == 0) && (conf1(0,0) > 0.5))
		{
			Mat col;
			cv::Mat_<double> nex(x.rows, m_nex.cols + 1);
			for (int c=0; c < m_nex.cols; ++c)
			{
				col = nex.col(c);
				m_nex.col(c).copyTo(col);
			}
			col = nex.col(m_nex.cols);
			x.col(i).copyTo(col);
			m_nex.create(nex.rows, nex.cols);
			nex.copyTo(m_nex);
	
		}
	}
	
	//=====test m_pex=====//
	//cv::namedWindow("m_PeX", 1);
	//cv::imshow("m_PeX", m_pex);
	
}

bool any(const cv::Mat_<bool>& mat)
{
	for (int r=0; r < mat.rows; ++r)
		for (int c=0; c < mat.cols; ++c)
			if (mat(r,c)) return true;
	return false;
}


void TargetLock::tldNN(const cv::Mat_<double>& x, cv::Mat_<double>& conf1, cv::Mat_<double>& conf2, cv::Mat_<double>& isin)
{
	//'conf1' ... full model (Relative Similarity)
	//'conf2' ... validated part of model (Conservative Similarity)
	//'isnin' ... inside positive ball, id positive ball, inside negative ball
	
	isin = cv::Mat_<double>(3, x.cols); // second column is NaN flag
	
	for (int i = 0; i < x.cols; i++) {
		isin(0,i) = NaN;
		isin(1,i) = NaN;
		isin(2,i) = NaN;
	}
	
	//if isempty(m_pex) % IF positive examples in the model are not defined THEN everything is negative
	if (m_pex.empty())
	{
		conf1 = cv::Mat_<double>::zeros(1, x.cols);
		conf2 = cv::Mat_<double>::zeros(1, x.cols);
		return;
	}
	
	
	//if isempty(m_nex) % IF negative examples in the model are not defined THEN everything is positive
	if (m_nex.empty())
	{
		conf1 = cv::Mat_<double>::ones(1, x.cols);
		conf2 = cv::Mat_<double>::ones(1, x.cols);
		return;
	}
	
	conf1 = cv::Mat_<double>(1, x.cols, -1); //NaN
	conf2 = cv::Mat_<double>(1, x.cols, -1); //NaN
	
	//for every patch that is tested
	for (int i=0; i < x.cols; ++i) //usually x.cols is 1 isn't?
	{
		//nccP measure NCC to positive examples
		//nccN measure NCC to negative examples
		cv::Mat_<double> nccP;
		cv::Mat_<double> patch;
		x.col(i).copyTo(patch);
		distance(patch, m_pex, 1, nccP);
		cv::Mat_<double> nccN;
		distance(patch, m_nex, 1, nccN);
		
		    
		//set isin
		//IF the query patch is highly correlated with any positive patch in the model THEN it is considered to be one of them;
		cv::Mat e = (Mat) (nccP > m_ncc_thesame);
		if (any(e)) isin(0, i) = 1;
		
		//get the index of the maximal correlated positive patch
		int index = 0;
		double max_nccP;
		maxMat<double>(nccP, index, max_nccP); //from utils.hpp
		isin(1, i) = index;
		
		//IF the query patch is highly correlated with any negative patch in the model THEN it is considered to be one of them
		e = (Mat) (nccN > m_ncc_thesame);
		if (any(e)) isin(2,i) = 1;
		
		//measure Relative Similarity
		double max_nccN;
		maxMat<double>(nccN, index, max_nccN);
		double dN = 1 - max_nccN;
		double dP = 1 - max_nccP;
		conf1(0, i) = dN / (dN + dP);
	
		//measure Conservative Similarity
		double maxP;
		double h = ceil(m_model_valid * m_pex.cols);
		if (h > 1) maxMat<double>(nccP.colRange(0, h - 1), index, maxP);
		else  maxMat<double>(nccP.col(0), index, maxP);
		dP = 1 - maxP;
		conf2(0, i) = dN / (dN + dP);
		
	}
	
}


// correlation
double ccorr(double *f1,double *f2,int numDim)
{

	double f = 0;
	for (int i = 0; i<numDim; i++) {
		f += f1[i]*f2[i];
	}
	return f;
}

// correlation normalized
double ccorr_normed(double *f1, int off1, double *f2, int off2, int numDim)
{
	
	double corr = 0;
	double norm1 = 0;
	double norm2 = 0;
	
	for (int i = 0; i<numDim; i++) {
		corr += f1[i*off1]*f2[i*off2];
		norm1 += f1[i*off1]*f1[i*off1];
		norm2 += f2[i*off2]*f2[i*off2];
	}
	// normalization to <0,1>
	return (corr / sqrt(norm1*norm2) + 1) / 2.0;
}

// euclidean distance
double euclidean(double *f1, int off1, double *f2, int off2, int numDim = 2)
{
	
	double sum = 0;
	for (int i = 0; i < numDim; i++)
		sum += (f1[i*off1]-f2[i*off2])*(f1[i*off1]-f2[i*off2]);
	return sqrt(sum);
}

void TargetLock::distance(const cv::Mat_<double>& x1, const cv::Mat_<double>& x2, int flag, cv::Mat_<double>& resp)
{
	double *x1Data = (double*) x1.data; int N1 = x1.cols; int M1 = x1.rows;
	double *x2Data = (double*) x2.data; int N2 = x2.cols; //int M2 = x2.rows;
	
	
	resp.create(N1, N2);
	resp = cv::Mat_<double>::zeros(N1, N2);
	double *respData = (double *) resp.data;

	
	switch (flag)
	{
		case 1 :
			for (int i = 0; i < N2; i++) {
				for (int ii = 0; ii < N1; ii++) {
					*respData++ = ccorr_normed(x1Data+ii,N1,x2Data+i,N2,M1);
				}
			}
			
			return;
		case 2 :
			
			for (int i = 0; i < N2; i++) {
				for (int ii = 0; ii < N1; ii++) {
					*respData++ = euclidean(x1Data+ii,N1,x2Data+i,N2,M1);
				}
			}
			
			return;
	}
}

template <class T>
bool TargetLock::randperm(int n, cv::Mat_<T>& outMat)
{
	if (n <= 0) return false;
	
	//std::vector<int> tmp;
	outMat.create(1, n);
	
	for (int c=0; c < n; ++c)
		outMat(0, c) = c;
	
	std::random_shuffle(outMat.begin(), outMat.end());
	
	return true;
}


void TargetLock::bbPredict(const cv::Mat_<double>& BB0, const cv::Mat_<double>& pt0, const cv::Mat_<double>& pt1, cv::Mat_<double>& BB1, cv::Mat_<double>& shift)
{

	cv::Mat_<double> of = pt1 - pt0;

	double dx = median<double>(of.row(0));
	double dy = median<double>(of.row(1));

	cv::Mat_<double> d1;
	pdist(pt0, d1);

	cv::Mat_<double> d2;
	pdist(pt1, d2);

	double s;
	cv::Mat_<double> tmp = d2 / d1;
	s = median<double>((vector<double>) tmp);

	double s1 = 0.5*(s-1)*bbWidth(BB0);
	double s2  = 0.5*(s-1)*bbHeight(BB0);

	BB1.create(4,1);
	BB1(0,0) = BB0(0,0) - s1 + dx;
	BB1(1,0) = BB0(1,0) - s2 + dy;
	BB1(2,0) = BB0(2,0) + s1 + dx;
	BB1(3,0) = BB0(3,0) + s2 + dy;
	
	shift.create(2,1);
	shift(0,0) = s1;
	shift(1,0) = s2;

}

void TargetLock::pdist(const cv::Mat_<double>& inMat, cv::Mat_<double>& outMat)
{
	
	double* inData = (double*) inMat.data;
	outMat.create(1, inMat.cols*(inMat.cols-1)/2);
	double *outData = (double*) outMat.data;
	for (int c=0; c < inMat.cols-1; ++c)
	{
		for (int c1=c+1; c1 < inMat.cols; ++c1)
		{
			*outData++ = euclidean(inData+c, inMat.cols, inData+c1, inMat.cols);
		}
	}
}




template<class T>
double TargetLock::median(std::vector<T> v)
{
	if (v.empty()) return NaN;
	
	double med;
	size_t size = v.size();
	std::sort(v.begin(), v.end());
	if (size  % 2 == 0)
		med = (v[size / 2 - 1] + v[size / 2]) / 2;
	else 
		med = v[size / 2];
	return med;
}

template<typename T>
double TargetLock::median2(std::vector<T> v)
{
	double med;
	std::vector<T> v1;

	typename std::vector<T>::iterator it;
	
	for (it=v.begin(); it != v.end(); ++it)
	{
		if (!isnan(*it))
			v1.push_back(*it);
	}
	if (!v1.empty()) med = median<T>(v1);
	else med = NaN;
	return med;
}


bool TargetLock::myIsNaN(double x)
{
	return x != x;
}

template<class T>
void TargetLock::isFinite(const cv::Mat_<T>& bb, cv::Mat_<int>& outMat)
{
	outMat.create(bb.rows, bb.cols);
	outMat = cv::Mat_<bool>::ones(bb.rows, bb.cols);
	for (int r=0; r < bb.rows; ++r)
		for (int c=0; c < bb.cols; ++c)
			if (isnan(bb(r,c)) || (bb(r,c) == inf) || (bb(r,c) == -inf))
				outMat(r,c) = 0;
	
}

template<class T>
bool TargetLock::bbIsDef(const cv::Mat_<T>& bb)
{
	// Info
	cv::Mat_<int> outMat;
	isFinite<T>(bb.row(1), outMat);
	for (int r=0; r < outMat.rows; ++r)
		for (int c=0; c < outMat.cols; ++c)
			if (outMat(r,c) == 0)
				return false;
	return true;
}

template<class T>
bool TargetLock::bbIsOut(const cv::Mat_<T>& bb, const cv::Mat_<T>& imsize)
{
	
	bool idx_out = ((bb(0,0) > imsize(0,1)) || (bb(1,0) > imsize(0,0)) ||
					(bb(2,0) < 1) || (bb(3,0) < 1));
	return idx_out;
}


void TargetLock::bbPoints(const cv::Mat_<double>& bb, double numM, double numN, double margin, cv::Mat_<double>& pt)
{

	//Generates numM x numN points on BBox.
	
	
	cv::Mat_<double> BB(4,1);
	BB(0,0) = bb(0,0) + margin;
	BB(1,0) = bb(1,0) + margin;
	BB(2,0) = bb(2,0) - margin;
	BB(3,0) = bb(3,0) - margin;
	
	if ((numM == 1) && (numN == 1))
	{
		bbCenter(BB, pt);
		return;
	}
	

	if ((numM == 1) && (numN > 1))
	{
		cv::Mat_<double> c;
		bbCenter(BB, c);
		int stepW = (BB(2,0) - BB(0,0)) / (numN - 1);
		std::vector<double> v;
		colon<double>(BB(0,0), stepW, BB(2,0), v);
		cv::Mat_<double> tmp(v, true);
		tmp = tmp.t();
		
		cv::Mat_<double> c_tmp = cv::Mat_<double>::zeros(1, c.cols);
		c_tmp += c.row(1);
		ntuples<cv::Mat_<double> >(tmp, c_tmp, pt);
		
		return;
	}
	
	
	if ((numM > 1) && (numN == 1))
	{
		cv::Mat_<double> c;
		bbCenter(BB, c);
		int stepH = (BB(3,0) - BB(1,0)) / (numM - 1);
		std::vector<double> v;
		colon<double>(BB(1,0), stepH, BB(3,0), v);
		cv::Mat_<double> tmp(v, true);
		tmp = tmp.t();
		cv::Mat_<double> c_tmp = cv::Mat_<double>::zeros(1, c.cols);
		c_tmp += c.row(0);
		ntuples<cv::Mat_<double> >(c_tmp, tmp, pt);
	
		return;
	}
	
	
	double stepW = (BB(2,0) - BB(0,0)) / (numN-1);
	double stepH = (BB(3,0) - BB(1,0)) / (numM-1);
	cv::Mat_<double> vwMat;
	colon<double>(BB(0,0), stepW, BB(2,0), vwMat);
	cv::Mat_<double> vhMat;
	colon<double>(BB(1,0), stepH, BB(3,0), vhMat);
	ntuples<cv::Mat_<double> >(vwMat, vhMat, pt);
}


void TargetLock::var(const cv::Mat_<double>& inMat, cv::Mat_<double>& outMat)
{
	
	outMat.create(1, inMat.cols);
	
	for (int c=0; c < inMat.cols; ++c)
	{
		double val = 0;
		double sum = 0;
		double mean = 0;
		double sum_squared = 0;
		double variance = 0;
		if (inMat.rows > 1)
		{
			for (int r=0; r < inMat.rows; ++r)
			{
				val = inMat(r,c);
				sum = sum + val;
				sum_squared = sum_squared + (val * val);
			}
			mean = sum / inMat.rows;
			variance = (sum_squared - (sum * mean)) / (inMat.rows - 1);
		}
		outMat(0, c) = variance;
	}
}

double TargetLock::bbHeight(const cv::Mat_<double>& bb)
{
	//Info
	return bb(3,0) - bb(1,0) + 1;
}

double TargetLock::bbWidth(const cv::Mat_<double>& bb)
{
	double w = bb(2,0) - bb(0,0) + 1;
	return w;
}

void TargetLock::bbSize(const cv::Mat_<double>& bb, cv::Mat_<double>& s)
{
	s = (cv::Mat_<double>(2, 1) << bb(3,0)-bb(1,0)+1, bb(2,0)-bb(0,0)+1);
}

void TargetLock::bbCenter(const cv::Mat_<double>& bb, cv::Mat_<double>& center)
{
	//Info
	if (bb.empty())
	{
		center.create(0,0);
		return;
	}
	
	center.create(2, bb.cols);
	center.row(0) = 0.5 * (bb.row(0) + bb.row(2));
	center.row(1) = 0.5 * (bb.row(1) + bb.row(3));
}



void TargetLock::bbHull(const cv::Mat_<double>& bb0, cv::Mat_<double>& hull)
{

	hull.create(4,1);
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(bb0.row(0), &minVal, &maxVal, &minLoc, &maxLoc);
	hull(0, 0) = minVal;
	cv::minMaxLoc(bb0.row(1), &minVal, &maxVal, &minLoc, &maxLoc);
	hull(1, 0) = minVal;
	cv::minMaxLoc(bb0.row(2), &minVal, &maxVal, &minLoc, &maxLoc);
	hull(2, 0) = maxVal;
	cv::minMaxLoc(bb0.row(3), &minVal, &maxVal, &minLoc, &maxLoc);
	hull(3, 0) = maxVal;
}
