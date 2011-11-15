/*
 *  fern.cpp
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/11/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */



#include "utils.hpp"
#include "fern.h"

#include <stdio.h>
#include <math.h>
#include <map>
#include <set>


using namespace std;

//#define sub2idx(row,col,height) ((int) (floor((row)+0.5) + floor((col)+0.5)*(height)))
#define sub2idx(row,col,width) ((int) (floor((col)+0.5) + floor((row)+0.5)*(width)))

Fern::Fern()
: BBOX(0)
, OFF(NULL)
, IIMG(0)
, IIMG2(0)
, BBOX_STEP(7)
, nBIT(1)
{
}

Fern::~Fern()
{
	reset();
}

void Fern::reset()
{
	delete [] IIMG; IIMG=0;
	delete [] IIMG2; IIMG2=0;
	delete [] BBOX; BBOX=0;
	delete [] OFF; OFF=0;
	WEIGHT.clear();
	nP.clear();
	nN.clear();
}

void Fern::printWeight()
{
	for (int i = 0; i<nTREES; i++) {
		stringstream text;
		text <<  "Weight/Weight_" << i;
		
		printVector(WEIGHT[i], text.str());
	}
	
}

bool Fern::init(const IplImage& image, const Mat_<double>& grid, const Mat_<double>& features, const Mat_<double>& scales)
{
	if (BBOX!=0) return false;
	
	m_inc = 0; // just to globally incerment to debug fern->getPattern
	
	iHEIGHT    = image.height; // 240
	iWIDTH     = image.width;  // 320
	nTREES     = features.cols;  // 10
	nFEAT      = features.rows / 4; // 13 // feature has 2 points: x1,y1,x2,y2
	thrN       = 0.5 * nTREES; // 5
	nSCALE     = scales.cols; // 12
	
	IIMG       = new double[iHEIGHT*iWIDTH*sizeof(double)];
	IIMG2      = new double[iHEIGHT*iWIDTH*sizeof(double)];
	
	// BBOX
	mBBOX      = grid.rows; // 6
	nBBOX      = grid.cols; // 22634
	BBOX	   = create_offsets_bbox((double*)grid.data); // 7 x nBBOX
	//cv::Mat_<int> matBBOX(7, nBBOX, BBOX);
	//cv::Mat_<int> matBBOXT = matBBOX.t();;
	//printMatrix(matBBOXT, "matBBOXT");
	//printMatrix(matBBOX, "matBBOX");
	double *x  = (double *)features.data;
	double *s  = (double *)scales.data;
	OFF		   = create_offsets(s,x);
	//cv::Mat_<int> matOFF(2, nSCALE*nTREES*nFEAT, OFF);
	//printMatrix(matOFF, "matOFF");
	//cv::Mat_<int> matOFFT = matOFF.t();
	//printMatrix(matOFFT, "matOFFT");
	
	for (int i = 0; i<nTREES; i++) {
		WEIGHT.push_back(vector<double>((int)pow(2.0,nBIT*nFEAT), 0));
		nP.push_back(vector<int>((int)pow(2.0,nBIT*nFEAT), 0));
		nN.push_back(vector<int>((int)pow(2.0,nBIT*nFEAT), 0));
	}
	
	for (int i = 0; i<nTREES; i++) {
		for (int j = 0; j < WEIGHT[i].size(); j++) {
			WEIGHT[i].at(j) = 0;
			nP[i].at(j) = 0;
			nN[i].at(j) = 0;
		}
	}
	
	return true;
	
}

int* Fern::create_offsets(double *scale0, double *x0)
{
	int *offsets = new int[nSCALE*nTREES*nFEAT*2*sizeof(int)];
	int *off = offsets;
	
	// assumes offsets is 2 rows
	double *scale = scale0;
	for (int k = 0; k < nSCALE; k++){
		for (int i = 0; i < nTREES; i++) {
			double *x  = x0 + i;
			for (int j = 0; j < nFEAT; j++) {
				/* TL-unchanged
				*off = sub2idx((scale[k]-1)*x[nTREES],(scale[k+nSCALE]-1)*x[0],iHEIGHT);
				*(off + (nSCALE*nTREES*nFEAT)) = sub2idx((scale[k]-1)*x[3*nTREES],(scale[k+nSCALE]-1)*x[2*nTREES],iHEIGHT);
				off++;
				x = x + 4 * nTREES;
				 */
				*off++ = sub2idx((scale[k]-1)*x[nTREES],(scale[k+nSCALE]-1)*x[0],iWIDTH);
				*off++ = sub2idx((scale[k]-1)*x[3*nTREES],(scale[k+nSCALE]-1)*x[2*nTREES],iWIDTH);
				x = x + 4 * nTREES;
			}
		}
	}
	//printMemoryBlock<int>(offsets, nSCALE*nTREES*nFEAT*2, "offsets");
	return offsets;
}

int* Fern::create_offsets_bbox(double *bb0)
{
	int *offsets = new int[BBOX_STEP*nBBOX*sizeof(int)]; // 7 x 22634
	int *off = offsets;
	
	for (int i = 0; i < nBBOX; i++)
	{
		/* TL-unchanged
		*off = sub2idx(bb0[i+1*nBBOX]-1,bb0[i+0*nBBOX]-1, iHEIGHT);
		*(off + nBBOX) = sub2idx(bb0[i+3*nBBOX]-1,bb0[i+0*nBBOX]-1, iHEIGHT);
		*(off + 2*nBBOX) = sub2idx(bb0[i+1*nBBOX]-1,bb0[i+2*nBBOX]-1, iHEIGHT);
		*(off + 3*nBBOX) = sub2idx(bb0[i+3*nBBOX]-1,bb0[i+2*nBBOX]-1, iHEIGHT);
		*(off + 4*nBBOX) = (int) ((bb0[i+2*nBBOX]-bb0[i+0*nBBOX])*(bb0[i+3*nBBOX]-bb0[i+1*nBBOX]));
		*(off + 5*nBBOX) = (int) (bb0[i+4*nBBOX]-1)*2*nFEAT*nTREES; // pointer to features for this scale
		*(off + 6*nBBOX) = (int) bb0[i+5*nBBOX]; // number of left-right bboxes, will be used for searching neighbours
		off++;
		 */
		*off++ = sub2idx(bb0[i+1*nBBOX]-1,bb0[i+0*nBBOX]-1, iWIDTH);
		*off++ = sub2idx(bb0[i+3*nBBOX]-1,bb0[i+0*nBBOX]-1, iWIDTH);
		*off++ = sub2idx(bb0[i+1*nBBOX]-1,bb0[i+2*nBBOX]-1, iWIDTH);
		*off++= sub2idx(bb0[i+3*nBBOX]-1,bb0[i+2*nBBOX]-1, iWIDTH);
		*off++ = (int) ((bb0[i+2*nBBOX]-bb0[i+0*nBBOX])*(bb0[i+3*nBBOX]-bb0[i+1*nBBOX]));
		*off++ = (int) (bb0[i+4*nBBOX]-1)*2*nFEAT*nTREES; // pointer to features for this scale
		*off++ = (int) bb0[i+5*nBBOX]; // number of left-right bboxes, will be used for searching neighbours
	}
	return offsets;
}



void Fern::getPatterns(const ImagePairType& input_, const cv::Mat_<unsigned>& idx_, double var, cv::Mat_<double>& patt_, cv::Mat_<double>& status_)
{
	// image
	unsigned char *input = (unsigned char *) input_.first->data;
	unsigned char *blur  = (unsigned char *) input_.second->data;
	
	/*
	m_inc++; // just to globally incerment to debug fern->getPattern
	
	std::stringstream str_fps;
	str_fps << "Pattern"<< m_inc ;
	namedWindow(str_fps.str(), 1);
	imshow(str_fps.str(), *input_.second);
	*/
	
	// bbox indexes
	//printMatrix(idx_, "idx_");
	unsigned *idx = (unsigned *) idx_.data; //I make value unsigned and double (pair), but sort using unsigned
	int numIdx = idx_.rows * idx_.cols;
	
	// minimal variance
	double minVar = var;
	if (minVar > 0) {
		iimg(input, IIMG, iHEIGHT, iWIDTH);
		iimg2(input, IIMG2, iHEIGHT, iWIDTH);
		//cv::Mat_<unsigned int> i(iHEIGHT, iWIDTH, input);
		//printMatrix(i, "1");
		//cv::Mat_<double> ii2(iHEIGHT, iWIDTH, IIMG2);
		//printMatrix(ii2, "ii2");
	}
	
	// output patterns
	patt_.create(nTREES, numIdx);
	double *patt = (double *) patt_.data;
	status_.create(1, numIdx);
	status_ = cv::Mat_<double>::zeros(1, numIdx);
	double *status = (double *) status_.data;
	
	for (int j = 0; j < numIdx; j++)
	{
		if (minVar > 0) {
			double bboxvar = bbox_var_offset(IIMG,IIMG2,BBOX+j*BBOX_STEP); //previously BBOX + j only
			if (bboxvar < minVar) continue;
		}
		status[j] = 1;
		double *tPatt = patt + j;
		for (int i = 0; i < nTREES; i++) {
			//std::cout<<"idx: "<<idx[j]<<std::endl;
			tPatt[i*numIdx] = (double) measure_tree_offset(blur, (unsigned) idx[j], i);//better double than unsigned
		}
	}
	return;
}

//==========test Correctness===========//
void Fern::getPatterns(const ImagePairType& input_, const cv::Mat_<double>& grid, const cv::Mat_<unsigned>& idx_, double var, cv::Mat_<double>& patt_, cv::Mat_<double>& status_)
{
	// image
	unsigned char *input = (unsigned char *) input_.first->data;
	unsigned char *blur  = (unsigned char *) input_.second->data;
	
	unsigned *idx = (unsigned *) idx_.data; //I make value unsigned and double (pair), but sort using unsigned
	int numIdx = idx_.rows * idx_.cols;
	
	// minimal variance
	double minVar = var;
	if (minVar > 0) {
		iimg(input, IIMG, iHEIGHT, iWIDTH);
		iimg2(input, IIMG2, iHEIGHT, iWIDTH);
	}
	
	// output patterns
	patt_.create(nTREES, numIdx);
	double *patt = (double *) patt_.data;
	status_.create(1, numIdx);
	status_ = cv::Mat_<double>::zeros(1, numIdx);
	double *status = (double *) status_.data;
	
	for (int j = 0; j < numIdx; j++)
	{
		if (minVar > 0) {
			double bboxvar = bbox_var_offset(IIMG,IIMG2,BBOX+j*BBOX_STEP);
			if (bboxvar < minVar) continue;
		}
		
		status[j] = 1;
		double *tPatt = patt + j;
		for (int i = 0; i < nTREES; i++) {
			//std::cout<<"idx: "<<idx[j]<<std::endl;
			tPatt[i*numIdx] = (double) measure_tree_offset(*input_.second, grid, (unsigned) idx[j], i);//better double than unsigned
		}
	}
	return;
	
}
//==========end of test Correctness===========//

void Fern::update(double *x, int C, int N, int offset)
{
	for (int i = 0; i < nTREES; i++) {
		
		int idx = (int) x[i*offset];
		
		(C==1) ? nP[i][idx] += N : nN[i][idx] += N;
		
		if (nP[i][idx]==0)
		{
			WEIGHT[i][idx] = 0;
		}
		else
		{
			WEIGHT[i][idx] = ((double) (nP[i][idx])) / (nP[i][idx] + nN[i][idx]);
		}
	}
}

void Fern::update(const cv::Mat_<double>& x, const cv::Mat_<double>& y, double thr_fern, int bootstrap, const cv::Mat_<double>* idx_)
{
	std::cout<<"update Fern->\n" ;
	
	double* X = (double *) x.data;
	int numX = x.cols;
	double* Y = (double *) y.data;
	double thrP = thr_fern * nTREES;
	int step = numX / 10;
	
	if (idx_ == NULL)
	{
		for (int j = 0; j < bootstrap; j++) {
			for (int i = 0; i < step; i++) {
				for (int k = 0; k < 10; k++) {
					
					int I = k*step + i;
					double *xi = X+I;
					if (Y[I] == 1) {
						if (measure_forest(xi, numX) <= thrP)
							update(xi,1,1,x.cols);
					} else {
						if (measure_forest(xi, numX) >= thrN)
							update(xi,0,1,x.cols);
					}
				}
			}
		}
	}
	else
	{
		double *idx = (double *) idx_->data;
		int nIdx = idx_->rows * idx_->cols;
		
		for (int j = 0; j < bootstrap; j++) {
			for (int i = 0; i < nIdx; i++) {
				int I = idx[i]-1;
				double *xi = X+I;
				if (Y[I] == 1) {
					if (measure_forest(xi, numX) <= thrP)
						update(xi,1,1,x.cols);
				} else {
					if (measure_forest(xi, numX) >= thrN)
						update(xi,0,1,x.cols);
				}
			}
		}
	}
}


void Fern::detect(const ImagePairType& img, double probability, double minVar, const cv::Mat_<double>& conf, const cv::Mat_<double>& patt)
{
	std::cout<<"detect Fern->\n" ;
	// Pointer to preallocated output matrixes
	double *confData = (double *) conf.data; 
	if (conf.cols != nBBOX) 
	{ 
		std::cout << "Fern::detect: Wrong input\n"; 
		return; 
	}
	double *pattData = (double *) patt.data; 
	
	if (patt.cols != nBBOX) 
	{ 
		std::cout << "Fern::detect: Wrong input\n"; 
		return; 
	}
	for (int i = 0; i < nBBOX; i++) { confData[i] = -1; }
	
	double nTest  = nBBOX * probability; 
	if (nTest <= 0) return;
	if (nTest > nBBOX) nTest = nBBOX;
	double pStep  = (double) nBBOX / nTest;
	double pState = randdouble() * pStep;
	
	// Input images
	unsigned char *input = (unsigned char *) img.first->data;
	unsigned char *blur  = (unsigned char *) img.second->data;
	
	// Integral images
	iimg(input, IIMG, iHEIGHT, iWIDTH);
	iimg2(input, IIMG2, iHEIGHT, iWIDTH);
	

	int I = 0;
	//int K = 2;
	
	while (1) 
	{
		// Get index of bbox
		I = (int) floor(pState);
		pState += pStep;
		if (pState >= nBBOX) { break; }
		
		// measure bbox
		double *tPatt = pattData + I;
		confData[I] = measure_bbox_offset(blur,I,minVar,tPatt);
		
	}

	return;
}

void Fern::evaluate(const cv::Mat_<double>& X, cv::Mat_<double>& resp0)
{
	std::cout<<"evaluate Fern->\n" ;
	
	int numX = X.cols;
	double* xData = (double *) X.data;
	resp0 = cv::Mat_<double>::zeros(1, numX);
	
	double* resp0Data = (double*) resp0.data;
	for (int i = 0; i < numX; i++) {
		*resp0Data++ = measure_forest(xData+i, numX);
	}
}


double Fern::measure_bbox_offset(unsigned char *blur, int idx_bbox, double minVar, double *tPatt)
{
	double conf = 0.0;
	double bboxvar = bbox_var_offset(IIMG,IIMG2,BBOX+idx_bbox);
	if (bboxvar < minVar) {	return conf; }
	
	for (int i = 0; i < nTREES; i++) { 
		int idx = measure_tree_offset(blur,idx_bbox,i);
		tPatt[i*nBBOX] = idx;
		conf += WEIGHT[i][idx];
	}
	return conf;
}

double Fern::measure_forest(double *idx, int offset)
{
	double votes = 0;
	for (int i = 0; i < nTREES; i++)
	{
		//std::cout<< " " <<i*offset <<" ";
		votes += WEIGHT[i][idx[i*offset]]; //unsigned? i put both type data - unsigned and double (pair)
	}
	//std::cout<<"\n";
	return votes;
}


int Fern::measure_tree_offset(unsigned char *img, int idx_bbox, int idx_tree)
{
	int index = 0;
	//== TL unchanged
	//int *bbox = BBOX + idx_bbox;
	//int *off = OFF + (bbox[5*nBBOX]/2) + idx_tree*nFEAT;
	 
	int *bbox = BBOX + idx_bbox*BBOX_STEP;
	int *off = OFF + bbox[5] + idx_tree*2*nFEAT;
	for (int i=0; i<nFEAT; i++) {
		index<<=1;
 
		//== TL unchanged
		//int idx1 = row2col(off[0]+bbox[0]);
		//int idx2 = row2col(off[nSCALE*nTREES*nFEAT]+bbox[0]);
		//int fp0 = img[idx1];
		//int fp1 = img[idx2];
		 
		int fp0 = img[off[0]+bbox[0]];
		int fp1 = img[off[1]+bbox[0]];
		if (fp0>fp1) { index |= 1;}
		//== TL unchanged
		//off++;
		 
		off += 2;
	}
	return index;	
}

//==========test Correctness===========//
int Fern::measure_tree_offset(const Mat& image, const cv::Mat_<double>& bb_out, int idx_bbox, int idx_tree)
{
	Mat disp_img;
	Mat_<uchar> disp_img2;
	disp_img = image;
	disp_img2.create(iHEIGHT,iWIDTH);
	//cvtColor(image, disp_img, CV_BGR2GRAY);
	//unsigned char *img = disp_img.data;
	unsigned char *img = image.data;
	
	namedWindow("fdTest", 1);
	
	int index = 0;
	
	int *bbox = BBOX + idx_bbox*BBOX_STEP;
	int *off = OFF + bbox[5] + idx_tree*2*nFEAT;
	for (int i=0; i<nFEAT; i++) {
		index<<=1;
		int fp0 = img[off[0]+bbox[0]];
		int fp1 = img[off[1]+bbox[0]];
		if (fp0>fp1) { index |= 1;}
		
		//plot star
		Mat_<uchar> disp_img3 = disp_img.clone();
		uchar* ptr_img = disp_img3.data;
		
		int off_one = off[0]+bbox[0];
		int off_two = off[1]+bbox[0];
		
		ptr_img[off_one ] = (uchar)255;
		ptr_img[off_one  + 1] = (uchar)255;
		if(off_one - 1 > 0 && off_one - 1 < iWIDTH*iHEIGHT )
			ptr_img[off_one  - 1] = (uchar)255;
		if(off_one + iWIDTH > 0 && off_one + iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_one  + iWIDTH] = (uchar)255;
		if(off_one - iWIDTH > 0 && off_one - iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_one  - iWIDTH] = (uchar)255;
		ptr_img[off_one  + 2] = (uchar)255;
		if(off_one - 2 > 0 && off_one - 2 < iWIDTH*iHEIGHT )
			ptr_img[off_one  - 2] = (uchar)255;
		if(off_one + 2*iWIDTH > 0 && off_one + 2*iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_one  + 2*iWIDTH] = (uchar)255;
		if(off_one - 2*iWIDTH > 0 && off_one - 2*iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_one - 2*iWIDTH] = (uchar)255;
		
		
		ptr_img[off_two ] = (uchar)150;
		ptr_img[off_two + 1] = (uchar)150;
		if(off_two - 1 > 0 && off_two - 1 < iWIDTH*iHEIGHT )
			ptr_img[off_two - 1] = (uchar)150;
		if(off_two + iWIDTH > 0 && off_two + iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_two + iWIDTH] = (uchar)150;
		if(off_two - iWIDTH > 0 && off_two - iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_two - iWIDTH] = (uchar)150;
		ptr_img[off_two + 2] = (uchar)150;
		if(off_two - 2 > 0 && off_two - 2 < iWIDTH*iHEIGHT )
			ptr_img[off_two - 2] = (uchar)150;
		if(off_two + 2*iWIDTH > 0 && off_two + 2*iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_two + 2*iWIDTH] = (uchar)150;
		if(off_two - 2*iWIDTH > 0 && off_two - 2*iWIDTH < iWIDTH*iHEIGHT )
			ptr_img[off_two - 2*iWIDTH] = (uchar)150;
		
		for (int j = 0; j < iHEIGHT; j++) {
			for (int k = 0; k < iWIDTH; k++) {
				disp_img3(j,k) = ptr_img[k+j*iWIDTH];
			}
		}
		
		cv::rectangle(disp_img3, Point(bb_out(idx_bbox,0), bb_out(idx_bbox,1)), Point(bb_out(idx_bbox,2), bb_out(idx_bbox,3)), Scalar(0,255,255), 1, CV_AA);
		
		imshow("fdTest", disp_img3);
		waitKey(2);
		
		
		off += 2;
	}
	return index;	
}
//==========end of test Correctness===========//

double Fern::bbox_var_offset(double *ii,double *ii2, int *off)
{
	// off[0-3] corners of bbox, off[4] area
	/* TL unchanged
	double mX  = (ii[row2col(off[3*nBBOX])] - ii[row2col(off[2*nBBOX])] - ii[row2col(off[nBBOX])] + ii[row2col(off[0])]) / (double) off[4*nBBOX];
	double mX2 = (ii2[row2col(off[3*nBBOX])] - ii2[row2col(off[2*nBBOX])] - ii2[row2col(off[nBBOX])] + ii2[row2col(off[0])]) / (double) off[4*nBBOX];
	return mX2 - mX*mX;
	 */
	/* hentam kromo
	double mX  = (ii[off[3*nBBOX]] - ii[off[2*nBBOX]] - ii[off[nBBOX]] + ii[off[0]]) / (double) off[4*nBBOX];
	double mX2 = (ii2[off[3*nBBOX]] - ii2[off[2*nBBOX]] - ii2[off[nBBOX]] + ii2[off[0]]) / (double) off[4*nBBOX];
	return mX2 - mX*mX;
	 */
	double mX  = (ii[off[3]] - ii[off[2]] - ii[off[1]] + ii[off[0]]) / (double) off[4];
	double mX2 = (ii2[off[3]] - ii2[off[2]] - ii2[off[1]] + ii2[off[0]]) / (double) off[4];
	return mX2 - mX*mX;
}

/*
int Fern::row2col(int ci)
{
	int ri = floor(((float) ci )/ iHEIGHT)+ ((ci % iHEIGHT) * iWIDTH);
	return ri;
}*/

double Fern::randdouble() 
{ 
	return rand()/(double(RAND_MAX)+1); 
}

void Fern::iimg(unsigned char *in, double *ii, int imH, int imW)
{
	ii[0] = in[0];
	
	for (int x = 1; x < imW; x++) {
		ii[x] = in[x] + ii[x-1];
	}
	
	for (int y = 1, Y = imW, YY=0; y < imH; y++, Y+=imW, YY+=imW)
	{
        // Keep track of the row sum
        double r = 0;
        for (int x = 0; x < imW; x++)
        {
            r += in[Y + x];
            ii[Y + x] = ii[YY + x] + r;
        }
	}
}

void Fern::iimg2(unsigned char *in, double *ii2, int imH, int imW)
{
	ii2[0] = in[0] * in[0];
	
	for (int y = 1; y < imH; y++) {
		ii2[y*imW] = ii2[(y-1)*imW] + in[y*imW]*in[y*imW];
	}
	
	double s;
	for (int x = 1; x < imW; x++) {
		s = 0;
		for (int y = 0; y < imH; y++) {
			s += in[x + y*imW] * in[x + y*imW];
			ii2[x + y*imW] = s + ii2[x - 1 + y*imW];
		}
	}
}