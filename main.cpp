#include "cvTimer.h"
#include "targetLock.h"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//============== Structures ================/
struct MouseEvent {
	MouseEvent() {
		event = -1;
		buttonState = 0;
	}
	Point pt;
	int event;
	int buttonState;
};

MouseEvent mouse;


//============Function Prototype============//
int GetTemplate_frompoints(Mat& img, const char*/*const string&*/ selWinName,
						   vector<Point2f>& maskpt, vector<Point2f>& imgpt);
static void onMouse(int event, int x, int y, int flags, void* userdata);


#define CAMERA_OUTPUT_WINDOW_NAME "TargetLock"

int main (int argc, char * const argv[]) {
    
	
	//Take note at TargetLock::imagePatch where i didn't use m_rng
	
	TargetLock* m_targetLock = new TargetLock;
	QRectF m_selection;
	IplImage *m_frame;
	
	CvCapture *camCapture;
	int OneFrameProcess = 0;
    int key;
	bool setTargetLock = false;
	

    //cv::namedWindow("TargetLockTest", 1); //<-- Mouse selection point resize test
	//cv::namedWindow("Debug", 1); // in predator::init() and processFrames()
	//cv::namedWindow("Debug2", 1);
	
	if (!(camCapture = cvCaptureFromCAM(CV_CAP_ANY))) {
        cout << "Failed to capture from camera" << endl;
    }
	//cap = cvCaptureFromFile("Vid_I_person_crossing.avi");
    
    cv::namedWindow(CAMERA_OUTPUT_WINDOW_NAME, CV_WINDOW_AUTOSIZE); // <----imshow operate at targetLock::display()
    
    IplImage *capture_img;
    
    while (true) {
        if ((capture_img = cvQueryFrame(camCapture))) {
            break;
        }
        
        // Bug fixes for opencv 2.6.1, you need to delay it to 60ms
        if(cvWaitKey(60) != -1) {
            cout << "Input" << endl;
            break;
        }
    }
	

	
	//cvNamedWindow("BobotTest", 1);
	//cvRectangle(capture_img, cvPoint(0.337266*360, 0.193971*240), cvPoint((0.337266+0.174041)*360, (0.193971+0.650066)*240), cvScalar(0, 255, 255), 2, CV_AA);
	//cvShowImage("BobotTest", capture_img);
	
	//http://opencv-users.1802565.n2.nabble.com/question-about-cvResize-td2772178.html
	//http://stackoverflow.com/questions/4770031/how-do-i-use-cvresize-resize-correctly
	
	//if 320x240 capture size
	//m_frame = cvCreateImage(cvSize(capture_img->width, capture_img->height), capture_img->depth, capture_img->nChannels);
	//if 640x480 capture size
	m_frame = cvCreateImage(cvSize(capture_img->width/2, capture_img->height/2), capture_img->depth, capture_img->nChannels);
	cvResize(capture_img, m_frame, CV_INTER_LANCZOS4);
	
	//========experiment purposed ===========//
	cv_timer FPS_timer;
	/*
	Mat_<float> m_timerExp;
	m_timerExp.create(2000, 2);
	 */
	int num_frameRun = 0;
    
	
	//=======================================//
	
	for (int fr = 0; camCapture ; fr++)
	{
        
        // Bug fixes for opencv 2.6.1 , you need to delay it to 60ms
        key = cvWaitKey(OneFrameProcess?0:60);
        
        capture_img = cvQueryFrame(camCapture);
		if (capture_img == NULL)
        {
            cout << "Failed to query frame" << endl;
            break;
        }
		
		// key 'r' = select region
		// key 's' = stop learning
		// key 'b' = begin learning
		
		if(key!=-1) //you have to make OneFrameProcess = 1 back after enter 'r' or else key always = -1 
        {
			//the trick here is if u did'nt put 'r' as key, the program will halt
            OneFrameProcess = 1;
            if(key=='r' || key == 's' || key == 'b')
			{
				OneFrameProcess = 0;
				m_targetLock->m_key = key;
				//cout << "mkey-main: "<< m_targetLock->m_key << endl;
			}
        }
		
		cvResize(capture_img, m_frame, CV_INTER_LANCZOS4);
		//m_frame = capture_img;
		
		//cvFlip(m_frame, NULL, 1);
		
		//cvNamedWindow("ResizeTest", 1);
		//cvShowImage("ResizeTest", m_frame);
		
		Mat img_mat = cvarrToMat(m_frame);
		
		//If press KEY
		if (key == 'r' && !setTargetLock) 
		{
			Mat img_resize;
	
			cv::resize(img_mat, img_resize, Size(),2, 2, INTER_LANCZOS4); //Lanczos4
			
			vector<Point2f> imgpt(4), maskpt(4);
			GetTemplate_frompoints(img_resize, CAMERA_OUTPUT_WINDOW_NAME, maskpt, imgpt); //img_mat
			
			if (!imgpt.empty())
			{
				
				m_selection.xp = imgpt[0].x/2.0; //Left
				m_selection.yp = imgpt[0].y/2.0; //Top
				m_selection.w = imgpt[1].x/2.0 - imgpt[0].x/2.0; // Width = Right - Left
				m_selection.h = imgpt[1].y/2.0 - imgpt[0].y/2.0; // Height = Bottom - Top
				
				//cout << m_selection.xp/320 << endl;
				//cout << m_selection.yp/240 << endl;
				//cout << m_selection.w/320 << endl;
				//cout << m_selection.h/240 << endl;
				
				//============Only for experiment sydney purposed=============//
				/*
				m_selection.xp = 0.101278*320.0;
				m_selection.yp = 0.134993*240.0;
				m_selection.w = 0.223206*320.0;
				m_selection.h = 0.323722*240.0;
				*/
				//============================================================//
				
				//cv::rectangle(img_mat, Point(m_selection.xp, m_selection.yp), Point(m_selection.w + m_selection.xp , m_selection.h + m_selection.yp), Scalar(0,255,255), 2, CV_AA);
				//cv::imshow("TargetLockTest", img_mat);
				
				//the color img m_frame will be passed and converted to gray in init
				//no need to hold color img for the first time, bcoz no display is made at 1st frame
				if (!m_targetLock->init(*m_frame, m_selection))
				{
					std::cout << "init wrong\n";
				}
				else {
					m_targetLock->display(0);
					setTargetLock = true;
					
				}
				
				
			}
		}
		else 
		{
			if (setTargetLock) {
				
				FPS_timer.start();
				m_targetLock->processFrames(*m_frame); //previous image is DELETED at the end of the process, 
				//don't call tld.img[I-1] (previous image) anymore
				FPS_timer.stop();
				std::stringstream str_fps;
				str_fps << FPS_timer.get_fps() << " fps";
				std::cout<< str_fps.str() << std::endl;
				/*
				m_timerExp(num_frameRun, 0) = num_frameRun;
				m_timerExp(num_frameRun, 1) = FPS_timer.get_fps()+ 8.0;
				 */
				num_frameRun++;
				
			}else {
				Mat img_resize;
				cv::resize(img_mat, img_resize, Size(),2, 2, INTER_LANCZOS4); //Lanczos4
				cv::imshow(CAMERA_OUTPUT_WINDOW_NAME, img_resize); //img_mat
			}
			
		}

	}
	
	//=======Print WEIGHT=========//
	//m_targetLock->m_fern->printWeight();
	
	//=====For experiment sydney =====//
	
	//printMatrix(m_targetLock->m_result, "m_result");
	//printMatrix(m_targetLock->m_resultConfidence, "m_result");
	//printMatrix(m_timerExp, "m_timer");
	
	//============end================//
	
	
	//=====Release Memory=====//
	
	delete m_targetLock;
    cvReleaseCapture(&camCapture);
		
    return 0;
}


int GetTemplate_frompoints(cv::Mat& img, const char*/*const string&*/ selWinName,
						   std::vector<Point2f>& maskpt, std::vector<Point2f>& imgpt) {
	//setMouseCallback(selWinName, onMouse, &mouse);
	cvSetMouseCallback(selWinName, onMouse, &mouse);
	
	cv::Mat temp;
	
	img.copyTo(temp);
	
	int nobjpt = 0;
	
	std::stringstream out;
	std::string tmp = out.str();
	const char* cstr = tmp.c_str();
	CvFont font;
	double hScale = 0.6;
	double vScale = 0.6;
	int lineWidth = 1;
	cvInitFont(&font, CV_FONT_HERSHEY_DUPLEX | CV_FONT_ITALIC, hScale,
			   vScale, 0, lineWidth);
	
	for (;;) {
		
		if (mouse.event == CV_EVENT_LBUTTONDOWN) {
			maskpt[nobjpt] = mouse.pt;
			//out << (8-nobjpt);
			//out << "points more";
			
			cv::circle(img, maskpt[nobjpt], 5, Scalar(255, 255, 255), -1, CV_AA);
			cv::circle(img, maskpt[nobjpt], 3, Scalar(255, 0, 0), -1, CV_AA);
			nobjpt++;
			//IplImage text_img = img;
			//cvPutText(&text_img, cstr, cvPoint(maskpt[nobjpt].x - 10,maskpt[nobjpt].y - 10), &font, cvScalar(255, 255, 0));
			//img = cvarrToMat(&text_img);
			//cout << nobjpt << endl;
		}
		mouse.event = -1;
		
		cv::imshow(selWinName, img);
		
		int c = waitKey(30);
		
		if (c == 27) {
			nobjpt = 0;
			temp.copyTo(img);
			//imgpt.clear();
		}
		if (nobjpt == 2) {
			cv::line(img, maskpt[0], maskpt[1], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
		}
		if (nobjpt == 3) {
			
			cv::line(img, maskpt[0], maskpt[1], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
			cv::line(img, maskpt[1], maskpt[2], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
		}
		if (nobjpt >= 4) {
			cv::line(img, maskpt[0], maskpt[1], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
			cv::line(img, maskpt[1], maskpt[2], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
			cv::line(img, maskpt[2], maskpt[3], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
			cv::line(img, maskpt[3], maskpt[0], Scalar(0, 0, 255), lineWidth,
				 CV_AA);
			
			float max_x = 0, min_x = img.rows;
			float max_y = 0, min_y = img.cols;
			for (int i = 0; i < maskpt.size(); i++) {
				if (maskpt[i].x >= max_x)
					max_x = maskpt[i].x;
				if (maskpt[i].x <= min_x)
					min_x = maskpt[i].x;
				
				if (maskpt[i].y >= max_y)
					max_y = maskpt[i].y;
				if (maskpt[i].y <= min_y)
					min_y = maskpt[i].y;
			}
			
			imgpt.resize(4);
			imgpt[0].x = min_x;
			imgpt[0].y = min_y;
			imgpt[1].x = max_x;
			imgpt[1].y = max_y;
			
			imgpt[2].x = imgpt[0].x;
			imgpt[2].y = imgpt[1].y;
			imgpt[3].x = imgpt[1].x;
			imgpt[3].y = imgpt[0].y;
			
			cv::line(img, imgpt[0], imgpt[2], Scalar(0, 255, 255), 3, CV_AA);
			cv::line(img, imgpt[2], imgpt[1], Scalar(0, 255, 255), 3, CV_AA);
			cv::line(img, imgpt[1], imgpt[3], Scalar(0, 255, 255), 3, CV_AA);
			cv::line(img, imgpt[3], imgpt[0], Scalar(0, 255, 255), 3, CV_AA);
			
		}
		
		cv::imshow(selWinName, img);
		if (c == '\r' || c == '\n' || c == 10) {
			std::cout << "New template is added" << std::endl;
			break;
		}
		if (c == 'a') {
			std::cout << "Template is not properly visible" << std::endl;
			maskpt.clear();
			imgpt.clear();
			break;
		}
		
	}
	
	temp.copyTo(img);
	return 1;
	
}
// Mouse movement
void onMouse(int event, int x, int y, int flags, void* userdata) {
	if (userdata != NULL) {
		MouseEvent* data = (MouseEvent*) userdata;
		data->event = event;
		data->pt = Point(x, y);
		data->buttonState = flags;
	}
}
