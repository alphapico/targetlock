/*
 *  cvTimer.h
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/8/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef CV_TIMER_H_
#define CV_TIMER_H_

#include "targetLock.h"

using namespace cv;

class cv_timer
{
	
public:
	cv_timer() : m_time(0)
	{  m_freq = 1e6 * cvGetTickFrequency(); }
	
	void start( bool a_reset=true )
	{ int64 l_tmp = cvGetTickCount(); if(a_reset) m_time=0; m_start=l_tmp; }
	
	void stop(void)
	{ int64  l_tmp = cvGetTickCount(); m_stop=l_tmp; m_time+=m_stop-m_start; }
	
	double get_time()
	{return(double)m_time/(double)m_freq;};
	
	double get_fps() 
	{return(double)m_freq/(double)m_time;};
	
	long get_clocks()
	{return m_time;};
	
private:
	double  m_freq;
	double  m_start, m_stop, m_time;
	
};

#define TIME(text,expr,l_timer) l_timer.start(); expr; l_timer.stop(); printf("%s %.7f sec\n",text,l_timer.get_time());
#define FPS(text,expr,l_timer) l_timer.start(); expr; l_timer.stop(); printf("%s %.2f fps\n",text,l_timer.get_fps());

#endif

