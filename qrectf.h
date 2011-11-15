/*
 *  qrectf.h
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/8/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */



#ifndef QRECTF_H_
#define QRECTF_H_

template <typename T>
const T &qMin(const T &a, const T &b) { if (a < b) return a; return b; }
template <typename T>
const T &qMax(const T &a, const T &b) { if (a < b) return b; return a; }

class QRectF
{
public:
	QRectF() { xp = yp = 0.; w = h = 0.; }
	QRectF(double left, double top, double width, double height);
	
	QRectF operator|(const QRectF &r) const;
    QRectF operator&(const QRectF &r) const;
    QRectF& operator|=(const QRectF &r);
    QRectF& operator&=(const QRectF &r);
	
	QRectF intersect(const QRectF &r) const;
	bool intersects(const QRectF &r) const;
	
	bool isNull() const;
    bool isEmpty() const;
    bool isValid() const;
	
	double width() const;
    double height() const;
	
	inline double x() const;
    inline double y() const;
	
	inline void setLeft(double pos);
    inline void setTop(double pos);
	inline void setX(double pos) { setLeft(pos); }
    inline void setY(double pos) { setTop(pos); }
	void setWidth(double w);
    void setHeight(double h);
	
public:
	//note: qreal type is double for QT
    double xp;
    double yp;
    double w;
    double h;
	

};

/*****************************************************************************
 QRectF inline member functions
 *****************************************************************************/

inline QRectF::QRectF(double aleft, double atop, double awidth, double aheight)
: xp(aleft), yp(atop), w(awidth), h(aheight)
{
}

inline bool QRectF::isNull() const
{ return w == 0. && h == 0.; }

inline bool QRectF::isEmpty() const
{ return w <= 0. || h <= 0.; }

inline bool QRectF::isValid() const
{ return w > 0. && h > 0.; }


inline QRectF& QRectF::operator|=(const QRectF &r)
{
    *this = *this | r;
    return *this;
}

inline QRectF& QRectF::operator&=(const QRectF &r)
{
    *this = *this & r;
    return *this;
}

inline QRectF QRectF::intersect(const QRectF &r) const
{
    return *this & r;
}

inline double QRectF::width() const
{ return w; }

inline double QRectF::height() const
{ return h; }

inline double QRectF::x() const
{ return xp; }

inline double QRectF::y() const
{ return yp; }

inline void QRectF::setLeft(double pos) { double diff = pos - xp; xp += diff; w -= diff; }
inline void QRectF::setTop(double pos) { double diff = pos - yp; yp += diff; h -= diff; }
inline void QRectF::setWidth(double aw)
{ this->w = aw; }

inline void QRectF::setHeight(double ah)
{ this->h = ah; }

#endif //QRECTF_H_