/*
 *  qrectf.cpp
 *  TargetLock
 *
 *  Created by Muhamad Hisham Wahab on 7/9/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "qrectf.h"

QRectF QRectF::operator&(const QRectF &r) const
{
    double l1 = xp;
    double r1 = xp;
    if (w < 0)
        l1 += w;
    else
        r1 += w;
    if (l1 == r1) // null rect
        return QRectF();
	
    double l2 = r.xp;
    double r2 = r.xp;
    if (r.w < 0)
        l2 += r.w;
    else
        r2 += r.w;
    if (l2 == r2) // null rect
        return QRectF();
	
    if (l1 >= r2 || l2 >= r1)
        return QRectF();
	
    double t1 = yp;
    double b1 = yp;
    if (h < 0)
        t1 += h;
    else
        b1 += h;
    if (t1 == b1) // null rect
        return QRectF();
	
    double t2 = r.yp;
    double b2 = r.yp;
    if (r.h < 0)
        t2 += r.h;
    else
        b2 += r.h;
    if (t2 == b2) // null rect
        return QRectF();
	
    if (t1 >= b2 || t2 >= b1)
        return QRectF();
	
    QRectF tmp;
    tmp.xp = qMax(l1, l2);
    tmp.yp = qMax(t1, t2);
    tmp.w = qMin(r1, r2) - tmp.xp;
    tmp.h = qMin(b1, b2) - tmp.yp;
    return tmp;
}


QRectF QRectF::operator|(const QRectF &r) const
{
    if (isNull())
        return r;
    if (r.isNull())
        return *this;
	
    double left = xp;
    double right = xp;
    if (w < 0)
        left += w;
    else
        right += w;
	
    if (r.w < 0) {
        left = qMin(left, r.xp + r.w);
        right = qMax(right, r.xp);
    } else {
        left = qMin(left, r.xp);
        right = qMax(right, r.xp + r.w);
    }
	
    double top = yp;
    double bottom = yp;
    if (h < 0)
        top += h;
    else
        bottom += h;
	
    if (r.h < 0) {
        top = qMin(top, r.yp + r.h);
        bottom = qMax(bottom, r.yp);
    } else {
        top = qMin(top, r.yp);
        bottom = qMax(bottom, r.yp + r.h);
    }
	
    return QRectF(left, top, right - left, bottom - top);
}

bool QRectF::intersects(const QRectF &r) const
{
    double l1 = xp;
    double r1 = xp;
    if (w < 0)
        l1 += w;
    else
        r1 += w;
    if (l1 == r1) // null rect
        return false;
	
    double l2 = r.xp;
    double r2 = r.xp;
    if (r.w < 0)
        l2 += r.w;
    else
        r2 += r.w;
    if (l2 == r2) // null rect
        return false;
	
    if (l1 >= r2 || l2 >= r1)
        return false;
	
    double t1 = yp;
    double b1 = yp;
    if (h < 0)
        t1 += h;
    else
        b1 += h;
    if (t1 == b1) // null rect
        return false;
	
    double t2 = r.yp;
    double b2 = r.yp;
    if (r.h < 0)
        t2 += r.h;
    else
        b2 += r.h;
    if (t2 == b2) // null rect
        return false;
	
    if (t1 >= b2 || t2 >= b1)
        return false;
	
    return true;
}