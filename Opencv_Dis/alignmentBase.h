#ifndef _ALIGNMENTBASE_H_
#define _ALIGNMENTBASE_H_
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>

using namespace std;
using namespace cv;

class alignmentBase
{
public:
	alignmentBase();
	~alignmentBase();
	virtual void calc(Mat &ref, Mat &tar, Mat &flow);
	virtual void setParam();
	virtual void meshgrid(int width, int height, float *grid_X, float *grid_Y);//matlab meshgrid
	virtual void warpRGB(Mat &ref,Mat &base, Mat &registerdBase, Mat &flowX,Mat &flowY);
	virtual void checkConsistency(const Mat &ref, const Mat &base, const Mat &refFlow, const Mat &baseFlow, int height, int width, int eps,Mat &consMap);
	virtual void makeCopyBorder(const Mat &src, Mat &dst, int borderSize);
private:

};





#endif // !_ALIGNMENTBASE_H_
