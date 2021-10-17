#ifndef _MUTIHOMOGRAPHY_H_
#define _MUTIHOMOGRAPHY_H_
#include"alignmentBase.h"


enum FeatureMode
{
	InfoDetectMode,
	GradientMode,
	GridSegmentMode
};


class mutiHomography : public alignmentBase
{
public:
	mutiHomography();
	~mutiHomography();

	void calc(Mat &ref,Mat &base,Mat &flow);
	void setParam();
	void getFeaturePointsByPatch(Mat &ref,Mat &base,vector<Point2f> &refPoints,vector<Point2f> &basePoints);


private:
	int gridStep;
	int scalePyr;
	FeatureMode detectionMode;

	Rect getPatchRange(int height, int width, int nBlockCnt,  int num_w, int num_h);
};

mutiHomography::mutiHomography()
{
}

mutiHomography::~mutiHomography()
{
}

void mutiHomography::calc(Mat &ref, Mat &base, Mat &flow)
{

}

void mutiHomography::setParam()
{
	gridStep = 16;
	scalePyr = 5;
}


Rect mutiHomography::getPatchRange(int height,int width,int nBlockCnt,int num_w,int num_h)
{
	int wBlock = width / nBlockCnt;
	int hBlock = height / nBlockCnt;

	int block_x = wBlock*num_w;
	int block_y = hBlock*num_h;

	int block_w = wBlock;
	int block_h = hBlock;

	if (block_x+wBlock>width-1)
	{
		block_w = width - block_x;
	}
	if (block_y+hBlock>height-1)
	{
		block_h = height - block_y;
	}
	return Rect(block_x,block_y,block_w,block_h);

}


//segement image to Patch and get feature points
void mutiHomography::getFeaturePointsByPatch(Mat &ref, Mat &base, vector<Point2f> &refPoints, vector<Point2f> &basePoints)
{
	//不同的获取ref帧块特征点的模式：1、根据信息熵检测特征点最大数值；2、根据梯度同上；3、根据ref网格划分特征点
	//根据1、2两种模式，分别有不同Block获取权重的函数
	int nBlockCnt = pow(2, scalePyr - 1);
	int height = ref.rows;
	int width = ref.cols;
	int wBlock = width / nBlockCnt;
	int hBlock = height / nBlockCnt;

	for (int y = 0; y < nBlockCnt; y++)
	{
		for (int x = 0; x < nBlockCnt; x++)
		{
			Rect curBlock = getPatchRange(height, width, nBlockCnt, x, y);

			switch (detectionMode)
			{
			case InfoDetectMode:
				break;
			case GradientMode:
				break;
			case GridSegmentMode:
				for (int i = 0; i < hBlock; i+=gridStep)
				{
					for (int j = 0; j < wBlock; x+=gridStep)
					{
						refPoints.push_back(Point2f(j, i));
					}
				}
				break;
			default:
				break;
			}

		}
	}






}



#endif // !_MUTIHOMOGRAPHY_H_

