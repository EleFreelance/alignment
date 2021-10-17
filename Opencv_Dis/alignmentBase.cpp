#include"alignmentBase.h"

alignmentBase::alignmentBase()
{
}

alignmentBase::~alignmentBase()
{

}

void alignmentBase::calc(Mat &ref, Mat &tar, Mat &flow)
{
	cout << "this is abstract class!" << endl;
}

void alignmentBase::setParam()
{
	cout << "this is abstract class!" << endl;
}


void alignmentBase::meshgrid(int width, int height, float *grid_X, float *grid_Y)
{
	Range rangeX(0, width);
	Range rangeY(0,height);

	Mat mat_X(height, width, CV_32F);
	Mat mat_Y(height, width, CV_32F);
	vector<float> x, y;
	//两个循环不要复制，对应X、Y分别敲
	for (int i = rangeX.start; i < rangeX.end; i++)
	{
		x.push_back(i);
	}
	for (int j = rangeY.start; j < rangeY.end; j++)
	{
		y.push_back(j);
	}
	repeat(Mat(x).t(), y.size(), 1, mat_X);
	repeat(Mat(y), 1, x.size(), mat_Y);
	//访问内存空间时，注意数据类型
	memcpy(grid_X, mat_X.data, width*height*sizeof(float));
	memcpy(grid_Y, mat_Y.data, width*height*sizeof(float));
}


void alignmentBase::warpRGB(Mat &ref, Mat &base,Mat &registerdBase, Mat &flowX, Mat &flowY)
{
	int height = ref.rows;
	int width = ref.cols;

	Mat YUV[3];
	split(base, YUV);
	Mat mat_X(height, width, CV_32F);
	Mat mat_Y(height, width, CV_32F);
	Mat registeredYBase = YUV[0].clone();

	float *pX = (float*)mat_X.data;
	float *pY = (float*)mat_Y.data;

	meshgrid(width, height, pX, pY);

	mat_X += flowX;
	mat_Y += flowY;

	//三通道是否能够同时remap?
	remap(registeredYBase, registeredYBase, mat_X, mat_Y, INTER_CUBIC, BORDER_CONSTANT);
	registeredYBase.copyTo(registerdBase);

}

float bicubic_at(Mat &flow,float x,float y)
{

	return 0;
}

template <typename T>
T distance(T x0,T x1)
{
	T res = sqrt(x0*x0-x1*x1);
	return  res;
}



void alignmentBase::checkConsistency(const Mat &ref, const Mat &base, const Mat &refFlow, const Mat &baseFlow, int height, int width, int eps, Mat &consMap)
{
	Mat _ref = ref.clone();
	Mat _base = base.clone();
	Mat _refFlow[2];
	Mat _baseFlow[2];
	split(refFlow, _refFlow);
	split(baseFlow, _baseFlow);

	uchar* ref_data = _ref.data;
	uchar* base_data = _base.data;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)      //表示tar帧的位置
		{
			//等效tar帧warp两次，最终再与自身的位置比较，来判断mv的正反向一致性
			float xx = *(_baseFlow[0].data + y*width + x)+x;		//获取base帧warp的绝对位置
			float yy = *(_baseFlow[1].data + y*width + x)+y;

			float xxx = xx + bicubic_at(_refFlow[0],xx,yy);       //表示ref帧warp后的位置
			float yyy = yy + bicubic_at(_refFlow[1], xx, yy);

			if (distance(xxx-x,yyy-y)<eps)
			{
				//opencv有point的位置则需要注意顺序
				consMap.at<uchar>(y,x)= 255;
			}
			else
			{
				consMap.at<uchar>(y, x) = 0;
			}
		}
	}
}

void alignmentBase::makeCopyBorder(const Mat &src, Mat &dst, int borderSize)
{
	int height = src.rows;
	int width = src.cols;

	dst.create(height + 2 * borderSize, width + 2 * borderSize, src.type());


	const uchar *ptrSrc;
	uchar *ptrDst;
	//从src中复制出来
	for (int y = borderSize; y < height+borderSize; y++)
	{
		ptrSrc = src.ptr<uchar>(y - borderSize);
		ptrDst = dst.ptr<uchar>(y);
		for (int x = 0; x < borderSize; x++)
		{
			ptrDst[x] = ptrSrc[borderSize - x];
		}
		for (int x = borderSize; x < height+borderSize; x++)
		{
			ptrDst[x] = ptrSrc[x - borderSize];
		}
		for (int x = height+borderSize; x < height+2*borderSize; x++)
		{
			ptrDst[x] = ptrSrc[2 * height + borderSize - 1 - x];
		}
	}

	//从dst中复制出来
	for (int y = 0; y < borderSize; y++)
	{
		ptrSrc = src.ptr<uchar>(borderSize - y);
		ptrDst = dst.ptr<uchar>(y);
		memcpy(ptrDst, ptrSrc, width + 2 * borderSize);
	}
	for (int y = height+borderSize; y < height+2*borderSize; y++)
	{
		ptrSrc = src.ptr<uchar>(2 * height + borderSize - 1 - y);
		ptrDst = dst.ptr<uchar>(y);
		memcpy(ptrDst, ptrSrc, width + 2 * borderSize);
	}
}