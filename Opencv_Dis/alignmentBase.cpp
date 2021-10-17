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
	//����ѭ����Ҫ���ƣ���ӦX��Y�ֱ���
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
	//�����ڴ�ռ�ʱ��ע����������
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

	//��ͨ���Ƿ��ܹ�ͬʱremap?
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
		for (int x = 0; x < width; x++)      //��ʾtar֡��λ��
		{
			//��Чtar֡warp���Σ��������������λ�ñȽϣ����ж�mv��������һ����
			float xx = *(_baseFlow[0].data + y*width + x)+x;		//��ȡbase֡warp�ľ���λ��
			float yy = *(_baseFlow[1].data + y*width + x)+y;

			float xxx = xx + bicubic_at(_refFlow[0],xx,yy);       //��ʾref֡warp���λ��
			float yyy = yy + bicubic_at(_refFlow[1], xx, yy);

			if (distance(xxx-x,yyy-y)<eps)
			{
				//opencv��point��λ������Ҫע��˳��
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
	//��src�и��Ƴ���
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

	//��dst�и��Ƴ���
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