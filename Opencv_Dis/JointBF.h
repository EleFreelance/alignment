
/***************************************************************/
/*
* Standard IO library is required.
* STL String library is required.
*
/***************************************************************/
#include <cstdio>
#include <string>

/***************************************************************/
/*
* OpenCV 2.4 is required.
* The following code is already built on OpenCV 2.4.2.
*
/***************************************************************/
#include<opencv2/opencv.hpp>
#include <time.h>


//Use the namespace of CV and STD
using namespace std;
using namespace cv;

class JointBF {

public:
	static Mat filter(Mat &I, Mat &feature, int r, float spaceSigma = 50, int colorSigma = 256, Mat mask = Mat())
	{
		assert(I.depth() == CV_32F || I.depth() == CV_8U);
		//assert(F.depth() == CV_8U && (F.channels() == 1 || F.channels() == 3));

		int height = I.rows;
		int width = I.cols;
		int diameter = 2 * r + 1;
		float *spaceCeoff = (float*)malloc(diameter * sizeof(float));
		float *colorCeoff = (float*)malloc(256 * sizeof(float));

		int *srcIndex = (int *)malloc(diameter * sizeof(int));
		int *jointIndex = (int *)malloc(diameter * sizeof(int));
		for (int i = 0; i < 256; i++)
		{
			colorCeoff[i] = exp(-(i*i) / (colorSigma*colorSigma));
		}
		//get spaceCoeff and I_index,feature_index
		for (int y = -r; y <= r; y++)
		{
			for (int x = -r; x <= r; x++)
			{

			}
		}

		free(spaceCeoff);
		free(colorCeoff);
		free(srcIndex);
		free(jointIndex); 
	}
};