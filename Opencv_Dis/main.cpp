
#include"alignmentBase.h"
#include"dis_flow.h"
#include"JointWMF.h"
#include"CPM.h"

char *filename_1 = "F:\\学术论文\\data\\MPI-Sintel-complete\\test\\clean\\ambush_1\\frame_0001.png";
char *filename_2 = "F:\\学术论文\\data\\MPI-Sintel-complete\\test\\clean\\ambush_1\\frame_0002.png";

// draw each match as a 3x3 color block
void Match2Flow(FImage& inMat, FImage& ou, FImage& ov, int w, int h)
{
	if (!ou.matchDimension(w, h, 1)) {
		ou.allocate(w, h, 1);
	}
	if (!ov.matchDimension(w, h, 1)) {
		ov.allocate(w, h, 1);
	}
	ou.setValue(0);
	ov.setValue(0);
	int cnt = inMat.height();
	for (int i = 0; i < cnt; i++) {
		float* p = inMat.rowPtr(i);
		float x = p[0];
		float y = p[1];
		float u = p[2] - p[0];
		float v = p[3] - p[1];
		for (int di = -1; di <= 1; di++) {
			for (int dj = -1; dj <= 1; dj++) {
				int tx = ImageProcessing::EnforceRange(x + dj, w);
				int ty = ImageProcessing::EnforceRange(y + di, h);
				ou[ty*w + tx] = u;
				ov[ty*w + tx] = v;
			}
		}
	}
}

int main()
{

	//Mat src = imread("E:\\TestImg\\1.jpg");
	//Mat gauss_5,gauss_11;
	//GaussianBlur(src, gauss_5, Size(5, 5), 0, 0);
	//GaussianBlur(src, gauss_11, Size(11,11), 0, 0);
	//Mat dst1 = JointWMF::filter(src, src, 3, 25.5, 256, 256, 1, "exp", Mat());
	//Mat dst2= JointWMF::filter(src, gauss_11, 3, 25.5,256, 256, 1, "exp", Mat());
	//medianBlur(src, dst2, 3);
	//Mat SobelX, SobelY;
	//Mat gray;
	//cvtColor(src, gray, COLOR_BGR2GRAY);
	//Sobel(gray, SobelX, CV_8U, 1, 0);
	//Sobel(gray, SobelY, CV_8U, 0, 1);

	//Mat joint = abs(SobelX) + abs(SobelY);
	//Mat dst3= JointWMF::filter(src, joint, 1);

	Mat I0 = imread(filename_1,0);
	Mat I1 = imread(filename_2,0);

	Mat f32I0, f32I1;
	I0.convertTo(f32I0, CV_32F);
	I1.convertTo(f32I1, CV_32F);

	int height = I0.rows;
	int width = I0.cols;

	FImage img1(I0.cols, I0.rows, 1);
	FImage img2(I0.cols, I0.rows, 1);
	memcpy(img1.data(), f32I0.data, sizeof(float)*height*width);
	memcpy(img2.data(), f32I1.data, sizeof(float)*height*width);
	FImage matches;



	FImage flow1, flow2;
	CPM cpm;
	cpm.SetStep(3);
	cpm.Matching(img1, img2, matches);
	Match2Flow(matches, flow1, flow2, width, height);

	Mat mDebug1(flow1.height(), flow1.width(), CV_32F, flow1.data());
	Mat mDebug2(flow2.height(), flow2.width(), CV_32F, flow2.data());
	//Mat I0_hsv,I1_hsv;
	//cvtColor(I0, I0_hsv, COLOR_RGB2HSV);
	//cvtColor(I1, I1_hsv, COLOR_RGB2HSV);


	//alignmentBase* dis = new DISOpticalFlowImpl();


	//int height = 5;
	//int width = 6;
	//float *X = (float*)malloc(height * width * sizeof(float));
	//float *Y = (float*)malloc(height * width * sizeof(float));
	//Mat flow;
	//dis->calc(I0_hsv, I1_hsv, flow);
	//Mat flowuv[2];
	//split(flow, flowuv);
	//Mat dst;
	//dis->warpRGB(I0_hsv, I1_hsv,dst, flowuv[0], flowuv[1]);

	//for (int i = 0; i < height; i++)
	//{
	//	for (int j = 0; j < width; j++)
	//	{
	//		//遍历图像时，分清楚控制变量与边界值
	//		cout << X[i*width +j];
	//	}
	//	cout << endl;
	//}

	
	waitKey(0);
	//free(X);
	//free(Y);
	return 0;
}