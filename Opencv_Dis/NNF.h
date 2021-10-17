#include<opencv2/opencv.hpp>
#include<vector>
#include<iostream>

using namespace cv;
using namespace std;

#define MAX_DISTANCE 65535
//#define MAX2(a,b) return a>b?a:b;
//#define MIN2(a,b) return a>b?b:a;

typedef struct NNF
{
	Mat ref;
	Mat match;

	int patch_radius;

	int ***field;
	int fieldH;
	int fieldW;
}NNF_T, *NNF_P;


NNF_P initNNF(Mat _ref, Mat _match, int _patch_radius);

void mallocFiled(NNF_P p_nnf);
void freeFiled(NNF_P p_nnf);

void randomize(NNF_P p_nnf);
void initialization(NNF_P p_nnf);

void minimize(NNF_P p_nnf, int max_iter);
void minimizeLink(NNF_P p_nnf, int x, int y, int dir);
int distance(NNF_P p_nnf, int xs, int ys, int xt, int yt);
