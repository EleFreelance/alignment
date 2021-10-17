#include<opencv2/opencv.hpp>
#include"alignmentBase.h"
#include<iostream>
#include<vector>

using namespace cv;
using namespace std;



class DISOpticalFlowImpl : public alignmentBase
{
public:
	DISOpticalFlowImpl();

	void calc(Mat& I0, Mat& I1, Mat& flow);
	void collectGarbage();

protected: //!< algorithm parameters
	int finest_scale, coarsest_scale;
	int patch_size;
	int patch_stride;
	int grad_descent_iter;
	int variational_refinement_iter;
	float variational_refinement_alpha;
	float variational_refinement_gamma;
	float variational_refinement_delta;
	bool use_mean_normalization;
	bool use_spatial_propagation;
	bool use_input_flow;

protected: //!< some auxiliary variables
	int border_size;
	int w, h;   //!< flow buffer width and height on the current scale
	int ws, hs; //!< sparse flow buffer width and height on the current scale

public:
	int getFinestScale() const { return finest_scale; }
	void setFinestScale(int val) { finest_scale = val; }
	int getPatchSize() const { return patch_size; }
	void setPatchSize(int val) { patch_size = val; }
	int getPatchStride() const { return patch_stride; }
	void setPatchStride(int val) { patch_stride = val; }
	int getGradientDescentIterations() const { return grad_descent_iter; }
	void setGradientDescentIterations(int val) { grad_descent_iter = val; }
	int getVariationalRefinementIterations() const { return variational_refinement_iter; }
	void setVariationalRefinementIterations(int val) { variational_refinement_iter = val; }
	float getVariationalRefinementAlpha() const { return variational_refinement_alpha; }
	void setVariationalRefinementAlpha(float val) { variational_refinement_alpha = val; }
	float getVariationalRefinementDelta() const { return variational_refinement_delta; }
	void setVariationalRefinementDelta(float val) { variational_refinement_delta = val; }
	float getVariationalRefinementGamma() const { return variational_refinement_gamma; }
	void setVariationalRefinementGamma(float val) { variational_refinement_gamma = val; }

	bool getUseMeanNormalization() const { return use_mean_normalization; }
	void setUseMeanNormalization(bool val) { use_mean_normalization = val; }
	bool getUseSpatialPropagation() const { return use_spatial_propagation; }
	void setUseSpatialPropagation(bool val) { use_spatial_propagation = val; }

protected:                      //!< internal buffers
	vector<Mat_<uchar> > I0s;     //!< Gaussian pyramid for the current frame
	vector<Mat_<uchar> > I1s;     //!< Gaussian pyramid for the next frame
	vector<Mat_<uchar> > I1s_ext; //!< I1s with borders

	vector<Mat_<short> > I0xs; //!< Gaussian pyramid for the x gradient of the current frame
	vector<Mat_<short> > I0ys; //!< Gaussian pyramid for the y gradient of the current frame

	vector<Mat_<float> > Ux; //!< x component of the flow vectors
	vector<Mat_<float> > Uy; //!< y component of the flow vectors

	vector<Mat_<float> > initial_Ux; //!< x component of the initial flow field, if one was passed as an input
	vector<Mat_<float> > initial_Uy; //!< y component of the initial flow field, if one was passed as an input

	Mat_<Vec2f> U; //!< a buffer for the merged flow

	Mat_<float> Sx; //!< intermediate sparse flow representation (x component)
	Mat_<float> Sy; //!< intermediate sparse flow representation (y component)

					/* Structure tensor components: */
	Mat_<float> I0xx_buf; //!< sum of squares of x gradient values
	Mat_<float> I0yy_buf; //!< sum of squares of y gradient values
	Mat_<float> I0xy_buf; //!< sum of x and y gradient products

						  /* Extra buffers that are useful if patch mean-normalization is used: */
	Mat_<float> I0x_buf; //!< sum of x gradient values
	Mat_<float> I0y_buf; //!< sum of y gradient values

						 /* Auxiliary buffers used in structure tensor computation: */
	Mat_<float> I0xx_buf_aux;
	Mat_<float> I0yy_buf_aux;
	Mat_<float> I0xy_buf_aux;
	Mat_<float> I0x_buf_aux;
	Mat_<float> I0y_buf_aux;

	vector<Ptr<VariationalRefinement> > variational_refinement_processors;

private: //!< private methods and parallel sections
	void prepareBuffers(Mat &I0, Mat &I1, Mat &flow, bool use_flow);
	void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x, Mat &dst_I0y, Mat &I0x,
		Mat &I0y);

	struct PatchInverseSearch_ParBody : public ParallelLoopBody
	{
		DISOpticalFlowImpl *dis;
		int nstripes, stripe_sz;
		int hs;
		Mat *Sx, *Sy, *Ux, *Uy, *I0, *I1, *I0x, *I0y;
		int num_iter, pyr_level;

		PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _hs, Mat &dst_Sx, Mat &dst_Sy,
			Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1, Mat &_I0x, Mat &_I0y, int _num_iter,
			int _pyr_level);
		void operator()(const Range &range) const;
	};

	struct Densification_ParBody : public ParallelLoopBody
	{
		DISOpticalFlowImpl *dis;
		int nstripes, stripe_sz;
		int h;
		Mat *Ux, *Uy, *Sx, *Sy, *I0, *I1;

		Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h, Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx,
			Mat &src_Sy, Mat &_I0, Mat &_I1);
		void operator()(const Range &range) const;
	};
};

class CV_EXPORTS_W vivoVariationalRefinement //: public DenseOpticalFlow
{
public:

	CV_WRAP virtual void calc(InputArray I0, InputArray I1, InputOutputArray flow) = 0;
	/** @brief Releases all inner buffers.
	*/
	CV_WRAP virtual void collectGarbage() = 0;
	/** @brief @ref calc function overload to handle separate horizontal (u) and vertical (v) flow components
	(to avoid extra splits/merges) */
	CV_WRAP virtual void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v) = 0;

	/** @brief Number of outer (fixed-point) iterations in the minimization procedure.
	@see setFixedPointIterations */
	CV_WRAP virtual int getFixedPointIterations() const = 0;
	/** @copybrief getFixedPointIterations @see getFixedPointIterations */
	CV_WRAP virtual void setFixedPointIterations(int val) = 0;

	/** @brief Number of inner successive over-relaxation (SOR) iterations
	in the minimization procedure to solve the respective linear system.
	@see setSorIterations */
	CV_WRAP virtual int getSorIterations() const = 0;
	/** @copybrief getSorIterations @see getSorIterations */
	CV_WRAP virtual void setSorIterations(int val) = 0;

	/** @brief Relaxation factor in SOR
	@see setOmega */
	CV_WRAP virtual float getOmega() const = 0;
	/** @copybrief getOmega @see getOmega */
	CV_WRAP virtual void setOmega(float val) = 0;

	/** @brief Weight of the smoothness term
	@see setAlpha */
	CV_WRAP virtual float getAlpha() const = 0;
	/** @copybrief getAlpha @see getAlpha */
	CV_WRAP virtual void setAlpha(float val) = 0;

	/** @brief Weight of the color constancy term
	@see setDelta */
	CV_WRAP virtual float getDelta() const = 0;
	/** @copybrief getDelta @see getDelta */
	CV_WRAP virtual void setDelta(float val) = 0;

	/** @brief Weight of the gradient constancy term
	@see setGamma */
	CV_WRAP virtual float getGamma() const = 0;
	/** @copybrief getGamma @see getGamma */
	CV_WRAP virtual void setGamma(float val) = 0;

	/** @brief Creates an instance of VariationalRefinement
	*/
	CV_WRAP static Ptr<vivoVariationalRefinement> create();
};


class vivoVariationalRefinementImpl : public vivoVariationalRefinement
{
public:
	vivoVariationalRefinementImpl();

	void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
	void calcUV(InputArray I0, InputArray I1, InputOutputArray flow_u, InputOutputArray flow_v) CV_OVERRIDE;
	void collectGarbage() CV_OVERRIDE;

protected: //!< algorithm parameters
	int fixedPointIterations, sorIterations;
	float omega;
	float alpha, delta, gamma;
	float zeta, epsilon;

public:
	int getFixedPointIterations() const CV_OVERRIDE { return fixedPointIterations; }
	void setFixedPointIterations(int val) CV_OVERRIDE { fixedPointIterations = val; }
	int getSorIterations() const CV_OVERRIDE { return sorIterations; }
	void setSorIterations(int val) CV_OVERRIDE { sorIterations = val; }
	float getOmega() const CV_OVERRIDE { return omega; }
	void setOmega(float val) CV_OVERRIDE { omega = val; }
	float getAlpha() const CV_OVERRIDE { return alpha; }
	void setAlpha(float val) CV_OVERRIDE { alpha = val; }
	float getDelta() const CV_OVERRIDE { return delta; }
	void setDelta(float val) CV_OVERRIDE { delta = val; }
	float getGamma() const CV_OVERRIDE { return gamma; }
	void setGamma(float val) CV_OVERRIDE { gamma = val; }

protected: //!< internal buffers
		   /* This struct defines a special data layout for Mat_<float>. Original buffer is split into two: one for "red"
		   * elements (sum of indices is even) and one for "black" (sum of indices is odd) in a checkerboard pattern. It
		   * allows for more efficient processing in SOR iterations, more natural SIMD vectorization and parallelization
		   * (Red-Black SOR). Additionally, it simplifies border handling by adding repeated borders to both red and
		   * black buffers.
		   */
	struct RedBlackBuffer
	{
		Mat_<float> red;   //!< (i+j)%2==0
		Mat_<float> black; //!< (i+j)%2==1

						   /* Width of even and odd rows may be different */
		int red_even_len, red_odd_len;
		int black_even_len, black_odd_len;

		RedBlackBuffer();
		void create(Size s);
		void release();
	};

	Mat_<float> Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;                            //!< image derivative buffers
	RedBlackBuffer Ix_rb, Iy_rb, Iz_rb, Ixx_rb, Ixy_rb, Iyy_rb, Ixz_rb, Iyz_rb; //!< corresponding red-black buffers

	RedBlackBuffer A11, A12, A22, b1, b2; //!< main linear system coefficients
	RedBlackBuffer weights;               //!< smoothness term weights in the current fixed point iteration

	Mat_<float> mapX, mapY; //!< auxiliary buffers for remapping

	RedBlackBuffer tempW_u, tempW_v; //!< flow buffers that are modified in each fixed point iteration
	RedBlackBuffer dW_u, dW_v;       //!< optical flow increment
	RedBlackBuffer W_u_rb, W_v_rb;   //!< red-black-buffer version of the input flow

private: //!< private methods and parallel sections
	void splitCheckerboard(RedBlackBuffer &dst, Mat &src);
	void mergeCheckerboard(Mat &dst, RedBlackBuffer &src);
	void updateRepeatedBorders(RedBlackBuffer &dst);
	void warpImage(Mat &dst, Mat &src, Mat &flow_u, Mat &flow_v);
	void prepareBuffers(Mat &I0, Mat &I1, Mat &W_u, Mat &W_v);

	/* Parallelizing arbitrary operations with 3 input/output arguments */
	typedef void (vivoVariationalRefinementImpl::*Op)(void *op1, void *op2, void *op3);
	struct ParallelOp_ParBody : public ParallelLoopBody
	{
		vivoVariationalRefinementImpl *var;
		vector<Op> ops;
		vector<void *> op1s;
		vector<void *> op2s;
		vector<void *> op3s;

		ParallelOp_ParBody(vivoVariationalRefinementImpl &_var, vector<Op> _ops, vector<void *> &_op1s,
			vector<void *> &_op2s, vector<void *> &_op3s);
		void operator()(const Range &range) const CV_OVERRIDE;
	};
	void gradHorizAndSplitOp(void *src, void *dst, void *dst_split)
	{
		Sobel(*(Mat *)src, *(Mat *)dst, -1, 1, 0, 1, 1, 0.00, BORDER_REPLICATE);
		splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
	}
	void gradVertAndSplitOp(void *src, void *dst, void *dst_split)
	{
		Sobel(*(Mat *)src, *(Mat *)dst, -1, 0, 1, 1, 1, 0.00, BORDER_REPLICATE);
		splitCheckerboard(*(RedBlackBuffer *)dst_split, *(Mat *)dst);
	}
	void averageOp(void *src1, void *src2, void *dst)
	{
		addWeighted(*(Mat *)src1, 0.5, *(Mat *)src2, 0.5, 0.0, *(Mat *)dst, CV_32F);
	}
	void subtractOp(void *src1, void *src2, void *dst)
	{
		subtract(*(Mat *)src1, *(Mat *)src2, *(Mat *)dst, noArray(), CV_32F);
	}

	struct ComputeDataTerm_ParBody : public ParallelLoopBody
	{
		vivoVariationalRefinementImpl *var;
		int nstripes, stripe_sz;
		int h;
		RedBlackBuffer *dW_u, *dW_v;
		bool red_pass;

		ComputeDataTerm_ParBody(vivoVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
			RedBlackBuffer &_dW_v, bool _red_pass);
		void operator()(const Range &range) const CV_OVERRIDE;
	};

	struct ComputeSmoothnessTermHorPass_ParBody : public ParallelLoopBody
	{
		vivoVariationalRefinementImpl *var;
		int nstripes, stripe_sz;
		int h;
		RedBlackBuffer *W_u, *W_v, *curW_u, *curW_v;
		bool red_pass;

		ComputeSmoothnessTermHorPass_ParBody(vivoVariationalRefinementImpl &_var, int _nstripes, int _h,
			RedBlackBuffer &_W_u, RedBlackBuffer &_W_v, RedBlackBuffer &_tempW_u,
			RedBlackBuffer &_tempW_v, bool _red_pass);
		void operator()(const Range &range) const CV_OVERRIDE;
	};

	struct ComputeSmoothnessTermVertPass_ParBody : public ParallelLoopBody
	{
		vivoVariationalRefinementImpl *var;
		int nstripes, stripe_sz;
		int h;
		RedBlackBuffer *W_u, *W_v;
		bool red_pass;

		ComputeSmoothnessTermVertPass_ParBody(vivoVariationalRefinementImpl &_var, int _nstripes, int _h,
			RedBlackBuffer &W_u, RedBlackBuffer &_W_v, bool _red_pass);
		void operator()(const Range &range) const CV_OVERRIDE;
	};

	struct RedBlackSOR_ParBody : public ParallelLoopBody
	{
		vivoVariationalRefinementImpl *var;
		int nstripes, stripe_sz;
		int h;
		RedBlackBuffer *dW_u, *dW_v;
		bool red_pass;

		RedBlackSOR_ParBody(vivoVariationalRefinementImpl &_var, int _nstripes, int _h, RedBlackBuffer &_dW_u,
			RedBlackBuffer &_dW_v, bool _red_pass);
		void operator()(const Range &range) const CV_OVERRIDE;
	};
};
