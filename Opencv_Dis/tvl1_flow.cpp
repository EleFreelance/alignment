#include"tvl1_flow.h"


OpticalFlowDual_TVL1::OpticalFlowDual_TVL1()
{
	tau = 0.25;
	lambda = 0.15;
	theta = 0.3;
	nscales = 5;
	warps = 5;
	epsilon = 0.01;
	gamma = 0.;
	innerIterations = 30;
	outerIterations = 10;
	useInitialFlow = false;
	medianFiltering = 5;
	scaleStep = 0.8;
}

void OpticalFlowDual_TVL1::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow)
{
	Mat I0 = _I0.getMat();
	Mat I1 = _I1.getMat();

	CV_Assert(I0.type() == CV_8UC1 || I0.type() == CV_32FC1);
	CV_Assert(I0.size() == I1.size());
	CV_Assert(I0.type() == I1.type());
	CV_Assert(!useInitialFlow || (_flow.size() == I0.size() && _flow.type() == CV_32FC2));
	CV_Assert(nscales > 0);
	bool use_gamma = gamma != 0;
	// allocate memory for the pyramid structure
	dm.I0s.resize(nscales);
	dm.I1s.resize(nscales);
	dm.u1s.resize(nscales);
	dm.u2s.resize(nscales);
	dm.u3s.resize(nscales);

	I0.convertTo(dm.I0s[0], dm.I0s[0].depth(), I0.depth() == CV_8U ? 1.0 : 255.0);
	I1.convertTo(dm.I1s[0], dm.I1s[0].depth(), I1.depth() == CV_8U ? 1.0 : 255.0);

	dm.u1s[0].create(I0.size());
	dm.u2s[0].create(I0.size());
	if (use_gamma) dm.u3s[0].create(I0.size());

	if (useInitialFlow)
	{
		Mat_<float> mv[] = { dm.u1s[0], dm.u2s[0] };
		split(_flow.getMat(), mv);
	}

	dm.I1x_buf.create(I0.size());
	dm.I1y_buf.create(I0.size());

	dm.flowMap1_buf.create(I0.size());
	dm.flowMap2_buf.create(I0.size());

	dm.I1w_buf.create(I0.size());
	dm.I1wx_buf.create(I0.size());
	dm.I1wy_buf.create(I0.size());

	dm.grad_buf.create(I0.size());
	dm.rho_c_buf.create(I0.size());

	dm.v1_buf.create(I0.size());
	dm.v2_buf.create(I0.size());
	dm.v3_buf.create(I0.size());

	dm.p11_buf.create(I0.size());
	dm.p12_buf.create(I0.size());
	dm.p21_buf.create(I0.size());
	dm.p22_buf.create(I0.size());
	dm.p31_buf.create(I0.size());
	dm.p32_buf.create(I0.size());

	dm.div_p1_buf.create(I0.size());
	dm.div_p2_buf.create(I0.size());
	dm.div_p3_buf.create(I0.size());

	dm.u1x_buf.create(I0.size());
	dm.u1y_buf.create(I0.size());
	dm.u2x_buf.create(I0.size());
	dm.u2y_buf.create(I0.size());
	dm.u3x_buf.create(I0.size());
	dm.u3y_buf.create(I0.size());

	// create the scales
	for (int s = 1; s < nscales; ++s)
	{
		resize(dm.I0s[s - 1], dm.I0s[s], Size(), scaleStep, scaleStep, INTER_LINEAR);
		resize(dm.I1s[s - 1], dm.I1s[s], Size(), scaleStep, scaleStep, INTER_LINEAR);

		if (dm.I0s[s].cols < 16 || dm.I0s[s].rows < 16)
		{
			nscales = s;
			break;
		}

		if (useInitialFlow)
		{
			resize(dm.u1s[s - 1], dm.u1s[s], Size(), scaleStep, scaleStep, INTER_LINEAR);
			resize(dm.u2s[s - 1], dm.u2s[s], Size(), scaleStep, scaleStep, INTER_LINEAR);

			multiply(dm.u1s[s], Scalar::all(scaleStep), dm.u1s[s]);
			multiply(dm.u2s[s], Scalar::all(scaleStep), dm.u2s[s]);
		}
		else
		{
			dm.u1s[s].create(dm.I0s[s].size());
			dm.u2s[s].create(dm.I0s[s].size());
		}
		if (use_gamma) dm.u3s[s].create(dm.I0s[s].size());
	}
	if (!useInitialFlow)
	{
		dm.u1s[nscales - 1].setTo(Scalar::all(0));
		dm.u2s[nscales - 1].setTo(Scalar::all(0));
	}
	if (use_gamma) dm.u3s[nscales - 1].setTo(Scalar::all(0));
	// pyramidal structure for computing the optical flow
	for (int s = nscales - 1; s >= 0; --s)
	{
		// compute the optical flow at the current scale
		procOneScale(dm.I0s[s], dm.I1s[s], dm.u1s[s], dm.u2s[s], dm.u3s[s]);

		// if this was the last scale, finish now
		if (s == 0)
			break;

		// otherwise, upsample the optical flow

		// zoom the optical flow for the next finer scale
		resize(dm.u1s[s], dm.u1s[s - 1], dm.I0s[s - 1].size(), 0, 0, INTER_LINEAR);
		resize(dm.u2s[s], dm.u2s[s - 1], dm.I0s[s - 1].size(), 0, 0, INTER_LINEAR);
		if (use_gamma) resize(dm.u3s[s], dm.u3s[s - 1], dm.I0s[s - 1].size(), 0, 0, INTER_LINEAR);

		// scale the optical flow with the appropriate zoom factor (don't scale u3!)
		multiply(dm.u1s[s - 1], Scalar::all(1 / scaleStep), dm.u1s[s - 1]);
		multiply(dm.u2s[s - 1], Scalar::all(1 / scaleStep), dm.u2s[s - 1]);
	}

	Mat uxy[] = { dm.u1s[0], dm.u2s[0] };
	merge(uxy, 2, _flow);
}


////////////////////////////////////////////////////////////
// buildFlowMap

struct BuildFlowMapBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> u1;
	Mat_<float> u2;
	mutable Mat_<float> map1;
	mutable Mat_<float> map2;
};

void BuildFlowMapBody::operator() (const Range& range) const
{
	for (int y = range.start; y < range.end; ++y)
	{
		const float* u1Row = u1[y];
		const float* u2Row = u2[y];

		float* map1Row = map1[y];
		float* map2Row = map2[y];

		for (int x = 0; x < u1.cols; ++x)
		{
			map1Row[x] = x + u1Row[x];
			map2Row[x] = y + u2Row[x];
		}
	}
}

static void buildFlowMap(const Mat_<float>& u1, const Mat_<float>& u2,
	Mat_<float>& map1, Mat_<float>& map2)
{
	CV_DbgAssert(u2.size() == u1.size());
	CV_DbgAssert(map1.size() == u1.size());
	CV_DbgAssert(map2.size() == u1.size());

	BuildFlowMapBody body;

	body.u1 = u1;
	body.u2 = u2;
	body.map1 = map1;
	body.map2 = map2;

	parallel_for_(Range(0, u1.rows), body);
}

////////////////////////////////////////////////////////////
// centeredGradient

struct CenteredGradientBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> src;
	mutable Mat_<float> dx;
	mutable Mat_<float> dy;
};

void CenteredGradientBody::operator() (const Range& range) const
{
	const int last_col = src.cols - 1;

	for (int y = range.start; y < range.end; ++y)
	{
		const float* srcPrevRow = src[y - 1];
		const float* srcCurRow = src[y];
		const float* srcNextRow = src[y + 1];

		float* dxRow = dx[y];
		float* dyRow = dy[y];

		for (int x = 1; x < last_col; ++x)
		{
			dxRow[x] = 0.5f * (srcCurRow[x + 1] - srcCurRow[x - 1]);
			dyRow[x] = 0.5f * (srcNextRow[x] - srcPrevRow[x]);
		}
	}
}

static void centeredGradient(const Mat_<float>& src, Mat_<float>& dx, Mat_<float>& dy)
{
	CV_DbgAssert(src.rows > 2 && src.cols > 2);
	CV_DbgAssert(dx.size() == src.size());
	CV_DbgAssert(dy.size() == src.size());

	const int last_row = src.rows - 1;
	const int last_col = src.cols - 1;

	// compute the gradient on the center body of the image
	{
		CenteredGradientBody body;

		body.src = src;
		body.dx = dx;
		body.dy = dy;

		parallel_for_(Range(1, last_row), body);
	}

	// compute the gradient on the first and last rows
	for (int x = 1; x < last_col; ++x)
	{
		dx(0, x) = 0.5f * (src(0, x + 1) - src(0, x - 1));
		dy(0, x) = 0.5f * (src(1, x) - src(0, x));

		dx(last_row, x) = 0.5f * (src(last_row, x + 1) - src(last_row, x - 1));
		dy(last_row, x) = 0.5f * (src(last_row, x) - src(last_row - 1, x));
	}

	// compute the gradient on the first and last columns
	for (int y = 1; y < last_row; ++y)
	{
		dx(y, 0) = 0.5f * (src(y, 1) - src(y, 0));
		dy(y, 0) = 0.5f * (src(y + 1, 0) - src(y - 1, 0));

		dx(y, last_col) = 0.5f * (src(y, last_col) - src(y, last_col - 1));
		dy(y, last_col) = 0.5f * (src(y + 1, last_col) - src(y - 1, last_col));
	}

	// compute the gradient at the four corners
	dx(0, 0) = 0.5f * (src(0, 1) - src(0, 0));
	dy(0, 0) = 0.5f * (src(1, 0) - src(0, 0));

	dx(0, last_col) = 0.5f * (src(0, last_col) - src(0, last_col - 1));
	dy(0, last_col) = 0.5f * (src(1, last_col) - src(0, last_col));

	dx(last_row, 0) = 0.5f * (src(last_row, 1) - src(last_row, 0));
	dy(last_row, 0) = 0.5f * (src(last_row, 0) - src(last_row - 1, 0));

	dx(last_row, last_col) = 0.5f * (src(last_row, last_col) - src(last_row, last_col - 1));
	dy(last_row, last_col) = 0.5f * (src(last_row, last_col) - src(last_row - 1, last_col));
}

////////////////////////////////////////////////////////////
// forwardGradient

struct ForwardGradientBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> src;
	mutable Mat_<float> dx;
	mutable Mat_<float> dy;
};

void ForwardGradientBody::operator() (const Range& range) const
{
	const int last_col = src.cols - 1;

	for (int y = range.start; y < range.end; ++y)
	{
		const float* srcCurRow = src[y];
		const float* srcNextRow = src[y + 1];

		float* dxRow = dx[y];
		float* dyRow = dy[y];

		for (int x = 0; x < last_col; ++x)
		{
			dxRow[x] = srcCurRow[x + 1] - srcCurRow[x];
			dyRow[x] = srcNextRow[x] - srcCurRow[x];
		}
	}
}

static void forwardGradient(const Mat_<float>& src, Mat_<float>& dx, Mat_<float>& dy)
{
	CV_DbgAssert(src.rows > 2 && src.cols > 2);
	CV_DbgAssert(dx.size() == src.size());
	CV_DbgAssert(dy.size() == src.size());

	const int last_row = src.rows - 1;
	const int last_col = src.cols - 1;

	// compute the gradient on the central body of the image
	{
		ForwardGradientBody body;

		body.src = src;
		body.dx = dx;
		body.dy = dy;

		parallel_for_(Range(0, last_row), body);
	}

	// compute the gradient on the last row
	for (int x = 0; x < last_col; ++x)
	{
		dx(last_row, x) = src(last_row, x + 1) - src(last_row, x);
		dy(last_row, x) = 0.0f;
	}

	// compute the gradient on the last column
	for (int y = 0; y < last_row; ++y)
	{
		dx(y, last_col) = 0.0f;
		dy(y, last_col) = src(y + 1, last_col) - src(y, last_col);
	}

	dx(last_row, last_col) = 0.0f;
	dy(last_row, last_col) = 0.0f;
}

////////////////////////////////////////////////////////////
// divergence

struct DivergenceBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> v1;
	Mat_<float> v2;
	mutable Mat_<float> div;
};

void DivergenceBody::operator() (const Range& range) const
{
	for (int y = range.start; y < range.end; ++y)
	{
		const float* v1Row = v1[y];
		const float* v2PrevRow = v2[y - 1];
		const float* v2CurRow = v2[y];

		float* divRow = div[y];

		for (int x = 1; x < v1.cols; ++x)
		{
			const float v1x = v1Row[x] - v1Row[x - 1];
			const float v2y = v2CurRow[x] - v2PrevRow[x];

			divRow[x] = v1x + v2y;
		}
	}
}

static void divergence(const Mat_<float>& v1, const Mat_<float>& v2, Mat_<float>& div)
{
	CV_DbgAssert(v1.rows > 2 && v1.cols > 2);
	CV_DbgAssert(v2.size() == v1.size());
	CV_DbgAssert(div.size() == v1.size());

	{
		DivergenceBody body;

		body.v1 = v1;
		body.v2 = v2;
		body.div = div;

		parallel_for_(Range(1, v1.rows), body);
	}

	// compute the divergence on the first row
	for (int x = 1; x < v1.cols; ++x)
		div(0, x) = v1(0, x) - v1(0, x - 1) + v2(0, x);

	// compute the divergence on the first column
	for (int y = 1; y < v1.rows; ++y)
		div(y, 0) = v1(y, 0) + v2(y, 0) - v2(y - 1, 0);

	div(0, 0) = v1(0, 0) + v2(0, 0);
}

////////////////////////////////////////////////////////////
// calcGradRho

struct CalcGradRhoBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> I0;
	Mat_<float> I1w;
	Mat_<float> I1wx;
	Mat_<float> I1wy;
	Mat_<float> u1;
	Mat_<float> u2;
	mutable Mat_<float> grad;
	mutable Mat_<float> rho_c;
};

void CalcGradRhoBody::operator() (const Range& range) const
{
	for (int y = range.start; y < range.end; ++y)
	{
		const float* I0Row = I0[y];
		const float* I1wRow = I1w[y];
		const float* I1wxRow = I1wx[y];
		const float* I1wyRow = I1wy[y];
		const float* u1Row = u1[y];
		const float* u2Row = u2[y];

		float* gradRow = grad[y];
		float* rhoRow = rho_c[y];

		for (int x = 0; x < I0.cols; ++x)
		{
			const float Ix2 = I1wxRow[x] * I1wxRow[x];
			const float Iy2 = I1wyRow[x] * I1wyRow[x];

			// store the |Grad(I1)|^2
			gradRow[x] = Ix2 + Iy2;

			// compute the constant part of the rho function
			rhoRow[x] = (I1wRow[x] - I1wxRow[x] * u1Row[x] - I1wyRow[x] * u2Row[x] - I0Row[x]);
		}
	}
}

static void calcGradRho(const Mat_<float>& I0, const Mat_<float>& I1w, const Mat_<float>& I1wx, const Mat_<float>& I1wy, const Mat_<float>& u1, const Mat_<float>& u2,
	Mat_<float>& grad, Mat_<float>& rho_c)
{
	CV_DbgAssert(I1w.size() == I0.size());
	CV_DbgAssert(I1wx.size() == I0.size());
	CV_DbgAssert(I1wy.size() == I0.size());
	CV_DbgAssert(u1.size() == I0.size());
	CV_DbgAssert(u2.size() == I0.size());
	CV_DbgAssert(grad.size() == I0.size());
	CV_DbgAssert(rho_c.size() == I0.size());

	CalcGradRhoBody body;

	body.I0 = I0;
	body.I1w = I1w;
	body.I1wx = I1wx;
	body.I1wy = I1wy;
	body.u1 = u1;
	body.u2 = u2;
	body.grad = grad;
	body.rho_c = rho_c;

	parallel_for_(Range(0, I0.rows), body);
}

////////////////////////////////////////////////////////////
// estimateV

struct EstimateVBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> I1wx;
	Mat_<float> I1wy;
	Mat_<float> u1;
	Mat_<float> u2;
	Mat_<float> u3;
	Mat_<float> grad;
	Mat_<float> rho_c;
	mutable Mat_<float> v1;
	mutable Mat_<float> v2;
	mutable Mat_<float> v3;
	float l_t;
	float gamma;
};

void EstimateVBody::operator() (const Range& range) const
{
	bool use_gamma = gamma != 0;
	for (int y = range.start; y < range.end; ++y)
	{
		const float* I1wxRow = I1wx[y];
		const float* I1wyRow = I1wy[y];
		const float* u1Row = u1[y];
		const float* u2Row = u2[y];
		const float* u3Row = use_gamma ? u3[y] : NULL;
		const float* gradRow = grad[y];
		const float* rhoRow = rho_c[y];

		float* v1Row = v1[y];
		float* v2Row = v2[y];
		float* v3Row = use_gamma ? v3[y] : NULL;

		for (int x = 0; x < I1wx.cols; ++x)
		{
			const float rho = use_gamma ? rhoRow[x] + (I1wxRow[x] * u1Row[x] + I1wyRow[x] * u2Row[x]) + gamma * u3Row[x] :
				rhoRow[x] + (I1wxRow[x] * u1Row[x] + I1wyRow[x] * u2Row[x]);
			float d1 = 0.0f;
			float d2 = 0.0f;
			float d3 = 0.0f;
			if (rho < -l_t * gradRow[x])
			{
				d1 = l_t * I1wxRow[x];
				d2 = l_t * I1wyRow[x];
				if (use_gamma) d3 = l_t * gamma;
			}
			else if (rho > l_t * gradRow[x])
			{
				d1 = -l_t * I1wxRow[x];
				d2 = -l_t * I1wyRow[x];
				if (use_gamma) d3 = -l_t * gamma;
			}
			else if (gradRow[x] > std::numeric_limits<float>::epsilon())
			{
				float fi = -rho / gradRow[x];
				d1 = fi * I1wxRow[x];
				d2 = fi * I1wyRow[x];
				if (use_gamma) d3 = fi * gamma;
			}

			v1Row[x] = u1Row[x] + d1;
			v2Row[x] = u2Row[x] + d2;
			if (use_gamma) v3Row[x] = u3Row[x] + d3;
		}
	}
}

static void estimateV(const Mat_<float>& I1wx, const Mat_<float>& I1wy, const Mat_<float>& u1, const Mat_<float>& u2, const Mat_<float>& u3, const Mat_<float>& grad, const Mat_<float>& rho_c,
	Mat_<float>& v1, Mat_<float>& v2, Mat_<float>& v3, float l_t, float gamma)
{
	CV_DbgAssert(I1wy.size() == I1wx.size());
	CV_DbgAssert(u1.size() == I1wx.size());
	CV_DbgAssert(u2.size() == I1wx.size());
	CV_DbgAssert(grad.size() == I1wx.size());
	CV_DbgAssert(rho_c.size() == I1wx.size());
	CV_DbgAssert(v1.size() == I1wx.size());
	CV_DbgAssert(v2.size() == I1wx.size());

	EstimateVBody body;
	bool use_gamma = gamma != 0;
	body.I1wx = I1wx;
	body.I1wy = I1wy;
	body.u1 = u1;
	body.u2 = u2;
	if (use_gamma) body.u3 = u3;
	body.grad = grad;
	body.rho_c = rho_c;
	body.v1 = v1;
	body.v2 = v2;
	if (use_gamma) body.v3 = v3;
	body.l_t = l_t;
	body.gamma = gamma;
	parallel_for_(Range(0, I1wx.rows), body);
}

////////////////////////////////////////////////////////////
// estimateU

static float estimateU(const Mat_<float>& v1, const Mat_<float>& v2, const Mat_<float>& v3,
	const Mat_<float>& div_p1, const Mat_<float>& div_p2, const Mat_<float>& div_p3,
	Mat_<float>& u1, Mat_<float>& u2, Mat_<float>& u3,
	float theta, float gamma)
{
	CV_DbgAssert(v2.size() == v1.size());
	CV_DbgAssert(div_p1.size() == v1.size());
	CV_DbgAssert(div_p2.size() == v1.size());
	CV_DbgAssert(u1.size() == v1.size());
	CV_DbgAssert(u2.size() == v1.size());

	float error = 0.0f;
	bool use_gamma = gamma != 0;
	for (int y = 0; y < v1.rows; ++y)
	{
		const float* v1Row = v1[y];
		const float* v2Row = v2[y];
		const float* v3Row = use_gamma ? v3[y] : NULL;
		const float* divP1Row = div_p1[y];
		const float* divP2Row = div_p2[y];
		const float* divP3Row = use_gamma ? div_p3[y] : NULL;

		float* u1Row = u1[y];
		float* u2Row = u2[y];
		float* u3Row = use_gamma ? u3[y] : NULL;


		for (int x = 0; x < v1.cols; ++x)
		{
			const float u1k = u1Row[x];
			const float u2k = u2Row[x];
			const float u3k = use_gamma ? u3Row[x] : 0;

			u1Row[x] = v1Row[x] + theta * divP1Row[x];
			u2Row[x] = v2Row[x] + theta * divP2Row[x];
			if (use_gamma) u3Row[x] = v3Row[x] + theta * divP3Row[x];
			error += use_gamma ? (u1Row[x] - u1k) * (u1Row[x] - u1k) + (u2Row[x] - u2k) * (u2Row[x] - u2k) + (u3Row[x] - u3k) * (u3Row[x] - u3k) :
				(u1Row[x] - u1k) * (u1Row[x] - u1k) + (u2Row[x] - u2k) * (u2Row[x] - u2k);
		}
	}

	return error;
}

////////////////////////////////////////////////////////////
// estimateDualVariables

struct EstimateDualVariablesBody : ParallelLoopBody
{
	void operator() (const Range& range) const CV_OVERRIDE;

	Mat_<float> u1x;
	Mat_<float> u1y;
	Mat_<float> u2x;
	Mat_<float> u2y;
	Mat_<float> u3x;
	Mat_<float> u3y;
	mutable Mat_<float> p11;
	mutable Mat_<float> p12;
	mutable Mat_<float> p21;
	mutable Mat_<float> p22;
	mutable Mat_<float> p31;
	mutable Mat_<float> p32;
	float taut;
	bool use_gamma;
};

void EstimateDualVariablesBody::operator() (const Range& range) const
{
	for (int y = range.start; y < range.end; ++y)
	{
		const float* u1xRow = u1x[y];
		const float* u1yRow = u1y[y];
		const float* u2xRow = u2x[y];
		const float* u2yRow = u2y[y];
		const float* u3xRow = u3x[y];
		const float* u3yRow = u3y[y];

		float* p11Row = p11[y];
		float* p12Row = p12[y];
		float* p21Row = p21[y];
		float* p22Row = p22[y];
		float* p31Row = p31[y];
		float* p32Row = p32[y];

		for (int x = 0; x < u1x.cols; ++x)
		{
			const float g1 = static_cast<float>(hypot(u1xRow[x], u1yRow[x]));
			const float g2 = static_cast<float>(hypot(u2xRow[x], u2yRow[x]));

			const float ng1 = 1.0f + taut * g1;
			const float ng2 = 1.0f + taut * g2;

			p11Row[x] = (p11Row[x] + taut * u1xRow[x]) / ng1;
			p12Row[x] = (p12Row[x] + taut * u1yRow[x]) / ng1;
			p21Row[x] = (p21Row[x] + taut * u2xRow[x]) / ng2;
			p22Row[x] = (p22Row[x] + taut * u2yRow[x]) / ng2;

			if (use_gamma)
			{
				const float g3 = static_cast<float>(hypot(u3xRow[x], u3yRow[x]));
				const float ng3 = 1.0f + taut * g3;
				p31Row[x] = (p31Row[x] + taut * u3xRow[x]) / ng3;
				p32Row[x] = (p32Row[x] + taut * u3yRow[x]) / ng3;
			}
		}
	}
}

static void estimateDualVariables(const Mat_<float>& u1x, const Mat_<float>& u1y,
	const Mat_<float>& u2x, const Mat_<float>& u2y,
	const Mat_<float>& u3x, const Mat_<float>& u3y,
	Mat_<float>& p11, Mat_<float>& p12,
	Mat_<float>& p21, Mat_<float>& p22,
	Mat_<float>& p31, Mat_<float>& p32,
	float taut, bool use_gamma)
{
	CV_DbgAssert(u1y.size() == u1x.size());
	CV_DbgAssert(u2x.size() == u1x.size());
	CV_DbgAssert(u3x.size() == u1x.size());
	CV_DbgAssert(u2y.size() == u1x.size());
	CV_DbgAssert(u3y.size() == u1x.size());
	CV_DbgAssert(p11.size() == u1x.size());
	CV_DbgAssert(p12.size() == u1x.size());
	CV_DbgAssert(p21.size() == u1x.size());
	CV_DbgAssert(p22.size() == u1x.size());
	CV_DbgAssert(p31.size() == u1x.size());
	CV_DbgAssert(p32.size() == u1x.size());

	EstimateDualVariablesBody body;

	body.u1x = u1x;
	body.u1y = u1y;
	body.u2x = u2x;
	body.u2y = u2y;
	body.u3x = u3x;
	body.u3y = u3y;
	body.p11 = p11;
	body.p12 = p12;
	body.p21 = p21;
	body.p22 = p22;
	body.p31 = p31;
	body.p32 = p32;
	body.taut = taut;
	body.use_gamma = use_gamma;

	parallel_for_(Range(0, u1x.rows), body);
}


void OpticalFlowDual_TVL1::procOneScale(const Mat_<float>& I0, const Mat_<float>& I1, Mat_<float>& u1, Mat_<float>& u2, Mat_<float>& u3)
{
	const float scaledEpsilon = static_cast<float>(epsilon * epsilon * I0.size().area());

	CV_DbgAssert(I1.size() == I0.size());
	CV_DbgAssert(I1.type() == I0.type());
	CV_DbgAssert(u1.size() == I0.size());
	CV_DbgAssert(u2.size() == u1.size());

	Mat_<float> I1x = dm.I1x_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> I1y = dm.I1y_buf(Rect(0, 0, I0.cols, I0.rows));
	centeredGradient(I1, I1x, I1y);

	Mat_<float> flowMap1 = dm.flowMap1_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> flowMap2 = dm.flowMap2_buf(Rect(0, 0, I0.cols, I0.rows));

	Mat_<float> I1w = dm.I1w_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> I1wx = dm.I1wx_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> I1wy = dm.I1wy_buf(Rect(0, 0, I0.cols, I0.rows));

	Mat_<float> grad = dm.grad_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> rho_c = dm.rho_c_buf(Rect(0, 0, I0.cols, I0.rows));

	Mat_<float> v1 = dm.v1_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> v2 = dm.v2_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> v3 = dm.v3_buf(Rect(0, 0, I0.cols, I0.rows));

	Mat_<float> p11 = dm.p11_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> p12 = dm.p12_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> p21 = dm.p21_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> p22 = dm.p22_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> p31 = dm.p31_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> p32 = dm.p32_buf(Rect(0, 0, I0.cols, I0.rows));
	p11.setTo(Scalar::all(0));
	p12.setTo(Scalar::all(0));
	p21.setTo(Scalar::all(0));
	p22.setTo(Scalar::all(0));
	bool use_gamma = gamma != 0.;
	if (use_gamma) p31.setTo(Scalar::all(0));
	if (use_gamma) p32.setTo(Scalar::all(0));

	Mat_<float> div_p1 = dm.div_p1_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> div_p2 = dm.div_p2_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> div_p3 = dm.div_p3_buf(Rect(0, 0, I0.cols, I0.rows));

	Mat_<float> u1x = dm.u1x_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> u1y = dm.u1y_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> u2x = dm.u2x_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> u2y = dm.u2y_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> u3x = dm.u3x_buf(Rect(0, 0, I0.cols, I0.rows));
	Mat_<float> u3y = dm.u3y_buf(Rect(0, 0, I0.cols, I0.rows));

	const float l_t = static_cast<float>(lambda * theta);
	const float taut = static_cast<float>(tau / theta);

	for (int warpings = 0; warpings < warps; ++warpings)
	{
		// compute the warping of the target image and its derivatives
		buildFlowMap(u1, u2, flowMap1, flowMap2);
		remap(I1, I1w, flowMap1, flowMap2, INTER_CUBIC);
		remap(I1x, I1wx, flowMap1, flowMap2, INTER_CUBIC);
		remap(I1y, I1wy, flowMap1, flowMap2, INTER_CUBIC);
		//calculate I1(x+u0) and its gradient
		calcGradRho(I0, I1w, I1wx, I1wy, u1, u2, grad, rho_c);

		float error = std::numeric_limits<float>::max();
		for (int n_outer = 0; error > scaledEpsilon && n_outer < outerIterations; ++n_outer)
		{
			if (medianFiltering > 1) {
				cv::medianBlur(u1, u1, medianFiltering);
				cv::medianBlur(u2, u2, medianFiltering);
			}
			for (int n_inner = 0; error > scaledEpsilon && n_inner < innerIterations; ++n_inner)
			{
				// estimate the values of the variable (v1, v2) (thresholding operator TH)
				estimateV(I1wx, I1wy, u1, u2, u3, grad, rho_c, v1, v2, v3, l_t, static_cast<float>(gamma));

				// compute the divergence of the dual variable (p1, p2, p3)
				divergence(p11, p12, div_p1);
				divergence(p21, p22, div_p2);
				if (use_gamma) divergence(p31, p32, div_p3);

				// estimate the values of the optical flow (u1, u2)
				error = estimateU(v1, v2, v3, div_p1, div_p2, div_p3, u1, u2, u3, static_cast<float>(theta), static_cast<float>(gamma));

				// compute the gradient of the optical flow (Du1, Du2)
				forwardGradient(u1, u1x, u1y);
				forwardGradient(u2, u2x, u2y);
				if (use_gamma) forwardGradient(u3, u3x, u3y);

				// estimate the values of the dual variable (p1, p2, p3)
				estimateDualVariables(u1x, u1y, u2x, u2y, u3x, u3y, p11, p12, p21, p22, p31, p32, taut, use_gamma);
			}
		}
	}
}

void OpticalFlowDual_TVL1::collectGarbage()
{
	//dataMat structure dm
	dm.I0s.clear();
	dm.I1s.clear();
	dm.u1s.clear();
	dm.u2s.clear();

	dm.I1x_buf.release();
	dm.I1y_buf.release();

	dm.flowMap1_buf.release();
	dm.flowMap2_buf.release();

	dm.I1w_buf.release();
	dm.I1wx_buf.release();
	dm.I1wy_buf.release();

	dm.grad_buf.release();
	dm.rho_c_buf.release();

	dm.v1_buf.release();
	dm.v2_buf.release();

	dm.p11_buf.release();
	dm.p12_buf.release();
	dm.p21_buf.release();
	dm.p22_buf.release();

	dm.div_p1_buf.release();
	dm.div_p2_buf.release();

	dm.u1x_buf.release();
	dm.u1y_buf.release();
	dm.u2x_buf.release();
	dm.u2y_buf.release();
}