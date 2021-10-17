#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

class OpticalFlowDual_TVL1
{
public:

	OpticalFlowDual_TVL1(double tau_, double lambda_, double theta_, int nscales_, int warps_,
		double epsilon_, int innerIterations_, int outerIterations_,
		double scaleStep_, double gamma_, int medianFiltering_,
		bool useInitialFlow_) :
		tau(tau_), lambda(lambda_), theta(theta_), gamma(gamma_), nscales(nscales_),
		warps(warps_), epsilon(epsilon_), innerIterations(innerIterations_),
		outerIterations(outerIterations_), useInitialFlow(useInitialFlow_),
		scaleStep(scaleStep_), medianFiltering(medianFiltering_)
	{
	}
	OpticalFlowDual_TVL1();

	void calc(InputArray I0, InputArray I1, InputOutputArray flow);
	void collectGarbage() ;

	inline double getTau() const  { return tau; }
	inline void setTau(double val) { tau = val; }
	inline double getLambda() const  { return lambda; }
	inline void setLambda(double val)  { lambda = val; }
	inline double getTheta() const { return theta; }
	inline void setTheta(double val) { theta = val; }
	inline double getGamma() const  { return gamma; }
	inline void setGamma(double val) { gamma = val; }
	inline int getScalesNumber() const  { return nscales; }
	inline void setScalesNumber(int val) { nscales = val; }
	inline int getWarpingsNumber() const { return warps; }
	inline void setWarpingsNumber(int val)  { warps = val; }
	inline double getEpsilon() const { return epsilon; }
	inline void setEpsilon(double val) { epsilon = val; }
	inline int getInnerIterations() const  { return innerIterations; }
	inline void setInnerIterations(int val) { innerIterations = val; }
	inline int getOuterIterations() const  { return outerIterations; }
	inline void setOuterIterations(int val) { outerIterations = val; }
	inline bool getUseInitialFlow() const { return useInitialFlow; }
	inline void setUseInitialFlow(bool val)  { useInitialFlow = val; }
	inline double getScaleStep() const { return scaleStep; }
	inline void setScaleStep(double val)  { scaleStep = val; }
	inline int getMedianFiltering() const{ return medianFiltering; }
	inline void setMedianFiltering(int val) { medianFiltering = val; }

protected:
	double tau;
	double lambda;
	double theta;
	double gamma;
	int nscales;
	int warps;
	double epsilon;
	int innerIterations;
	int outerIterations;
	bool useInitialFlow;
	double scaleStep;
	int medianFiltering;

private:
	void procOneScale(const Mat_<float>& I0, const Mat_<float>& I1, Mat_<float>& u1, Mat_<float>& u2, Mat_<float>& u3);

	struct dataMat
	{
		std::vector<Mat_<float> > I0s;
		std::vector<Mat_<float> > I1s;
		std::vector<Mat_<float> > u1s;
		std::vector<Mat_<float> > u2s;
		std::vector<Mat_<float> > u3s;

		Mat_<float> I1x_buf;
		Mat_<float> I1y_buf;

		Mat_<float> flowMap1_buf;
		Mat_<float> flowMap2_buf;

		Mat_<float> I1w_buf;
		Mat_<float> I1wx_buf;
		Mat_<float> I1wy_buf;

		Mat_<float> grad_buf;
		Mat_<float> rho_c_buf;

		Mat_<float> v1_buf;
		Mat_<float> v2_buf;
		Mat_<float> v3_buf;

		Mat_<float> p11_buf;
		Mat_<float> p12_buf;
		Mat_<float> p21_buf;
		Mat_<float> p22_buf;
		Mat_<float> p31_buf;
		Mat_<float> p32_buf;

		Mat_<float> div_p1_buf;
		Mat_<float> div_p2_buf;
		Mat_<float> div_p3_buf;

		Mat_<float> u1x_buf;
		Mat_<float> u1y_buf;
		Mat_<float> u2x_buf;
		Mat_<float> u2y_buf;
		Mat_<float> u3x_buf;
		Mat_<float> u3y_buf;
	} dm;
};
