#define _USE_MATH_DEFINES
#include <math.h>
#include "MathHelper.h"
namespace Neural {
	using namespace std;
	double MathHelper::Sigmoid(double x) {
		return 1.0 / (1.0 + pow(M_E, -x));
	}
}