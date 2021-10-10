//
//  Description:		source for computing Z-order curve value
//  Created:			2016/12/11
//  Last updated:		2019/10/17
//  Author:				Yanchuan Chang
//  Mail:				changyanchuan@gmail.com
//

#include "z.H"
#include <assert.h>
#include <math.h>
#include <iostream>
__uint128_t compute_Z_value(long long x[], const size_t &x_len, const long long &bits)
{
	assert(bits > 0);
	assert(x_len > 0);
	// for (size_t i = 0; i < x_len; ++i)
	// 	{ assert(x[i] > 0); }

	__uint128_t z = 0;
	for (long long i = 0; i < bits; ++i)
	{
		for (size_t j = 0; j < x_len; ++j)
		{
			z += (x[j] % 2) * (__uint128_t)2 << (x_len * i + j);
			x[j] /= 2;
		}
	}
	return z;
}

long long compute_Z_value(long long x, long long y, int bit_num)
{
	long long result = 0;
	long seed = 1;
	for (int i = 0; i < bit_num; i++)
	{
		result += (((seed & y) << 1) + (seed & x)) << i;
		seed = seed << 1;
	}
	return result;
}
