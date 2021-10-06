#ifndef Config_H
#define Config_H

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <ctime>

namespace config
{
    float sampling_rate = 0.1;
    int bit_num = 8;
    int rs_m = 100;
    int cluster_k = 100;

    float lambda = 0.8;

}

#endif