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
    map<int, int> method_pool;
    float sampling_rate = 0.0001;
    int bit_num = 8;
    int rs_m = 100;
    int cluster_k = 100;

    int cardinality_l = 1e4;
    int cardinality_u = 1e8;

    float lambda = 0.8;
    float query_frequency = 1.0;

    void init_method_pool(vector<int> methods)
    {
        for (size_t i = 0; i < methods.size(); i++)
        {
            method_pool.insert(pair<int, int>(i, methods[i]));
        }
    }

}

#endif