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
    bool is_systematic_sampling = true;
    int bit_num = 8;
    int rs_m = 100;
    int cluster_k = 100;

    int cardinality_l = 1e4;
    int cardinality_u = 1e8;

    float lambda = 0.8;
    float query_frequency = 1.0;

    int build_time_model_training_cardinality_bound = 8;

    string build_time_model_path = Constants::BUILD_TIME_MODEL_PATH;
    string query_time_model_path = Constants::QUERY_TIME_MODEL_PATH;

    void init_method_pool(vector<int> methods)
    {
        for (size_t i = 0; i < methods.size(); i++)
        {
            method_pool.insert(pair<int, int>(i, methods[i]));
        }
    }

    void set_method_value(int method, float value)
    {
        if (method == Constants::CL)
            cluster_k = (int)value;
        if (method == Constants::MR)
            sampling_rate = value;
        if (method == Constants::RL)
            bit_num = (int)value;
        if (method == Constants::RS)
            rs_m = (int)value;
        if (method == Constants::SP)
            sampling_rate = value;
    }
}
#endif