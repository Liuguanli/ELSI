#ifndef STAT_H
#define STAT_H
#include <vector>
#include <string>
#include "../utils/Constants.h"
#include "DataSetInfo.h"

using namespace std;

template <typename D, typename T>
class Statistices
{
public:
    DataSetInfo<T> current_info;
    DataSetInfo<T> initial_info;
    long cardinality;
    long inserted;
    long deleted;
    float cdf_change;
    int initial_depth;
    int current_depth;
    float relative_depth;
    float update_ratio;
    vector<float> distribution_encoding;

    void insert() { inserted++; }
    void remove() { deleted++; }

    void get_input()
    {
        vector<float> parameters;

        vector<float> distribution_list = current_info.get_distribution();
        parameters.push_back(cardinality);

        cdf_change = current_info.cal_similarity(initial_info.cdf);
        parameters.push_back(cdf_change);

        relative_depth = (float)current_depth / initial_depth;
        parameters.push_back(relative_depth);

        update_ratio = (float)inserted / cardinality;
        parameters.push_back(update_ratio);
        parameters.insert(parameters.end(), distribution_list.begin(), distribution_list.end());
    }
};

#endif