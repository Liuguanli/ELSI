#ifndef STAT_H
#define STAT_H
#include <vector>
#include <string>
#include <sstream>
#include "../utils/Constants.h"
#include "DataSetInfo.h"

using namespace std;

class Statistics
{
public:
    float cardinality;
    long inserted;
    long deleted;
    float cdf_change;

    int initial_depth;
    int current_depth;
    float relative_depth;

    float update_ratio;
    float distribution;

    bool is_rebuild = false;

    void insert() { inserted++; }
    void remove() { deleted++; }

    vector<float> get_input()
    {
        vector<float> parameters;

        parameters.push_back(cardinality);

        parameters.push_back(cdf_change);

        // relative_depth = (initial_depth == 0 || current_depth == 0) ? 1 : (float)current_depth / initial_depth;
        parameters.push_back(relative_depth);

        parameters.push_back(update_ratio);

        parameters.push_back(distribution);

        return parameters;
    }

    string get_Statistics()
    {
        stringstream ss;
        vector<float> parameters = get_input();
        copy(parameters.begin(), parameters.end(), ostream_iterator<int>(ss, ","));
        string s = ss.str();
        // s = s.substr(0, s.length() - 1);
        return s + to_string(is_rebuild ? 1 : 0);
    }
};

#endif