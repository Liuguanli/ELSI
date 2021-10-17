#ifndef SP_H
#define SP_H

#include <vector>
#include <string.h>
#include <string>
#include <cstdlib>
#include "../entities/DataSet.h"

template <typename D, typename T>
class SP
{
public:
    DataSet<D, T> do_sp(DataSet<D, T> &data_set, float sampling_rate)
    {
        vector<D> points;
        int sample_gap = 1 / sampling_rate;
        long long counter = 0;
        // int start = (rand() % (sample_gap));
        int start = 0;
        int count = data_set.points.size();
        for (size_t i = start; i < count; i += sample_gap)
        {
            points.push_back(data_set.points[i]);
        }
        DataSet<D, T> sampled_data_set(points);
        cout << "size sp:" << sampled_data_set.points.size() << endl;
        return sampled_data_set;
    }
};

#endif