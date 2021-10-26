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
    DataSet<D, T> do_sp(DataSet<D, T> &dataset, float sampling_rate, int dimension)
    {
        if (dimension == 1)
        {
            return do_sp(dataset, sampling_rate);
        }
        if (dimension == 2)
        {
            return do_sp_2d(dataset, sampling_rate);
        }
        return dataset;
    }

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

    DataSet<D, T> do_sp_2d(DataSet<D, T> &data_set, float sampling_rate)
    {
        vector<D> points;
        vector<float> normalized_keys;
        vector<float> labels;
        int sample_gap = 1 / sampling_rate;
        long long counter = 0;
        // int start = (rand() % (sample_gap));
        int start = 0;
        int count = data_set.points.size();
        for (size_t i = start; i < count; i += sample_gap)
        {
            points.push_back(data_set.points[i]);
            normalized_keys.push_back(data_set.normalized_keys[2 * i]);
            normalized_keys.push_back(data_set.normalized_keys[2 * i + 1]);
            labels.push_back(data_set.labels[i]);
        }
        DataSet<D, T> sampled_data_set;
        sampled_data_set.points = points;
        sampled_data_set.normalized_keys = normalized_keys;
        sampled_data_set.labels = labels;
        cout << "size sp:" << sampled_data_set.points.size() << endl;
        return sampled_data_set;
    }
};

#endif