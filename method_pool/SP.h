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
    DataSet<D, T> do_sp(DataSet<D, T> &dataset, float sampling_rate, bool is_systematic_sampling, int dimension)
    {
        if (dimension == 1)
        {
            return do_sp(dataset, sampling_rate, is_systematic_sampling);
        }
        if (dimension == 2)
        {
            return do_sp_2d(dataset, sampling_rate, is_systematic_sampling);
        }
        return dataset;
    }

    DataSet<D, T> do_sp(DataSet<D, T> &data_set, float sampling_rate, bool is_systematic_sampling)
    {

        vector<D> points;
        DataSet<D, T> sampled_data_set;
        if (is_systematic_sampling)
        {
            int sample_gap = 1 / sampling_rate;
            long long counter = 0;
            // int start = (rand() % (sample_gap));
            int start = 0;
            int count = data_set.points.size();
            for (size_t i = start; i < count; i += sample_gap)
            {
                points.push_back(data_set.points[i]);
            }
            // DataSet<D, T> sampled_data_set(points);
            sampled_data_set.points = points;
            sampled_data_set.read_keys_and_labels();
            // cout << "size sp:" << sampled_data_set.points.size() << endl;
        }
        else
        {
            int count = data_set.points.size();
            int n = count * sampling_rate;
            while (points.size() < n)
            {
                int index = (rand() % (count));
                points.push_back(data_set.points[index]);
            }
            sort(points.begin(), points.end(), sort_label());
            sampled_data_set.points = points;
            sampled_data_set.read_keys_and_labels();
        }
        return sampled_data_set;
    }

    DataSet<D, T> do_sp_2d(DataSet<D, T> &data_set, float sampling_rate, bool is_systematic_sampling)
    {
        vector<D> points;
        vector<float> normalized_keys;
        vector<float> labels;
        DataSet<D, T> sampled_data_set;
        if (is_systematic_sampling)
        {
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
            sampled_data_set.points = points;
            sampled_data_set.normalized_keys = normalized_keys;
            sampled_data_set.labels = labels;
            // cout << "size sp:" << sampled_data_set.points.size() << endl;
        }
        else
        {
            int count = data_set.points.size();
            int n = count * sampling_rate;
            while (points.size() < n)
            {
                int index = (rand() % (count));
                points.push_back(data_set.points[index]);
                normalized_keys.push_back(data_set.normalized_keys[2 * index]);
                normalized_keys.push_back(data_set.normalized_keys[2 * index + 1]);
                labels.push_back(data_set.labels[index]);
            }
            
            sampled_data_set.points = points;
            sampled_data_set.read_keys_and_labels();
        }

        return sampled_data_set;
    }
};

#endif