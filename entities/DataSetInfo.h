#ifndef DATASETINFO_CPP
#define DATASETINFO_CPP

#include <vector>
#include <map>

#include "../utils/Constants.h"
#include <cmath>
#include <assert.h>

using namespace std;

template <typename T>
class DataSetInfo
{
private:
    // long long binary_search(float);

    int bin_num = Constants::DEFAULT_BIN_NUM;

    T start_x;
    T end_x;
    T key_gap;

    int N = 0;

public:
    vector<float> hist;
    vector<float> cdf;
    vector<float> uniform_cdf;

    DataSetInfo(int bin_num, vector<T> &data)
    {
        N = data.size();
        assert(N > 0);
        start_x = data[0];
        end_x = data[N - 1];

        this->bin_num = bin_num;
        key_gap = (end_x - start_x) * 1.0 / bin_num;
        int cdf_index = 0;
        for (size_t i = 1; i < bin_num; i++)
        {
            int inner_index = 0;
            while (cdf_index < N && data[cdf_index] <= (start_x + i * key_gap))
            {
                inner_index++;
                cdf_index++;
            }
            hist.push_back(inner_index * 1.0 / N);
            cdf.push_back(cdf_index * 1.0 / N);

            uniform_cdf.push_back((float)i / bin_num);
        }
        hist.push_back(1 - cdf_index * 1.0 / N);
        cdf.push_back(1.0);
        uniform_cdf.push_back(1.0);
    }

    void update(T value)
    {
        int index = ceil((value - start_x) * 1.0 / key_gap);
        for (size_t i = index; i < hist.size(); i++)
        {
            hist[i] = hist[i] + 1.0 / N;
        }
        for (size_t i = 0; i < hist.size(); i++)
        {
            hist[i] = hist[i] * N / (N + 1);
        }
        N++;
    }

    // Histogram(std::vector<float>);
    // float cal_dist(std::vector<float>);
    float cal_dist(vector<float> source_cdf)
    {
        float dist = 0;
        for (size_t i = 0; i < source_cdf.size(); i++)
        {
            float temp = abs(source_cdf[i] - cdf[i]);

            if (dist < temp)
            {
                dist = temp;
            }
        }
        return dist;
    }

    float cal_similarity(vector<float> source_cdf)
    {
        return 1 - cal_dist(source_cdf);
    }

    float get_distribution()
    {
        return cal_dist(uniform_cdf);
    }
};

#endif