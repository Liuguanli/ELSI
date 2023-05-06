#ifndef QUERY_H
#define QUERY_H
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "../utils/Constants.h"
#include "../entities/Mbr.h"

using namespace std;
std::random_device rd;
std::mt19937 rg(rd());
template <typename D>
class Query
{
private:
    int type = 0;
    // vector<D> query_points;
    // vector<D> knn_query_points;
    // vector<Mbr> query_windows;
    int k = 1;
    int iteration_num = 1;

public:
    vector<D> query_points;
    vector<D> knn_query_points;
    vector<Mbr> query_windows;
    vector<D> results;
    vector<D> extra_storage_results;
    bool is_point() { return type == Constants::QUERY_TYPE_POINT; }
    bool is_window() { return type == Constants::QUERY_TYPE_WINDOW; }
    bool is_knn() { return type == Constants::QUERY_TYPE_KNN; }
    int get_k() { return k; }
    int get_iteration_num() { return iteration_num; }
    // vector<D> get_query_points() { return query_points; }
    // vector<D> get_knn_query_points() { return knn_query_points; }
    // vector<Mbr> get_query_windows() { return query_windows; }

    void shuffle()
    {
        if (is_point())
        {
            std::shuffle(query_points.begin(), query_points.end(), rg);
        }
        else if (is_window())
        {
            std::shuffle(query_windows.begin(), query_windows.end(), rg);
        }
        else if (is_knn())
        {
            std::shuffle(knn_query_points.begin(), knn_query_points.end(), rg);
        }
    }

    Query *set_point_query()
    {
        type = Constants::QUERY_TYPE_POINT;
        return this;
    }

    Query *set_window_query()
    {
        type = Constants::QUERY_TYPE_WINDOW;
        return this;
    }

    Query *set_knn_query()
    {
        type = Constants::QUERY_TYPE_KNN;
        return this;
    }

    Query *set_k(int k)
    {
        this->k = k;
        return this;
    }

    // Query *set_query_points(vector<D> points)
    // {
    //     this->query_points = points;
    //     return this;
    // }

    // Query *set_knn_query_points(vector<D> points)
    // {
    //     this->knn_query_points = points;
    //     return this;
    // }

    // Query *set_query_windows(vector<Mbr> windows)
    // {
    //     this->query_windows = windows;
    //     return this;
    // }

    Query *set_iterations(int iteration)
    {
        this->iteration_num = iteration;
        return this;
    }
};

#endif