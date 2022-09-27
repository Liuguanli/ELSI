#ifndef FLOOD_H
#define FLOOD_H
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <queue>
#include "../entities/Point.h"
#include "../entities/Bucket.h"
#include "../entities/Query.h"
#include "../entities/LeafNode.h"
#include "../entities/DataSet.h"
#include "../utils/FileReader.h"
#include "../utils/SortTools.h"
#include "../utils/Config.h"
#include "../utils/Log.h"
#include "../utils/ExpRecorder.h"
#include "../curves/z.H"

#include "../ELSI.h"

using namespace std;
using namespace config;
using namespace logger;

namespace flood
{
    ELSI<Point, long long> framework;
    vector<float> split_points;

    vector<Bucket> buckets;

    vector<LeafNode> storage_leafnodes;
    DataSet<Point, long long> dataset;

    int learned_dim = -1;
    int partition_size = -1;
    long N;

    vector<Point> read_data(string filename, string delimeter, double &min_x, double &min_y, double &max_x, double &max_y)
    {
        ifstream file(filename);

        vector<Point> points;

        string line = "";
        while (getline(file, line))
        {
            vector<string> vec;
            boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
            Point point(stod(vec[0]), stod(vec[1]));

            max_x = max_x > point.x ? max_x : point.x;
            max_y = max_y > point.y ? max_y : point.y;
            min_x = min_x < point.x ? min_x : point.x;
            min_y = min_y < point.y ? min_y : point.y;

            points.push_back(point);
        }

        file.close();

        return points;
    }

    void mapping(vector<Point> &points, vector<long long> &keys)
    {
    }

    void save_data(vector<Point> points, string filename)
    {
        ofstream write;
        write.open(filename, ios::out);
        for (Point point : points)
        {
            write << to_string(point.x) + "," + to_string(point.y) + "\n";
        }
        write.close();
    }

    int get_point_index(Point &query_point, long &front, long &back)
    {
    }

    bool point_query(Point &query_point)
    {
        // float key = learned_dim == 0 ? query_point.y : query_point.x;

        // int index = 0;
        // if (key < split_points[0])
        // {
        //     index = 0;
        // }
        // else if (key > split_points[split_points.size() - 1])
        // {
        //     index = buckets.size() - 1;
        // }
        // else
        // {
        //     vector<int>::iterator target;
        //     target = std::upper_bound(split_points.begin(), split_points.end(), to);
        //     index = target - split_points.begin();
        // }
        
        // return buckets[index].point_query(query_point);
        return true;
      
    }

    void point_query(Query<Point> &query)
    {
        int point_not_found = 0;
        vector<Point> query_points = query.query_points;
        int query_num = query_points.size();
        for (size_t i = 0; i < query_num; i++)
        {
            if (!point_query(query.query_points[i]))
            {
                if (!framework.point_query(query.query_points[i]))
                {
                    point_not_found++;
                }
            }
        }
        printf("point_not_found %d\n", point_not_found);
    }

    void window_query(vector<Point> &results, Mbr &query_window)
    {
        // float from = learned_dim == 0 ? query_window.y1 : query_window.x1;
        // float to = learned_dim == 0 ? query_window.y2 : query_window.x2;
        // int begin = 0;
        // int end = buckets.size() - 1;
        // if (from >= split_points[0])
        // {
        //     vector<int>::iterator lower;
        //     lower = std::lower_bound(split_points.begin(), split_points.end(), from);
        //     begin = lower - split_points.begin();
        // }
        // if (to < split_points[split_points.size() - 1])
        // {
        //     vector<int>::iterator upper;
        //     upper = std::upper_bound(split_points.begin(), split_points.end(), to);
        //     end = upper - split_points.begin();
        // }
        // for (size_t i = begin; i <= end; i++)
        // {
        //     buckets[i].window_query(query_window, results);
        // }
    }

    void window_query(Query<Point> &query)
    {
        query.results.clear();
        query.results.shrink_to_fit();
        int query_num = query.query_windows.size();
        for (size_t i = 0; i < query_num; i++)
        {
            window_query(query.results, query.query_windows[i]);
            query.results.clear();
            query.results.shrink_to_fit();
        }
    }

    void kNN_query(vector<Point> &results, Point query_point, int k)
    {
    }

    void kNN_query(Query<Point> &query)
    {
        query.results.clear();
        query.results.shrink_to_fit();
        int query_num = query.knn_query_points.size();
        int k = query.get_k();
        for (size_t i = 0; i < query_num; i++)
        {
            kNN_query(query.results, query.knn_query_points[i], k);
        }
    }

    void insert(Point point)
    {
    }

    bool delete_(Point &query_point)
    {
    }

    void build_Flood(ExpRecorder &exp_recorder)
    {
        assert(learned_dim >= 0);
        assert(partition_size > 0);

        if (learned_dim == 0)
        {
            sort(dataset.points.begin(), dataset.points.end(), sortY());
        }
        else
        {
            sort(dataset.points.begin(), dataset.points.end(), sortX());
        }
        N = dataset.points.size();
        int bucket_volume = ceil(N / partition_size);
        for (size_t i = 0; i < partition_size; i++)
        {
            long first = i * bucket_volume;
            long last = min((long)i * bucket_volume + bucket_volume, N);
            auto bn = dataset.points.begin() + first;
            auto en = dataset.points.begin() + last;
            vector<Point> points(bn, en);
            Bucket bucket(learned_dim);
            bucket.build(exp_recorder, dataset.points, framework);
            buckets.push_back(bucket);
            if (i != partition_size - 1)
            {
                if (learned_dim == 0)
                {
                    split_points.push_back((dataset.points[last - 1].y + dataset.points[last].y) / 2);
                }
                else
                {
                    split_points.push_back((dataset.points[last - 1].x + dataset.points[last].x) / 2);
                }
            }
        }
    }

    void query(Query<Point> &query, ExpRecorder &exp_recorder)
    {
        exp_recorder.timer_begin();
        framework.query(query);
        exp_recorder.timer_end();
    }

    void init(string _dataset_name, ExpRecorder &exp_recorder)
    {
        exp_recorder.name = "Flood";
        exp_recorder.timer_begin();

        vector<int> methods{Constants::CL, Constants::MR, Constants::OG, Constants::RL, Constants::RS, Constants::SP};
        config::init_method_pool(methods);

        framework.config_method_pool();
        framework.dimension = 1;
        framework.point_query_p = point_query;
        framework.window_query_p = window_query;
        framework.knn_query_p = kNN_query;
        // framework.init_storage_p = init_underlying_data_storage;
        framework.insert_p = insert;
        // framework.generate_points_p = generate_points;
        framework.max_cardinality = cardinality_u;

        DataSet<Point, long long>::read_data_pointer = read_data;
        DataSet<Point, long long>::mapping_pointer = mapping;
        DataSet<Point, long long>::save_data_pointer = save_data;

        // stages.push_back(1);

        if (exp_recorder.is_framework)
        {
            framework.init();
        }

        exp_recorder.timer_end();
        print("framework init time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");
        exp_recorder.timer_begin();

        dataset.dataset_name = _dataset_name;
        dataset.read_data();
        exp_recorder.timer_end();
        print("read data time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");

        N = dataset.points.size();
    }
}
#endif // use_gpu
