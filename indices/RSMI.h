#ifndef RSMI_H
#define RSMI_H
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
#include "../entities/Query.h"
#include "../entities/LeafNode.h"
#include "../entities/DataSet.h"
#include "../entities/Partition.h"
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

namespace rsmi
{
    ELSI<Point, long long> framework;
    long N;
    DataSet<Point, long long> dataset;
    int page_size = Constants::PAGESIZE;
    long long first_key, last_key, gap;
    int threshold = Constants::THRESHOLD;

    string dataset_name;
    Partition root;

    int cardinality_l = 10000;
    int cardinality_u = 10000000;

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

    // // TODO change mappinng
    // void mapping(vector<Point> &points, vector<long long> &keys)
    // {
    //     N = points.size();
    //     int index_bit_num = ceil((log(N)) / log(2));
    //     // cout << "index_bit_num:" << index_bit_num << endl;
    //     for (long i = 0; i < N; i++)
    //     {
    //         long long xs[2] = {(long long)(points[i].x * N), (long long)(points[i].y * N)};
    //         points[i].curve_val = compute_Z_value(xs, 2, index_bit_num);
    //     }
    //     sort(points.begin(), points.end(), sort_curve_val());

    //     first_key = points[0].curve_val;
    //     last_key = points[N - 1].curve_val;
    //     gap = last_key - first_key;

    //     for (long i = 0; i < N; i++)
    //     {
    //         points[i].index = i;
    //         points[i].label = (float)i / N;
    //         points[i].normalized_key = (float)(points[i].curve_val - first_key) / gap;
    //         keys.push_back(points[i].curve_val);
    //     }
    // }

    void gen_input_keys(vector<Point> &points, vector<float> &keys)
    {
        for (size_t i = 0; i < points.size(); i++)
        {
            keys.push_back(points[i].x);
            keys.push_back(points[i].y);
        }
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

    void point_query(Query<Point> &query)
    {
        int point_not_found = 0;
        vector<Point> query_points = query.query_points;
        int query_num = query_points.size();
        for (size_t i = 0; i < query_num; i++)
        {
            if (!root.point_query(query_points[i]))
            {
                point_not_found++;
            }
        }
        cout << "point_not_found: " << point_not_found << endl;
    }

    void window_query()
    {
    }

    void knn_query()
    {
    }

    void build_single_RSMI(DataSet<Point, long long> dataset, int method)
    {
        Partition partition;
        vector<float> locations;
        vector<float> labels;
        vector<long long> keys;
        partition.is_last = true;
        partition.init_last(dataset.points, locations, labels, keys);
        dataset.normalized_keys = locations;
        dataset.labels = labels;

        std::shared_ptr<MLP> mlp = framework.build_with_method(dataset, method);
        int max_error = 0;
        int min_error = 0;
        for (size_t i = 0; i < dataset.points.size(); i++)
        {
            float x1 = (dataset.points[i].x - partition.x_0) * partition.x_scale + partition.x_0;
            float x2 = (dataset.points[i].y - partition.y_0) * partition.y_scale + partition.y_0;
            int predicted_index = (int)(mlp->predict(x1, x2) * partition.leaf_node_num);
            predicted_index = max(predicted_index, 0);
            predicted_index = min(predicted_index, partition.leaf_node_num - 1);

            int error = i / page_size - predicted_index;
            max_error = max(max_error, error);
            min_error = min(min_error, error);
        }
        mlp->max_error = max_error;
        mlp->min_error = min_error;
        partition.mlp = mlp;
    }

    Partition build_RSMI(ExpRecorder &exp_recorder, vector<Point> points)
    {
        // cout << "build_RSMI: " << points.size() << endl;
        Partition partition;
        DataSet<Point, long long> original_data_set;
        vector<float> locations;
        vector<float> labels;
        vector<long long> keys;
        if (points.size() <= Constants::THRESHOLD)
        {
            partition.is_last = true;
            partition.init_last(points, locations, labels, keys);
            int method = exp_recorder.build_method;
            original_data_set.points = points;
            original_data_set.normalized_keys = locations;
            original_data_set.keys = keys;
            original_data_set.labels = labels;
            if (exp_recorder.is_framework)
            {
                method = framework.build_predict_method(exp_recorder.upper_level_lambda, query_frequency, original_data_set);
            }

            exp_recorder.record_method_nums(method);

            std::shared_ptr<MLP> mlp = framework.build_with_method(original_data_set, method);
            int max_error = 0;
            int min_error = 0;
            for (size_t i = 0; i < points.size(); i++)
            {
                float x1 = (points[i].x - partition.x_0) * partition.x_scale + partition.x_0;
                float x2 = (points[i].y - partition.y_0) * partition.y_scale + partition.y_0;
                int predicted_index = (int)(mlp->predict(x1, x2) * partition.leaf_node_num);
                predicted_index = max(predicted_index, 0);
                predicted_index = min(predicted_index, partition.leaf_node_num - 1);

                int error = i / page_size - predicted_index;
                max_error = max(max_error, error);
                min_error = min(min_error, error);
            }
            mlp->max_error = max_error;
            mlp->min_error = min_error;
            partition.mlp = mlp;
        }
        else
        {
            partition.init(points, locations, labels, keys);
            map<int, vector<Point>> points_map;
            original_data_set.points = points;
            original_data_set.normalized_keys = locations;
            original_data_set.keys = keys;
            original_data_set.labels = labels;

            int method = exp_recorder.build_method;
            if (exp_recorder.is_framework)
            {
                method = framework.build_predict_method(exp_recorder.upper_level_lambda, query_frequency, original_data_set);
            }
            exp_recorder.record_method_nums(method);

            std::shared_ptr<MLP> mlp = framework.build_with_method(original_data_set, method);

            for (size_t i = 0; i < points.size(); i++)
            {
                float x1 = (points[i].x - partition.x_0) * partition.x_scale + partition.x_0;
                float x2 = (points[i].y - partition.y_0) * partition.y_scale + partition.y_0;
                int predicted_index = (int)(mlp->predict(x1, x2) * partition.width);
                predicted_index = max(predicted_index, 0);
                predicted_index = min(predicted_index, partition.width);
                points_map[predicted_index].push_back(points[i]);
            }
            partition.mlp = mlp;
            map<int, vector<Point>>::iterator iter;
            iter = points_map.begin();

            while (iter != points_map.end())
            {
                if (iter->second.size() > 0)
                {
                    // cout << iter->first << "    " << iter->second.size() << endl;
                    partition.children.insert(pair<int, Partition>(iter->first, build_RSMI(exp_recorder, iter->second)));
                }
                iter++;
            }
        }
        return partition;
    }

    void query(Query<Point> query, ExpRecorder &exp_recorder)
    {
        print("query--------------------");
        exp_recorder.timer_begin();
        framework.query(query);
        exp_recorder.timer_end();
        // if (query.is_window())
        // {
        //     long query_size = query.results.size();
        //     query.results.clear();
        //     query.results.shrink_to_fit();
        //     long size = acc_window_query_num(query);
        //     exp_recorder.accuracy = (float)query_size / size;
        //     cout << "accuracy: " << exp_recorder.accuracy << endl;
        // }
        // if (query.is_knn())
        // {
        //     vector<Point> res;
        //     cout << "knn query size: " << query.results.size() << endl;
        //     int found_num = 0;
        //     for (size_t i = 0; i < query.knn_query_points.size(); i++)
        //     {
        //         priority_queue<Point, vector<Point>, sortForKNN2> pq;
        //         int k = query.get_k();
        //         float knn_query_side = sqrt((float)k / N) * 4;
        //         Mbr window = Mbr::get_mbr(query.knn_query_points[i], knn_query_side);
        //         for (size_t j = 0; j < storage_leafnodes.size(); j++)
        //         {
        //             if (storage_leafnodes[j].mbr.interact(window))
        //             {
        //                 for (Point point : storage_leafnodes[j].children)
        //                 {
        //                     if (window.contains(point))
        //                     {
        //                         point.temp_dist = point.cal_dist(query.knn_query_points[i]);
        //                         if (pq.size() < k)
        //                         {
        //                             pq.push(point);
        //                         }
        //                         else
        //                         {
        //                             if (pq.top().temp_dist > point.temp_dist)
        //                             {
        //                                 pq.pop();
        //                                 pq.push(point);
        //                             }
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //         vector<Point> query_res(query.results.begin() + i * k, query.results.begin() + i * k + k);
        //         while (!pq.empty())
        //         {
        //             vector<Point>::iterator iter = find(query_res.begin(), query_res.end(), pq.top());
        //             if (iter != query_res.end())
        //             {
        //                 found_num++;
        //             }

        //             pq.pop();
        //         }
        //     }
        //     exp_recorder.accuracy = (float)found_num / query.results.size();
        //     cout << "accuracy: " << exp_recorder.accuracy << endl;
        // }
    }

    void init(string _dataset_name, ExpRecorder &exp_recorder)
    {
        dataset_name = _dataset_name;
        // generate_points();
        exp_recorder.name = "RSMI";
        exp_recorder.timer_begin();
        framework.point_query_p = point_query;
        // framework.window_query_p = window_query;
        // framework.knn_query_p = kNN_query;
        framework.build_index_p = build_single_RSMI;
        // framework.init_storage_p = init_underlying_data_storage;
        // framework.insert_p = insert;
        // framework.generate_points_p = generate_points;
        framework.max_cardinality = cardinality_u;
        framework.dimension = 2;

        DataSet<Point, long long>::read_data_pointer = read_data;
        // DataSet<Point, long long>::mapping_pointer = mapping;
        DataSet<Point, long long>::gen_input_keys_pointer = gen_input_keys;
        DataSet<Point, long long>::save_data_pointer = save_data;
        framework.index_name = "RSMI";
        framework.config_method_pool();
        if (exp_recorder.is_framework)
        {
            framework.init();
        }

        exp_recorder.timer_end();
        print("framework init time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");

        dataset.dataset_name = _dataset_name;
        dataset.read_data();
        exp_recorder.timer_end();
        print("read data time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");
        N = dataset.points.size();
    }

}
#endif // use_gpu
