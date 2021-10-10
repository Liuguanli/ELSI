#ifndef ZM_H
#define ZM_H
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
#include "../utils/FileReader.h"
#include "../utils/SortTools.h"
#include "../utils/Config.h"
#include "../utils/Log.h"
#include "../curves/z.H"

#include "../ELSI.h"

using namespace std;
using namespace config;
using namespace logger;

namespace zm
{
    ELSI<Point, long long> framework;
    vector<int> stages;
    vector<vector<std::shared_ptr<MLP>>> index;
    long N;
    vector<vector<vector<Point>>> records;
    vector<LeafNode> storage_leafnodes;
    DataSet<Point, long long> dataset;
    int page_size = Constants::PAGESIZE;
    int error_shift = 0;
    long long first_key, last_key, gap;

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
        int N = points.size();
        bit_num = ceil((log(N)) / log(2));
        for (long i = 0; i < N; i++)
        {
            long long xs[2] = {(long long)(points[i].x * N), (long long)(points[i].y * N)};
            points[i].curve_val = compute_Z_value(xs, 2, bit_num);
        }
        sort(points.begin(), points.end(), sort_curve_val());

        first_key = points[0].curve_val;
        last_key = points[N - 1].curve_val;
        gap = last_key - first_key;

        for (long i = 0; i < N; i++)
        {
            points[i].index = i;
            points[i].label = (float)i / N;
            points[i].normalized_key = (float)(points[i].curve_val - first_key) / gap;
            keys.push_back(points[i].curve_val);
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

    void init_underlying_data_storage(DataSet<Point, long long> dataset)
    {
        storage_leafnodes.clear();
        // storage_leafnodes.shrink_to_fit();
        N = N == 0 ? dataset.points.size() : N;
        int leaf_node_num = ceil((float)N / page_size);
        storage_leafnodes.resize(leaf_node_num);
        for (int i = 0; i < leaf_node_num; i++)
        {
            auto bn = dataset.points.begin() + i * page_size;
            auto en = dataset.points.begin() + min((long)i * page_size + page_size, N);
            vector<Point> points(bn, en);
            LeafNode leaf_node(points);
            storage_leafnodes[i] = leaf_node;
        }
    }

    int get_point_index(Point query_point, long index_low, long index_high)
    {
        long long xs[2] = {(long long)(query_point.x * N), (long long)(query_point.y * N)};
        long long curve_val = compute_Z_value(xs, 2, bit_num);
        query_point.curve_val = curve_val;
        float key = (curve_val - first_key) * 1.0 / gap;
        query_point.normalized_key = key;
        int predicted_index = 0;
        int next_stage_length = 1;
        int min_error = 0;
        int max_error = 0;
        int level = stages.size();
        for (int j = 0; j < level - 1; j++)
        {
            next_stage_length = stages[j + 1];
            predicted_index = index[j][predicted_index]->predict_ZM(key) * next_stage_length;
            predicted_index = max(predicted_index, 0);
            predicted_index = min(predicted_index, next_stage_length - 1);
        }

        next_stage_length = N;
        min_error = index[level - 1][predicted_index]->min_error;
        max_error = index[level - 1][predicted_index]->max_error;

        long front = predicted_index + min_error - error_shift;
        front = max((long)0, front);
        long back = predicted_index + max_error + error_shift;
        back = max(N - 1, back);
        front = front / page_size;
        back = back / page_size;
        // cout << "predicted_index: " << predicted_index << " max_error: " << max_error << " min_error: " << min_error << " error_shift: " << error_shift << endl;
        return predicted_index;
    }

    void point_query(Query<Point> query)
    {
        int point_not_found = 0;
        vector<Point> query_points = query.get_query_points();
        int query_num = query_points.size();
        for (size_t i = 0; i < query_num; i++)
        {
            long front, back;
            int predicted_index = get_point_index(query_points[i], front, back);

            // cout << "front: " << front << " back: " << back << endl;
            while (front <= back)
            {
                int mid = (front + back) / 2;
                LeafNode leafnode = storage_leafnodes[mid];
                long long first_curve_val = leafnode.children[0].curve_val;
                long long last_curve_val = leafnode.children[leafnode.children.size() - 1].curve_val;
                // cout << "first_curve_val: " << first_curve_val << " last_curve_val: " << last_curve_val << endl;

                if (first_curve_val <= query_points[i].curve_val && query_points[i].curve_val <= last_curve_val)
                {
                    vector<Point>::iterator iter = find(leafnode.children.begin(), leafnode.children.end(), query_points[i]);
                    if (iter != leafnode.children.end())
                    {
                        break;
                    }
                    else
                    {
                        point_not_found++;
                    }
                }
                else
                {
                    if (leafnode.children[0].curve_val < query_points[i].curve_val)
                    {
                        front = mid + 1;
                    }
                    else
                    {
                        back = mid - 1;
                    }
                }
            }
        }
        cout << "point_not_found: " << point_not_found << endl;
    }

    vector<Point> window_query(Query<Point> query)
    {
        int point_not_found = 0;
        vector<Mbr> query_windows = query.get_query_windows();
        int query_num = query_windows.size();
        vector<Point> window_query_results;
        for (size_t i = 0; i < query_num; i++)
        {
            vector<Point> vertexes = query_windows[i].get_corner_points();
            vector<long> indices;
            for (Point point : vertexes)
            {
                long index_low, index_high;
                get_point_index(point, index_low, index_high);
                indices.push_back(index_low);
                indices.push_back(index_high);
            }
            sort(indices.begin(), indices.end());

            long front = indices.front();
            long back = indices.back();

            for (size_t j = front; j <= back; j++)
            {
                if (storage_leafnodes[j].mbr.interact(query_windows[i]))
                {
                    for (Point point : storage_leafnodes[j].children)
                    {
                        if (query_windows[i].contains(point))
                        {
                            window_query_results.push_back(point);
                        }
                    }
                }
            }
        }
        return window_query_results;
    }

    vector<Point> kNN_query(Query<Point> query)
    {
        vector<Point> knn_query_points = query.get_knn_query_points();
        int query_num = knn_query_points.size();
        vector<Point> result;
        Query<Point> window_query_for_knn;
        window_query_for_knn.set_window_query();
        int k = query.get_k();
        for (size_t i = 0; i < query_num; i++)
        {
            priority_queue<Point, vector<Point>, sortForKNN2> pq;
            float knn_query_side = sqrt((float)k / N) * 2;
            while (true)
            {
                vector<Mbr> window = {Mbr::get_mbr(knn_query_points[i], knn_query_side)};
                window_query_for_knn.set_query_windows(window);
                vector<Point> temp_result = window_query(window_query_for_knn);
                if (temp_result.size() >= k)
                {
                    for (size_t j = 0; j < temp_result.size(); j++)
                    {
                        double temp_dist = temp_result[j].cal_dist(knn_query_points[i]);
                        temp_result[j].temp_dist = temp_dist;
                        if (pq.size() < k)
                        {
                            pq.push(temp_result[j]);
                        }
                        else
                        {
                            if (pq.top().temp_dist > temp_dist)
                            {
                                pq.pop();
                                pq.push(temp_result[j]);
                            }
                        }
                    }
                    if (pq.top().temp_dist <= knn_query_side)
                    {
                        break;
                    }
                }
                knn_query_side *= 2;
            }
        }

        return result;
    }

    vector<Point> knn_query(Query<Point> query)
    {
        vector<Point> res;
        return res;
    }

    void build_single_ZM(DataSet<Point, long long> dataset, int method)
    {
        print("build_single_ZM");

        vector<std::shared_ptr<MLP>> temp_index;
        int next_stage_length = dataset.points.size();
        std::shared_ptr<MLP> mlp = framework.build_with_method(dataset, method);

        temp_index.push_back(mlp);
        int max_error = 0;
        int min_error = 0;
        for (Point point : dataset.points)
        {
            float pred = mlp->predict_ZM(point.normalized_key);
            int pos = pred * next_stage_length;
            pos = pos < 0 ? 0 : pos;
            pos = pos >= next_stage_length ? next_stage_length - 1 : pos;

            int error = point.index - pos;
            max_error = max(max_error, error);
            min_error = min(min_error, error);
            mlp->max_error = max_error;
            mlp->min_error = min_error;
        }
        index.push_back(temp_index);
    }

    void init(string _dataset_name)
    {
        // dataset_name = _dataset_name;
        // generate_points();

        framework.point_query_p = point_query;
        framework.window_query_p = window_query;
        framework.knn_query_p = knn_query;
        framework.build_index_p = build_single_ZM;
        framework.init_storage_p = init_underlying_data_storage;

        DataSet<Point, long long>::read_data_pointer = read_data;
        DataSet<Point, long long>::mapping_pointer = mapping;
        DataSet<Point, long long>::save_data_pointer = save_data;
        stages.push_back(1);
        framework.config_method_pool("ZM");
        framework.init();

        // dataset.dataset_name = _dataset_name;
        // dataset.read_data();
        // dataset.mapping();
        // N = dataset.points.size();

        // stages.push_back(5000);
        // init_underlying_data_storage(dataset);
    }

    void build_ZM()
    {

        records.resize(stages.size());
        vector<vector<Point>> stage1(stages[0]);
        stage1[0] = dataset.points;
        records[0] = stage1;

        print("build_ZM");

        for (size_t i = 0; i < stages.size(); i++)
        {
            vector<std::shared_ptr<MLP>> temp_index;

            bool is_last_level = i == stages.size() - 1;
            int next_stage_length = is_last_level ? N : stages[i + 1];
            if (!is_last_level)
            {
                vector<vector<Point>> temp_points(next_stage_length);
                records[i + 1] = temp_points;
            }

            for (size_t j = 0; j < stages[i]; j++)
            {
                if (records[i][j].size() == 0)
                {
                    auto mlp = std::make_shared<MLP>(1);
#ifdef use_gpu
                    mlp->to(torch::kCUDA);
#endif
                    temp_index.push_back(mlp);
                    continue;
                }
                // TODO change records[i][j] to Dataset
                // std::shared_ptr<MLP> mlp = framework.build(config::lambda, records[i][j]);
                DataSet<Point, long long> original_data_set(records[i][j]);
                int method = framework.build_predict_method(lambda, original_data_set);
                std::shared_ptr<MLP> mlp = framework.build_with_method(original_data_set, method);

                temp_index.push_back(mlp);
                int max_error = 0;
                int min_error = 0;
                for (Point point : records[i][j])
                {
                    float pred = mlp->predict_ZM(point.normalized_key);
                    int pos = pred * next_stage_length;
                    // if (j > 50)
                    // {
                    //     cout << "normalized_key: " << point.normalized_key << " pred: " << pred << " pos: " << pos << "index: " << point.index << endl;
                    // }
                    pos = pos < 0 ? 0 : pos;
                    pos = pos >= next_stage_length ? next_stage_length - 1 : pos;

                    if (is_last_level)
                    {
                        int error = point.index - pos;
                        max_error = max(max_error, error);
                        min_error = min(min_error, error);
                        mlp->max_error = max_error;
                        mlp->min_error = min_error;
                        // cout << "min_error:" << min_error << " max_error:" << max_error << endl;
                    }
                    else
                    {
                        records[i + 1][pos].push_back(point);
                    }
                }

                records[i][j].clear();
                records[i][j].shrink_to_fit();
            }

            index.push_back(temp_index);
        }
        print("begin finish");
    }

    void query()
    {
        print("begin query");
        Query<Point> point_query;
        point_query.set_point_query()->set_query_points(dataset.points);
        framework.query(point_query);

        // Query<Point> window_query;
        // vector<Mbr> windows;
        // window_query.set_window_query()->set_query_windows(windows);
    }

    void generate_points()
    {
        print("generate_points");
        int cardinality_l = 10000;
        int cardinality_u = 10000000;

        int bin_num_synthetic = 10;

        while (cardinality_u >= cardinality_l)
        {
            int N = cardinality_u;
            int bit_num = 8;
            // long long xs_max[2] = {(long long)(N), (long long)(N)};
            // long long max_key = compute_Z_value(xs_max, 2, bit_num);

            long max_edge = pow(2, bit_num - 1) - 1;

            long long max_key = compute_Z_value(max_edge, max_edge, bit_num);

            // long long xs_min[2] = {(long long)(2), (long long)(2)};
            // long long min_key = compute_Z_value(xs_min, 2, bit_num);

            long long min_key = compute_Z_value(0, 0, bit_num);

            long long gap = (max_key - min_key) / bin_num_synthetic;
            int *counter_array = new int[bin_num_synthetic];

            for (size_t i = 0; i < bin_num_synthetic; i++)
            {
                vector<Point> points;
                float dist = i * 0.1;
                counter_array[0] = (dist + 0.1) * cardinality_u;
                int other_num = (cardinality_u - counter_array[0]) / (bin_num_synthetic - 1);
                // cout << "counter_array[0]:" << counter_array[0] << endl;
                // cout << "other_num:" << other_num << endl;

                for (size_t j = 1; j < 10; j++)
                {
                    counter_array[j] = other_num;
                }
                int temp = 0;
                for (size_t j = 0; j < 10; j++)
                {
                    temp += counter_array[j];
                }

                while (temp > 0)
                {
                    int x = (rand() % (max_edge + 1));
                    int y = (rand() % (max_edge + 1));
                    // long long xs[2] = {(long long)(x), (long long)(y)};
                    // long long key = compute_Z_value(xs, 2, bit_num);
                    long long key = compute_Z_value((long long)(x), (long long)(y), bit_num);
                    int bin = (key - min_key) / gap;
                    // cout << "x:" << x << " y:" << y << " key:" << key << " bin:" << bin << endl;
                    if (bin < bin_num_synthetic && counter_array[bin] > 0)
                    {
                        counter_array[bin]--;
                        temp--;
                        Point point((float)x / max_edge, (float)y / max_edge);
                        points.push_back(point);
                    }
                    // cout << "bin:" << bin << endl;
                    // cout << "temp:" << temp << endl;
                }
                string name = "/home/research/datasets/BASE/synthetic/" + to_string(cardinality_u) + "_" + to_string(dist) + ".csv";
                save_data(points, name);
            }
            cardinality_u /= 10;
        }
    }
}
#endif // use_gpu
