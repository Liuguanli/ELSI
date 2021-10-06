#ifndef ZM_H
#define ZM_H
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
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
    int N;
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
        // vector<long> keys(N);

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
        // return keys;
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
        int leaf_node_num = ceil((float)dataset.points.size() / page_size);
        storage_leafnodes.resize(leaf_node_num);
        for (int i = 0; i < leaf_node_num; i++)
        {
            auto bn = dataset.points.begin() + i * page_size;
            auto en = dataset.points.begin() + min(i * page_size + page_size, N);
            vector<Point> points(bn, en);
            LeafNode leaf_node(points);
            storage_leafnodes[i] = leaf_node;
        }
        cout << "init_underlying_data_storage::storage_leafnodes: " << storage_leafnodes.size() << endl;
    }

    void point_query(Query<Point> query)
    {
        int point_not_found = 0;
        vector<Point> query_points = query.get_query_points();
        int query_num = query_points.size();
        for (size_t i = 0; i < query_num; i++)
        {
            int next_stage_length;
            int min_error;
            int max_error;
            int predicted_index = 0;

            long long xs[2] = {(long long)(query_points[i].x * N), (long long)(query_points[i].y * N)};

            query_points[i].curve_val = compute_Z_value(xs, 2, bit_num);
            query_points[i].normalized_key = (float)(query_points[i].curve_val - first_key) / gap;

            float key = query_points[i].normalized_key;
            // cout << "bit_num: " << bit_num << " query_point.curve_val: " << query_points[i].curve_val << endl;
            // cout << "key: " << key << " N: " << N << endl;
            for (int j = 0; j < stages.size(); j++)
            {
                if (j == stages.size() - 1)
                {
                    next_stage_length = N;
                    min_error = index[j][predicted_index]->min_error;
                    max_error = index[j][predicted_index]->max_error;
                }
                else
                {
                    next_stage_length = stages[j + 1];
                }
                predicted_index = index[j][predicted_index]->predict_ZM(key) * next_stage_length;
                if (predicted_index < 0)
                {
                    predicted_index = 0;
                }
                if (predicted_index >= next_stage_length)
                {
                    predicted_index = next_stage_length - 1;
                }
            }

            long front = predicted_index + min_error - error_shift;
            front = front < 0 ? 0 : front;
            long back = predicted_index + max_error + error_shift;
            back = back >= N ? N - 1 : back;
            front = front / page_size;
            back = back / page_size;
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
                if (front > back)
                {
                    point_not_found++;
                }
            }
        }
        cout << "point_not_found: " << point_not_found << endl;
    }

    vector<Point> window_query(Query<Point> query)
    {
        vector<Point> res;
        return res;
    }

    vector<Point> knn_query(Query<Point> query)
    {
        vector<Point> res;
        return res;
    }

    void init(string _dataset_name)
    {
        // dataset_name = _dataset_name;
        framework.point_query_p = point_query;
        framework.window_query_p = window_query;
        framework.knn_query_p = knn_query;
        DataSet<Point, long long>::read_data_pointer = read_data;
        DataSet<Point, long long>::mapping_pointer = mapping;
        DataSet<Point, long long>::save_data_pointer = save_data;

        dataset.dataset_name = _dataset_name;
        dataset.read_data();
        dataset.mapping();
        N = dataset.points.size();

        stages.push_back(1);
        stages.push_back(100);
        records.resize(stages.size());
        vector<vector<Point>> stage1(stages[0]);
        stage1[0] = dataset.points;
        records[0] = stage1;

        init_underlying_data_storage(dataset);
    }

    void build_ZM()
    {
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
                std::shared_ptr<MLP> mlp = framework.build(config::lambda, records[i][j]);
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

        // Query<long> window_query;
        // vector<Mbr> windows;
        // window_query.set_window_query()->set_query_windows(windows)->set_iterations(10);
        // framework.query(window_query);

        // Query<long> knn_query;
        // vector<long> knn_query_points;
        // knn_query.set_knn_query()->set_query_points(knn_query_points)->set_k(10)->set_iterations(10);
        // framework.query(knn_query);

        // framework.is_rebuild();
    }

    void query()
    {
        Query<Point> point_query;
        point_query.set_point_query()->set_query_points(dataset.points);
        framework.query(point_query);
    }
}
#endif // use_gpu
