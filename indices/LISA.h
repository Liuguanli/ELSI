#ifndef LISA_H
#define LISA_H
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
#include "../entities/Shard.h"
#include "../entities/LeafNode.h"
#include "../entities/DataSet.h"
#include "../utils/FileReader.h"
#include "../utils/SortTools.h"
#include "../utils/Config.h"
#include "../utils/Log.h"
#include "../utils/ExpRecorder.h"

#include "../ELSI.h"

using namespace std;
using namespace config;
using namespace logger;

namespace lisa
{

    ELSI<Point, double> framework;
    long N;
    vector<LeafNode> storage_leafnodes;
    DataSet<Point, double> dataset;
    int page_size = Constants::PAGESIZE;
    int error_shift = 0;
    double first_key, last_key, gap;

    string model_path_root;

    int n_parts = 240;
    // int n_parts = 1;
    float eta = 0.01;
    // int n_models = 200;
    int n_models = 1000;
    float max_value_x = 1.0;
    int shard_id = 0;
    int page_id = 0;
    vector<Point> split_points;
    vector<float> split_index_list;
    vector<std::shared_ptr<MLP>> SP;
    vector<map<int, Shard>> shards;

    vector<double> mappings;
    vector<float> borders;
    vector<float> gaps;
    vector<float> x_split_points;
    vector<double> model_split_mapping;
    vector<int> shard_start_id_each_model;
    vector<int> partition_sizes;
    vector<double> min_key_list;

    vector<int> model_split_idxes;

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

    int get_model_index(double key)
    {
        if (model_split_mapping[0] > key)
        {
            return 0;
        }
        if (model_split_mapping[n_models - 2] <= key)
        {
            return n_models - 1;
        }
        int begin = 1;
        int end = n_models - 1;
        while (begin < end)
        {
            int mid = (begin + end) / 2;
            if (model_split_mapping[mid - 1] <= key && key < model_split_mapping[mid])
            {
                return mid;
            }
            else if (model_split_mapping[mid - 1] > key)
            {
                end = mid;
            }
            else
            {
                begin = mid;
            }
        }
        return n_models - 1;
    }

    int get_partition_index(Point point)
    {
        if (x_split_points[0] >= point.x)
        {
            return 0;
        }
        if (x_split_points[n_parts - 2] < point.x)
        {
            return n_parts - 1;
        }
        int begin = 1;
        int end = n_parts - 1;
        while (begin < end)
        {
            int mid = (begin + end) / 2;
            if (x_split_points[mid - 1] < point.x && point.x <= x_split_points[mid])
            {
                return mid;
            }
            else if (x_split_points[mid - 1] >= point.x)
            {
                end = mid;
            }
            else
            {
                begin = mid;
            }
        }
        return n_parts - 1;
    }

    double get_mapped_key(Point point, int i)
    {
        // double mapped_val = point.x / max_value_x + (point.x - borders[i]) / gaps[i] + i * 2;
        // return mapped_val;

        // double measure = (point.x - borders[i]) / gaps[i];
        double mapped_val = point.y + i * 2;

        // double mapped_val = measure * eta * 0.000001 + point.y + i * 2;
        // double mapped_val = (point.y / 1 * (n_parts - 1)) + i * n_parts;
        // cout << "mapped_val: " << mapped_val << endl;
        return mapped_val;
    }

    void mapping(vector<Point> &points, vector<double> &keys)
    {
        N = points.size();
        int partition_size = floor(N / n_parts);
        int remainder = N - n_parts * partition_size;
        vector<int> x_split_idxes;

        sort(points.begin(), points.end(), sortX());
        // this->max_value_x = points[N - 1].x;
        // cout << "max_value_x: " << max_value_x << endl;

        borders.push_back(0.0);
        for (size_t i = 0; i < remainder; i++)
        {
            int idx = (i + 1) * (partition_size + 1);
            while (points[idx - 1].x == points[idx].x)
            {
                idx++;
            }
            x_split_idxes.push_back(idx);
            x_split_points.push_back(points[idx - 1].x);
            borders.push_back(points[idx - 1].x);
            gaps.push_back(x_split_points[i] - borders[i]);
        }

        for (size_t i = remainder; i < n_parts - 1; i++)
        {
            int idx = (i + 1) * partition_size + remainder;
            while (points[idx - 1].x == points[idx].x)
            {
                idx++;
            }
            x_split_idxes.push_back(idx);
            x_split_points.push_back(points[idx - 1].x);
            borders.push_back(points[idx - 1].x);
            gaps.push_back(x_split_points[i] - borders[i]);
        }

        x_split_points.push_back(max_value_x);
        x_split_idxes.push_back(N);
        gaps.push_back(x_split_points[n_parts - 1] - borders[n_parts - 1]);
        // for (size_t i = 0; i < n_parts; i++)
        // {
        //     cout << x_split_points[i] << " " << x_split_idxes[i] << " " << borders[i] << " " << gaps[i] << endl;
        // }
        int start = 0;
        int index = 0;

        vector<float> float_mappings;

        for (size_t i = 0; i < x_split_idxes.size(); i++)
        {
            // cout << "-----------" << i << "-----------" << endl;
            vector<Point> part_data(points.begin() + start, points.begin() + x_split_idxes[i]);
            start = x_split_idxes[i];

            sort(part_data.begin(), part_data.end(), sortY());

            for (size_t j = 0; j < part_data.size(); j++)
            {
                // TODO change mapping
                // double mapped_val = ((part_data[j].x - borders[i]) / gaps[i]) * eta + (part_data[j].x / max_value_x * (n_parts - 1)) + (i * n_parts) * 1.0;
                // double mapped_val = part_data[j].x / max_value_x + (part_data[j].x - borders[i]) / gaps[i] + i * 2;
                int partition_index = get_partition_index(part_data[j]);
                // if (239 == i)
                // {
                //     cout << "index: " << index << " j: " << j << endl;
                // }
                double mapped_val = get_mapped_key(part_data[j], partition_index);
                mappings.push_back(mapped_val);
                float_mappings.push_back(mapped_val);
                part_data[j].key = mapped_val;
                points[index++] = part_data[j];
            }
        }

        int offset = N / n_models;
        for (size_t i = 1; i < n_models; i++)
        {
            // cout << "-----------" << i << "-----------" << endl;
            int idx = i * offset;
            while (mappings[idx] == mappings[idx + 1])
            {
                idx++;
            }
            // cout << "idx:" << idx << endl;
            model_split_mapping.push_back(mappings[idx]);
            model_split_idxes.push_back(idx);
        }
        model_split_mapping.push_back(mappings[N - 1]);
        model_split_idxes.push_back(N);
        keys = mappings;
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
            double key = get_mapped_key(query_points[i], get_partition_index(query_points[i]));
            query_points[i].key = key;
            int model_index = get_model_index(key);

            int partition_size = partition_sizes[model_index];
            int shard_index = SP[model_index]->predict_ZM(key - min_key_list[model_index]) * partition_size / page_size;

            auto it = shards[model_index].find(shard_index);
            if (!it->second.search_point(query_points[i]))
            {
                if (model_index > 0)
                {
                    model_index--;
                    partition_size = partition_sizes[model_index];
                    shard_index = SP[model_index]->predict_ZM(key - min_key_list[model_index]) * partition_size / page_size;
                    auto it_left = shards[model_index].find(shard_index);
                    if (!it_left->second.search_point(query_points[i]))
                    {
                        point_not_found++;
                    }
                }
            }
        }
        cout << "point_not_found: " << point_not_found << endl;
    }

    void window_query(Mbr query_window, int partition_index, vector<Point> &result)
    {
        Point pl(query_window.x1, query_window.y1);
        Point pu(query_window.x2, query_window.y2);
        int partition_l = partition_index == -1 ? get_partition_index(pl) : partition_index;
        int partition_u = partition_index == -1 ? get_partition_index(pu) : partition_index;
        if (partition_l == partition_u)
        {
            double key_l = get_mapped_key(pl, partition_l);
            int start_model_index = get_model_index(key_l);
            start_model_index = start_model_index > 0 ? start_model_index - 1 : start_model_index;
            int start_partition_size = partition_sizes[start_model_index];
            int start_shard_index = SP[start_model_index]->predict_ZM(key_l - min_key_list[start_model_index]) * start_partition_size / page_size + shard_start_id_each_model[start_model_index];
            // auto it = shards[start_model_index].find(start_shard_index);
            double key_u = get_mapped_key(pu, partition_u);
            int end_model_index = get_model_index(key_u);
            end_model_index = end_model_index == (partition_sizes.size() - 1) ? end_model_index : end_model_index + 1;
            int end_partition_size = partition_sizes[end_model_index];
            int end_shard_index = SP[end_model_index]->predict_ZM(key_u - min_key_list[end_model_index]) * end_partition_size / page_size + shard_start_id_each_model[end_model_index];

            auto it_start = shards[start_model_index].find(start_shard_index);
            auto it_end = shards[start_model_index].end();
            while (it_start != shards[start_model_index].end())
            {
                vector<Point> res = it_start->second.window_query(query_window);
                result.insert(result.end(), res.begin(), res.end());
                it_start++;
            }

            for (size_t i = start_model_index + 1; i < end_model_index; i++)
            {
                it_start = shards[i].begin();
                it_end = shards[i].end();
                while (it_start != it_end)
                {
                    vector<Point> res = it_start->second.window_query(query_window);
                    result.insert(result.end(), res.begin(), res.end());
                    it_start++;
                }
            }

            it_start = shards[end_model_index].begin();
            it_end = shards[end_model_index].find(start_shard_index);
            it_end++;
            while (it_start != it_end)
            {
                vector<Point> res = it_start->second.window_query(query_window);
                result.insert(result.end(), res.begin(), res.end());
                it_start++;
            }
            // it_end->second.window_query(exp_recorder, query_window);
        }
        else
        {
            Mbr first_query_window = query_window;
            first_query_window.x2 = x_split_points[partition_l];
            window_query(first_query_window, partition_l, result);

            for (size_t i = partition_l + 1; i < partition_u; i++)
            {
                Mbr new_query_window = query_window;
                new_query_window.x1 = x_split_points[i - 1];
                new_query_window.x2 = x_split_points[i];
                window_query(new_query_window, i, result);
                // vector<Point> temp_result = window_query(exp_recorder, new_query_window);
                // result.insert(result.end(), temp_result.begin(), temp_result.end());
            }
            Mbr last_query_window = query_window;
            last_query_window.x1 = x_split_points[partition_u - 1];
            window_query(last_query_window, partition_u, result);
            // vector<Point> last_result = window_query(exp_recorder, last_query_window);
            // result.insert(result.end(), last_result.begin(), last_result.end());
        }
    }

    void window_query(Query<Point> &query)
    {
        query.results.clear();
        query.results.shrink_to_fit();

        int point_not_found = 0;
        int query_num = query.query_windows.size();
        for (size_t i = 0; i < query_num; i++)
        {
            window_query(query.query_windows[i], -1, query.results);
            query.results.clear();
            query.results.shrink_to_fit();
        }
    }

    vector<Point> kNN_query(Point query_point, int k)
    {
        float knn_query_side = sqrt((float)k / N) * 2;
        vector<Point> result;
        while (true)
        {
            vector<Point> temp_result;
            Mbr mbr = Mbr::get_mbr(query_point, knn_query_side);
            vector<Point> window_result;
            window_query(mbr, -1, window_result);

            for (size_t i = 0; i < window_result.size(); i++)
            {
                if (window_result[i].cal_dist(query_point) <= knn_query_side)
                {
                    temp_result.push_back(window_result[i]);
                }
            }
            if (temp_result.size() < k)
            {
                if (temp_result.size() == 0)
                {
                    knn_query_side *= 2;
                }
                else
                {
                    knn_query_side *= sqrt((float)k / temp_result.size());
                }
                continue;
            }
            else
            {
                sort(temp_result.begin(), temp_result.end(), sort_for_kNN(query_point));
                vector<Point> vec(temp_result.begin(), temp_result.begin() + k);
                result = vec;
                break;
            }
        }
        return result;
    }

    void kNN_query(Query<Point> &query)
    {
        int query_num = query.knn_query_points.size();
        Query<Point> window_query_for_knn;
        window_query_for_knn.set_window_query();
        int k = query.get_k();
        for (size_t i = 0; i < query_num; i++)
        {
            vector<Point> res = kNN_query(query.knn_query_points[i], k);
            query.results.insert(query.results.end(), res.begin(), res.end());
        }
    }

    void insert(Point point)
    {
        double key = get_mapped_key(point, get_partition_index(point));
        point.key = key;
        int model_index = get_model_index(key);

        int partition_size = partition_sizes[model_index];
        int shard_index = SP[model_index]->predict_ZM(key - min_key_list[model_index]) * partition_size / page_size;

        auto it = shards[model_index].find(shard_index);
        if (it != shards[model_index].end())
        {
            it->second.insert(point);
        }
        else
        {
            vector<Point> shard_points;
            shard_points.push_back(point);
            Shard shard(0, page_size);
            shard.gen_local_model(shard_points, shard_id);
            shards[model_index].insert(pair<int, Shard>(shard_index, shard));
        }
    }

    // void build_single_LISA(DataSet<Point, double> dataset, int method)
    // {
    // }

    void build_LISA(ExpRecorder &exp_recorder)
    {

        int start = 0;
        for (size_t i = 0; i < n_models; i++)
        {
            vector<Point> part_data(dataset.points.begin() + start, dataset.points.begin() + model_split_idxes[i]);
            vector<double> keys;
            vector<float> labels;
            // float min_key = part_data[0].key;
            double min_key = mappings[start];
            min_key_list.push_back(min_key);
            int part_data_size = part_data.size();
            partition_sizes.push_back(part_data_size);
            keys.push_back(0.0);
            labels.push_back(0.0);

            for (size_t j = 1; j < part_data_size; j++)
            {
                keys.push_back(mappings[start + j] - min_key);
                labels.push_back(j * 1.0 / part_data_size);
            }

            DataSet<Point, double> original_data_set(part_data);
            original_data_set.read_keys_and_labels();
            original_data_set.keys = keys;
            int method = exp_recorder.build_method;
            if (exp_recorder.is_framework)
            {
                method = framework.build_predict_method(exp_recorder.upper_level_lambda, query_frequency, original_data_set);
            }
            if (method == Constants::CL || method == Constants::RL)
            {
                method = Constants::OG;
            }
            if (exp_recorder.is_single_build)
            {
                method = exp_recorder.build_method;
            }
            if (exp_recorder.is_original)
            {
                method = Constants::OG;
            }

            exp_recorder.record_method_nums(method);

            std::shared_ptr<MLP> mlp = framework.get_build_method(original_data_set, method);

            map<int, int> entries_count;

            vector<int> positions;
            // vector<float> predicts;
            for (size_t j = 0; j < part_data_size; j++)
            {
                // predicts.push_back(net->predict_ZM(keys[j]));
                int idx = mlp->predict_ZM(keys[j]) * part_data_size / page_size;
                positions.push_back(idx);
                entries_count[idx]++;
            }
            map<int, int>::iterator iter = entries_count.begin();
            int start_idx = 0;
            int end_idx = 0;
            shard_start_id_each_model.push_back(shard_id);
            map<int, Shard> model_shards;
            while (iter != entries_count.end())
            {
                int idx = iter->first;
                int shard_size = iter->second;
                end_idx += shard_size;

                vector<Point> shard_points(part_data.begin() + start_idx, part_data.begin() + end_idx);
                Shard shard(shard_id++, page_size);
                shard.gen_local_model(shard_points, page_id);
                start_idx = end_idx;
                model_shards.insert(pair<int, Shard>(idx, shard));
                iter++;
                // cout << "------------idx-----------: " << idx << endl;
            }

            start = model_split_idxes[i];
            SP.push_back(mlp);
            shards.push_back(model_shards);
        }
        exp_recorder.timer_end();
        print("build time:" + to_string(exp_recorder.time / 1e9) + " s");
    }

    void query(Query<Point> query, ExpRecorder &exp_recorder)
    {
        exp_recorder.timer_begin();
        framework.query(query);
        exp_recorder.timer_end();
        // if (query.is_window())
        // {
        //     long res_size = 0;
        //     if (exp_recorder.distribution == "OSM")
        //     {
        //         res_size = 823500207;
        //     }
        //     if (exp_recorder.distribution == "SA")
        //     {
        //         res_size = 844801538;
        //     }
        //     if (exp_recorder.distribution == "uniform")
        //     {
        //         res_size = 12798879;
        //     }
        //     if (exp_recorder.distribution == "skewed")
        //     {
        //         res_size = 85430460;
        //     }

        //     // cout << "window query size: " << query.results.size() << endl;
        //     exp_recorder.accuracy = (float)query.results.size() / res_size;
        //     // cout << "accuracy: " << exp_recorder.accuracy << endl;
        // }
        // if (query.is_knn())
        // {
        //     vector<Point> res;
        //     // cout << "knn query size: " << query.results.size() << endl;
        //     int found_num = 0;
        //     for (size_t i = 0; i < query.knn_query_points.size(); i++)
        //     {
        //         priority_queue<Point, vector<Point>, sortForKNN2> pq;
        //         int k = query.get_k();

        //         for (size_t j = 0; j < dataset.points.size(); j++)
        //         {
        //             Point point = dataset.points[j];
        //             point.temp_dist = point.cal_dist(query.knn_query_points[i]);
        //             if (pq.size() < k)
        //             {
        //                 pq.push(point);
        //             }
        //             else
        //             {
        //                 if (pq.top().temp_dist > point.temp_dist)
        //                 {
        //                     pq.pop();
        //                     pq.push(point);
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
        //     // cout << "accuracy: " << exp_recorder.accuracy << endl;
        // }
    }

    // TODO opt
    DataSet<Point, double> generate_points(long cardinality, float dist)
    {
        DataSet<Point, double> dataset;
        int bin_num_synthetic = 10;

        int N = cardinality_u;
        // int bit_num = 8;

        long max_edge = pow(2, bit_num - 1) - 1;

        float max_key = 0;

        // // long long xs_min[2] = {(long long)(2), (long long)(2)};
        // // long long min_key = compute_Z_value(xs_min, 2, bit_num);

        n_models = N / Constants::THRESHOLD;
        float min_key = 1;

        float gap = (max_key - min_key) / bin_num_synthetic;
        int *counter_array = new int[bin_num_synthetic];

        vector<Point> points;
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
            float x = (float)rand() / (RAND_MAX);
            float y = (float)rand() / (RAND_MAX);
            int bin = (x - min_key) / gap;
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
        dataset.points = points;
        dataset.mapping();
        return dataset;
    }

    void generate_all_points()
    {
        // print("generate_points");
        int bin_num_synthetic = 10;
        while (cardinality_u >= cardinality_l)
        {
            for (size_t i = 0; i < bin_num_synthetic; i++)
            {
                float dist = i * 0.1;
                string name = Constants::SYNTHETIC_DATA_PATH + to_string(cardinality_u) + "_" + to_string(dist) + ".csv";
                save_data(generate_points(cardinality_u, dist).points, name);
            }
            cardinality_u /= 10;
        }
    }

    void init(string _dataset_name, ExpRecorder &exp_recorder)
    {
        // dataset_name = _dataset_name;
        // generate_points();
        exp_recorder.name = "LISA";
        exp_recorder.timer_begin();
        // vector<int> methods{Constants::MR, Constants::OG, Constants::RS, Constants::SP};
        vector<int> methods{Constants::CL, Constants::MR, Constants::OG, Constants::RL, Constants::RS, Constants::SP};
        config::init_method_pool(methods);
        framework.dimension = 1;
        framework.point_query_p = point_query;
        framework.window_query_p = window_query;
        framework.knn_query_p = kNN_query;
        // framework.build_index_p = build_single_LISA;
        framework.insert_p = insert;
        framework.generate_points_p = generate_points;
        framework.max_cardinality = cardinality_u;

        DataSet<Point, double>::read_data_pointer = read_data;
        DataSet<Point, double>::mapping_pointer = mapping;
        DataSet<Point, double>::save_data_pointer = save_data;
        // framework.index_name = "LISA";
        framework.config_method_pool();
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
        exp_recorder.timer_begin();
        dataset.mapping()->generate_normalized_keys()->generate_labels();
        exp_recorder.timer_end();
        print("mapping data time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");

        N = dataset.points.size();

        exp_recorder.timer_end();
        print("data init time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");
    }
}
#endif // use_gpu
