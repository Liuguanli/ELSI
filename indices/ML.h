#ifndef ML_H
#define ML_H
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
#include "../utils/ExpRecorder.h"
#include "../ELSI.h"

using namespace std;
using namespace config;
using namespace logger;

namespace ml
{
    ELSI<Point, double> framework;
    vector<int> stages;
    vector<vector<std::shared_ptr<MLP>>> index;
    long N;
    int index_bit_num;
    vector<vector<vector<Point>>> records;
    vector<LeafNode> storage_leafnodes;
    DataSet<Point, double> dataset;
    int page_size = Constants::PAGESIZE;
    int error_shift = 0;
    double first_key = 0, last_key = 0, gap = 0;

    int k = 100;
    vector<double> offsets;
    vector<int> partition_size;
    vector<Point> reference_points;
    vector<vector<Point>> partitions;

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

    void gen_reference_points(ExpRecorder &exp_recorder)
    {
        string python_path = "./method_pool/CL/cluster.py";
        string result_points_path = "/home/research/datasets/" + exp_recorder.get_file_name() + "_k_" + to_string(k) + "_minibatchkmeans_auto.csv";
        string commandStr = "python " + python_path + " -i " + exp_recorder.get_dataset_name() + " -k " + to_string(k) + " -o " + result_points_path;
        // print(commandStr);
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        FileReader reader;
        reference_points = reader.get_points(result_points_path, ",");
    }

    int get_partition_id(Point point)
    {
        int partition_index = 0;
        double minimal_dist = numeric_limits<double>::max();
        for (size_t j = 0; j < k; j++)
        {
            double temp_dist = reference_points[j].cal_dist(point);
            if (temp_dist < minimal_dist)
            {
                minimal_dist = temp_dist;
                partition_index = j;
            }
        }
        return partition_index;
    }

    void mapping(vector<Point> &points, vector<double> &keys)
    {
        N = points.size();

        for (size_t i = 0; i < N; i++)
        {
            int partition_index = get_partition_id(points[i]);
            points[i].partition_id = partition_index;
            partitions[partition_index].push_back(points[i]);
        }
        offsets[0] = 0;
        partition_size[0] = 0;

        for (size_t i = 0; i < k; i++)
        {
            double maximal_dist = numeric_limits<double>::min();
            for (Point point : partitions[i])
            {
                double temp_dist = reference_points[i].cal_dist(point);
                if (maximal_dist < temp_dist)
                {
                    maximal_dist = temp_dist;
                }
            }
            partition_size[i + 1] = partition_size[i] + partitions[i].size();
            offsets[i + 1] = offsets[i] + maximal_dist;
            // cout << "partition_size[i + 1]: " << partition_size[i + 1] << endl;
        }

        // 5 order the data
        for (size_t i = 0; i < N; i++)
        {
            points[i].key = offsets[points[i].partition_id] + points[i].cal_dist(reference_points[points[i].partition_id]);
        }

        sort(points.begin(), points.end(), sort_key());

        first_key = points[0].key;
        last_key = points[N - 1].key;
        gap = last_key - first_key;

        // cout << "first_key: " << first_key << endl;
        // cout << "last_key: " << last_key << endl;

        for (long i = 0; i < N; i++)
        {
            points[i].index = i;
            points[i].label = (float)i / N;
            points[i].normalized_key = (points[i].key - first_key) / gap;
            // if (abs(5.08001 - points[i].key) < 1e-5)
            // {
            //     cout << "-----i:" << i << " key:" << points[i].key << endl;
            // }
            keys.push_back(points[i].key);
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

    void init_underlying_data_storage(DataSet<Point, double> &dataset)
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
            // cout << "points[0].key:" << points[0].key << " points[-1].key:" << points[page_size - 1].key << endl;
            LeafNode leaf_node(points);
            storage_leafnodes[i] = leaf_node;
        }
    }

    int get_point_index(double key, long &front, long &back)
    {
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
        predicted_index = index[level - 1][predicted_index]->predict_ZM(key) * next_stage_length;

        predicted_index = max(predicted_index, 0);
        predicted_index = min(predicted_index, next_stage_length - 1);

        // cout << "min_error: " << min_error << " max_error: " << max_error << endl;
        // cout << "predicted_index: " << predicted_index << endl;
        front = predicted_index + min_error - error_shift;
        front = min(N - 1, max((long)0, front));
        back = predicted_index + max_error + error_shift;
        back = min(N - 1, back);
        // cout << "front: " << front << " back: " << back << endl;

        return predicted_index;
    }

    int get_point_index(Point &query_point, long &front, long &back)
    {
        int partition_id = get_partition_id(query_point);
        double dist = query_point.cal_dist(reference_points[partition_id]);
        query_point.key = offsets[partition_id] + dist;
        double key = (query_point.key - first_key) / gap;
        query_point.normalized_key = key;

        return get_point_index(key, front, back);
    }

    long predict_closest_position(double key, bool is_upper, int partition)
    {
        double kkey = (key - first_key) / gap;
        long s = 0, t = 0;
        // get_point_index(kkey, s, t);
        int mid = 0;
        s = partition_size[partition];
        t = partition_size[partition + 1];
        // LeafNode leafnode_first = storage_leafnodes[partition_size[partition] / page_size];
        // LeafNode leafnode_last = storage_leafnodes[(partition_size[partition + 1]-1) / page_size];
        LeafNode leafnode_first = storage_leafnodes.front();
        LeafNode leafnode_last = storage_leafnodes.back();
        if (key <= leafnode_first.children[0].key)
        {
            return partition_size[partition];
        }
        if (key >= leafnode_last.children[leafnode_last.children.size() - 1].key)
        {
            return partition_size[partition + 1] - 1;
        }
        while (s <= t)
        {
            mid = ceil((s + t) / 2);
            int leaf_node_index = mid / page_size;
            LeafNode leafnode = storage_leafnodes[leaf_node_index];
            if (leafnode.children[0].key <= key && key <= leafnode.children[leafnode.children.size() - 1].key)
            {
                int offset = 0;
                for (size_t i = 0; i < leafnode.children.size() - 1; i++)
                {
                    if (leafnode.children[i].key <= key && key <= leafnode.children[i + 1].key)
                    {
                        if (is_upper)
                        {
                            return leaf_node_index * page_size + i;
                        }
                        else
                        {
                            return leaf_node_index * page_size + i + 1;
                        }
                    }
                }
            }
            int offset = mid - page_size * (leaf_node_index);
            Point mid_point = leafnode.children[offset];
            if (key < mid_point.key)
            {
                t = mid - 1;
            }
            else
            {
                s = mid + 1;
            }
        }
        return 0;
    }

    bool point_query(Point &query_point)
    {
        long front = 0, back = 0;
        int predicted_index = get_point_index(query_point, front, back);

        front /= page_size;
        back /= page_size;
        front = max((long)0, front--);
        back = min(back++, (long)storage_leafnodes.size() - 1);
        while (front <= back)
        {
            int mid = (front + back) / 2;
            double first_ml_key = storage_leafnodes[mid].children[0].key;
            double last_ml_key = storage_leafnodes[mid].children[storage_leafnodes[mid].children.size() - 1].key;

            // cout << "first_ml_key: " << first_ml_key << endl;
            // cout << "last_ml_key: " << last_ml_key << endl;
            // cout << "query_point.key: " << query_point.key << endl;
            // cout << "front: " << front << " mid: " << mid << " back: " << back << endl;

            if (first_ml_key <= query_point.key && query_point.key <= last_ml_key)
            {
                vector<Point>::iterator iter = find(storage_leafnodes[mid].children.begin(), storage_leafnodes[mid].children.end(), query_point);
                if (iter == storage_leafnodes[mid].children.end())
                {
                    int inner_front = mid - 1;

                    while (inner_front >= 0 && (storage_leafnodes[inner_front].children[0].key <= query_point.key && query_point.key <= storage_leafnodes[inner_front].children[storage_leafnodes[inner_front].children.size() - 1].key))
                    {
                        iter = find(storage_leafnodes[inner_front].children.begin(), storage_leafnodes[inner_front].children.end(), query_point);
                        if (iter != storage_leafnodes[inner_front].children.end())
                        {
                            return true;
                        }
                        inner_front--;
                    }
                    int inner_back = mid + 1;
                    while (inner_back < storage_leafnodes.size() && (storage_leafnodes[inner_back].children[0].key <= query_point.key && query_point.key <= storage_leafnodes[inner_back].children[storage_leafnodes[inner_back].children.size() - 1].key))
                    {
                        iter = find(storage_leafnodes[inner_back].children.begin(), storage_leafnodes[inner_back].children.end(), query_point);
                        if (iter != storage_leafnodes[inner_back].children.end())
                        {
                            return true;
                        }
                        inner_back++;
                    }
                }
                else
                {
                    return true;
                }
            }
            else
            {
                if (storage_leafnodes[mid].children[0].key < query_point.key)
                {
                    front = mid + 1;
                }
                else
                {
                    back = mid - 1;
                }
            }
        }
        return false;
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
                    // cout << "i :" << i << endl;
                    // break;
                }
            }
        }
        // cout << "point_not_found: " << point_not_found << endl;
        printf("point_not_found %d\n", point_not_found);
    }

    vector<Point> find_closet_points(Mbr query_window)
    {
        vector<Point> closest_points;
        for (size_t i = 0; i < k; i++)
        {
            Point point;
            Point reference_point = reference_points[i];
            if (reference_point.x < query_window.x1)
            {
                point.x = query_window.x1;
            }
            else if (reference_point.x < query_window.x2)
            {
                point.x = reference_point.x;
            }
            else
            {
                point.x = query_window.x2;
            }

            if (reference_point.y < query_window.y1)
            {
                point.y = query_window.y1;
            }
            else if (reference_point.y < query_window.y2)
            {
                point.y = reference_point.y;
            }
            else
            {
                point.y = query_window.y2;
            }
            closest_points.push_back(point);
        }
        return closest_points;
    }

    vector<Point> find_furthest_points(Mbr query_window)
    {
        vector<Point> vertexes = query_window.get_corner_points();
        vector<Point> furthest_points;

        for (size_t i = 0; i < k; i++)
        {
            double max_dist = numeric_limits<double>::min();
            Point furthest_point;
            for (Point point : vertexes)
            {
                double temp_dist = point.cal_dist(reference_points[i]);
                if (max_dist < temp_dist)
                {
                    max_dist = temp_dist;
                    furthest_point = point;
                }
            }
            furthest_points.push_back(furthest_point);
        }
        return furthest_points;
    }

    void window_query(vector<Point> &results, Mbr &query_window)
    {
        vector<Point> closest_points = find_closet_points(query_window);
        vector<Point> furthest_points = find_furthest_points(query_window);
        // vector<Point> res;
        for (size_t j = 0; j < k; j++)
        {
            double lower_key = reference_points[j].cal_dist(closest_points[j]) + offsets[j];
            double upper_key = reference_points[j].cal_dist(furthest_points[j]) + offsets[j];

            long index_low = 0, index_high = 0;
            int lower_i = predict_closest_position(lower_key, false, j);
            int upper_i = predict_closest_position(upper_key, true, j);

            lower_i = lower_i < partition_size[j] ? partition_size[j] : lower_i;
            upper_i = upper_i > partition_size[j + 1] ? partition_size[j + 1] : upper_i;
            upper_i = upper_i < lower_i ? partition_size[j + 1] : upper_i;
            long front = lower_i / page_size - 1;
            long back = upper_i / page_size + 1;
            front = front < 0 ? 0 : front;
            back = back >= storage_leafnodes.size() ? storage_leafnodes.size() - 1 : back;
            for (size_t l = front; l <= back; l++)
            {
                if (storage_leafnodes[l].mbr.interact(query_window))
                {
                    for (size_t i = 0; i < storage_leafnodes[l].children.size(); i++)
                    {
                        if (query_window.contains(storage_leafnodes[l].children[i]))
                        {
                            results.push_back(storage_leafnodes[l].children[i]);
                        }
                    }
                }
            }
        }
    }

    void window_query(Query<Point> &query)
    {
        query.results.clear();
        query.results.shrink_to_fit();
        int query_num = query.query_windows.size();
        vector<Point> window_query_results;
        for (size_t i = 0; i < query_num; i++)
        {
            // std::cout << "window query: " << i << std::endl;

            window_query(query.results, query.query_windows[i]);
            // framework.window_query(query.results, query.query_windows[i]);
            // the number of results is not necessary
            query.results.clear();
            query.results.shrink_to_fit();
        }
    }

    void search_inward(int &lnode_index, int lower_bound, double key, vector<Point> &S, int kk, Point query_point)
    {
        // std::cout << "begin: " << lnode_index << std::endl;
        while (true)
        {
            // cout << "lnode_index: " << lnode_index << " lower_bound: " << lower_bound << endl;
            if (lnode_index < lower_bound)
            {
                // std::cout << "end:" << lower_bound << std::endl;
                lnode_index = -1;
                return;
            }
            for (Point point : storage_leafnodes[lnode_index].children)
            {
                if (S.size() == kk)
                {
                    sort(S.begin(), S.end(), sortForKNN2());
                    // double dist_furthest = 0;
                    // int dist_furthest_i = 0;
                    // for (size_t i = 0; i < kk; i++)
                    // {
                    //     double temp_dist = S[i].cal_dist(query_point);
                    //     if (temp_dist > dist_furthest)
                    //     {
                    //         dist_furthest = temp_dist;
                    //         dist_furthest_i = i;
                    //     }
                    // }
                    if (point.cal_dist(query_point) < S[kk - 1].temp_dist && std::find(S.begin(), S.end(), point) == S.end())
                    {
                        S[kk - 1] = point;
                        // S[dist_furthest_i] = point;
                    }
                }
                else if (S.size() < kk)
                {
                    point.cal_dist(query_point);
                    S.push_back(point);
                }
            }
            if (storage_leafnodes[lnode_index].children[0].key > key)
            {
                if (lnode_index > lower_bound)
                {
                    lnode_index--;
                }
                else
                {
                    lnode_index = -1;
                    break;
                }
            }
            else
            {
                lnode_index = -1;
                break;
            }
        }
    }

    void search_outward(int &lnode_index, int upper_bound, double key, vector<Point> &S, int kk, Point query_point)
    {
        while (true)
        {
            if (lnode_index > upper_bound)
            {
                lnode_index = -1;
                return;
            }
            if (lnode_index >= storage_leafnodes.size())
            {
                return;
            }

            for (Point point : storage_leafnodes[lnode_index].children)
            {
                if (S.size() == kk)
                {
                    sort(S.begin(), S.end(), sortForKNN2());
                    // double dist_furthest = 0;
                    // int dist_furthest_i = 0;
                    // for (size_t i = 0; i < kk; i++)
                    // {
                    //     double temp_dist = S[i].cal_dist(query_point);
                    //     if (temp_dist > dist_furthest)
                    //     {
                    //         dist_furthest = temp_dist;
                    //         dist_furthest_i = i;
                    //     }
                    // }
                    if (point.cal_dist(query_point) < S[kk - 1].temp_dist && std::find(S.begin(), S.end(), point) == S.end())
                    {
                        S[kk - 1] = point;
                    }
                }
                else if (S.size() < kk)
                {
                    point.cal_dist(query_point);
                    S.push_back(point);
                }
            }
            if (storage_leafnodes[lnode_index].children[storage_leafnodes[lnode_index].children.size() - 1].key < key)
            {
                if (lnode_index < upper_bound)
                {
                    lnode_index++;
                }
                else
                {
                    lnode_index = -1;
                    break;
                }
            }
            else
            {
                lnode_index = -1;
                break;
            }
        }
    }

    // TODO KNN accuracy bugs
    void kNN_query(vector<Point> &results, Point query_point, int kk)
    {
        vector<int> lp(k, -1); // stores the index of node
        vector<int> rp(k, -1);
        vector<bool> oflag(k, false);
        double delta_r = sqrt((float)kk / N);
        double r = delta_r;
        vector<Point> S;
        while (true)
        {
            if (S.size() == kk)
            {
                double dist_furthest = 0;
                int dist_furthest_i = 0;
                for (size_t i = 0; i < kk; i++)
                {
                    double temp_dist = S[i].cal_dist(query_point);
                    if (temp_dist > dist_furthest)
                    {
                        dist_furthest = temp_dist;
                        dist_furthest_i = i;
                    }
                }
                if (dist_furthest < r)
                {
                    break;
                }
            }
            for (size_t i = 0; i < k; i++)
            {
                double dis = reference_points[i].cal_dist(query_point);
                if (oflag[i] == false)
                {
                    if (offsets[i + 1] - offsets[i] >= dis) // shpere contains q
                    {
                        oflag[i] = true;

                        int lnode_index = ceil(predict_closest_position(dis + offsets[i], false, i) / page_size);
                        int upper_bound = ceil(partition_size[i + 1] / page_size);
                        int lower_bound = floor(partition_size[i] / page_size);

                        lnode_index = lnode_index < lower_bound ? lower_bound : lnode_index;
                        lnode_index = lnode_index > upper_bound ? upper_bound : lnode_index;

                        lp[i] = lnode_index;
                        rp[i] = lnode_index;
                        double key = dis + offsets[i] - r;

                        search_inward(lp[i], lower_bound, key, S, kk, query_point);
                        key = dis + offsets[i] + r;
                        search_outward(rp[i], upper_bound, key, S, kk, query_point);
                    }
                    else if (offsets[i + 1] - offsets[i] + r >= dis)
                    {
                        oflag[i] = true;
                        int lnode_index = predict_closest_position(offsets[i + 1], false, i) / page_size;
                        int lower_bound = partition_size[i] / page_size;
                        lower_bound = lower_bound > 0 ? lower_bound - 1 : lower_bound;
                        if (lnode_index < lower_bound)
                        {
                            lnode_index = lower_bound;
                        }
                        lp[i] = lnode_index;
                        double key = dis + offsets[i] - r;
                        search_inward(lp[i], lower_bound, key, S, kk, query_point);
                    }
                }
                else
                {
                    if (lp[i] != -1)
                    {
                        double key = dis + offsets[i] - r;
                        int lower_bound = partition_size[i] / page_size;
                        lower_bound = lower_bound > 0 ? lower_bound - 1 : lower_bound;
                        search_inward(lp[i], lower_bound, key, S, kk, query_point);
                    }
                    if (rp[i] != -1)
                    {
                        double key = dis + offsets[i + 1] + r;
                        int upper_bound = partition_size[i + 1] / page_size;
                        upper_bound = upper_bound < storage_leafnodes.size() - 1 ? upper_bound + 1 : upper_bound;
                        upper_bound = upper_bound > storage_leafnodes.size() - 1 ? storage_leafnodes.size() - 1 : upper_bound;
                        search_outward(rp[i], upper_bound, key, S, kk, query_point);
                    }
                }
            }
            r *= 2;
        }
        vector<Point> extra_storage_S;
        framework.kNN_query(extra_storage_S, query_point, kk);
        if (extra_storage_S.size() != 0)
        {
            S.insert(S.end(), extra_storage_S.begin(), extra_storage_S.end());
            sort(S.begin(), S.end(), sortForKNN2());
        }
        results.insert(results.end(), S.begin(), S.begin() + kk);
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
        framework.insert(point);
    }

    bool delete_(Point &query_point)
    {
        long front = 0, back = 0;
        int predicted_index = get_point_index(query_point, front, back);
        front /= page_size;
        back /= page_size;
        while (front <= back)
        {
            int mid = (front + back) / 2;
            double first_ml_key = storage_leafnodes[mid].children[0].key;
            double last_ml_key = storage_leafnodes[mid].children[storage_leafnodes[mid].children.size() - 1].key;
            if (first_ml_key <= query_point.key && query_point.key <= last_ml_key)
            {
                for (size_t i = 0; i < storage_leafnodes[mid].children.size(); i++)
                {
                    if (storage_leafnodes[mid].children[i] == query_point)
                    {
                        storage_leafnodes[mid].children[i].is_deleted = true;
                        return true;
                    }
                }
                break;
            }
            else
            {
                if (storage_leafnodes[mid].children[0].key < query_point.key)
                {
                    front = mid + 1;
                }
                else
                {
                    back = mid - 1;
                }
            }
        }
        return framework.delete_(query_point);
    }

    void build_single_ML(DataSet<Point, double> dataset, int method)
    {
        error_shift = 0;
        print("build_single_ML");

        vector<std::shared_ptr<MLP>> temp_index;
        int next_stage_length = dataset.points.size();
        std::shared_ptr<MLP> mlp = framework.get_build_method(dataset, method);

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

    void build_ML(ExpRecorder &exp_recorder)
    {
        records.resize(stages.size());
        vector<vector<Point>> stage1(stages[0]);
        stage1[0] = dataset.points;
        records[0] = stage1;

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
                DataSet<Point, double> original_data_set(records[i][j]);
                original_data_set.read_keys_and_labels();
                int method = exp_recorder.build_method;
                if (exp_recorder.is_framework)
                {
                    method = framework.build_predict_method(exp_recorder.upper_level_lambda, query_frequency, original_data_set);
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

                temp_index.push_back(mlp);
                int max_error = 0;
                int min_error = 0;
                for (Point point : records[i][j])
                {
                    float pred = mlp->predict_ZM(point.normalized_key);
                    int pos = pred * next_stage_length;
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
        exp_recorder.timer_end();
    }

    long acc_window_query_num(Query<Point> &query)
    {
        vector<Point> res;
        for (size_t i = 0; i < query.query_windows.size(); i++)
        {
            for (size_t j = 0; j < storage_leafnodes.size(); j++)
            {
                if (storage_leafnodes[j].mbr.interact(query.query_windows[i]))
                {
                    for (Point point : storage_leafnodes[j].children)
                    {
                        if (query.query_windows[i].contains(point))
                        {
                            res.push_back(point);
                        }
                    }
                }
            }
        }
        return res.size();
    }

    void query(Query<Point> query, ExpRecorder &exp_recorder)
    {
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
        //         for (size_t j = 0; j < storage_leafnodes.size(); j++)
        //         {
        //             for (Point point : storage_leafnodes[j].children)
        //             {
        //                 point.cal_dist(query.knn_query_points[i]);
        //                 if (pq.size() < k)
        //                 {
        //                     pq.push(point);
        //                 }
        //                 else
        //                 {
        //                     if (pq.top().temp_dist > point.temp_dist)
        //                     {
        //                         pq.pop();
        //                         pq.push(point);
        //                     }
        //                 }
        //             }
        //         }

        //         vector<Point> query_res(query.results.begin() + i * k, query.results.begin() + i * k + k);
        //         // for (size_t j = 0; j < query_res.size(); j++)
        //         // {
        //         //     cout << query_res[j].x << ", " << query_res[j].y << " " << query_res[j].temp_dist << endl;
        //         // }
        //         // cout << "---------------------------" << endl;
        //         while (!pq.empty())
        //         {
        //             // cout << pq.top().x << ", " << pq.top().y << " " << pq.top().temp_dist << endl;

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

    DataSet<Point, double> generate_points(long cardinality, float dist)
    {
        DataSet<Point, double> dataset;
        int bin_num_synthetic = 10;

        int N = cardinality_u;
        // int bit_num = 8;
        // float xs_max[2] = {(float)(N), (float)(N)};
        // float max_key = compute_Z_value(xs_max, 2, bit_num);

        double max_key = sqrt(2);

        double min_key = 0;

        double gap = (max_key - min_key) / bin_num_synthetic;
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
            double key = ((double)rand() / (RAND_MAX));
            double x = ((double)rand() / (RAND_MAX));
            double y = ((double)rand() / (RAND_MAX));
            int bin = (key - min_key) / gap;
            // cout << "x:" << x << " y:" << y << " key:" << key << " bin:" << bin << endl;
            if (bin < bin_num_synthetic && counter_array[bin] > 0)
            {
                counter_array[bin]--;
                temp--;
                Point point(x, y);
                points.push_back(point);
            }
            // cout << "bin:" << bin << endl;
            // cout << "temp:" << temp << endl;
        }
        dataset.points = points;
        dataset.mapping()->generate_normalized_keys()->generate_labels();
        return dataset;
    }

    void generate_all_points()
    {
        print("generate_points");
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
        exp_recorder.name = "ML";
        exp_recorder.timer_begin();
        vector<int> methods{Constants::CL, Constants::MR, Constants::OG, Constants::RL, Constants::RS, Constants::SP};
        config::init_method_pool(methods);
        framework.config_method_pool();

        framework.dimension = 1;
        framework.point_query_p = point_query;
        framework.window_query_p = window_query;
        framework.knn_query_p = kNN_query;
        framework.build_index_p = build_single_ML;
        framework.init_storage_p = init_underlying_data_storage;
        framework.insert_p = insert;
        framework.generate_points_p = generate_points;
        framework.max_cardinality = cardinality_u;
        partitions.resize(k);
        offsets.resize(k + 1);
        partition_size.resize(k + 1);
        partitions.resize(k);
        gen_reference_points(exp_recorder);

        DataSet<Point, double>::read_data_pointer = read_data;
        DataSet<Point, double>::mapping_pointer = mapping;
        DataSet<Point, double>::save_data_pointer = save_data;
        stages.push_back(1);
        // framework.index_name = "ML";
        // framework.config_method_pool();
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
        stages.push_back(N / Constants::THRESHOLD);

        init_underlying_data_storage(dataset);
        exp_recorder.timer_end();
        print("data init time:" + to_string((int)(exp_recorder.time / 1e9)) + "s");
    }
}
#endif // use_gpu
