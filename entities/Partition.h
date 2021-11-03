#ifndef PARTITION_H
#define PARTITION_H
#include <iostream>
#include <vector>
#include <queue>
#include "../entities/Point.h"
#include "../entities/Mbr.h"
#include "../entities/LeafNode.h"
#include <typeinfo>
#include "../curves/hilbert.H"
#include "../curves/hilbert4.H"
#include <map>
#include <boost/smart_ptr/make_shared_object.hpp>
#include "../utils/SortTools.h"
#include "../utils/ModelTools.h"

class Partition
{

private:
    int level;
    int index;
    long long N = 0;
    int bit_num;
    Mbr mbr;
    bool is_reused = false;
    long long side = 0;

public:
    int max_error = 0;
    int min_error = 0;
    int leaf_node_num = 0;
    int width = 0;
    int max_partition_num = Constants::MAX_WIDTH;
    int page_size = Constants::PAGESIZE;
    bool is_last = false;
    float x_gap = 1.0;
    float x_scale = 1.0;
    float x_0 = 0;
    float x_1 = 0;
    float y_gap = 1.0;
    float y_scale = 1.0;
    float y_0 = 0;
    float y_1 = 0;
    std::shared_ptr<MLP> mlp;
    string model_path;
    static string model_path_root;
    map<int, Partition> children;
    vector<LeafNode> leafnodes;
    vector<float> points_x;
    vector<float> points_y;

    Partition() {}

    void init_last(vector<Point> &points, vector<float> &locations, vector<float> &labels, vector<long long> &keys)
    {
        N = points.size();
        side = pow(2, ceil(log(N) / log(2)));
        sort(points.begin(), points.end(), sortX());
        x_gap = 1.0 / (points[N - 1].x - points[0].x);
        x_scale = 1.0 / x_gap;
        x_0 = points[0].x;
        x_1 = points[N - 1].x;
        for (int i = 0; i < N; i++)
        {
            points[i].x_i = i;
            // points_x.push_back(points[i].x);
            mbr.update(points[i].x, points[i].y);
        }
        sort(points.begin(), points.end(), sortY());
        y_gap = points[N - 1].y - points[0].y;
        y_scale = 1.0 / y_gap;
        y_0 = points[0].y;
        y_1 = points[N - 1].y;
        for (int i = 0; i < N; i++)
        {
            points[i].y_i = i;
            points[i].key = compute_Hilbert_value(points[i].x_i, points[i].y_i, side);
        }
        sort(points.begin(), points.end(), sort_key());
        for (int i = 0; i < N; i++)
        {
            keys.push_back(points[i].key);
            labels.push_back((float)i / N);
            locations.push_back((points[i].x - x_0) * x_scale + x_0);
            locations.push_back((points[i].y - y_0) * y_scale + y_0);
        }
        long long h_min = points[0].key;
        long long h_max = points[N - 1].key;
        long long h_gap = h_max - h_min + 1;

        width = N - 1;
        for (long i = 0; i < N; i++)
        {
            points[i].index = i;
        }

        leaf_node_num = points.size() / page_size;
        for (int i = 0; i < leaf_node_num; i++)
        {
            LeafNode leafNode;
            auto bn = points.begin() + i * page_size;
            auto en = points.begin() + i * page_size + page_size;
            vector<Point> vec(bn, en);
            leafNode.add_points(vec);
            leafnodes.push_back(leafNode);
        }

        if (points.size() > page_size * leaf_node_num)
        {
            LeafNode leafNode;
            auto bn = points.begin() + page_size * leaf_node_num;
            auto en = points.end();
            vector<Point> vec(bn, en);
            leafNode.add_points(vec);
            leafnodes.push_back(leafNode);
            leaf_node_num++;
        }
    }

    void init(vector<Point> &points, vector<float> &locations, vector<float> &labels, vector<long long> &keys)
    {
        N = points.size();
        // print("Partition init");
        int partition_size = ceil(points.size() * 1.0 / pow(max_partition_num, 2));
        // cout << "partition_size:" << partition_size << endl;
        sort(points.begin(), points.end(), sortY());
        y_gap = points[N - 1].y - points[0].y;
        // cout << "y_gap:" << y_gap << endl;
        y_scale = 1.0 / y_gap;
        // cout << "y_scale:" << y_scale << endl;
        y_0 = points[0].y;
        // cout << "y_0:" << y_0 << endl;
        y_1 = points[N - 1].y;
        // cout << "y_1:" << y_1 << endl;
        sort(points.begin(), points.end(), sortX());
        x_gap = points[N - 1].x - points[0].x;
        // cout << "x_gap:" << x_gap << endl;

        x_scale = 1.0 / x_gap;
        // cout << "x_scale:" << x_scale << endl;

        x_0 = points[0].x;
        // cout << "x_0:" << x_0 << endl;

        x_1 = points[N - 1].x;
        // cout << "x_1:" << x_1 << endl;

        long long side = pow(max_partition_num, 2);
        width = side - 1;
        // cout << "width: " << width << endl;
        int each_item_size = partition_size * max_partition_num;
        long long point_index = 0;

        long long z_min = compute_Z_value(0, 0, 4);
        long long z_max = compute_Z_value(max_partition_num - 1, max_partition_num - 1, 4);
        long long z_gap = z_max - z_min;
        for (size_t i = 0; i < max_partition_num; i++)
        {
            long long bn_index = i * each_item_size;
            long long end_index = bn_index + each_item_size;
            if (bn_index >= N)
            {
                break;
            }
            else
            {
                if (end_index > N)
                {
                    end_index = N;
                }
            }
            auto bn = points.begin() + bn_index;
            auto en = points.begin() + end_index;
            vector<Point> vec(bn, en);
            sort(vec.begin(), vec.end(), sortY());
            for (size_t j = 0; j < max_partition_num; j++)
            {
                long long sub_bn_index = j * partition_size;
                long long sub_end_index = sub_bn_index + partition_size;
                if (sub_bn_index >= vec.size())
                {
                    break;
                }
                else
                {
                    if (sub_end_index > vec.size())
                    {
                        sub_end_index = vec.size();
                    }
                }
                auto sub_bn = vec.begin() + sub_bn_index;
                auto sub_en = vec.begin() + sub_end_index;
                vector<Point> sub_vec(sub_bn, sub_en);
                long long Z_value = compute_Z_value(i, j, 4);
                int sub_point_index = 1;
                long sub_size = sub_vec.size();
                int counter = 0;
                for (Point point : sub_vec)
                {
                    point.key = Z_value;
                    point.label = (float)(Z_value - z_min) / z_gap;
                    // point.label =
                    // keys.push_back(Z_value);
                    // labels.push_back(point.label);
                    // locations.push_back((point.x - x_0) * x_scale + x_0);
                    // locations.push_back((point.y - y_0) * y_scale + y_0);
                    mbr.update(point.x, point.y);
                    points[point_index++] = point;
                }
            }
        }

        sort(points.begin(), points.end(), sort_key());
        point_index = 0;
        for (Point point : points)
        {
            keys.push_back(point.key);
            labels.push_back(point.label);
            locations.push_back((point.x - x_0) * x_scale + x_0);
            locations.push_back((point.y - y_0) * y_scale + y_0);
        }
    }

    bool point_query_bs(Point &query_point)
    {
        // cout << "query_point.curve: " << query_point.key << endl;
        int predicted_index = 0;
        float x1 = (query_point.x - x_0) * x_scale + x_0;
        float x2 = (query_point.y - y_0) * y_scale + y_0;
        predicted_index = mlp->predict(x1, x2) * leaf_node_num;

        predicted_index = predicted_index < 0 ? 0 : predicted_index;
        predicted_index = predicted_index >= leaf_node_num ? leaf_node_num - 1 : predicted_index;
        int front = predicted_index + min_error;
        front = front < 0 ? 0 : front;
        int back = predicted_index + max_error;
        back = back >= leaf_node_num ? leaf_node_num - 1 : back;
        while (front <= back)
        {
            int mid = (front + back) / 2;
            long long first_curve_val = leafnodes[mid].children[0].key;
            long long last_curve_val = leafnodes[mid].children[leafnodes[mid].children.size() - 1].key;

            if (first_curve_val <= query_point.key && query_point.key <= last_curve_val)
            {
                vector<Point>::iterator iter = find(leafnodes[mid].children.begin(), leafnodes[mid].children.end(), query_point);
                if (iter == leafnodes[mid].children.end())
                {
                    return false;
                }
                else
                {
                    return true;
                }
            }
            else
            {
                if (leafnodes[mid].children[0].key < query_point.key)
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

    bool point_query(Point &query_point)
    {
        if (is_last)
        {
            if (leafnodes.size() == 0)
            {
                return false;
            }

            int predicted_index = 0;
            float x1 = (query_point.x - x_0) * x_scale + x_0;
            float x2 = (query_point.y - y_0) * y_scale + y_0;
            // predicted_index = (int)(net->predict(query_point, x_scale, y_scale, x_0, y_0) * width);
            // predicted_index = net->predict(query_point, x_scale, y_scale, x_0, y_0) * width;
            predicted_index = mlp->predict(x1, x2) * leaf_node_num;

            predicted_index = predicted_index < 0 ? 0 : predicted_index;
            predicted_index = predicted_index >= leaf_node_num ? leaf_node_num - 1 : predicted_index;
            // LeafNode leafnode = leafnodes[predicted_index];

            if (leafnodes[predicted_index].mbr.contains(query_point))
            {
                vector<Point>::iterator iter = find(leafnodes[predicted_index].children.begin(), leafnodes[predicted_index].children.end(), query_point);
                if (iter != leafnodes[predicted_index].children.end())
                {
                    // cout<< "find it" << endl;
                    return true;
                }
            }

            // predicted result is not correct
            int front = predicted_index + min_error;
            front = front < 0 ? 0 : front;
            int back = predicted_index + max_error;
            back = back >= leafnodes.size() ? leafnodes.size() - 1 : back;

            int gap = 1;
            int predicted_index_left = predicted_index - gap;
            int predicted_index_right = predicted_index + gap;
            while (predicted_index_left >= front && predicted_index_right <= back)
            {
                // search left
                // LeafNode leafnode = leafnodes[predicted_index_left];
                if (leafnodes[predicted_index_left].mbr.contains(query_point))
                {
                    vector<Point>::iterator iter = find(leafnodes[predicted_index_left].children.begin(), leafnodes[predicted_index_left].children.end(), query_point);
                    if (iter != leafnodes[predicted_index_left].children.end())
                    {
                        // cout<< "find it" << endl;
                        return true;
                    }
                }

                // search right
                // leafnode = leafnodes[predicted_index_right];
                if (leafnodes[predicted_index_right].mbr.contains(query_point))
                {
                    vector<Point>::iterator iter = find(leafnodes[predicted_index_right].children.begin(), leafnodes[predicted_index_right].children.end(), query_point);
                    if (iter != leafnodes[predicted_index_right].children.end())
                    {
                        // cout<< "find it" << endl;
                        return true;
                    }
                }
                gap++;
                predicted_index_left = predicted_index - gap;
                predicted_index_right = predicted_index + gap;
            }

            while (predicted_index_left >= front)
            {
                // LeafNode leafnode = leafnodes[predicted_index_left];
                if (leafnodes[predicted_index_left].mbr.contains(query_point))
                {
                    vector<Point>::iterator iter = find(leafnodes[predicted_index_left].children.begin(), leafnodes[predicted_index_left].children.end(), query_point);
                    if (iter != leafnodes[predicted_index_left].children.end())
                    {
                        // cout<< "find it" << endl;
                        return true;
                    }
                }
                gap++;
                predicted_index_left = predicted_index - gap;
            }

            while (predicted_index_right <= back)
            {
                // LeafNode leafnode = leafnodes[predicted_index_right];
                if (leafnodes[predicted_index_right].mbr.contains(query_point))
                {
                    vector<Point>::iterator iter = find(leafnodes[predicted_index_right].children.begin(), leafnodes[predicted_index_right].children.end(), query_point);
                    if (iter != leafnodes[predicted_index_right].children.end())
                    {
                        // cout<< "find it" << endl;
                        return true;
                    }
                }
                gap++;
                predicted_index_right = predicted_index + gap;
            }

            // point_not_found++;
            return false;
        }
        else
        {
            int predicted_index = 0;
            float x1 = (query_point.x - x_0) * x_scale + x_0;
            float x2 = (query_point.y - y_0) * y_scale + y_0;

            predicted_index = mlp->predict(x1, x2) * width;

            predicted_index = max(predicted_index, 0);
            predicted_index = min(predicted_index, width);

            return children[predicted_index].point_query(query_point);
        }
    }
};

#endif
