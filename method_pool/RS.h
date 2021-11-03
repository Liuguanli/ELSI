#ifndef RS_H
#define RS_H

#include "method.h"
#include <vector>
#include <string.h>
#include <string>
#include "../entities/DataSet.h"

template <typename T>
class RS
{

public:
    DataSet<Point, T> do_rs(DataSet<Point, T> &dataset, int m, int dimension)
    {
        if (dimension == 1)
        {
            return do_rs(dataset, m);
        }
        if (dimension == 2)
        {
            return do_rs_2d(dataset, m);
        }
        return dataset;
    }

    DataSet<Point, T> do_rs(DataSet<Point, T> &data_set, int m)
    {
        double start_x = data_set.min_x;
        double start_y = data_set.min_y;
        double x_edge_length = (data_set.max_x - data_set.min_x) / 2;
        double y_edge_length = (data_set.max_y - data_set.min_y) / 2;
        vector<Point> points;
        get_rep_set_space(m, start_x, start_y, x_edge_length, y_edge_length, data_set.points, points);
        sort(points.begin(), points.end(), sort_label());
        DataSet<Point, T> sampled_data_set(points);
        sampled_data_set.read_keys_and_labels();
        return sampled_data_set;
    }

    DataSet<Point, T> do_rs_2d(DataSet<Point, T> &data_set, int m)
    {
        double start_x = data_set.min_x;
        double start_y = data_set.min_y;
        double x_edge_length = (data_set.max_x - data_set.min_x) / 2;
        double y_edge_length = (data_set.max_y - data_set.min_y) / 2;
        vector<Point> points;
        get_rep_set_space(m, start_x, start_y, x_edge_length, y_edge_length, data_set.points, points);
        sort(points.begin(), points.end(), sort_label());
        DataSet<Point, T> sampled_data_set;
        for (size_t i = 0; i < points.size(); i++)
        {
            sampled_data_set.points.push_back(points[i]);
            sampled_data_set.normalized_keys.push_back(points[i].x);
            sampled_data_set.normalized_keys.push_back(points[i].y);
            sampled_data_set.labels.push_back(points[i].label);
        }
        cout << "sampled_data_set.size(): " << sampled_data_set.points.size() << endl;
        return sampled_data_set;
    }

    void get_rep_set_space(int m, double start_x, double start_y, double x_edge_length, double y_edge_length, vector<Point> &all_points, vector<Point> &rs)
    {
        // cout << "start_x: " << start_x << " start_y: " << start_y << " x_edge_length: " << x_edge_length << " y_edge_length: " << y_edge_length << " all_points: " << all_points.size() << endl;
        long long N = all_points.size();
        // vector<Point> rs;
        if (all_points.size() == 0)
        {
            return;
        }
        int key_num = 4;
        map<int, vector<Point>> points_map;
        for (size_t i = 0; i < N; i++)
        {
            int key = 0;
            if (all_points[i].x - start_x <= x_edge_length)
            {
                if (all_points[i].y - start_y <= y_edge_length)
                {
                    key = 0;
                }
                else
                {
                    key = 2;
                }
            }
            else
            {
                if (all_points[i].y - start_y <= y_edge_length)
                {
                    key = 1;
                }
                else
                {
                    key = 3;
                }
            }
            points_map[key].push_back(all_points[i]);
        }
        for (size_t i = 0; i < key_num; i++)
        {
            // cout << "get_rep_set_space 4 i= " << i << endl;
            if (points_map[i].size() == 0)
            {
                continue;
            }
            double start_x_temp = start_x;
            double start_y_temp = start_y;
            if (points_map[i].size() > m)
            {
                // cout << "get_rep_set_space 5" << endl;
                if (i == 1)
                {
                    start_x_temp = start_x + x_edge_length;
                }
                if (i == 2)
                {
                    start_y_temp = start_y + y_edge_length;
                }
                if (i == 3)
                {
                    start_x_temp = start_x + x_edge_length;
                    start_y_temp = start_y + y_edge_length;
                }
                if (points_map[i].size() == N)
                {
                    rs.insert(rs.end(), points_map[i].begin(), points_map[i].end());
                }
                else
                {
                    get_rep_set_space(m, start_x_temp, start_y_temp, x_edge_length / 2, y_edge_length / 2, points_map[i], rs);
                }
            }
            else if (points_map[i].size() > 0)
            {
                int middle_point = (points_map[i].size() - 1) / 2;
                rs.push_back(points_map[i][middle_point]);
            }
        }
    }
};

#endif