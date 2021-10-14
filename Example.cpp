#ifndef use_gpu
#define use_gpu
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
#include <chrono>

#include "entities/Point.h"
#include "entities/Query.h"
#include "entities/DataSet.h"
#include "utils/FileReader.h"
#include "utils/FileWriter.h"
#include "utils/SortTools.h"
#include "curves/z.H"

#include "indices/ZM.h"
#include "ELSI.h"

using namespace std;
using namespace zm;

// vector<Point> read_data(string filename, string delimeter, double &min_x, double &min_y, double &max_x, double &max_y)
// {
//     ifstream file(filename);

//     vector<Point> points;

//     string line = "";
//     while (getline(file, line))
//     {
//         vector<string> vec;
//         boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
//         Point point(stod(vec[0]), stod(vec[1]));

//         max_x = max_x > point.x ? max_x : point.x;
//         max_y = max_y > point.y ? max_y : point.y;
//         min_x = min_x < point.x ? min_x : point.x;
//         min_y = min_y < point.y ? min_y : point.y;

//         points.push_back(point);
//     }
//     // Close the File
//     file.close();

//     return points;
// }

// vector<long> generate_keys(vector<Point> points)
// {
//     int N = points.size();
//     int bit_num = ceil((log(N)) / log(2));
//     for (long i = 0; i < N; i++)
//     {
//         long long xs[2] = {(long long)(points[i].x * N), (long long)(points[i].y * N)};
//         points[i].curve_val = compute_Z_value(xs, 2, bit_num);
//     }
//     sort(points.begin(), points.end(), sort_curve_val());
//     vector<long> keys(N);
//     for (long i = 0; i < N; i++)
//     {
//         keys[i] = points[i].curve_val;
//     }
//     return keys;
// }

void parse(int argc, char **argv, ExpRecorder &exp_recorder)
{
    int c;
    static struct option long_options[] =
        {
            {"cardinality", required_argument, NULL, 'c'},
            {"distribution", required_argument, NULL, 'd'},
            {"skewness", required_argument, NULL, 's'},
            {"lambda", required_argument, NULL, 'l'},
            {"name", required_argument, NULL, 'n'},
            {"is_framework", no_argument, NULL, 'f'},
            {"update", no_argument, NULL, 'u'},
        };

    while (1)
    {
        int opt_index = 0;
        c = getopt_long(argc, argv, "c:d:s:l:n:fu:r", long_options, &opt_index);

        if (-1 == c)
        {
            break;
        }
        switch (c)
        {
        case 'c':
            exp_recorder.dataset_cardinality = atoll(optarg);
            break;
        case 'd':
            exp_recorder.distribution = optarg;
            break;
        case 's':
            exp_recorder.skewness = atoi(optarg);
            break;
        case 'l':
            exp_recorder.upper_level_lambda = atof(optarg);
            exp_recorder.lower_level_lambda = atof(optarg);
            break;
        case 'n':
            exp_recorder.name = optarg;
            break;
        case 'f':
            exp_recorder.is_framework = true;
            break;
        case 'u':
            exp_recorder.is_update = true;
            exp_recorder.insert_points_distribution = optarg;
            break;
        case 'r':
            exp_recorder.is_rebuildable = true;
            break;
        }
    }
}

float areas[] = {0.000006, 0.000025, 0.0001, 0.0004, 0.0016};
float ratios[] = {0.25, 0.5, 1, 2, 4};
int window_length = sizeof(areas) / sizeof(areas[0]);
int ratio_length = sizeof(ratios) / sizeof(ratios[0]);
int query_num = 1000;

void get_mbrs(map<string, vector<Mbr>> &mbrs_map, ExpRecorder &exp_recorder)
{
    FileReader query_filereader;

    for (size_t i = 0; i < window_length; i++)
    {
        for (size_t j = 0; j < ratio_length; j++)
        {
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[j];
            vector<Mbr> mbrs = query_filereader.get_mbrs(exp_recorder.get_query_mbrs_name(), ",");
            mbrs_map.insert(pair<string, vector<Mbr>>(to_string(areas[i]) + to_string(ratios[j]), mbrs));
        }
    }
}

int main(int argc, char **argv)
{
    torch::manual_seed(0);
    ExpRecorder exp_recorder;
    parse(argc, argv, exp_recorder);
    map<string, vector<Mbr>> mbrs_map;
    get_mbrs(mbrs_map, exp_recorder);
    // string dataset_name = "/home/research/datasets/skewed_2000000_4_2_.csv";
    // string dataset_name = "/home/research/datasets/OSM_100000000_1_2_.csv";
    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;
    zm::init(dataset_name, exp_recorder);
    zm::build_ZM(exp_recorder);
    exp_recorder.time /= 1e9;
    file_writer.write_build(exp_recorder);

    Query<Point> query;
    query.set_point_query()->set_query_points(zm::dataset.points);
    zm::query(query, exp_recorder);
    exp_recorder.time /= dataset.points.size();
    file_writer.write_point_query(exp_recorder);

    vector<Mbr> mbrs = mbrs_map[to_string(areas[2]) + to_string(ratios[2])];
    query.set_window_query()->set_query_windows(mbrs);
    zm::query(query, exp_recorder);
    exp_recorder.time /= query_num;
    file_writer.write_window_query(exp_recorder);

    vector<Point> knn_query_points;
    for (int i = 0; i < query_num; i++)
    {
        int index = rand() % zm::dataset.points.size();
        knn_query_points.push_back(zm::dataset.points[index]);
    }
    query.set_knn_query()->set_knn_query_points(knn_query_points)->set_k(25);
    zm::query(query, exp_recorder);
    exp_recorder.time /= query_num;
    file_writer.write_kNN_query(exp_recorder);

    // TODO point query !!!
    // TODO knn query!!!!


    // TODO RSMI
    // TODO lisa
    // TODO ML-index
}

#endif // use_gpu
