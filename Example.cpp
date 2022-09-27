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
#include "indices/Flood.h"
#include "indices/ML.h"
#include "indices/RSMI.h"
#include "indices/LISA.h"
#include "ELSI.h"

using namespace std;
using namespace zm;
using namespace flood;
// using namespace rsmi;

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
//         points[i].key = compute_Z_value(xs, 2, bit_num);
//     }
//     sort(points.begin(), points.end(), sort_key());
//     vector<long> keys(N);
//     for (long i = 0; i < N; i++)
//     {
//         keys[i] = points[i].key;
//     }
//     return keys;
// }

float areas[] = {0.000006, 0.000025, 0.0001, 0.0004, 0.0016};
float ratios[] = {0.25, 0.5, 1, 2, 4};
int methods[6] = {Constants::MR, Constants::CL, Constants::RS, Constants::RL, Constants::SP, Constants::OG};
int window_length = sizeof(areas) / sizeof(areas[0]);
int ratio_length = sizeof(ratios) / sizeof(ratios[0]);
int query_num = 1000;

void get_mbrs(map<string, vector<Mbr>> &mbrs_map, ExpRecorder &exp_recorder, vector<Point> &points)
{
    FileReader query_filereader;

    for (size_t i = 0; i < window_length; i++)
    {
        for (size_t j = 0; j < ratio_length; j++)
        {
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[j];
            vector<Mbr> mbrs = query_filereader.get_mbrs(exp_recorder.get_query_mbrs_name(), ",");
            if (mbrs.size() == 0)
            {
                break;
            }
            mbrs_map.insert(pair<string, vector<Mbr>>(to_string(areas[i]) + to_string(ratios[j]), mbrs));
        }
    }
    if (mbrs_map.size() == 0)
    {
        FileWriter query_file_writer(Constants::QUERYPROFILES);

        for (size_t i = 0; i < window_length; i++)
        {
            for (size_t j = 0; j < ratio_length; j++)
            {
                exp_recorder.window_size = areas[i];
                exp_recorder.window_ratio = ratios[j];
                vector<Mbr> mbrs = Mbr::get_mbrs(points, exp_recorder.window_size, query_num, exp_recorder.window_ratio);
                query_file_writer.write_mbrs(mbrs, exp_recorder);
                mbrs_map.insert(pair<string, vector<Mbr>>(to_string(areas[i]) + to_string(ratios[j]), mbrs));
            }
        }
    }
}

void test_ZM_single(ExpRecorder &exp_recorder)
{
}

void test_ZM(ExpRecorder &exp_recorder)
{
    print("---------ZM----------");

    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;
    zm::init(dataset_name, exp_recorder);

    map<string, vector<Mbr>> mbrs_map;
    get_mbrs(mbrs_map, exp_recorder, zm::dataset.points);

    zm::stages.push_back(1);
    zm::stages.push_back(zm::dataset.points.size() / Constants::THRESHOLD);

    zm::build_ZM(exp_recorder);
    exp_recorder.time /= 1e9;
    print("build time:" + to_string(exp_recorder.time) + " s");
    print(exp_recorder.get_build_result());
    file_writer.write_build(exp_recorder);

    Query<Point> query;
    long N = zm::dataset.points.size();
    if (exp_recorder.test_point)
    {
        query.set_point_query()->query_points = zm::dataset.points;
        zm::query(query, exp_recorder);
        exp_recorder.time /= N;
        cout << "point query time:" << exp_recorder.time << endl;
        file_writer.write_point_query(exp_recorder);
    }

    if (exp_recorder.test_window)
    {
        for (int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = mbrs_map[to_string(areas[i]) + to_string(ratios[2])];
            query.set_window_query()->query_windows = mbrs;
            // ->set_query_windows(mbrs);
            zm::query(query, exp_recorder);
            exp_recorder.time /= query_num;
            print("window query time:" + to_string(exp_recorder.time) + " ns");
            file_writer.write_window_query(exp_recorder);
        }
    }

    if (exp_recorder.test_knn)
    {
        vector<Point> knn_query_points;
        for (int i = 0; i < query_num; i++)
        {
            int index = rand() % zm::dataset.points.size();
            knn_query_points.push_back(zm::dataset.points[index]);
        }
        query.set_knn_query()->set_k(25)->knn_query_points = knn_query_points;
        // set_knn_query_points(knn_query_points)->
        zm::query(query, exp_recorder);
        exp_recorder.time /= query_num;
        print("knn query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_kNN_query(exp_recorder);
    }
}

void test_ML_single(ExpRecorder &exp_recorder)
{
}

void test_ML(ExpRecorder &exp_recorder)
{
    print("---------ML----------");

    map<string, vector<Mbr>> mbrs_map;
    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;
    ml::init(dataset_name, exp_recorder);
    get_mbrs(mbrs_map, exp_recorder, ml::dataset.points);

    ml::build_ML(exp_recorder);
    exp_recorder.time /= 1e9;
    print("build time:" + to_string(exp_recorder.time) + " s");
    print(exp_recorder.get_build_result());

    file_writer.write_build(exp_recorder);

    Query<Point> query;
    if (exp_recorder.test_point)
    {
        query.set_point_query()->query_points = ml::dataset.points;
        ml::query(query, exp_recorder);
        exp_recorder.time /= ml::N;
        print("point query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_point_query(exp_recorder);
    }

    if (exp_recorder.test_window)
    {
        for (int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = mbrs_map[to_string(areas[i]) + to_string(ratios[2])];
            query.set_window_query()->query_windows = mbrs;
            ml::query(query, exp_recorder);
            exp_recorder.time /= query_num;
            print("window query time:" + to_string(exp_recorder.time) + " ns");
            file_writer.write_window_query(exp_recorder);
        }
    }

    if (exp_recorder.test_knn)
    {
        vector<Point> knn_query_points;
        for (int i = 0; i < query_num; i++)
        {
            int index = rand() % ml::dataset.points.size();
            knn_query_points.push_back(ml::dataset.points[index]);
        }
        query.set_knn_query()->set_k(25)->knn_query_points = knn_query_points;
        ml::query(query, exp_recorder);
        exp_recorder.time /= query_num;
        print("knn query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_kNN_query(exp_recorder);
    }
}

void test_RSMI_single(ExpRecorder &exp_recorder)
{
    map<string, vector<Mbr>> mbrs_map;
    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;

    int length = sizeof(methods) / sizeof(methods[0]);
    for (size_t i = 0; i < length; i++)
    {
        exp_recorder.build_method = methods[i];
        rsmi::init(dataset_name, exp_recorder);
        get_mbrs(mbrs_map, exp_recorder, rsmi::dataset.points);
        rsmi::root = rsmi::build_single_RSMI(exp_recorder, rsmi::dataset);
        exp_recorder.timer_end();
        print("build time:" + to_string(exp_recorder.time_to_second()) + " s");
        file_writer.write_build(exp_recorder);

        Query<Point> query;
        query.set_point_query()->query_points = rsmi::dataset.points;
        rsmi::query(query, exp_recorder);
        exp_recorder.time /= rsmi::N;
        print("query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_point_query(exp_recorder);

        for (int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = mbrs_map[to_string(areas[i]) + to_string(ratios[2])];

            query.set_window_query()->query_windows = mbrs;
            rsmi::query(query, exp_recorder);
            exp_recorder.time /= query_num;
            print("window query time:" + to_string(exp_recorder.time) + " ns");
            file_writer.write_window_query(exp_recorder);
        }

        vector<Point> knn_query_points;
        for (int i = 0; i < query_num; i++)
        {
            int index = rand() % rsmi::dataset.points.size();
            knn_query_points.push_back(rsmi::dataset.points[index]);
        }
        query.set_knn_query()->set_k(25)->knn_query_points = knn_query_points;
        rsmi::query(query, exp_recorder);
        exp_recorder.time /= query_num;
        print("knn query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_kNN_query(exp_recorder);
    }
}

void test_RSMI(ExpRecorder &exp_recorder)
{
    print("---------RSMI----------");

    map<string, vector<Mbr>> mbrs_map;
    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;

    // exp_recorder.build_method = Constants::MR;
    rsmi::init(dataset_name, exp_recorder);
    get_mbrs(mbrs_map, exp_recorder, rsmi::dataset.points);

    rsmi::build_RSMI(exp_recorder, rsmi::dataset.points);
    exp_recorder.timer_end();
    exp_recorder.time /= 1e9;
    print("build time:" + to_string(exp_recorder.time) + " s");
    print(exp_recorder.get_build_result());

    Query<Point> query;
    if (exp_recorder.test_point)
    {
        query.set_point_query()->query_points = rsmi::dataset.points;
        rsmi::query(query, exp_recorder);
        exp_recorder.time /= rsmi::N;
        cout << "point query time:" << exp_recorder.time << endl;
        file_writer.write_point_query(exp_recorder);
    }

    if (exp_recorder.test_window)
    {
        for (int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = mbrs_map[to_string(areas[i]) + to_string(ratios[2])];
            query.set_window_query()->query_windows = mbrs;
            rsmi::query(query, exp_recorder);
            exp_recorder.time /= query_num;
            print("window query time:" + to_string(exp_recorder.time) + " ns");
            file_writer.write_window_query(exp_recorder);
        }
    }

    if (exp_recorder.test_knn)
    {
        vector<Point> knn_query_points;
        for (int i = 0; i < query_num; i++)
        {
            int index = rand() % rsmi::dataset.points.size();
            knn_query_points.push_back(rsmi::dataset.points[index]);
        }
        query.set_knn_query()->set_k(25)->knn_query_points = knn_query_points;
        rsmi::query(query, exp_recorder);
        exp_recorder.time /= query_num;
        print("knn query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_kNN_query(exp_recorder);
    }
}

void test_LISA_single(ExpRecorder &exp_recorder)
{
}

void test_LISA(ExpRecorder &exp_recorder)
{
    print("---------LISA----------");

    map<string, vector<Mbr>> mbrs_map;
    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;

    lisa::init(dataset_name, exp_recorder);
    get_mbrs(mbrs_map, exp_recorder, lisa::dataset.points);

    lisa::build_LISA(exp_recorder);
    exp_recorder.time /= 1e9;
    print("build time:" + to_string(exp_recorder.time) + " s");
    print(exp_recorder.get_build_result());
    file_writer.write_build(exp_recorder);

    Query<Point> query;
    if (exp_recorder.test_point)
    {
        query.set_point_query()->query_points = lisa::dataset.points;
        lisa::query(query, exp_recorder);
        exp_recorder.time /= lisa::N;
        cout << "point query time:" << exp_recorder.time << endl;
        file_writer.write_point_query(exp_recorder);
    }

    if (exp_recorder.test_window)
    {
        for (int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = mbrs_map[to_string(areas[i]) + to_string(ratios[2])];
            query.set_window_query()->query_windows = mbrs;
            lisa::query(query, exp_recorder);
            exp_recorder.time /= query_num;
            print("window query time:" + to_string(exp_recorder.time) + " ns");
            file_writer.write_window_query(exp_recorder);
        }
    }

    if (exp_recorder.test_knn)
    {
        vector<Point> knn_query_points;
        for (int i = 0; i < query_num; i++)
        {
            int index = rand() % lisa::dataset.points.size();
            knn_query_points.push_back(lisa::dataset.points[index]);
        }
        query.set_knn_query()->set_k(25)->knn_query_points = knn_query_points;
        lisa::query(query, exp_recorder);
        exp_recorder.time /= query_num;
        print("knn query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_kNN_query(exp_recorder);
    }
}

void test_Flood(ExpRecorder &exp_recorder)
{
    print("---------Flood----------");

    string dataset_name = exp_recorder.get_dataset_name();
    print("dataset_name:" + dataset_name);
    FileWriter file_writer;

    // exp_recorder.build_method = Constants::MR;
    flood::init(dataset_name, exp_recorder);

    flood::build_Flood(exp_recorder);
    exp_recorder.timer_end();
    exp_recorder.time /= 1e9;
    print("build time:" + to_string(exp_recorder.time) + " s");
    print(exp_recorder.get_build_result());

    Query<Point> query;
    if (exp_recorder.test_point)
    {
        query.set_point_query()->query_points = rsmi::dataset.points;
        rsmi::query(query, exp_recorder);
        exp_recorder.time /= rsmi::N;
        cout << "point query time:" << exp_recorder.time << endl;
        file_writer.write_point_query(exp_recorder);
    }

    if (exp_recorder.test_window)
    {
        map<string, vector<Mbr>> mbrs_map;
        get_mbrs(mbrs_map, exp_recorder, rsmi::dataset.points);
        for (int i = 0; i < 5; i++)
        {
            vector<Mbr> mbrs = mbrs_map[to_string(areas[i]) + to_string(ratios[2])];
            query.set_window_query()->query_windows = mbrs;
            rsmi::query(query, exp_recorder);
            exp_recorder.time /= query_num;
            print("window query time:" + to_string(exp_recorder.time) + " ns");
            file_writer.write_window_query(exp_recorder);
        }
    }

    if (exp_recorder.test_knn)
    {
        vector<Point> knn_query_points;
        for (int i = 0; i < query_num; i++)
        {
            int index = rand() % rsmi::dataset.points.size();
            knn_query_points.push_back(rsmi::dataset.points[index]);
        }
        query.set_knn_query()->set_k(25)->knn_query_points = knn_query_points;
        rsmi::query(query, exp_recorder);
        exp_recorder.time /= query_num;
        print("knn query time:" + to_string(exp_recorder.time) + " ns");
        file_writer.write_kNN_query(exp_recorder);
    }
}

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
            {"rebuild", no_argument, NULL, 'r'},
            {"original build", no_argument, NULL, 'o'},
            {"single build", no_argument, NULL, 'b'},
            {"single build parameter", no_argument, NULL, 'p'},
            {"query method(s)", no_argument, NULL, 'q'},
        };

    while (1)
    {
        int opt_index = 0;
        c = getopt_long(argc, argv, "c:d:s:l:n:fu:rob:p:q:", long_options, &opt_index);

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
        case 'o':
            exp_recorder.is_original = true;
            break;
        case 'b':
            exp_recorder.is_single_build = true;
            exp_recorder.build_method = atoi(optarg);
            std::cout << "exp_recorder.build_method: " << exp_recorder.build_method << std::endl;
            break;
        case 'p':
            // if is_single_build is true, we fix the parameter
            config::set_method_value(exp_recorder.build_method, atof(optarg));
            break;
        case 'q':
            int num = atoi(optarg);
            exp_recorder.test_point = num & 1;
            exp_recorder.test_window = num & 2;
            exp_recorder.test_knn = num & 4;
            // std::cout << "exp_recorder.test_point: " << exp_recorder.test_point << std::endl;
            // std::cout << "exp_recorder.test_window: " << exp_recorder.test_window << std::endl;
            // std::cout << "exp_recorder.test_knn: " << exp_recorder.test_knn << std::endl;
            break;
        }
    }
}

int main(int argc, char **argv)
{
    torch::manual_seed(0);
    ExpRecorder exp_recorder;
    parse(argc, argv, exp_recorder);
    map<string, int> names = {{"ml", 3}, {"zm", 1}, {"rsmi", 4}, {"lisa", 2}, {"test", 5}, {"ml-single", 6}, {"zm-single", 7}, {"rsmi-single", 8}, {"lisa-single", 9}, {"flood", 10}};

    // map<int, string> methods = {{0, "CL"}, {1, "MR"}, {2, "OG"}, {3, "RL"}, {4, "RS"}, {5, "SP"}};
    // if (exp_recorder.is_single_build)
    // {
    //     print("test method: " + methods[exp_recorder.build_method]);
    // }
    switch (names[exp_recorder.name])
    {
    case 1:
        test_ZM(exp_recorder);
        break;
    case 2:
        test_LISA(exp_recorder);
        break;
    case 3:
        test_ML(exp_recorder);
        break;
    case 4:
        test_RSMI(exp_recorder);
        break;
    case 5:
        print("test");
        // test_RSMI_single(exp_recorder);
        break;
    case 6:
        test_ML_single(exp_recorder);
        break;
    case 7:
        test_ZM_single(exp_recorder);
        break;
    case 8:
        test_RSMI_single(exp_recorder);
        break;
    case 9:
        test_LISA_single(exp_recorder);
        break;
    case 10:
        test_Flood(exp_recorder);
        break;
    default:
        break;
    }
}

#endif // use_gpu
