#ifndef EXPRECORDER_H
#define EXPRECORDER_H

#include <vector>
#include <string>
#include <chrono>
#include <queue>
#include "../entities/Point.h"
#include "../entities/Statistics.h"
#include "Constants.h"
#include "SortTools.h"

using namespace std;

class ExpRecorder
{

public:
    long long point_not_found = 0;

    long long index_high;
    long long index_low;

    long long leaf_node_num;
    long long non_leaf_node_num;

    long long max_error = 0;
    long long min_error = 0;

    int depth = 1;

    int new_depth = 1;

    int N = Constants::THRESHOLD;

    bool is_framework = false;
    bool is_update = false;
    bool is_rebuildable = false;

    string name;

    long long top_error = 0;
    long long bottom_error = 0;
    float loss = 0;

    int last_level_model_num = 0;
    long long non_leaf_model_num = 0;

    string structure_name;
    string distribution;
    long dataset_cardinality;

    int update_num = 0;

    long long insert_num = 0;
    int insert_times = 0;
    long long previous_insert_num = 0;
    long delete_num;
    float window_size;
    float window_ratio;
    int k_num = 25;
    int skewness = 1;

    long time;
    long top_level_time;
    long bottom_level_time;
    long insert_time = 0;
    long delete_time;
    long long rebuild_time;
    int rebuild_num;
    double page_access = 0.0;
    double accuracy;
    long size;
    long top_rl_time;
    long extra_time;

    long prediction_time = 0;
    long search_time = 0;
    long sfc_cal_time = 0;
    long ordering_cost = 0;
    long training_cost = 0;

    long long search_length = 0;

    long cost_model_time = 0;

    long search_steps = 0;

    float sampling_rate = 1.0;

    int rs_threshold_m = 10000;
    double model_reuse_threshold = 0.1;

    int build_method = Constants::SP;

    string cluster_method = "kmeans";
    int cluster_size = 100;
    string insert_points_distribution = "normal";

    long window_query_result_size;
    long acc_window_query_result_size;
    vector<Point> knn_query_results;
    vector<Point> acc_knn_query_results;
    vector<Point> window_query_results;
    vector<Point> acc_window_query_results;

    vector<Point> inserted_points;

    double upper_level_lambda = 0.8;
    double lower_level_lambda = 0.8;

    int sp_num = 0;
    int model_reuse_num = 0;
    int rl_num = 0;
    int cluster_num = 0;
    int rs_num = 0;
    int original_num = 0;

    int insert_rebuild_index = 0;

    long traverse_time = 0;

    int bit_num = 0;

    bool is_rebuild = false;
    bool is_knn = false;
    bool is_window = false;
    bool is_point = false;
    bool is_insert = false;

    void record_method_nums(int method)
    {
        switch (method)
        {
        case Constants::OG:
            original_num++;
            break;
        case Constants::MR:
            model_reuse_num++;
            break;
        case Constants::CL:
            cluster_num++;
            break;
        case Constants::RS:
            rs_num++;
            break;
        case Constants::RL:
            rl_num++;
            break;
        case Constants::SP:
            sp_num++;
            break;
        default:
            break;
        }
    }

    string get_current_time()
    {
        time_t now = std::time(0);
        char *dt = ctime(&now);
        std::string str(dt);
        return str;
    }

    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point finish;

    void timer_begin()
    {
        start = chrono::high_resolution_clock::now();
    }

    void timer_end()
    {
        finish = chrono::high_resolution_clock::now();
        time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }

    long time_to_second()
    {
        time /= 1e9;
        return time;
    }

    long time_to_millisecond()
    {
        time /= 1e6;
        return time;
    }

    long time_to_microsecond()
    {
        time /= 1e3;
        return time;
    }

    

    string get_file_name()
    {
        return distribution + "_" + to_string(dataset_cardinality) + "_" + to_string(skewness);
    }

    string get_dataset_name()
    {
        return Constants::DATASETS + get_file_name() + "_2_.csv";
    }

    string get_query_mbrs_name()
    {
        return Constants::QUERYPROFILES + Constants::WINDOW + distribution + "_" + to_string(dataset_cardinality) + "_" + to_string(skewness) + "_" + to_string(window_size) + "_" + to_string(window_ratio) + ".csv";
    }

    string get_build_result()
    {
        string result = "--------------------" + get_current_time();
        result += "time:" + to_string(time) + "\n";

        if (is_framework)
        {
            result += "FRAMEWORK\n";
            result += "lambda:" + to_string(upper_level_lambda) + "\n";
        }
        result += "og_num:" + to_string(original_num) + "\n";
        result += "mr_num:" + to_string(model_reuse_num) + "\n";
        result += "sp_num:" + to_string(sp_num) + "\n";
        result += "rl_num:" + to_string(rl_num) + "\n";
        result += "rs_num:" + to_string(rs_num) + "\n";
        result += "cl_num:" + to_string(cluster_num) + "\n";
        return result;
    }

    string get_point_query_result()
    {
        string result = "--------------------" + get_current_time();
        result += "time:" + to_string(time) + "\n";
        if (is_framework)
        {
            result += "FRAMEWORK\n";
            result += "lambda:" + to_string(upper_level_lambda) + "\n";
        }
        result += "og_num:" + to_string(original_num) + "\n";
        result += "mr_num:" + to_string(model_reuse_num) + "\n";
        result += "sp_num:" + to_string(sp_num) + "\n";
        result += "rl_num:" + to_string(rl_num) + "\n";
        result += "rs_num:" + to_string(rs_num) + "\n";
        result += "cl_num:" + to_string(cluster_num) + "\n";

        return result;
    }

    string get_window_query_result()
    {
        string result = "--------------------" + get_current_time();
        result += "time:" + to_string(time) + "\n";
        result += "accuracy:" + to_string(accuracy) + "\n";

        if (is_framework)
        {
            result += "FRAMEWORK\n";
            result += "lambda:" + to_string(upper_level_lambda) + "\n";
        }
        result += "og_num:" + to_string(original_num) + "\n";
        result += "mr_num:" + to_string(model_reuse_num) + "\n";
        result += "sp_num:" + to_string(sp_num) + "\n";
        result += "rl_num:" + to_string(rl_num) + "\n";
        result += "rs_num:" + to_string(rs_num) + "\n";
        result += "cl_num:" + to_string(cluster_num) + "\n";
        return result;
    }

    string get_knn_query_result()
    {
        string result = "--------------------" + get_current_time();
        result += "time:" + to_string(time) + "\n";
        result += "accuracy:" + to_string(accuracy) + "\n";

        if (is_framework)
        {
            result += "FRAMEWORK\n";
            result += "lambda:" + to_string(upper_level_lambda) + "\n";
        }
        result += "og_num:" + to_string(original_num) + "\n";
        result += "mr_num:" + to_string(model_reuse_num) + "\n";
        result += "sp_num:" + to_string(sp_num) + "\n";
        result += "rl_num:" + to_string(rl_num) + "\n";
        result += "rs_num:" + to_string(rs_num) + "\n";
        result += "cl_num:" + to_string(cluster_num) + "\n";
        return result;
    }
};

#endif