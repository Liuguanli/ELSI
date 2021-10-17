#ifndef ELSI_H
#define ELSI_H

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <ctime>

#include "entities/DataSet.h"
#include "entities/DataSetInfo.h"

#include "method_pool/CL.h"
#include "method_pool/SP.h"
#include "method_pool/OG.h"
#include "method_pool/RS.h"
#include "method_pool/RL.h"
#include "method_pool/MR.h"
#include "method_pool/method.h"

#include "utils/MethodScorer.h"
#include "utils/Log.h"
#include "utils/Util.h"
#include "utils/Config.h"
#include "utils/Constants.h"

#include "entities/Statistics.h"
#include "entities/ScorerItem.h"
#include "entities/ExtraStorageBlock.h"

using namespace logger;
using namespace config;
template <typename D, typename T>
class ELSI
{
public:
    int status = -1;
    std::shared_ptr<MLP> query_cost_model = std::make_shared<MLP>(8, 32);
    std::shared_ptr<MLP> build_cost_model = std::make_shared<MLP>(8, 32);
    std::shared_ptr<MLP> rebuild_model = std::make_shared<MLP>(5, 32);

    void (*point_query_p)(Query<D> &);
    void (*build_index_p)(DataSet<D, T>, int);
    void (*init_storage_p)(DataSet<D, T>);
    void (*window_query_p)(Query<D> &);
    void (*knn_query_p)(Query<D> &);
    void (*insert_p)(D);
    DataSet<D, T> (*generate_points_p)(long, float);
    map<int, vector<float>> methods;
    ExpRecorder exp_recorder;

    long max_cardinality = 10000000;

    // vector<ExtraStorageBlock<D>> extra_storage;
    vector<D> extra_storage;

    void config_method_pool(string index_name)
    {
        methods.insert(pair<int, vector<float>>(0, {1, 0, 0, 0, 0, 0}));
        methods.insert(pair<int, vector<float>>(1, {0, 1, 0, 0, 0, 0}));
        methods.insert(pair<int, vector<float>>(2, {0, 0, 1, 0, 0, 0}));
        methods.insert(pair<int, vector<float>>(5, {0, 0, 0, 0, 0, 1}));
        if (index_name != "LISA")
        {
            methods.insert(pair<int, vector<float>>(3, {0, 0, 0, 1, 0, 0}));
            methods.insert(pair<int, vector<float>>(4, {0, 0, 0, 0, 1, 0}));
        }
    }

    void init()
    {
        status = Constants::STATUS_FRAMEWORK_INIT;
        print("ELSI::init ELSI");
        init_build_processor();
        init_rebuild_processor();
        MR<D, T>::load_pre_trained_model();
        status = Constants::STATUS_FRAMEWORK_INIT_DONE;
    }

    int build_predict_method(float lambda, float query_frequency, DataSet<D, T> &original_data_set)
    {
        int method_pool_size = methods.size();
        DataSetInfo<T> info(Constants::DEFAULT_BIN_NUM, original_data_set.keys);
        float *build_score = new float[method_pool_size];
        float *query_score = new float[method_pool_size];
        float max = numeric_limits<float>::min();
        int max_index = 0;
        float distribution = info.get_distribution();
        vector<float> parameters;
        parameters.push_back((float)original_data_set.keys.size() / max_cardinality);
        parameters.push_back(distribution);
        for (size_t i = 0; i < method_pool_size; i++)
        {
            vector<float> temp = parameters;
            temp.insert(temp.end(), methods[i].begin(), methods[i].end());
#ifdef use_gpu
            torch::Tensor x = torch::tensor(temp, at::kCUDA).reshape({1, 8});
            build_cost_model->to(torch::kCUDA);
            query_cost_model->to(torch::kCUDA);
#else
            torch::Tensor x = torch::tensor(temp).reshape({1, 8});
#endif
            build_score[i] = build_cost_model->predict(x).item().toFloat();
            query_score[i] = query_cost_model->predict(x).item().toFloat();

            float score = build_score[i] * lambda + query_score[i] * query_frequency * (1 - lambda);
            if (score > max)
            {
                max = score;
                max_index = i;
            }
        }
        return max_index;
    }

    std::shared_ptr<MLP> build_with_method(DataSet<D, T> &original_data_set, int method_index)
    {
        DataSetInfo<T> info(Constants::DEFAULT_BIN_NUM, original_data_set.keys);
        DataSet<D, T> shrinked_data_set;
        switch (method_index)
        {
        case Constants::CL:
            CL<D, T> cl;
            shrinked_data_set = cl.do_cl(original_data_set, config::cluster_k);
            break;
        case Constants::MR:
            MR<D, T> mr;
            {
                string model_path = Constants::DEFAULT_PRE_TRAIN_MODEL_PATH;
                if (mr.is_reusable(info, model_path))
                {
                    // TODO here the input is 1,  net is for model reuse
                    auto net = std::make_shared<MLP>(1);
#ifdef use_gpu
                    net->to(torch::kCUDA);
#endif
                    // cout << "model_path" << model_path << endl;
                    torch::load(net, model_path);
                    net->get_parameters_ZM();

                    return net;
                }
            }
            shrinked_data_set = original_data_set;
            break;
        case Constants::OG:
            shrinked_data_set = original_data_set;
            break;
        case Constants::RL:
            RL<D, T> rl;
            shrinked_data_set = rl.do_rl(original_data_set, config::bit_num);
            break;
        case Constants::RS:
            RS<T> rs;
            shrinked_data_set = rs.do_rs(original_data_set, config::rs_m);
            break;
        case Constants::SP:
            SP<D, T> sp;
            shrinked_data_set = sp.do_sp(original_data_set, config::sampling_rate);
            break;
        default:
            shrinked_data_set = original_data_set;
            break;
        }
        auto mlp = model_training::real_train_1d(shrinked_data_set.normalized_keys, shrinked_data_set.labels);

        status = Constants::STATUS_FRAMEWORK_BUILD_DONE;
        return mlp;
    }

    void query(Query<D> &query)
    {
        status = Constants::STATUS_FRAMEWORK_BEGIN_QUERY;

        if (query.is_point())
        {
            print("point");

            point_query(query);
        }
        if (query.is_window())
        {
            print("window");

            window_query(query);
        }
        if (query.is_knn())
        {
            print("knn");
            knn_query(query);
        }

        status = Constants::STATUS_FRAMEWORK_BEGIN_QUERY_DONE;
    }

    void point_query(Query<D> &query)
    {
        if (point_query_p != NULL)
        {
            print("point_query");

            point_query_p(query);
        }
        // if (extra_storage.size() > 0)
        // {
        //     for (D query_point : query.query_points)
        //     {
        //         // vector<D>::iterator iter = find(extra_storage.begin(), extra_storage.end(), query_point);
        //         // if (iter != extra_storage.end())
        //         // {
        //         // }
        //     }
        // }
    }

    void window_query(Query<D> &query)
    {

        if (window_query_p != NULL)
        {
            print("window_query");

            window_query_p(query);
        }
    }

    void knn_query(Query<D> &query)
    {
        if (knn_query_p != NULL)
        {
            print("knn_query");

            knn_query_p(query);
        }
    }

    void insert(D point)
    {
        if (insert_p == NULL)
        {
            // if (extra_storage[extra_storage.size() - 1].is_full())
            // {
            //     ExtraStorageBlock<D> new_block;
            //     new_block.add_point(point);
            //     extra_storage.push_back(new_block);
            // }
            // else
            // {
            //     extra_storage[extra_storage.size() - 1].add_point(point);
            // }
            extra_storage.push_back(point);
        }
        else
        {
            insert_p(point);
        }
    }

    bool is_rebuild(Statistics statistics)
    {
        status = 11;
#ifdef use_gpu
        torch::Tensor x = torch::tensor(statistics.get_input(), at::kCUDA).reshape({1, 5});
        rebuild_model->to(torch::kCUDA);
#else
        torch::Tensor x = torch::tensor(statistics.get_input()).reshape({1, 5});
#endif
        bool is_rebuild = rebuild_model->predict(x).item().toFloat() >= 0.5;
        status = 12;
        return false;
    }

private:
    void init_build_processor()
    {
        status = Constants::STATUS_FRAMEWORK_INIT_BUILD_PROCESSOR;
        // string build_time_model_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/cost_model/build_time_model_zm.pt";
        // string query_time_model_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/cost_model/query_time_model_zm.pt";
        string build_time_model_path = Constants::BUILD_TIME_MODEL_PATH;
        string query_time_model_path = Constants::QUERY_TIME_MODEL_PATH;
        string raw_data_path = Constants::RAW_DATA_PATH;

        std::ifstream fin_build(build_time_model_path);
        std::ifstream fin_query(query_time_model_path);
        if (fin_build && fin_query)
        {
            torch::load(build_cost_model, build_time_model_path);
            torch::load(query_cost_model, query_time_model_path);
            print("init_build_processor-->load models finish");
            status = Constants::STATUS_FRAMEWORK_INIT_BUILD_PROCESSOR_LOAD_DONE;
        }
        else
        {
            string ppath = Constants::SYNTHETIC_DATA_PATH;
            struct dirent *ptr;
            DIR *dir;
            dir = opendir(ppath.c_str());
            vector<ScorerItem> records;
            long shortest_build_time = numeric_limits<long>::max();
            long shortest_query_time = numeric_limits<long>::max();
            while ((ptr = readdir(dir)) != NULL)
            {
                if (ptr->d_name[0] == '.')
                    continue;
                string path = ptr->d_name;
                int find_result = path.find(".csv");
                if (find_result > 0 && find_result <= path.length())
                {
                    print(ppath + path);
                    string prefix = path.substr(0, path.find(".csv"));

                    vector<string> sub_string;
                    boost::algorithm::split(sub_string, prefix, boost::is_any_of("_"));

                    DataSet<D, T> dataset;
                    dataset.dataset_name = ppath + path;

                    dataset.read_data();
                    dataset.mapping();
                    init_storage_p(dataset);

                    for (std::map<int, vector<float>>::iterator iter = methods.begin(); iter != methods.end(); ++iter)
                    {
                        int method = iter->first;
                        ScorerItem item(stof(sub_string[0]), stof(sub_string[1]), iter->second);
                        exp_recorder.timer_begin();
                        build_index_p(dataset, method);
                        exp_recorder.timer_end();
                        long build_time = exp_recorder.time;
                        Query<D> point_query;
                        point_query.set_point_query()->query_points = dataset.points;
                        // ->set_query_points(dataset.points);
                        exp_recorder.timer_begin();
                        point_query_p(point_query);
                        exp_recorder.timer_end();
                        long query_time = exp_recorder.time / dataset.points.size();
                        item.build_time = build_time;
                        item.query_time = query_time;
                        shortest_build_time = min(shortest_build_time, build_time);
                        shortest_query_time = min(shortest_query_time, query_time);
                        records.push_back(item);
                    }
                }
                // break;
            }
            vector<float> parameters;
            vector<float> build_time_labels;
            vector<float> query_time_labels;

            for (size_t i = 0; i < records.size(); i++)
            {
                records[i].cardinality = (float)records[i].cardinality / max_cardinality;
                records[i].build_time = (float)shortest_build_time / records[i].build_time;
                records[i].query_time = (float)shortest_query_time / records[i].query_time;
                parameters.push_back(records[i].cardinality);
                parameters.push_back(records[i].dist);
                parameters.insert(parameters.end(), records[i].method.begin(), records[i].method.end());
                build_time_labels.push_back(records[i].build_time);
                query_time_labels.push_back(records[i].query_time);
            }

            FileWriter writer;
            writer.write_score_items(records, raw_data_path);

            // save
#ifdef use_gpu
            query_cost_model->to(torch::kCUDA);
            build_cost_model->to(torch::kCUDA);
#endif
            build_cost_model->train_model(parameters, build_time_labels);
            query_cost_model->train_model(parameters, query_time_labels);
            torch::save(build_cost_model, build_time_model_path);
            torch::save(query_cost_model, query_time_model_path);
            // learn

            status = Constants::STATUS_FRAMEWORK_INIT_BUILD_PROCESSOR_TRAIN_DONE;
        }
    }

    void init_rebuild_processor()
    {
        status = Constants::STATUS_FRAMEWORK_INIT_REBUILD_PROCESSOR;
        string raw_data_path = Constants::REBUILD_RAW_DATA_PATH;
        string path = Constants::REBUILD_DATA_PATH;
        string rebuild_model_path = Constants::REBUILD_MODEL_PATH;
        std::ifstream fin_rebuild(rebuild_model_path);
        if (fin_rebuild)
        {
            torch::load(rebuild_model, rebuild_model_path);
            print("init_rebuild_processor-->load model finish");
            status = Constants::STATUS_FRAMEWORK_INIT_REBUILD_PROCESSOR_LOAD_DONE;
        }
        else
        {
            print("init rebuild models");
            string ppath = Constants::SYNTHETIC_DATA_PATH;
            struct dirent *ptr;
            DIR *dir;
            dir = opendir(ppath.c_str());
            vector<Statistics> records;
            while ((ptr = readdir(dir)) != NULL)
            {
                if (ptr->d_name[0] == '.')
                    continue;
                string path = ptr->d_name;
                int find_result = path.find(".csv");
                if (find_result > 0 && find_result <= path.length())
                {
                    string prefix = path.substr(0, path.find(".csv"));

                    vector<string> sub_string;
                    boost::algorithm::split(sub_string, prefix, boost::is_any_of("_"));

                    float dist = stof(sub_string[1]);
                    // only consider the maximal data sets
                    if (stol(sub_string[0]) != max_cardinality)
                    {
                        continue;
                    }
                    print(ppath + path);

                    DataSet<D, T> dataset;
                    dataset.dataset_name = ppath + path;
                    dataset.read_data();
                    dataset.mapping();
                    init_storage_p(dataset);

                    // TODO generate synthetic points
                    long base = max_cardinality / 100;
                    for (size_t i = 0; i < 9; i++)
                    {
                        Query<D> point_query;
                        point_query.set_point_query()->query_points = dataset.points;
                        // set_query_points(dataset.points);

                        DataSetInfo<T> original_info(Constants::DEFAULT_BIN_NUM, dataset.keys);
                        DataSetInfo<T> current_info(Constants::DEFAULT_BIN_NUM, dataset.keys);
                        // DataSet<D, T> inserted_dateset = generate_points_p(base, dist);
                        DataSet<D, T> inserted_dateset = generate_points_p(base, 0);
                        Statistics statistics;
                        statistics.cardinality = (float)dataset.points.size() / max_cardinality;
                        statistics.relative_depth = 1.0;

                        build_index_p(dataset, Constants::OG);
                        exp_recorder.timer_begin();
                        point_query_p(point_query);
                        exp_recorder.timer_end();

                        long before_insert_query_time = exp_recorder.time / dataset.points.size();

                        for (size_t i = 0; i < base; i++)
                        {
                            insert(inserted_dateset.points[i]);
                            statistics.insert();
                            current_info.update(inserted_dateset.keys[i]);
                        }

                        statistics.cdf_change = original_info.cal_dist(current_info.cdf);
                        statistics.relative_depth = 1.0;
                        statistics.update_ratio = (float)(statistics.inserted + statistics.deleted) / dataset.points.size();
                        statistics.distribution = current_info.get_distribution();

                        point_query.query_points = dataset.points;
                        // set_query_points(inserted_dateset.points);
                        exp_recorder.timer_begin();
                        point_query_p(point_query);
                        exp_recorder.timer_end();
                        long after_insert_query_time = exp_recorder.time / dataset.points.size();
                        cout << "(float)(statistics.inserted + statistics.deleted) : " << (float)(statistics.inserted + statistics.deleted) << endl;
                        cout << "dataset.points.size() : " << dataset.points.size() << endl;
                        cout << "before_insert_query_time: " << before_insert_query_time << endl;
                        cout << "after_insert_query_time: " << after_insert_query_time << endl;
                        statistics.is_rebuild = ((float)after_insert_query_time / before_insert_query_time) > 1.1;
                        cout << statistics.statistics_to_string() << endl;
                        records.push_back(statistics);
                        base *= 2;
                    }
                }
            }

            // TODO setup extra storage and query methods.!
            vector<float> parameters;
            vector<float> labels;

            for (size_t i = 0; i < records.size(); i++)
            {
                vector<float> temp = records[i].get_input();
                parameters.insert(parameters.end(), temp.begin(), temp.end());
                labels.push_back(records[i].is_rebuild ? 1 : 0);
            }

            FileWriter writer;
            writer.write_statistics_items(records, raw_data_path);

            // save
#ifdef use_gpu
            rebuild_model->to(torch::kCUDA);
#endif
            rebuild_model->train_model(parameters, labels);
            torch::save(rebuild_model, rebuild_model_path);
        }

        status = Constants::STATUS_FRAMEWORK_INIT_REBUILD_PROCESSOR_TRAIN_DONE;
    }
};

#endif
