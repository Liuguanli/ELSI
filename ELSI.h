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

// #include "utils/MethodScorer.h"
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
    // string index_name = "";
    int dimension = 1;
    int status = -1;
    // std::shared_ptr<MLP> query_cost_model = std::make_shared<MLP>(8, 32);
    // std::shared_ptr<MLP> build_cost_model = std::make_shared<MLP>(8, 32);
    std::shared_ptr<MLP> query_cost_model;
    std::shared_ptr<MLP> build_cost_model;
    std::shared_ptr<MLP> rebuild_model = std::make_shared<MLP>(5, Constants::SCORER_WIDTH);

    void (*point_query_p)(Query<D> &);
    void (*build_index_p)(DataSet<D, T>, int);
    void (*init_storage_p)(DataSet<D, T>);
    void (*window_query_p)(Query<D> &);
    void (*knn_query_p)(Query<D> &);
    void (*insert_p)(D);
    DataSet<D, T> (*generate_points_p)(long, float);
    map<int, vector<float>> methods;
    ExpRecorder exp_recorder;

    long max_cardinality = cardinality_u;

    // vector<ExtraStorageBlock<D>> extra_storage;
    vector<LeafNode> extra_storage;

    void config_method_pool()
    {
        int pool_size = config::method_pool.size();
        assert(pool_size > 0);
        for (size_t i = 0; i < pool_size; i++)
        {
            vector<float> one_hot(pool_size);
            for (size_t j = 0; j < pool_size; j++)
            {
                one_hot[j] = i == j ? 1 : 0;
            }
            methods.insert(pair<int, vector<float>>(i, one_hot));
        }
        query_cost_model = std::make_shared<MLP>(2 + pool_size, Constants::SCORER_WIDTH);
        build_cost_model = std::make_shared<MLP>(2 + pool_size, Constants::SCORER_WIDTH);
    }

    void init()
    {
        // assert((DataSet<Point, long long>::read_data_pointer != NULL));
        // assert((DataSet<Point, long long>::gen_input_keys_pointer != NULL));
        // assert((DataSet<Point, long long>::save_data_pointer != NULL));
        int pool_size = config::method_pool.size();
        assert(pool_size > 0);
        print("ELSI::init ELSI");
        init_build_processor();
        init_rebuild_processor();
        MR<D, T>::load_pre_trained_model(dimension);
    }

    int build_predict_method(float lambda, float query_frequency, DataSet<D, T> &original_data_set)
    {

        assert(original_data_set.keys.size() > 0);
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
        // return max_index;
        return Constants::SP;
    }

    std::shared_ptr<MLP> get_build_method(DataSet<D, T> &original_data_set, int method_index)
    {
        DataSet<D, T> shrinked_data_set;
        switch (config::method_pool[method_index])
        {
        case Constants::CL:
            CL<D, T> cl;
            shrinked_data_set = cl.do_cl(original_data_set, config::cluster_k, dimension);
            break;
        case Constants::MR:
            MR<D, T> mr;
            {
                DataSetInfo<T> info(Constants::DEFAULT_BIN_NUM, original_data_set.keys);

                if (dimension == 1)
                {
                    string model_path = Constants::DEFAULT_PRE_TRAIN_MODEL_PATH;
                    if (mr.is_reusable(info, model_path))
                    {
                        // TODO here the input is 1,  net is for model reuse
                        auto net = std::make_shared<MLP>(dimension);
#ifdef use_gpu
                        net->to(torch::kCUDA);
#endif
                        torch::load(net, model_path);
                        if (dimension == 1)
                        {
                            net->get_parameters_ZM();
                        }
                        return net;
                    }
                }
                if (dimension == 2)
                {
                    string model_path = Constants::DEFAULT_PRE_TRAIN_MODEL_PATH_RSMI;
                    if (original_data_set.points.size() > Constants::THRESHOLD)
                    {
                        if (mr.is_reusable_2d(info, model_path))
                        {
                            // TODO here the input is 1,  net is for model reuse
                            auto net = std::make_shared<MLP>(dimension);
#ifdef use_gpu
                            net->to(torch::kCUDA);
#endif
                            // cout << "model_path: " << model_path << endl;
                            torch::load(net, model_path);
                            net->get_parameters();
                            return net;
                        }
                    }
                    else
                    {
                        model_path = Constants::DEFAULT_PRE_TRAIN_MODEL_PATH_RSMI_H;
                        if (mr.is_reusable_2d_leafnode(info, model_path))
                        {
                            // TODO here the input is 1,  net is for model reuse
                            auto net = std::make_shared<MLP>(dimension);
#ifdef use_gpu
                            net->to(torch::kCUDA);
#endif
                            // cout << "model_path: " << model_path << endl;
                            torch::load(net, model_path);
                            net->get_parameters();
                            return net;
                        }
                    }
                }
            }
            shrinked_data_set = original_data_set;
            break;
        case Constants::OG:
            shrinked_data_set = original_data_set;
            break;
        case Constants::RL:
            RL<D, T> rl;
            shrinked_data_set = rl.do_rl(original_data_set, config::bit_num, dimension);
            break;
        case Constants::RS:
            RS<T> rs;
            shrinked_data_set = rs.do_rs(original_data_set, config::rs_m, dimension);
            break;
        case Constants::SP:
            SP<D, T> sp;
            shrinked_data_set = sp.do_sp(original_data_set, config::sampling_rate, dimension);
            break;
        default:
            shrinked_data_set = original_data_set;
            break;
        }
        auto mlp = model_training::real_train(shrinked_data_set.normalized_keys, shrinked_data_set.labels);
        return mlp;
    }

    void query(Query<D> &query)
    {
        if (query.is_point())
        {
            point_query(query);
        }
        if (query.is_window())
        {
            window_query(query);
        }
        if (query.is_knn())
        {
            knn_query(query);
        }
    }

    void point_query(Query<D> &query)
    {
        if (point_query_p != NULL)
        {
            point_query_p(query);
        }
    }

    bool point_query(Point &query_point)
    {
        long front = 0, back = extra_storage.size() - 1;
        while (front <= back)
        {
            int mid = (front + back) / 2;
            double first_key = extra_storage[mid].children[0].key;
            double last_key = extra_storage[mid].children[extra_storage[mid].children.size() - 1].key;
            if (first_key <= query_point.key && query_point.key <= last_key)
            {
                vector<Point>::iterator iter = find(extra_storage[mid].children.begin(), extra_storage[mid].children.end(), query_point);
                return iter != extra_storage[mid].children.end();
            }
            else
            {
                if (extra_storage[mid].children[0].key < query_point.key)
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

    void window_query(vector<Point> &results, Mbr &query_window)
    {
        for (size_t i = 0; i < extra_storage.size(); i++)
        {
            if (extra_storage[i].mbr.interact(query_window))
            {
                for (Point point : extra_storage[i].children)
                {
                    if (query_window.contains(point))
                    {
                        results.push_back(point);
                    }
                }
            }
        }
    }

    void window_query(Query<D> &query)
    {
        if (window_query_p != NULL)
        {
            window_query_p(query);
        }
    }

    void kNN_query(vector<Point> &results, Point query_point, int kk)
    {
        priority_queue<Point, vector<Point>, sortForKNN2> pq;
        for (size_t i = 0; i < extra_storage.size(); i++)
        {
            for (Point point : extra_storage[i].children)
            {
                point.cal_dist(query_point);
                if (pq.size() >= kk)
                {
                    if (pq.top().temp_dist > point.temp_dist)
                    {
                        pq.pop();
                        pq.push(point);
                    }
                }
                else
                {
                    pq.push(point);
                }
            }
        }
        while (!pq.empty())
        {
            results.push_back(pq.top());
            pq.pop();
        }
    }

    void knn_query(Query<D> &query)
    {
        if (knn_query_p != NULL)
        {
            knn_query_p(query);
        }
    }

    void insert(D point)
    {
        if (insert_p == NULL)
        {
            long front = 0, back = extra_storage.size() - 1;
            while (front <= back)
            {
                int mid = (front + back) / 2;
                double first_key = extra_storage[mid].children[0].key;
                double last_key = extra_storage[mid].children[extra_storage[mid].children.size() - 1].key;

                if (first_key <= point.key && point.key <= last_key)
                {
                    if (extra_storage[mid].is_full())
                    {
                        extra_storage[mid].add_point(point);
                        sort(extra_storage[mid].children.begin(), extra_storage[mid].children.end(), sort_key());
                        LeafNode right = extra_storage[mid].split();
                        extra_storage.insert(extra_storage.begin() + mid + 1, right);
                    }
                    else
                    {
                        extra_storage[mid].add_point(point);
                        sort(extra_storage[mid].children.begin(), extra_storage[mid].children.end(), sort_key());
                    }
                    return;
                }
                else
                {
                    if (first_key < point.key)
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
        else
        {
            insert_p(point);
        }
    }

    bool delete_(D &query_point)
    {
        long front = 0, back = extra_storage.size() - 1;
        while (front <= back)
        {
            int mid = (front + back) / 2;
            double first_key = extra_storage[mid].children[0].key;
            double last_key = extra_storage[mid].children[extra_storage[mid].children.size() - 1].key;
            if (first_key <= query_point.key && query_point.key <= last_key)
            {
                for (size_t i = 0; i < extra_storage[mid].children.size(); i++)
                {
                    if (extra_storage[mid].children[i] == query_point)
                    {
                        extra_storage[mid].children[i].is_deleted = true;
                        return true;
                    }
                }
                break;
            }
            else
            {
                if (extra_storage[mid].children[0].key < query_point.key)
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

    bool is_rebuild(Statistics statistics)
    {
#ifdef use_gpu
        torch::Tensor x = torch::tensor(statistics.get_input(), at::kCUDA).reshape({1, 5});
        rebuild_model->to(torch::kCUDA);
#else
        torch::Tensor x = torch::tensor(statistics.get_input()).reshape({1, 5});
#endif
        bool is_rebuild = rebuild_model->predict(x).item().toFloat() >= 0.5;
        return is_rebuild;
    }

private:
    void init_build_processor()
    {
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
            print("LOAD BUILD MODULE---------------DONE");
        }
        else
        {
            print("TRAIN BUILD MODULE---------------START");

            string ppath = Constants::SYNTHETIC_DATA_PATH;
            struct dirent *ptr;
            DIR *dir;
            dir = opendir(ppath.c_str());
            vector<ScorerItem> records;
            long og_build_time = numeric_limits<long>::max();
            long og_query_time = numeric_limits<long>::max();
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

                    dataset.read_data()->mapping()->generate_normalized_keys()->generate_labels();
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
                        if (method == Constants::OG)
                        {
                            og_build_time = build_time;
                            og_query_time = query_time;
                        }

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
                records[i].build_time = (float)og_build_time / records[i].build_time;
                records[i].query_time = (float)og_query_time / records[i].query_time;
                parameters.push_back(records[i].cardinality);
                parameters.push_back(records[i].dist);
                parameters.insert(parameters.end(), records[i].method.begin(), records[i].method.end());
                build_time_labels.push_back(records[i].build_time);
                query_time_labels.push_back(records[i].query_time);
            }

            FileWriter writer;
            writer.write_score_items(records, raw_data_path);

#ifdef use_gpu
            query_cost_model->to(torch::kCUDA);
            build_cost_model->to(torch::kCUDA);
#endif

            // build_cost_model->train_model(parameters, build_time_labels);
            // query_cost_model->train_model(parameters, query_time_labels);

            // torch::save(build_cost_model, build_time_model_path);
            // torch::save(query_cost_model, query_time_model_path);

            int k = 5;
            int parameter_gap = parameters.size() / k;
            int label_gap = build_time_labels.size() / k;
            float loss = numeric_limits<float>::max();

            vector<vector<float>> all_parameters;
            vector<vector<float>> all_build_time_labels;
            vector<vector<float>> all_query_time_labels;

            for (size_t i = 0; i < k; i++)
            {
                vector<float> parameters_temp(parameters.begin() + parameter_gap * i, parameters.end() + parameter_gap * i + parameter_gap);
                vector<float> build_time_labels_temp(build_time_labels.begin() + label_gap * i, build_time_labels.end() + label_gap * i + label_gap);
                vector<float> query_time_labels_temp(query_time_labels.begin() + label_gap * i, query_time_labels.end() + label_gap * i + label_gap);
                all_parameters.push_back(parameters_temp);
                all_build_time_labels.push_back(build_time_labels_temp);
                all_query_time_labels.push_back(query_time_labels_temp);
            }

            for (size_t i = 0; i < k; i++)
            {

                vector<float> train_parameters;
                vector<float> test_parameters;

                vector<float> train_build_time_labels;
                vector<float> test_build_time_labels;

                for (size_t j = 0; j < k; j++)
                {
                    if (i == j)
                    {
                        test_parameters = all_parameters[i];
                        test_build_time_labels = all_build_time_labels[i];
                    }
                    else
                    {
                        train_parameters.insert(train_parameters.end(), all_parameters[i].begin(), all_parameters[i].end());
                        train_build_time_labels.insert(train_build_time_labels.end(), all_build_time_labels[i].begin(), all_build_time_labels[i].end());
                    }
                }
                std::shared_ptr<MLP> build_cost_model_temp = std::make_shared<MLP>(8, 32);
                build_cost_model_temp->train_model(train_parameters, train_build_time_labels);

                float temp_loss = 0;
                for (size_t j = 0; j < label_gap; j++)
                {
                    vector<float> temp(test_parameters.begin() + 8 * j, test_parameters.begin() + 8 * j + 8);
#ifdef use_gpu
                    torch::Tensor x = torch::tensor(temp, at::kCUDA).reshape({1, 8});
                    build_cost_model_temp->to(torch::kCUDA);
#else
                    torch::Tensor x = torch::tensor(temp).reshape({1, 8});
#endif
                    float res = build_cost_model_temp->predict(x).item().toFloat();
                    temp_loss += abs(res - test_build_time_labels[j]);
                }
                if (temp_loss < loss)
                {
                    loss = temp_loss;
                    build_cost_model = build_cost_model_temp;
                }
            }
            torch::save(build_cost_model, build_time_model_path);

            loss = 0;
            for (size_t i = 0; i < k; i++)
            {

                vector<float> train_parameters;
                vector<float> test_parameters;

                vector<float> train_query_time_labels;
                vector<float> test_query_time_labels;

                for (size_t j = 0; j < k; j++)
                {
                    if (i == j)
                    {
                        test_parameters = all_parameters[i];
                        test_query_time_labels = all_query_time_labels[i];
                    }
                    else
                    {
                        train_parameters.insert(train_parameters.end(), all_parameters[i].begin(), all_parameters[i].end());
                        train_query_time_labels.insert(train_query_time_labels.end(), all_query_time_labels[i].begin(), all_query_time_labels[i].end());
                    }
                }
                std::shared_ptr<MLP> query_cost_model_temp = std::make_shared<MLP>(8, 32);
                query_cost_model_temp->train_model(train_parameters, train_query_time_labels);

                float temp_loss = 0;
                for (size_t j = 0; j < label_gap; j++)
                {
                    vector<float> temp(test_parameters.begin() + 8 * j, test_parameters.begin() + 8 * j + 8);
#ifdef use_gpu
                    torch::Tensor x = torch::tensor(temp, at::kCUDA).reshape({1, 8});
                    query_cost_model_temp->to(torch::kCUDA);
#else
                    torch::Tensor x = torch::tensor(temp).reshape({1, 8});
#endif
                    float res = query_cost_model_temp->predict(x).item().toFloat();
                    temp_loss += abs(res - test_query_time_labels[j]);
                }
                if (temp_loss < loss)
                {
                    loss = temp_loss;
                    query_cost_model = query_cost_model_temp;
                }
            }
            torch::save(query_cost_model, query_time_model_path);
            print("TRAIN BUILD MODULE---------------DONE");
        }
    }

    void init_rebuild_processor()
    {
        string raw_data_path = Constants::REBUILD_RAW_DATA_PATH;
        string path = Constants::REBUILD_DATA_PATH;
        string rebuild_model_path = Constants::REBUILD_MODEL_PATH;
        std::ifstream fin_rebuild(rebuild_model_path);
        if (fin_rebuild)
        {
            torch::load(rebuild_model, rebuild_model_path);
            print("LOAD REBUILD MODULE---------------DONE");
        }
        else
        {
            print("TRAIN REBUILD MODEL---------------START");
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
                    dataset.read_data()->mapping()->generate_normalized_keys()->generate_labels();
                    init_storage_p(dataset);

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
                        // cout << "(float)(statistics.inserted + statistics.deleted) : " << (float)(statistics.inserted + statistics.deleted) << endl;
                        // cout << "dataset.points.size() : " << dataset.points.size() << endl;
                        // cout << "before_insert_query_time: " << before_insert_query_time << endl;
                        // cout << "after_insert_query_time: " << after_insert_query_time << endl;
                        statistics.is_rebuild = ((float)after_insert_query_time / before_insert_query_time) > 1.1;
                        cout << statistics.statistics_to_string() << endl;
                        records.push_back(statistics);
                        base *= 2;
                    }
                }
            }

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
            print("TRAIN REBUILD MODEL---------------DONE");
        }
    }
};

#endif
