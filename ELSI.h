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

#include "utils/Rebuild.h"
#include "utils/MethodScorer.h"
#include "utils/Log.h"
#include "utils/Config.h"
#include "utils/Constants.h"

#include "entities/Statistics.h"
using namespace logger;
using namespace config;
template <typename D, typename T>
class ELSI
{
public:
    std::shared_ptr<MLP> query_cost_model = std::make_shared<MLP>(10, 32);
    std::shared_ptr<MLP> build_cost_model = std::make_shared<MLP>(10, 32);
    std::shared_ptr<MLP> rebuild_model = std::make_shared<MLP>(7, 32);

    void (*point_query_p)(Query<D>);
    vector<D> (*window_query_p)(Query<D>);
    vector<D> (*knn_query_p)(Query<D>);

    void init()
    {
        status = 0;
        print("ELSI::init ELSI");
        init_build();
        init_rebuild();
        MR<D, T>::load_pre_trained_model();
        status = 2;
    }

    std::shared_ptr<MLP> build(float lambda, vector<D> points)
    {
        DataSet<D, T> original_data_set(points);

        status = 7;
        print("ELSI::build step 1: get CDF");

        DataSetInfo<T> info(Constants::DEFAULT_BIN_NUM, original_data_set.keys);
        print("ELSI::build step 2: run for all methods");

        float *build_score = new float[6];
        float *query_score = new float[6];
        float max = numeric_limits<float>::min();
        int max_index = 0;
        vector<float> distribution_list = info.get_distribution();
        vector<float> parameters;
        parameters.push_back((float)original_data_set.keys.size() / Constants::MAX_CARDINALITY);
        parameters.insert(parameters.end(), distribution_list.begin(), distribution_list.end());
        map<int, vector<float>> methods = {
            {0, {1, 0, 0, 0, 0, 0}}, {1, {0, 1, 0, 0, 0, 0}}, {2, {0, 0, 1, 0, 0, 0}}, {3, {0, 0, 0, 1, 0, 0}}, {4, {0, 0, 0, 0, 1, 0}}, {5, {0, 0, 0, 0, 0, 1}}};
        for (size_t i = 0; i < 6; i++)
        {
            vector<float> temp = parameters;
            temp.insert(temp.end(), methods[i].begin(), methods[i].end());
#ifdef use_gpu
            torch::Tensor x = torch::tensor(temp, at::kCUDA).reshape({1, 10});
            build_cost_model->to(torch::kCUDA);
            query_cost_model->to(torch::kCUDA);
#else
            torch::Tensor x = torch::tensor(temp).reshape({1, 10});
#endif
            build_score[i] = build_cost_model->predict(x).item().toFloat();
            query_score[i] = query_cost_model->predict(x).item().toFloat();

            float score = build_score[i] * lambda + query_score[i] * (1 - lambda);
            if (score > max)
            {
                max = score;
                max_index = i;
            }
        }
        print("ELSI::build step 3: choose one method:" + to_string(max_index));
        DataSet<D, T> shrinked_data_set;
        max_index = Constants::RS;

        switch (max_index)
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
                    cout << "model_path" << model_path << endl;
                    torch::load(net, model_path);
                    return net;
                }
            }
            shrinked_data_set = original_data_set;
            break;
        case Constants::OG:
            shrinked_data_set = original_data_set;
            break;
        // case Constants::RL:
        //     RL<D, T> rl;
        //     shrinked_data_set = rl.do_rl(original_data_set, info, config::bit_num);
        //     break;
        case Constants::RS:
            RS<T> rs;
            shrinked_data_set = rs.do_rs(original_data_set, config::rs_m);
            break;
        case Constants::SP:
            SP<D, T> sp;
            shrinked_data_set = sp.do_sp(original_data_set, config::sampling_rate);
            break;
        default:
            break;
        }
        auto mlp = model_training::real_train_1d(shrinked_data_set.normalized_keys, shrinked_data_set.labels);
        print("ELSI::build step 4: get a model");

        status = 8;
        return mlp;
    }

    void query(Query<D> query)
    {
        status = 9;
        print("ELSI::query");

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
        status = 10;
    }

    void point_query(Query<D> query)
    {
        if (point_query_p == NULL)
        {
            int count = query.get_query_points().size();
            for (size_t i = 0; i < count; i++)
            {
            }
        }
        else
        {
            point_query_p(query);
        }
    }

    void window_query(Query<D> query)
    {
        if (window_query_p == NULL)
        {
            int count = query.get_query_windows().size();
            for (size_t i = 0; i < count; i++)
            {
            }
        }
        else
        {
            window_query_p(query);
        }
    }

    void knn_query(Query<D> query)
    {
        if (knn_query_p == NULL)
        {
            int count = query.get_query_points().size();
            for (size_t i = 0; i < count; i++)
            {
            }
        }
        else
        {
            knn_query_p(query);
        }
    }

    bool is_rebuild()
    {
        status = 11;

        status = 12;
        return false;
    }

private:
    vector<D> extra_storage;
    int status = -1;
    void init_build()
    {
        status = 3;
        print("   init build models");

        string build_time_model_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/cost_model/build_time_model_zm.pt";
        string query_time_model_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/cost_model/query_time_model_zm.pt";

        std::ifstream fin_build(build_time_model_path);
        std::ifstream fin_query(query_time_model_path);
        if (fin_build && fin_query)
        {
            torch::load(build_cost_model, build_time_model_path);
            torch::load(query_cost_model, query_time_model_path);
            print("    init_build-->load models finish");
        }
        else
        {
            method_scorer::generate_data_set();
            method_scorer::build_simple_models_and_query();
            method_scorer::generate_training_set();
            method_scorer::learn_build_time_prediction_model();
            method_scorer::learn_query_time_prediction_model();
        }

        status = 4;
    }

    void init_rebuild()
    {
        status = 5;

        print("   init rebuild models");
        string path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/rebuild_model/train_set_formatted.csv";
        string rebuild_model_path = "/home/liuguanli/Dropbox/shared/VLDB20/codes/rsmi/rebuild_model/rebuild_model.pt";
        std::ifstream fin_rebuild(rebuild_model_path);
        if (fin_rebuild)
        {
            torch::load(rebuild_model, rebuild_model_path);
            print("    init_rebuild-->load model finish");
        }
        else
        {
            rebuild::generate_updates_data_set();
            rebuild::build_simple_models_and_updates();
            rebuild::generate_training_set();
            rebuild::learn_rebuild_model();
        }

        status = 6;
    }
};

#endif
