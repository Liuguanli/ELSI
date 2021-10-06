#ifndef MR_H
#define MR_H

#include "method.h"
#include <vector>
#include <map>
#include <iterator>
#include <string.h>
#include <string>
#include "../entities/DataSet.h"
// #include "../entities/DataSetInfo.h"
#include "../utils/Log.h"

using namespace logger;
template <typename D, typename T>
class MR
{

public:
    inline static map<string, DataSetInfo<T>> pre_trained_dataset_info;

    bool is_reusable(DataSetInfo<T> target, string &model_path)
    {
        double min_dist = 1.0;
        typename std::map<string, DataSetInfo<T>>::iterator iter; // need typename iq
        iter = pre_trained_dataset_info.begin();

        while (iter != pre_trained_dataset_info.end())
        {
            double temp_dist = target.cal_similarity(iter->second.cdf);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + iter->first + ".pt";
            }
            iter++;
        }
        return true;
    }

    inline static void load_pre_trained_model()
    {
        print("    MR::load_pre_trained_model-->load");
        if (pre_trained_dataset_info.size() > 0)
        {
            return;
        }

        string ppath = Constants::PRE_TRAIN_MODEL_PATH_ZM;
        cout << "load_pre_trained_model_zm: ppath:" << ppath << endl;

        // struct dirent *ptr;
        // DIR *dir;
        // dir = opendir(ppath.c_str());
        // while ((ptr = readdir(dir)) != NULL)
        // {
        //     if (ptr->d_name[0] == '.')
        //         continue;
        //     string file_name_s = ptr->d_name;
        //     int find_result = file_name_s.find(".pt");
        //     if (find_result > 0 && find_result <= file_name_s.length())
        //     {
        //         string prefix = file_name_s.substr(0, file_name_s.find(".pt"));
        //         string feature_path = Constants::FEATURES_PATH_ZM + prefix + ".csv";
        //         FileReader reader;
        //         string path = Constants::PRE_TRAIN_1D_DATA;
        //         vector<D> keys = get_keys(path, prefix + ".csv");
        //         DataSetInfo<D> info(Constants::DEFAULT_BIN_NUM, keys);
        //         pre_trained_dataset_info.insert(pair<string, DataSetInfo<D>>(prefix, info));
        //     }
        // }
        cout << "load finish..." << pre_trained_dataset_info.size() << endl;
    }

    inline static vector<T> get_keys(string folder, string file_name)
    {
        FileReader filereader(folder + file_name, ",");
        vector<T> features = filereader.read_features();
        return features;
    }
};

#endif