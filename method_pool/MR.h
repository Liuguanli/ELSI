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
    inline static map<string, DataSetInfo<T>> pre_trained_dataset_info_leafnode; // For RSMI H

    bool is_reusable(DataSetInfo<T> target, string &model_path)
    {
        double min_dist = 1.0;
        typename std::map<string, DataSetInfo<T>>::iterator iter; // need typename iq
        iter = pre_trained_dataset_info.begin();

        while (iter != pre_trained_dataset_info.end())
        {
            double temp_dist = target.cal_dist(iter->second.cdf);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_ZM + iter->first + ".pt";
            }
            iter++;
        }
        return true;
    }

    bool is_reusable_2d(DataSetInfo<T> target, string &model_path)
    {
        double min_dist = 1.0;
        typename std::map<string, DataSetInfo<T>>::iterator iter; // need typename iq
        iter = pre_trained_dataset_info.begin();

        while (iter != pre_trained_dataset_info.end())
        {
            double temp_dist = target.cal_dist(iter->second.cdf);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_RSMI + "Z/" + iter->first + ".pt";
            }
            iter++;
        }
        return true;
    }

    bool is_reusable_2d_leafnode(DataSetInfo<T> target, string &model_path)
    {
        double min_dist = 1.0;
        typename std::map<string, DataSetInfo<T>>::iterator iter; // need typename iq
        iter = pre_trained_dataset_info_leafnode.begin();

        while (iter != pre_trained_dataset_info_leafnode.end())
        {
            double temp_dist = target.cal_dist(iter->second.cdf);
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                model_path = Constants::PRE_TRAIN_MODEL_PATH_RSMI + "H/" + iter->first + ".pt";
            }
            iter++;
        }
        return true;
    }

    inline static void load_pre_trained_model(int dimension)
    {
        print("MR::load_pre_trained_model-->load " + to_string(dimension));

        if (pre_trained_dataset_info.size() > 0)
        {
            if (dimension == 1)
            {
                return;
            }
            if (dimension == 2)
            {
                if (pre_trained_dataset_info_leafnode.size() > 0)
                {
                    return;
                }
            }
        }

        string ppath = "";
        string ppath1 = "";
        string f_path = "";
        if (dimension == 1)
        {
            ppath = Constants::PRE_TRAIN_MODEL_PATH_ZM;
            f_path = Constants::PRE_TRAIN_1D_DATA;
        }
        if (dimension == 2)
        {
            ppath = Constants::PRE_TRAIN_MODEL_PATH_RSMI + "Z/";
            f_path = Constants::FEATURES_PATH_RSMI + "Z/";
        }
        struct dirent *ptr;
        DIR *dir;
        dir = opendir(ppath.c_str());
        while ((ptr = readdir(dir)) != NULL)
        {
            if (ptr->d_name[0] == '.')
                continue;
            string file_name_s = ptr->d_name;
            int find_result = file_name_s.find(".pt");
            if (find_result > 0 && find_result <= file_name_s.length())
            {
                string prefix = file_name_s.substr(0, file_name_s.find(".pt"));
                string feature_path = f_path + prefix + ".csv";
                FileReader reader;
                // string path = Constants::PRE_TRAIN_1D_DATA;
                vector<float> cdf = get_cdf(f_path, prefix + ".csv");
                DataSetInfo<T> info(cdf);
                pre_trained_dataset_info.insert(pair<string, DataSetInfo<T>>(prefix, info));
            }
        }

        // TODO loading distance!!! 1

        if (dimension == 2)
        {
            ppath = Constants::PRE_TRAIN_MODEL_PATH_RSMI + "H/";
            f_path = Constants::FEATURES_PATH_RSMI + "H/";
            struct dirent *ptr_2;
            DIR *dir_2;
            dir_2 = opendir(ppath.c_str());
            while ((ptr_2 = readdir(dir_2)) != NULL)
            {
                if (ptr_2->d_name[0] == '.')
                    continue;
                string file_name_s = ptr_2->d_name;
                int find_result = file_name_s.find(".pt");
                if (find_result > 0 && find_result <= file_name_s.length())
                {
                    string prefix = file_name_s.substr(0, file_name_s.find(".pt"));
                    string feature_path = f_path + prefix + ".csv";
                    FileReader reader;
                    // string path = Constants::PRE_TRAIN_1D_DATA;
                    vector<float> cdf = get_cdf(f_path, prefix + ".csv");
                    DataSetInfo<T> info(cdf);
                    pre_trained_dataset_info_leafnode.insert(pair<string, DataSetInfo<T>>(prefix, info));
                }
            }
        }

        cout << "load finish..." << pre_trained_dataset_info.size() << endl;
        cout << "load finish..." << pre_trained_dataset_info_leafnode.size() << endl;
    }

    inline static vector<float> get_cdf(string folder, string file_name)
    {
        FileReader filereader(folder + file_name, ",");
        vector<float> cdf = filereader.read_features();
        return cdf;
    }
};

#endif