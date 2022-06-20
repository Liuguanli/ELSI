#ifndef CL_H
#define CL_H

#include <vector>
#include <string.h>
#include <string>
#include "../entities/DataSet.h"
#include "../utils/Constants.h"

template <typename D, typename T>
class CL
{

public:
    DataSet<D, T> do_cl(DataSet<D, T> &dataset, int k, int dimension)
    {
        if (dimension == 1)
        {
            return do_cl(dataset, k);
        }
        if (dimension == 2)
        {
            return do_cl_2d(dataset, k);
        }
        return dataset;
    }

    DataSet<D, T> do_cl(DataSet<D, T> &dataset, int k)
    {
        // TODO 1 store as a temp dataset file, use python to get centroids
        dataset.save_temp_data();
        string out_file_name = "./data/generated.csv";
        string commandStr = "python " + Constants::CLUSTER_FILE + " -k " + to_string(k) + " -i " + dataset.path + " -o " + out_file_name;
        // cout << "commandStr: " << commandStr << endl;
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        dataset.remove_temp_data();
        DataSet<D, T> generated_dataset(out_file_name);
        generated_dataset.read_data()->mapping()->generate_normalized_keys()->generate_labels();
        return generated_dataset;
    }

    DataSet<D, T> do_cl_2d(DataSet<D, T> &dataset, int k)
    {
        cout << "do_cl:" << dataset.path << endl;
        dataset.save_temp_data();
        string out_file_name = "./data/generated.csv";
        string commandStr = "python " + Constants::CLUSTER_FILE + " -k " + to_string(k) + " -i " + dataset.path + " -o " + out_file_name;
        // cout << "commandStr: " << commandStr << endl;
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        dataset.remove_temp_data();
        DataSet<D, T> generated_dataset(out_file_name);
        // generated_dataset.read_data()->mapping()->generate_normalized_keys()->generate_labels();
        generated_dataset.read_data();
        generated_dataset.mapping();
        generated_dataset.generate_normalized_keys();
        generated_dataset.generate_labels();
        return generated_dataset;
    }
};

#endif