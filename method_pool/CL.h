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
    DataSet<D, T> do_cl(DataSet<D, T> &dataset, int k)
    {
        // TODO 1 store as a temp dataset file, use python to get centroids
        cout << "do_cl:" << dataset.path << endl;
        dataset.save_temp_data();
        string out_file_name = "./data/generated.csv";
        string commandStr = "python " + Constants::CLUSTER_FILE + " -k " + to_string(k) + " -i " + dataset.path + " -o " + out_file_name;
        cout << "commandStr: " << commandStr << endl;
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        dataset.remove_temp_data();

        DataSet<D, T> generated_dataset(out_file_name);
        generated_dataset.read_data();
        generated_dataset.mapping();
        return generated_dataset;
        // TODO needs a function pointer to get the DataSet
        // e.g., dataset = function pointer (out_file_name);
    }
};

#endif