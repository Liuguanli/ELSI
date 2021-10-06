#ifndef RL_H
#define RL_H

#include "method.h"
#include <vector>
#include <string.h>
#include <string>
#include "../entities/DataSet.h"
#include "../utils/Constants.h"

template <typename D, typename T>
class RL
{

public:
    DataSet<D, T> do_rl(DataSet<D, T> &dataset, DataSetInfo<T> &info, int bit_num)
    {
        cout << "do_cl:" << dataset.path << endl;
        dataset.save_temp_data();
        string input_file = dataset.path;
        string output_file = "./data/generated.csv";
        string commandStr = "python " + Constants::RL_FILE + " -b " + to_string(bit_num) +
                            " -i " + input_file + " -o " + output_file;
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        dataset.remove_temp_data();
        DataSet<D, T> generated_dataset(output_file);

        generated_dataset.read_data();
        generated_dataset.mapping();
        return generated_dataset;
    }
    // TODO needs a function pointer to get the DataSet
    // e.g., data_set = function pointer (out_file_name);
};

#endif