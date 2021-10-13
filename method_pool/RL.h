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
    DataSet<D, T> do_rl(DataSet<D, T> &dataset, int bit_num)
    {
        cout << "do_rl:" << dataset.path << endl;
        dataset.save_temp_data();
        string input_file = "./data/original_cdf.csv";

        DataSetInfo<T> info(dataset.keys.size() / 10, dataset.keys);

        // TODO optimize write file
        ofstream write;
        write.open(input_file, ios::out);
        for (size_t i = 0; i < info.cdf.size(); i++)
        {
            write << to_string(info.hist[i]) + "," + to_string(info.cdf[i]) + "\n";
        }
        write.close();
        cout << "write finish:" << endl;

        string output_file = "./data/generated.csv";
        string commandStr = "python " + Constants::RL_FILE + " -b " + to_string(bit_num * bit_num) +
                            " -i " + input_file + " -o " + output_file;
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        dataset.remove_temp_data();
        DataSet<D, T> generated_dataset(output_file);

        // TODO optimize remove file

        char s[input_file.length() + 1];
        strcpy(s, input_file.c_str());
        remove(s);

        generated_dataset.read_cdf(output_file);
        return generated_dataset;
    }
    // TODO needs a function pointer to get the DataSet
    // e.g., data_set = function pointer (out_file_name);
};

#endif