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
    DataSet<D, T> do_rl(DataSet<D, T> &dataset, int bit_num, int dimension)
    {
        if (dimension == 1)
        {
            return do_rl(dataset, bit_num);
        }
        if (dimension == 2)
        {
            return do_rl_2d(dataset, bit_num);
        }
        return dataset;
    }

    DataSet<D, T> do_rl(DataSet<D, T> &dataset, int bit_num)
    {
        // cout << "do_rl:" << dataset.path << endl;
        // dataset.save_temp_data();
        string input_file = "./data/original_cdf.csv";

        // cout << dataset.keys[0] << " " << dataset.keys[dataset.keys.size() - 1] << endl;
        // cout << "dataset.keys.size():" << dataset.keys.size() << endl;
        DataSetInfo<T> info(dataset.keys.size() / 10, dataset.keys);
        // cout << "info.cdf.size():" << info.cdf.size() << endl;

        // DataSetInfo<T> info(dataset.keys.size() / 10, dataset.normalized_keys);

        ofstream write;
        write.open(input_file, ios::out);
        for (size_t i = 0; i < info.cdf.size(); i++)
        {
            write << to_string(info.hist[i]) + "," + to_string(info.cdf[i]) + "\n";
        }
        write.close();
        // cout << "write finish:" << endl;

        string output_file = "./data/generated.csv";
        string commandStr = "python " + Constants::RL_FILE + " -b " + to_string(bit_num * bit_num) +
                            " -i " + input_file + " -o " + output_file;
        char command[1024];
        strcpy(command, commandStr.c_str());

        // cout << "command:" << command << endl;


        int res = system(command);
        // dataset.remove_temp_data();
        DataSet<D, T> generated_dataset(output_file);

        char s[input_file.length() + 1];
        strcpy(s, input_file.c_str());
        remove(s);

        generated_dataset.read_cdf(output_file);
        return generated_dataset;
    }

    DataSet<D, T> do_rl_2d(DataSet<D, T> &dataset, int bit_num)
    {
        // cout << "do_rl:" << dataset.path << endl;
        // dataset.save_temp_data();
        string input_file = "./data/original_cdf.csv";

        DataSetInfo<T> info(dataset.keys.size() / 10, dataset.keys);
        // cout << "info.cdf.size():" << info.cdf.size() << endl;

        ofstream write;
        write.open(input_file, ios::out);
        for (size_t i = 0; i < info.cdf.size(); i++)
        {
            write << to_string(info.hist[i]) + "," + to_string(info.cdf[i]) + "\n";
        }
        write.close();
        // cout << "write finish:" << endl;

        string output_file = "./data/generated.csv";

        string commandStr = "python " + Constants::RL_FILE_RSMI + " -b " + to_string(bit_num * bit_num) +
                            " -i " + input_file + " -o " + output_file;
        char command[1024];
        strcpy(command, commandStr.c_str());
        int res = system(command);
        // dataset.remove_temp_data();
        DataSet<D, T> generated_dataset(output_file);

        char s[input_file.length() + 1];
        strcpy(s, input_file.c_str());
        remove(s);

        generated_dataset.read_cdf_2d(output_file);
        return generated_dataset;
    }
};

#endif