#ifndef METHOD_SCORER_H
#define METHOD_SCORER_H

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

#include "../method_pool/method.h"
#include "../entities/Point.h"

namespace method_scorer
{

    // void generate_data_set()
    // {
    //     float dist = 0.1;
    //     string input_path = "./data";
    //     cout << "   method_scorer::generate_data_set-->invoke python code to generate data set" << endl;
    // }

    void build_simple_models_and_query()
    {
        string path = "/home/research/datasets/BASE/synthetic/";
        std::ifstream fin(path);
        assert(fin);

        int file_num = 100;
        for (size_t i = 0; i < file_num; i++)
        {
            // TODO read data, 
        }
        

        cout << "   method_scorer::build_simple_models_and_query-->build models with different methods and query" << endl;

        // assert()
    }

    void generate_training_set()
    {
        std::ifstream fin("/home/research/datasets/BASE/synthetic/");
        assert(fin);
    }

    void learn_build_time_prediction_model()
    {
        cout << "   method_scorer::learn_build_time_prediction_model" << endl;
    }

    void learn_query_time_prediction_model()
    {
        cout << "   method_scorer::learn_query_time_prediction_model" << endl;
    }
}

#endif