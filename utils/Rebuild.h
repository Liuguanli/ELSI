#ifndef REBUILD_H
#define REBUILD_H

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

namespace rebuild
{
    int status = -1;
    void generate_updates_data_set()
    {
        status = 0;
        cout << "   rebuild::generate_updates_data_set" << endl;

        string path = "";
        // assert()
        status = 1;
    }

    void build_simple_models_and_updates()
    {
        status = 2;
        cout << "   rebuild::build_simple_models_and_updates" << endl;

        string path = "";
        // assert()
        status = 3;
    }

    void generate_training_set()
    {
        status = 4;
        cout << "   rebuild::generate_training_set" << endl;

        status = 5;
    }

    void learn_rebuild_model()
    {
        status = 6;
        cout << "   rebuild::learn_rebuild_model" << endl;

        status = 7;
    }
}

#endif